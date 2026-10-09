"""
Universal LLM Runtime — local (Ollama) or any hosted API
------------------------------------------------------------
Everything in this app (router.py, main.py) calls get_llm() and only ever
uses the returned object's .invoke()/.stream() — never anything provider-
specific — so switching providers is a config change, not a code change.
Set LLM_PROVIDER in .env to one of:

    local      (default) — Ollama, fully offline, see OLLAMA_* below
    anthropic  — Claude API, needs ANTHROPIC_API_KEY
    openai     — ChatGPT / OpenAI API, needs OPENAI_API_KEY
    google     — Gemini API, needs GOOGLE_API_KEY
    custom     — any OpenAI-compatible API (Groq, Together, Fireworks,
                 DeepSeek, OpenRouter, a self-hosted vLLM server, ...),
                 needs CUSTOM_API_KEY + CUSTOM_BASE_URL + CUSTOM_MODEL

Switching away from "local" means questions/data leave this machine (sent
to whichever provider you pick) — this project's default has always been
"local" for that reason. Only switch if you've made that tradeoff
deliberately.

IMPORTANT — this file controls WHICH MODEL answers, never WHAT IT'S ASKED
OR ALLOWED TO DO. Every prompt (router.py's _SQL_PROMPT, _CLASSIFY_PROMPT,
_SYNTHESIS_PROMPT, the column glossary, ...) is passed byte-identical to
whichever provider is configured — nothing here rewrites, shortens, or
softens them per provider. And the guardrails that actually matter for
safety (profanity blocking, out-of-scope detection, and — most importantly
— the BA-vs-admin data scoping check in router.py's _sql_is_properly_scoped)
are plain deterministic Python that runs whether or not the LLM cooperated,
not something a "helpful" or non-compliant model could bypass. Swapping
providers changes answer quality/speed/cost; it cannot loosen what the
app allows a Business Associate to see, because that check never trusted
the model in the first place.
"""
import os
import re
import threading
import time

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# ============================================================================
# CHANGING THE MODEL LATER: this default is the ONLY place the model name is
# hardcoded in Python — everywhere else (main.py, router.py) just calls
# get_llm() and never mentions a model name. To switch models you only need
# to touch:
#   1. This default (or, without touching code at all, set OLLAMA_MODEL in
#      .env — that always wins over this default, so most of the time you
#      don't need to edit this file).
#   2. `ollama pull <the new model>` on whichever machine runs Ollama.
#   3. OLLAMA_NUM_CTX below/in .env, if the new model's real usable context
#      is smaller than 8192, or you want to raise it for a bigger model.
# Nothing in router.py, database.py, or rag.py needs to change — they're all
# model-agnostic, they just call whatever get_llm() returns.
#
# qwen3.5:4b: same Qwen3.5 instruction-following/Hindi-Hinglish strengths as
# the 9b tier, at ~3.4GB (Q4_K_M) instead of ~6.6GB — chosen specifically to
# fit comfortably in VRAM-constrained GPUs (e.g. a 6GB laptop card) alongside
# OLLAMA_NUM_CTX's KV cache, where the 9b model was spilling into much slower
# CPU inference. If you have more VRAM to spare (8GB+) and want stronger
# reasoning at the cost of speed, qwen3.5:9b is the natural step up — nothing
# else in this codebase needs to change either way (see below).
# ============================================================================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

# How much GPU VRAM to plan around when auto-sizing OLLAMA_NUM_CTX below (see
# _default_ollama_num_ctx). Defaults to 6 — this project's defaults were
# originally tuned for a 6GB laptop GPU (e.g. an RTX 4050) — change this if
# you're on different hardware and want the auto-sized context to adjust
# with it, without hand-computing a new OLLAMA_NUM_CTX yourself.
OLLAMA_GPU_VRAM_GB = float(os.getenv("OLLAMA_GPU_VRAM_GB", "6"))

# Percentage of OLLAMA_GPU_VRAM_GB deliberately kept OUT of the sizing
# calculation below — i.e. never planned around as available for the model
# at all. Ollama isn't the only thing that wants this GPU: Windows' own
# desktop compositor, an open browser with hardware acceleration, or any
# other app can all be using a slice of it at the same time, and sizing the
# context as if the whole card were free for this one model leaves zero
# margin for any of that. 15% of a 6GB card is ~0.9GB reserved — raise this
# if you regularly run other GPU-heavy things alongside the chatbot, lower
# it (down to 0) if this machine is dedicated to just this app.
OLLAMA_VRAM_RESERVE_PCT = float(os.getenv("OLLAMA_VRAM_RESERVE_PCT", "15"))


def _default_ollama_num_ctx(model_name: str, vram_gb: float, reserve_pct: float = 15.0) -> int:
    """Pick a context-window default from the model's own parameter count
    (parsed out of its Ollama tag — e.g. "qwen3.5:4b" -> 4, "qwen3.5:9b" ->
    9), the configured VRAM budget, and a reserved slice of that budget kept
    off the table entirely (see OLLAMA_VRAM_RESERVE_PCT) — instead of one
    fixed context size for every model, and instead of planning as if this
    one model gets the entire GPU to itself. A bigger model's weights alone
    eat more of the remaining budget, leaving less room for the KV cache
    that OLLAMA_NUM_CTX actually sizes — push the context too high for a
    given model+VRAM combination and Ollama silently offloads layers to
    CPU, which is the exact "everything got much slower with no error"
    failure mode this project has hit before with a 9B model at a context
    size sized for a 4B one.

    This is a rough sizing heuristic (assumes roughly 0.65GB of VRAM per
    billion parameters at Q4 quantization — the common default for Ollama
    pulls), not a precise VRAM calculator, and it only ever applies when
    OLLAMA_NUM_CTX is left unset in .env — set it explicitly there to
    override this entirely. `ollama ps` remains the real source of truth
    for whether a given combination is actually landing 100% on GPU."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_name, re.I)
    params_b = float(match.group(1)) if match else 8.0  # unparseable tag -> assume mid-size, stay conservative
    usable_vram_gb = vram_gb * (1 - reserve_pct / 100.0)
    weight_gb = params_b * 0.65
    headroom_gb = max(usable_vram_gb - weight_gb, 0.5)
    if headroom_gb >= 4:
        return 8192
    if headroom_gb >= 2:
        return 4096
    if headroom_gb >= 1:
        return 2048
    return 1024


# The SQL-generation prompt carries a compact column-level data dictionary
# for all 3 tables (~2000 tokens) plus the schema (~1000 tokens) — comfortably
# under even the smallest auto-sized default below. Leave OLLAMA_NUM_CTX
# unset in .env to let _default_ollama_num_ctx() pick a size from OLLAMA_MODEL
# + OLLAMA_GPU_VRAM_GB above (e.g. qwen3.5:4b on a 6GB card -> 4096, a
# heavier qwen3.5:9b on the same 6GB card -> a much smaller 1024, since its
# larger weights leave far less VRAM for the KV cache this setting sizes).
# Set OLLAMA_NUM_CTX explicitly here to bypass the heuristic entirely.
OLLAMA_NUM_CTX = int(
    os.getenv("OLLAMA_NUM_CTX")
    or _default_ollama_num_ctx(OLLAMA_MODEL, OLLAMA_GPU_VRAM_GB, OLLAMA_VRAM_RESERVE_PCT)
)

# Ollama's default is to unload a model from VRAM after 5 minutes idle — on a
# VRAM-constrained card, reloading it back in can itself take many seconds,
# which shows up indistinguishably from "the model is just slow" on the very
# next question after a short pause. Keeping it resident for longer (default
# here: 30 minutes) trades idle VRAM usage for consistently fast responses
# during an active session. Set OLLAMA_KEEP_ALIVE=-1 in .env to never unload
# it at all while Ollama is running, or lower it if you need that VRAM back
# for something else between uses.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Optional, explicit GPU-layer override. Leave unset (the default) to let
# Ollama decide automatically — only set OLLAMA_NUM_GPU in .env if you've
# confirmed via `ollama ps` that layers are landing on CPU when they shouldn't
# be and want to force the issue; an incorrect value here can make things
# worse; None means "don't pass this parameter at all."
OLLAMA_NUM_GPU = os.getenv("OLLAMA_NUM_GPU")

# Requires langchain-ollama>=1.0 (this project pins 1.1.0). Reasoning-capable
# models (Qwen3/Qwen3.5, DeepSeek-R1, ...) can spend their ENTIRE output
# budget on internal <think>...</think> deliberation before ever writing a
# real answer — on a small OLLAMA_NUM_CTX this can mean no visible answer at
# all (see router.py's answer_stream, which has to defensively strip/handle
# exactly that). This app always wants a direct, cited answer, never exposed
# chain-of-thought, so reasoning defaults OFF here rather than leaving it at
# the model's own default. Override via OLLAMA_REASONING in .env: "true" to
# force reasoning on (slower, occasionally better on genuinely hard
# analytical questions), or "default"/"" to defer to whatever the model
# itself would normally do.
_REASONING_RAW = os.getenv("OLLAMA_REASONING", "false").strip().lower()
if _REASONING_RAW in ("false", "0", "no", "off"):
    OLLAMA_REASONING = False
elif _REASONING_RAW in ("true", "1", "yes", "on"):
    OLLAMA_REASONING = True
else:
    OLLAMA_REASONING = None  # let the model use its own default behavior


# ============================================================================
# Provider switch. "local" (default) never leaves this machine. Every other
# value sends requests to that provider's API over the internet — put the
# relevant API key in .env (see each block below for exactly which variable).
# Nothing else in the codebase needs to know which provider is active;
# get_llm() below is the only place that branches on it.
# ============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").strip().lower()

# --- Anthropic Claude (LLM_PROVIDER=anthropic) --------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
# Conservative guess at a typical free/trial-tier requests-per-minute cap —
# every account/tier is different, so check your actual plan in Anthropic's
# console and set this to match. 0 disables rate limiting for this provider.
ANTHROPIC_RATE_LIMIT_RPM = int(os.getenv("ANTHROPIC_RATE_LIMIT_RPM", "5"))

# --- OpenAI / ChatGPT (LLM_PROVIDER=openai) -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
# Same caveat as ANTHROPIC_RATE_LIMIT_RPM above — check your actual OpenAI
# usage tier/limits page and adjust. 0 disables rate limiting.
OPENAI_RATE_LIMIT_RPM = int(os.getenv("OPENAI_RATE_LIMIT_RPM", "3"))

# --- Google Gemini (LLM_PROVIDER=google) --------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
GOOGLE_MAX_TOKENS = int(os.getenv("GOOGLE_MAX_TOKENS", "4096"))
# Google's free tier is commonly more generous than Anthropic's/OpenAI's for
# flash-tier models, but still check https://ai.google.dev/gemini-api/docs/rate-limits
# for your actual key. 0 disables rate limiting.
GOOGLE_RATE_LIMIT_RPM = int(os.getenv("GOOGLE_RATE_LIMIT_RPM", "15"))

# --- Any other OpenAI-compatible API (LLM_PROVIDER=custom) --------------
# Covers hosted open-weight-model providers (Groq, Together, Fireworks,
# DeepSeek, OpenRouter, ...) and self-hosted OpenAI-compatible servers
# (vLLM, LM Studio, text-generation-webui, ...) — anything that speaks the
# same /v1/chat/completions shape as OpenAI, which the vast majority of
# these do. CUSTOM_BASE_URL is that server's API base (e.g.
# https://api.groq.com/openai/v1); CUSTOM_MODEL is whatever model name that
# specific provider expects.
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
CUSTOM_BASE_URL = os.getenv("CUSTOM_BASE_URL")
CUSTOM_MODEL = os.getenv("CUSTOM_MODEL")
CUSTOM_MAX_TOKENS = int(os.getenv("CUSTOM_MAX_TOKENS", "4096"))
# Free tiers on hosted open-weight providers vary widely (Groq's free tier is
# often 30/min for smaller models, others are much lower) — check your
# specific provider's docs/dashboard and set this to match. 0 disables
# rate limiting (e.g. for a self-hosted server with no real limit).
CUSTOM_RATE_LIMIT_RPM = int(os.getenv("CUSTOM_RATE_LIMIT_RPM", "30"))

# A single question can trigger up to ~4 LLM calls in a row (follow-up
# rewrite, intent classification, SQL generation — possibly retried once,
# see router.py's _SQL_RETRY_PROMPT — and the final synthesis), so even a
# generous per-minute quota can be exceeded by ONE question if those calls
# fire back-to-back with no spacing. _RateLimiter enforces a minimum gap
# between consecutive calls (60s / RPM) so that burst on its own can't blow
# through the quota — this is the "delay" requested for hosted providers.
# Never applied to LLM_PROVIDER=local, since Ollama has no real per-minute
# quota to protect and the delay would just slow down local usage for
# nothing.
class _RateLimiter:
    def __init__(self, requests_per_minute: int):
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._last_call_at = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            remaining = self._min_interval - (now - self._last_call_at)
            if remaining > 0:
                time.sleep(remaining)
            self._last_call_at = time.monotonic()


class _RateLimitedChatModel:
    """Transparent wrapper — exposes the same .invoke()/.stream() surface as
    the model it wraps (plus every other attribute, via __getattr__), so
    nothing outside this file needs to know a call is being paced. Delays
    just long enough before each call to stay under the configured
    requests-per-minute; does NOT catch or hide errors — if the provider's
    limit is still hit despite the delay (e.g. actual usage briefly exceeds
    the configured RPM, or the account's real limit is lower than
    configured), the error still propagates to router.py, which is what
    recognizes it as a rate-limit failure and returns the fallback answer
    described in router.py's _looks_like_rate_limit_error."""

    def __init__(self, inner, limiter: _RateLimiter):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_limiter", limiter)

    def invoke(self, *args, **kwargs):
        self._limiter.wait()
        return self._inner.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        self._limiter.wait()
        return self._inner.stream(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


_RATE_LIMIT_RPM_BY_PROVIDER = {
    "anthropic": ANTHROPIC_RATE_LIMIT_RPM,
    "openai": OPENAI_RATE_LIMIT_RPM,
    "google": GOOGLE_RATE_LIMIT_RPM,
    "custom": CUSTOM_RATE_LIMIT_RPM,
}
_rate_limiters = {
    provider: _RateLimiter(rpm) for provider, rpm in _RATE_LIMIT_RPM_BY_PROVIDER.items()
}


def _get_ollama_llm(temperature: float):
    from langchain_ollama import ChatOllama

    kwargs = dict(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=temperature,
        num_ctx=OLLAMA_NUM_CTX,
        keep_alive=OLLAMA_KEEP_ALIVE,
        reasoning=OLLAMA_REASONING,
    )
    if OLLAMA_NUM_GPU:
        kwargs["num_gpu"] = int(OLLAMA_NUM_GPU)
    return ChatOllama(**kwargs)


def _get_anthropic_llm(temperature: float):
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set in .env."
        )
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "LLM_PROVIDER=anthropic requires the langchain-anthropic package — "
            "run: pip install -r requirements.txt"
        ) from exc

    return ChatAnthropic(
        api_key=ANTHROPIC_API_KEY,
        model=ANTHROPIC_MODEL,
        temperature=temperature,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        # See the module docstring + router.py's _extract_text: this app
        # always wants a direct, plain-text, cited answer. Extended
        # thinking is on by default for several current Claude models,
        # and left on, response content can arrive as a list of blocks
        # (thinking + text) instead of a plain string — disabled here
        # unconditionally rather than exposed as a setting, since this
        # app's synthesis step has no use for exposed chain-of-thought.
        thinking={"type": "disabled"},
    )


def _get_openai_llm(temperature: float):
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set in .env."
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "LLM_PROVIDER=openai requires the langchain-openai package — "
            "run: pip install -r requirements.txt"
        ) from exc

    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=temperature,
        max_tokens=OPENAI_MAX_TOKENS,
    )


def _get_google_llm(temperature: float):
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=google requires GOOGLE_API_KEY to be set in .env."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "LLM_PROVIDER=google requires the langchain-google-genai package — "
            "run: pip install -r requirements.txt"
        ) from exc

    return ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model=GOOGLE_MODEL,
        temperature=temperature,
        max_output_tokens=GOOGLE_MAX_TOKENS,
    )


def _get_custom_llm(temperature: float):
    missing = [
        name
        for name, value in (
            ("CUSTOM_API_KEY", CUSTOM_API_KEY),
            ("CUSTOM_BASE_URL", CUSTOM_BASE_URL),
            ("CUSTOM_MODEL", CUSTOM_MODEL),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "LLM_PROVIDER=custom requires " + ", ".join(missing) + " to be set in .env."
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "LLM_PROVIDER=custom requires the langchain-openai package (used for "
            "any OpenAI-compatible API, not just OpenAI itself) — "
            "run: pip install -r requirements.txt"
        ) from exc

    return ChatOpenAI(
        api_key=CUSTOM_API_KEY,
        base_url=CUSTOM_BASE_URL,
        model=CUSTOM_MODEL,
        temperature=temperature,
        max_tokens=CUSTOM_MAX_TOKENS,
    )


_PROVIDERS = {
    "local": _get_ollama_llm,
    "anthropic": _get_anthropic_llm,
    "openai": _get_openai_llm,
    "google": _get_google_llm,
    "custom": _get_custom_llm,
}


def get_llm(temperature: float = 0.1):
    try:
        factory = _PROVIDERS[LLM_PROVIDER]
    except KeyError:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' in .env — must be one of: "
            + ", ".join(_PROVIDERS)
        )
    model = factory(temperature)
    limiter = _rate_limiters.get(LLM_PROVIDER)
    if limiter is not None:
        model = _RateLimitedChatModel(model, limiter)
    return model
