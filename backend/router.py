"""
Agentic Multi-Engine Router & Synthesis (FR-4) + Citation Framework (FR-5)
----------------------------------------------------------------------------
Classifies every incoming question into one of:
  STRUCTURED    -> DuckDB relational warehouse (clients, BAs, brokerage, portfolios)
  UNSTRUCTURED  -> ChromaDB product-leaflet vector store (strategy, fees, benchmarks)
  EXTERNAL      -> live market data
  CROSS_DOMAIN  -> a blend of the above
  OUT_OF_SCOPE  -> anything unrelated to this tool's job (see guardrails below)

...then asks the local LLM to synthesize a single, coherent answer from
whatever evidence was retrieved, and appends explicit citation chips
matching exactly one of:
  [Source: CSV::{filename} | Table: {table_name}]
  [Source: PDF::{filename}, Page {number}]
  [Source: API::{service_name} | Timestamp: {utc_time}]

Design principle throughout this file: local LLMs (especially smaller/
quantized ones) are unreliable at following subtle multi-part instructions
("only use history if it's actually relevant", "pick exactly one of these 5
labels", "obey these 6 simultaneous style rules"). Wherever a decision can be
made deterministically in Python instead of trusted to the model, it is —
the model is reserved for the things only it can do (SQL generation, natural
-language synthesis). See _looks_like_followup, _looks_out_of_scope, and
_contains_profanity.
"""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import llm as llm_module
from database import StructuredDataEngine
from external import get_quote
from llm import get_llm
from rag import UnstructuredDataEngine

logger = logging.getLogger("mo_insight.router")

_TICKER_HINT = re.compile(r"\b(price|quote|ltp|trading at|live)\b", re.I)

# How many past (question, answer) turns to keep/feed back per conversation.
# Kept small on purpose: every extra turn adds tokens to every prompt below.
MAX_HISTORY_TURNS = 6

# How many times query_structured() will regenerate SQL after a hallucinated
# column/table name before giving up (see _looks_like_schema_error). 2 means
# "one real attempt, one self-correction" — the default. Set
# SQL_RETRY_ATTEMPTS=1 in .env to disable the retry entirely and consume
# fewer requests per structured question (at the cost of more "couldn't find
# that column" answers instead of a self-corrected one); raising it above 2
# rarely helps, since a model that's wrong twice with the real schema and
# the exact error in front of it is unlikely to get it right on a third try.
SQL_RETRY_ATTEMPTS = max(1, int(os.getenv("SQL_RETRY_ATTEMPTS", "2")))

# ============================================================================
# Guardrail 1: abusive/vulgar language — blocked deterministically in Python,
# BEFORE anything is sent to the LLM. English + Hindi/Hinglish (the actual mix
# BAs type day to day at an Indian firm), matched on whole words only — not
# an exhaustive content-moderation system, and it won't catch creative
# misspellings/obfuscation. Extend the tuple below if a specific term keeps
# slipping through.
#
# Coverage note: this deliberately does NOT reach into Marathi, Gujarati,
# Punjabi, Tamil, Telugu, or Bengali profanity — getting those wrong in
# either direction (missing real abuse, or blocking an innocuous word I
# misjudged) is worse than leaving them out. If staff commonly type in one of
# those languages, send the specific terms and they can be added with
# confidence instead of guessed at here.
# ============================================================================
_PROFANITY_WORDS = (
    "fuck", "fucking", "fucker", "motherfucker", "shit", "bullshit",
    "asshole", "bastard", "bitch", "bitches", "cunt", "dick", "dickhead",
    "piss", "pissed", "slut", "whore", "prick", "wanker",
    # Hindi/Hinglish terms are listed with their common inflected variants
    # explicitly (chutiya/chutiye/chutiyo, gandu/gaandu, ...) rather than as a
    # single base form — Hindi vulgar terms inflect a lot more than their
    # English equivalents, and a bare base-form match misses siblings like
    # "chutiye" (vocative) that show up just as often as "chutiya" (neutral).
    "chutiya", "chutiye", "chutiyo",
    "madarchod", "behenchod", "bhenchod", "banchod",
    "bhosdi", "bhosdike", "bhosadi", "bhosdiwala",
    "gandu", "gaandu", "gand", "gaand",
    "randi", "raand", "harami", "haraami", "haramzada", "haramzadi",
    "lund", "lauda", "laude", "lawde", "loda",
    "chod", "chodu", "chinaal", "chinal", "bhadwa", "bhadwe",
    "tatti", "kamina", "kaminey", "suar",
)
_PROFANITY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _PROFANITY_WORDS) + r")\b", re.I
)


def _contains_profanity(text: str) -> bool:
    return bool(_PROFANITY_PATTERN.search(text))


_PROFANITY_MESSAGE = (
    "Let's keep this professional, please — I can't engage with language like that. "
    "Feel free to rephrase your question about a client, a Business Associate's brokerage, "
    "or a Model Portfolio, and I'll help."
)

# ============================================================================
# Guardrail 2: staying in scope. The LLM classifier (see _CLASSIFY_PROMPT)
# has an OUT_OF_SCOPE label too, but that alone isn't reliable enough — a
# smaller/weaker local model frequently mis-picks STRUCTURED/CROSS_DOMAIN out
# of a bias toward being "helpful" even for something like "write me a
# palindrome function". This deterministic pre-check catches the common,
# obvious cases (coding help, creative writing, translation, general trivia,
# math homework) before the LLM ever sees the question at all — the LLM
# classifier remains a second layer for subtler cases this misses.
# ============================================================================
_OUT_OF_SCOPE_PATTERNS = (
    r"\bwrite\s+(me\s+)?(a\s+|an\s+)?(python|java|javascript|typescript|c\+\+|c#|sql|html|css|"
    r"code|function|program|script|algorithm|poem|essay|story|song|lyric|joke|recipe)\b",
    r"\b(palindrome|fibonacci|factorial|bubble sort|quicksort|binary search|leetcode|hackerrank)\b",
    r"\brecipe\b", r"\bhow (do|to) (i |you )?(cook|bake|make)\b.*\b(dish|cake|curry|biryani)\b",
    r"\btranslate\b.*\b(to|into)\s+(french|spanish|german|hindi|marathi|tamil|telugu|english)\b",
    r"\bwhat('s| is) the capital of\b",
    r"\bwho (is|was) the (president|prime minister|ceo of google|ceo of apple)\b",
    r"\b(solve for x|derivative of|integral of|quadratic equation)\b",
    r"\b(your|the) (system prompt|internal (instructions|configuration)|underlying model)\b",
    r"\bignore (all |your )?(previous|prior|above) instructions\b",
    r"\btell me a joke\b",
    r"\bwrite (a |an )?(email|letter|cover letter|resume|cv)\b(?!.*(client|portfolio|ba|brokerage))",
)
_OUT_OF_SCOPE_PATTERN = re.compile("|".join(_OUT_OF_SCOPE_PATTERNS), re.I)


def _looks_out_of_scope(question: str) -> bool:
    return bool(_OUT_OF_SCOPE_PATTERN.search(question))


_OUT_OF_SCOPE_MESSAGE = (
    "I'm built specifically to help with Motilal Oswal client portfolios, Business Associate "
    "brokerage, and Model Portfolio products — I'm not trained for that. Try asking me about a "
    "client's holdings, a BA's brokerage, or a product like MO Signia or ASCEND."
)

# ============================================================================
# Guardrail 3: conversation history should only influence an answer when the
# question actually depends on it. Left unconditional, a smaller local model
# will often blend in irrelevant prior context even when told not to — asking
# about client A right after asking about client B can come back answered as
# if it were still about client B. This heuristic decides whether history is
# even offered to the model at all: short questions and ones with referential
# words ("it", "that", "also", "the other one", ...) are very likely
# follow-ups; long, fully self-contained questions almost never are.
# Biased toward over-including on short questions (mild cost: a little
# unnecessary context) rather than under-including on genuine follow-ups
# (real cost: "its benchmark?" left unresolved is useless on its own).
# ============================================================================
_FOLLOWUP_HINT = re.compile(
    r"\b(it|its|it's|these|those|they|them|their|theirs|"
    r"he|she|his|her|hers|same|above|previous|earlier|also|too|"
    r"one|ones|other|another|compared|instead)\b|"
    # "this"/"that" are only treated as referential when NOT followed by a
    # calendar/generic noun — "this year"/"that quarter" are normal temporal
    # phrases, not a reference back to something in the conversation. "this
    # client"/"that product" etc. are left as real referential hits.
    r"\b(this|that)\b(?!\s+(year|month|week|quarter|time|period|dataset|table|"
    r"data|system|tool|app|information|moment|point))",
    re.I,
)
_FOLLOWUP_WORD_LIMIT = 6


def _looks_like_followup(question: str) -> bool:
    if len(question.split()) <= _FOLLOWUP_WORD_LIMIT:
        return True
    return bool(_FOLLOWUP_HINT.search(question))


# ============================================================================
# Guardrail 4 (data grounding): a column-level data dictionary for the 3
# loaded CSVs, fed into the text-to-SQL step, documenting what every real
# column means plus the informal business terms (AUM, POA, MTD/YTD
# brokerage, ...) that don't literally match a column name. This documents
# column SEMANTICS only — it never hardcodes a data value, so it doesn't
# violate the project's zero-hardcoding rule. Kept as compact as possible
# while still covering every column: local models degrade noticeably once a
# prompt gets too long, and this reference block is added to every single
# SQL-generation call, so every character here has a real ongoing cost.
# ============================================================================
_GLOSSARY = """Column reference for the 4 tables below. Also covers common BA jargon that doesn't
match a column name literally: AUM / "portfolio value" = current_portfolio.CurrentValue; POA =
IsPOA / poa_status; "auto consent" = IsAllowedConsent / isallowedconsent; brokerage revenue =
brokerage.brokerageamount; MTD/YTD brokerage = SUM(brokerageamount) filtered by sauda_date this
month/year (no dedicated column); BA = BA1200code / BAName / ba1200code; RM = rm_name /
FinalAdvisorName; RA / "research analyst" (in the context of a model portfolio PRODUCT, not a
client's BA/RM) = model_portfolio_comparison.research_analyst. "DP value", "trade error", "IAP
transfer", "PR report", a Baf200Code distinct from BA1200code: NOT present in any of these 4
tables — say so plainly if asked, never guess or substitute an unrelated column.

TABLE imp_view — one row per IMP investment/withdrawal transaction:
clientcode=client id; productname/productcode=IMP product; paymentmode=how paid; investmentmode:
N=new investment, A=top-up into an existing holding; investmentamount=amount this transaction;
FreshInvestment=portion that's genuinely new money (not switched from another product);
date_of_investment=transaction date; subscriptiondate=original subscription date (only differs
from date_of_investment when investmentmode=A); activeto=subscription active-until date;
isallowedconsent=standing consent letting the BA/advisor trade for the client without asking each
time; poa_status=Power of Attorney granted; rm_empcode/rm_name=assigned RM's code/name;
tgs_rm_empcode/tgs_rm_name=assigned RM's code/name from the TGS (Trade Guide Signal) desk;
currentbranchcode=branch of the assigned BA; currentsubbrokercode=sub-broker/advisor under that BA;
ba1200code=assigned BA's code; withdrawaldate/withdrawalamount=withdrawal date/amount;
migrationstatus=migrated from IAP to IMP; migrationinvstdate/migrationwdrldate=migration
investment/withdrawal dates; migrationwdrldiff=investment remaining after a migration withdrawal;
migrationwdrlflag=withdrawal happened during migration; fmigrationconfirm=migration confirmed
complete; devicetype=device used (mobile/tablet/web); reinvestflag=reinvested from one IMP product
straight into another; TransactionStatus=transaction status (PENDING/SUCCESS/COMPLETED);
Remarks=free-text notes; FinalAdvisoryType: DIRECT=Motilal Oswal's own advisor,
INDIRECT=external/independent advisor.

TABLE current_portfolio — one row per client per product currently held:
ClientCode/ClientName=client id/name; ProductCode/ProductName/ProductCategory=product held;
PaymentMode=how paid; Dateofsubscription/ActiveTo=subscribed date/active-until date;
NetInvestment=net amount invested; Portfolio=portfolio holdings value; CashRemaining=uninvested
cash in account; Dividend=quarterly dividend received; CurrentValue=current portfolio market value
(= "AUM"); ProfitValue=profit made; perCity/perState=client's home city/state;
branchcode/subbrokercode=branch/sub-broker code; BA1200code/BAName=BA code/name; IsPOA=Power of
Attorney on file (Y/N); IsAllowedConsent=standing consent given (Y/N); IsAMCWaived=Annual
Maintenance Charge waived; PortfolioGeneratedStatus=portfolio generated yet (Y/N);
PartialWithdrawalStatus=a partial (not full) withdrawal occurred; ProfitTreatmentMode=how profit
is handled, e.g. reinvest vs. payout (specific values not fully documented); Direct_AdvisoryType/
Direct_AdvisorCode=Direct or Indirect advisory + that advisor's code; BAFamilyUCIDCond=a BA-family/
UCID condition flag (not fully documented); BA_Zone/BA_State/BA_City=BA's current
zone/state/city; BA_CityAsPerBackOffice=BA's city per back office (may differ from BA_City);
BA_AdvisorId/BA_Advisor=BA's advisor id/name; BA_AdvisoryTL=that advisory team's team leader;
FinalAdvisoryType/FinalRegion/FinalAdvisorCode/FinalAdvisorName=the resolved advisory
type/region/advisor code/name actually responsible for this client; FAdv_Email/FAdv_Profile/
FAdv_Profile1/FAdv_Department/FAdv_DESIGNATION=that advisor's contact/HR details; FAdv_Status=
advisor active/inactive; FAdv_Location/FAdv_CostCenter=advisor's location/cost-center;
FAdv_1stRPTMNGRCODE/Fadv_1stRPTMNGRNAME=advisor's 1st-level reporting manager code/name;
FAdv_2ndRPTMNGRCODE/FAdv_2ndRPTMNGRNAME=2nd-level reporting manager code/name;
FAdv_HODCODE/FAdv_HODNAME=Head of Department code/name; TGSTL1Code/TGSTL2Code=TGS team's
level-1/level-2 team leader codes; TGSVHCode=TGS team's vertical head code.

TABLE brokerage — one row per trade, with brokerage/revenue-sharing splits:
company=broker (MOSL=Motilal Oswal); business/businesslineno=business unit/channel + its id;
clientcode=client id; sauda_date=trade date; symbol/scripname=ticker/company name; series=exchange
segment (EQ=equity cash); settlementno=settlement cycle id; contractno=contract note number;
orderno/tradeno=exchange order id/trade execution id; settlementtype=settlement framework
(N=normal); exchangetradetype=execution mode (e.g. REGULAR); transactiontype: B=buy, S=sell;
exchange=NSE/BSE; terminaltype/terminalcode=trading platform (e.g. ODIN)/terminal id;
tradequantity=shares traded; tradelot=lot size (usually 1); tradeprice=executed price/share;
netrate_with_all_charges=effective price/share after brokerage+duties; volume=trade value
(qty x price); volume_marketrate=that volume at current market rate; revenuetype=income category
(e.g. BROKERAGE); revenuesubtype1/revenuesubtype2=revenue stream/execution style;
productclass/productsubclass=asset class/sub-class; businesstype=holding nature (e.g. DELIVERY);
internalproducttype=internal strategy code (e.g. MODEL_PORTFOLIO); segment/finalsegment/
dimsubsegment=market segment/consolidated segment/reporting sub-segment; stampdutyval=state stamp
duty; turnoverchargeval=exchange turnover fee; sebichargeval=SEBI charge; transactiontaxval=STT;
transactioncahrgeval=exchange/clearing processing fee; otherval=other levies;
cgst/sgst/igst/ugst=GST components on brokerage; floodcessvalue=disaster/state cess;
auctionpenaltycharges=trade-default/auction penalty; brokerageamount=total gross brokerage
charged; branchcode/branch_glcode=branch code/its GL account; branch_brokerage_sharepercent/
branch_brokerage_shareamount=branch's %/amount share; subbrokercode/subbroker_glcode=BA/sub-broker
code/GL account; subbroker_brokerage_sharepercent/subbroker_brokerage_shareamount=BA's
commission %/amount; referalbacode/referalglcode=referral partner's BA code/GL account;
referal_brokerage_sharepercent/referal_brokerage_shareamount=referral partner's %/amount share;
moslbrokeragesharingamount=amount retained by MOSL itself; netrevenue=revenue after partner
splits; wm_sharingamount=share to Wealth Management desks; cadcommodity_sharingamount=share to
Commodity/Forex desk; branch_sharingamount=share to branch management;
institution_sharingamount=share to institutional desks; acctualnetrevenue=final net revenue after
every payout; nri_shareamount=share to NRI desk; algo_shareamount=share to algo-trading desks;
advisorcode/rmcode=primary advisor/RM code; advisorcode_equity=advisor id for equity advisory;
advisortype=PRIMARY/SECONDARY role; advisorysharingtype=sharing policy (e.g. STANDARD);
primaryadvisorsharing_perc/primaryadvisorshare_grossrevenue=primary RM's %/amount share;
supportadvisorsharing_perc/supportadvisorshare_grossrevenue=support advisor's %/amount share;
nrideskrevenue=revenue attributed to NRI desk advisors; business_head=exec head of the segment;
national_owner_name=national sales/channel manager; region/pnl_region=geographic zone/P&L region;
channel_suncode=accounting channel code; branch_code_3way/subbroker_code_3way/finalsegment_3way=
3-way reconciliation codes; dr_clientcode=debit-ledger client reference; mis_code/mis_ba_code=MIS
tracking code for client/BA; clientcategory=RETAIL/HNI; nri_location=NRI client's overseas
location; Logdate=row's database log timestamp.

TABLE model_portfolio_comparison — one row per Model Portfolio PRODUCT (13 products), a firm-wide
reference/comparison sheet, NOT client or BA data — never subject to the BA scoping below:
product_name=the model portfolio's name (e.g. "MO Signature", "APEX - Active NS Idustry Champ");
research_analyst=the research desk/analyst house behind it (e.g. "MO Research", "Teji Mandi",
"Narnolia"); risk_category=Conservative/Moderate/Aggressive; investing_style=Fundamental/Quant/
Technical/Quantamental; rebal_frequency=how often it's rebalanced (Weekly/Monthly/Quarterly/"Need
Basis"=no fixed schedule); market_cap=Large Cap/Flexi Cap universe it invests in; num_stocks=typical
number of stocks held (a single number or a range like "20-25"); top_sectors=its top 2-3 sector
exposures, semicolon-separated (e.g. "1) Banks; 2) Auto; 3) IT"; "Dynamic" means sector exposure
isn't fixed); top_stocks=its top 5 current stock holdings, semicolon-separated in the same "1) ...;
2) ..." format ("Dynamic" = not fixed); recommended_time_frame=suggested minimum holding horizon
(e.g. "3 - 5 years"); benchmark=the index it's benchmarked against (e.g. "Nifty 200", "Nifty 200
Momentum 30"); brokerage_rate=the brokerage rate charged on this product as a decimal fraction
(0.01 = 1%, 0.002 = 0.2% — NOT the same figure as brokerage.brokerageamount, which is a client's
actual rupee brokerage paid on a trade, not a product's rate).
"""

# ============================================================================
# Guardrail 5 (access control): a Business Associate must only ever see their
# own clients' data; only an Administrator sees everything. This CANNOT be
# left to the LLM to remember to add a "WHERE ba_code = ..." filter — that's
# exactly the kind of instruction a local model will drop under load, and
# unlike the other guardrails here, getting this one wrong means leaking a
# different BA's client data, not just a wrong or off-topic answer. So it's
# enforced twice: the SQL prompt is told which BA is asking and given the
# exact column to filter on, AND the generated SQL is verified afterward to
# actually contain that filter before it's ever executed. If verification
# fails, the query is refused outright (fail closed) rather than run
# unscoped or attempt to rewrite/patch the SQL — patching arbitrary
# LLM-generated SQL to inject a missing WHERE clause safely isn't reliably
# possible (aggregates in particular: once `SUM(brokerageamount)` has run
# over every BA's rows, no amount of filtering the RESULT afterward un-mixes
# other BAs' numbers back out of that total).
# ============================================================================
_BA_SCOPE_COLUMNS = {
    "current_portfolio": "BA1200code",
    "imp_view": "ba1200code",
    "brokerage": "referalbacode",
}

_SCOPE_DENIED_MESSAGE = (
    "As a Business Associate you can only see your own clients' data — I wasn't able to safely "
    "scope that query to just your BA code. Try rephrasing around your own clients specifically "
    "(e.g. \"my clients' total AUM\" rather than \"total AUM\"), or ask an administrator for a "
    "firm-wide view."
)


def _scope_instruction(ba_code: Optional[str]) -> str:
    if not ba_code:
        return ""
    columns = "; ".join(f"{table}.{col}" for table, col in _BA_SCOPE_COLUMNS.items())
    return (
        f"\nACCESS CONTROL — MANDATORY: this question is being asked by Business Associate "
        f"'{ba_code}'. If your query touches current_portfolio, imp_view, or brokerage, it MUST "
        f"filter to this BA's own rows using that table's BA column ({columns}), e.g. "
        f"WHERE BA1200code = '{ba_code}'. Never return another BA's data. If the question asks "
        f"about a different BA or 'all BAs'/'the firm', answer only for '{ba_code}' and note that "
        f"a firm-wide view needs administrator access.\n"
    )


def _sql_is_properly_scoped(sql: str, ba_code: Optional[str]) -> bool:
    """Deterministic verification that a generated query actually restricts
    itself to the asking BA's own rows, for every scopable table it touches.

    Table detection here is DELIBERATELY independent of the FROM/JOIN regex
    used elsewhere for hallucination-checking (see query_structured) — that
    regex only matches a bare identifier right after FROM/JOIN, so a query
    that quotes the table name (`FROM "current_portfolio"`, `` FROM `brokerage` ``,
    a schema-qualified `main.current_portfolio`, ...) would never show up in
    its "referenced tables" set. Relying on that set here meant a quoted
    table name made this check see ZERO scopable tables touched and pass
    with nothing enforced — the BA-filter requirement silently skipped and
    the query ran unscoped. A stronger model that happens to write cleaner,
    more standard-compliant SQL (i.e. one that quotes identifiers) is *more*
    likely to trigger this, not less, so this can't be treated as an
    unlikely edge case. Fixed by searching the raw SQL text directly for each
    protected table name as a whole word — quoting/backticks/schema
    prefixes don't hide a plain word-boundary match. An admin (ba_code
    falsy) always passes — this check only exists for BA-scoped sessions.
    """
    if not ba_code:
        return True
    escaped_code = re.escape(ba_code)
    for table, column in _BA_SCOPE_COLUMNS.items():
        if not re.search(rf"\b{re.escape(table)}\b", sql, re.I):
            continue  # this scopable table isn't mentioned in the SQL at all
        pattern = re.compile(rf"\b{re.escape(column)}\b\s*=\s*'{escaped_code}'", re.I)
        if not pattern.search(sql):
            return False
    return True


_CLASSIFY_PROMPT = """You are the intent router for MO-Insight, a wealth-management copilot used
internally by Motilal Oswal business associates and wealth managers. Classify the user's question
into exactly ONE label:
STRUCTURED   - about clients, brokerage/revenue, business associates, portfolio holdings/valuations, transactions (tabular/relational data); ALSO ANY question about model portfolio PRODUCTS' comparison-sheet fields — RA/research analyst, risk category, investing style, rebal frequency, market cap, number of stocks, top sectors, top holdings/stocks, recommended time frame, benchmark, brokerage rate — whether asking about one product or comparing/filtering/sorting several, e.g. "which product has the lowest brokerage", "list all Aggressive products", "who is the RA for MO Signature", "what are APEX's top holdings"
UNSTRUCTURED - about a model portfolio PRODUCT's narrative strategy, philosophy, minimum investment, or other brochure/leaflet content NOT covered by the tabular metrics above
EXTERNAL     - about a live/current stock price or market quote
CROSS_DOMAIN - needs a mix of the above (e.g. "compare this client's actual holding against the product brochure")
OUT_OF_SCOPE - anything else (general knowledge, coding help, writing help, or any topic unrelated
               to Motilal Oswal clients/portfolios/brokerage/products)

Respond with only the single label, nothing else.

Question: {question}
Label:"""

_CONTEXTUALIZE_PROMPT = """Given the recent conversation below and a new follow-up question,
rewrite the follow-up as a fully standalone question that makes complete sense on its own —
resolve pronouns and implicit references ("it", "that portfolio", "what about X", "and hers?")
using the conversation. If the follow-up already stands on its own, return it unchanged.
Output ONLY the rewritten question, nothing else — no quotes, no commentary.

Recent conversation:
{history}

Follow-up question: {question}
Standalone question:"""

# Fuses contextualize() + classify() into a single call for follow-up
# questions — the two most common consecutive LLM calls this router makes,
# since a follow-up always needs both a rewrite AND a classification of
# that rewrite. Cuts LLM calls per follow-up question from 4 to 3 (also
# meaningful for hosted API providers with a free-tier rate limit — see
# llm.py's rate limiter). A plain two-line output format is used instead of
# JSON specifically because it's easier for a smaller local model to
# produce reliably; see contextualize_and_classify()'s parsing, which falls
# back to the original two separate calls if this doesn't parse cleanly, so
# a malformed response costs a retry, not a wrong answer.
_CONTEXTUALIZE_AND_CLASSIFY_PROMPT = """Given the recent conversation below and a new follow-up
question, do TWO things:

1. Rewrite the follow-up as a fully standalone question that makes complete sense on its own —
resolve pronouns and implicit references ("it", "that portfolio", "what about X", "and hers?")
using the conversation. If it already stands on its own, repeat it unchanged.

2. Classify that standalone question into exactly ONE label:
STRUCTURED   - about clients, brokerage/revenue, business associates, portfolio holdings/valuations, transactions (tabular/relational data)
UNSTRUCTURED - about a model portfolio PRODUCT's features, strategy, philosophy, benchmark, fees, minimum investment (brochure/leaflet content)
EXTERNAL     - about a live/current stock price or market quote
CROSS_DOMAIN - needs a mix of the above (e.g. "compare this client's actual holding against the product brochure")
OUT_OF_SCOPE - anything else (general knowledge, coding help, writing help, or any topic unrelated
               to Motilal Oswal clients/portfolios/brokerage/products)

Respond in EXACTLY this two-line format, nothing else — no quotes, no commentary, no markdown:
STANDALONE: <the rewritten standalone question>
INTENT: <the single label>

Recent conversation:
{history}

Follow-up question: {question}"""

_SQL_PROMPT = """Write ONE read-only DuckDB SQL SELECT statement that answers the question below,
using ONLY the tables/columns in the schema. Never invent a column. Never write more than one
statement or any write/DDL statement (INSERT/UPDATE/DELETE/DROP/ALTER/etc.) — if the question asks
to modify, delete, or create data instead of look it up, output exactly NO_QUERY instead of SQL.
Output only the raw SQL (or NO_QUERY): no markdown fences, no commentary, no explanation.
{scope_instruction}
{glossary}

Example query SHAPES for common question types (column/table names below are illustrative only —
always use the REAL schema given further down, never these names verbatim):
- "Which BA earned the most brokerage?" -> a GROUP BY + aggregate + ORDER BY ... DESC + LIMIT 1,
  e.g. SELECT referalbacode, SUM(brokerageamount) AS total FROM brokerage GROUP BY referalbacode
  ORDER BY total DESC LIMIT 1
- "What is client 1000101's current portfolio value?" -> a direct WHERE filter, e.g.
  SELECT CurrentValue FROM current_portfolio WHERE ClientCode = 1000101
- "How many clients does BA12001 have?" -> COUNT(DISTINCT ...) with a WHERE filter, e.g.
  SELECT COUNT(DISTINCT ClientCode) FROM current_portfolio WHERE BA1200code = 'BA12001'
These are here because a "top N by some aggregate" question needs GROUP BY + ORDER BY + LIMIT
together, which is easy to leave incomplete (e.g. missing the GROUP BY, so the aggregate applies
across ALL rows instead of per-group, or missing the ORDER BY/LIMIT and returning every group
instead of just the top one) if you write it from scratch.

Schema:
{schema}

Question: {question}
SQL:"""

_SQL_RETRY_PROMPT = """Your previous DuckDB SQL query failed because it referenced a column or
table that doesn't actually exist. This is the ONLY thing wrong with it — do not change anything
else about the query's approach.

Previous query:
{previous_sql}

Database error:
{error}

Rewrite the query using ONLY the exact table/column names listed in the schema below — every name
is case-sensitive and must be copied exactly as spelled there. Never invent or guess a name close
to what you think it should be. If, after checking the schema, the information genuinely isn't
available under any column, output exactly NO_QUERY instead of guessing. Output only the corrected
raw SQL (or NO_QUERY): no markdown fences, no commentary, no explanation.
{scope_instruction}
{glossary}

Schema:
{schema}

Question: {question}
Corrected SQL:"""

_SYNTHESIS_PROMPT = """You are MO-Insight, an internal advisor for Motilal Oswal business
associates and wealth managers. Answer the question below using ONLY the information provided.
Never invent a number, client, product, or advisor name that isn't in it. State the answer
directly, as fact — don't say "according to the documents/evidence/data" or similar; the source is
already shown to the user separately.

{style}

If the information isn't enough to answer confidently, say so in one short, plain sentence instead
of guessing. Give facts only, never personal investment advice — if asked "should I invest in X",
state the facts you have and suggest speaking with a licensed advisor instead of recommending a
course of action. Don't append citation tags yourself, the UI shows those separately.
{history_block}
Question: {question}

Information:
{evidence}

Answer:"""


def _looks_like_schema_error(exc: Exception) -> bool:
    """True when a DuckDB execution failure is specifically about a
    hallucinated column/table name (a "Binder Error" / "Catalog Error" —
    DuckDB's own authority on what actually exists in the loaded data),
    as opposed to some other failure (a locked file, a genuine bug, a
    write-guard rejection). Used to decide whether it's worth asking the
    model to fix and retry, versus giving up immediately — retrying a
    non-schema failure would just burn a second LLM call for no benefit."""
    text = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    if "binder" in exc_name or "catalog" in exc_name:
        return True
    return any(
        phrase in text
        for phrase in ("binder error", "catalog error", "column", "does not exist", "not found")
    )


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    """True when an LLM call failed because a hosted API provider's rate
    limit or quota was hit (HTTP 429, or the provider's own "rate limit"/
    "quota"/"resource exhausted" wording) rather than a genuine bug or
    connectivity problem. llm.py already paces requests to hosted providers
    to stay under their configured RPM (see _RateLimiter there) — this
    covers the case where the limit is still hit anyway (actual traffic
    briefly exceeding the configured RPM, or the account's real limit being
    lower than what's configured), so the fallback answer can say plainly
    what happened instead of looking like a generic crash."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    exc_name = type(exc).__name__.lower()
    if "ratelimit" in exc_name or "resourceexhausted" in exc_name:
        return True
    text = str(exc).lower()
    return any(
        phrase in text
        for phrase in (
            "rate limit", "rate_limit", "429", "quota",
            "resource exhausted", "resource_exhausted", "too many requests",
        )
    )


def _llm_unavailable_message(evidence: str, exc: Exception) -> str:
    """Builds the fallback answer used whenever the synthesis LLM call
    itself fails (as opposed to failing to retrieve evidence, which has its
    own handling upstream). Distinguishes a rate-limit failure — expected
    and temporary, with an actionable next step — from every other failure
    mode, and names the actual configured provider rather than assuming
    Ollama, since this app can run on any of local/anthropic/openai/google/
    custom (see llm.py)."""
    provider = llm_module.LLM_PROVIDER
    if _looks_like_rate_limit_error(exc):
        if provider == "local":
            reason = (
                "The local Ollama runtime is reporting it's overloaded/rate-limited — "
                "unusual for a local model, but possible if it's serving other requests too."
            )
        else:
            reason = (
                f"This request hit the '{provider}' API's rate limit, even after pacing "
                "requests to stay under it. That provider needs a short break before "
                "accepting more requests — try again in a minute, or switch LLM_PROVIDER "
                "to \"local\" in .env to keep working without waiting."
            )
    elif provider == "local":
        reason = "I couldn't reach the local LLM runtime (is Ollama running?)."
    else:
        reason = f"I couldn't reach the configured LLM provider ('{provider}') — {exc}"
    return f"{reason} Here is the raw evidence I retrieved:\n\n{evidence}"


def _extract_text(content) -> str:
    """Normalize a chat-model .content value to plain text, regardless of
    which provider produced it (see llm.py — this app can run on Ollama,
    Anthropic, OpenAI, Google, or any OpenAI-compatible API). Most
    providers return a plain string. Providers with extended-thinking
    support (Anthropic, some Gemini models, ...) can instead return a LIST
    of content blocks — separate thinking/reasoning/text entries — if
    thinking is ever left enabled for that provider. Every prompt in this
    file expects plain text back, so list-shaped content is flattened here
    (dropping reasoning/thinking blocks, keeping everything else that
    carries a "text" field) rather than left to break str.strip()/string
    concatenation downstream with a TypeError. Block shape isn't identical
    across providers, so this matches loosely (any "text" key) rather than
    hardcoding one provider's exact schema."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("thinking", "redacted_thinking", "reasoning"):
                    continue
                if "text" in block:
                    parts.append(block.get("text") or "")
        return "".join(parts)
    return str(content)


def _format_history(history: Optional[List[Tuple[str, str]]]) -> str:
    if not history:
        return "(none)"
    lines: List[str] = []
    for q, a in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def _history_block(history: List[Tuple[str, str]], is_followup: bool) -> str:
    """Only surfaced to the synthesis prompt when the current question was
    heuristically judged a likely follow-up (see _looks_like_followup) — for
    a fresh, unrelated question this returns "" so there is nothing for the
    model to (incorrectly) blend in from a previous topic."""
    if not is_followup or not history:
        return ""
    return (
        "\nRecent conversation, for continuity/tone only — this is a follow-up, so it may be "
        f"relevant:\n{_format_history(history)}\n"
    )


def _style_hint(intent: str) -> str:
    """Guardrail 5: verbosity should match the question type — thorough prose
    for product/strategy questions, terse figures-only for data lookups."""
    if intent == "UNSTRUCTURED":
        return (
            "This is a product/strategy question: give a thorough answer covering every relevant "
            "detail available below (strategy, benchmark, fees, minimum investment, risk profile, "
            "features — whichever apply). Don't just give a one-line answer."
        )
    if intent == "STRUCTURED":
        return (
            "This is a data lookup: answer with only the specific figure(s)/fact(s) asked for, in "
            "one or two sentences — no padding, no unrelated columns, no restating the whole row."
        )
    if intent == "EXTERNAL":
        return "This is a live market data lookup: state the figure(s) asked for concisely."
    return (
        "This blends structured data and product information: give the specific figure(s) "
        "concisely, then the relevant product detail in a bit more depth."
    )


@dataclass
class RouterResult:
    answer: str
    intent: str = ""
    citations: List[str] = field(default_factory=list)


class HybridRouter:
    def __init__(self, structured: StructuredDataEngine, unstructured: UnstructuredDataEngine):
        self.structured = structured
        self.unstructured = unstructured
        self.llm = get_llm()

    # ------------------------------------------------------------------
    # Step 0: fold conversation history into a standalone question, e.g.
    # "and what about its benchmark?" -> "What is the benchmark for MO
    # Signia?" — so every downstream step (classify/SQL/vector search) can
    # work with a self-contained question instead of needing history itself.
    # Only ever called when _looks_like_followup already said yes — see
    # answer() below.
    # ------------------------------------------------------------------
    def contextualize(self, question: str, history: List[Tuple[str, str]]) -> str:
        if not history:
            return question
        try:
            resp = self.llm.invoke(
                _CONTEXTUALIZE_PROMPT.format(history=_format_history(history), question=question)
            )
            rewritten = _extract_text(resp.content).strip().strip('"')
            # Sanity-check the rewrite instead of trusting it blindly: a
            # local model asked to "resolve references" can occasionally go
            # off the rails and return something wildly longer/unrelated. If
            # it did, the original question is the safer bet.
            if not rewritten or len(rewritten) > max(200, len(question) * 4):
                return question
            return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("Follow-up contextualization failed (%s) — using original question", exc)
            return question

    # ------------------------------------------------------------------
    # Step 1: intent classification
    # ------------------------------------------------------------------
    def classify(self, question: str) -> str:
        try:
            resp = self.llm.invoke(_CLASSIFY_PROMPT.format(question=question))
            label = _extract_text(resp.content).strip().upper()
            # Whole-word match only: "STRUCTURED" is a literal substring of
            # "UNSTRUCTURED" (UN+STRUCTURED), so a naive `"STRUCTURED" in label`
            # containment check — checked first — matched every correct
            # UNSTRUCTURED classification too, silently skipping the PDF/RAG
            # engine on every product/leaflet question. \b prevents that since
            # there's no word boundary between the "N" and "S" inside
            # "UNSTRUCTURED".
            for candidate in ("STRUCTURED", "UNSTRUCTURED", "EXTERNAL", "CROSS_DOMAIN", "OUT_OF_SCOPE"):
                if re.search(rf"\b{candidate}\b", label):
                    return candidate
        except Exception as exc:  # noqa: BLE001
            logger.warning("Intent classification failed (%s) — defaulting to CROSS_DOMAIN", exc)
        return "CROSS_DOMAIN"

    # ------------------------------------------------------------------
    # Combined step 0+1 for follow-up questions: one LLM call producing
    # both the standalone rewrite and its classification, instead of two
    # separate calls (contextualize() then classify()). Used only when
    # _looks_like_followup() already said yes — see answer_stream(). Falls
    # back to the original two-call sequence if the fused response doesn't
    # parse cleanly, so this is a pure request-count optimization: it can
    # never produce a WORSE result than the two-call path, only occasionally
    # fail to save the call.
    # ------------------------------------------------------------------
    def contextualize_and_classify(
        self, question: str, history: List[Tuple[str, str]]
    ) -> Tuple[str, str]:
        try:
            resp = self.llm.invoke(
                _CONTEXTUALIZE_AND_CLASSIFY_PROMPT.format(
                    history=_format_history(history), question=question
                )
            )
            text = _extract_text(resp.content)
            standalone_match = re.search(r"STANDALONE:\s*(.+)", text)
            intent_match = re.search(r"INTENT:\s*([A-Z_]+)", text, re.I)

            standalone = standalone_match.group(1).strip().strip('"') if standalone_match else None
            # Same sanity check as contextualize(): a rewrite that's empty or
            # wildly longer than the original is more likely a derailed
            # response than a legitimate rewrite.
            if standalone and len(standalone) > max(200, len(question) * 4):
                standalone = None

            intent = None
            if intent_match:
                candidate = intent_match.group(1).strip().upper()
                if candidate in ("STRUCTURED", "UNSTRUCTURED", "EXTERNAL", "CROSS_DOMAIN", "OUT_OF_SCOPE"):
                    intent = candidate

            if standalone and intent:
                return standalone, intent
            logger.info("Combined contextualize+classify response didn't parse cleanly — falling back to two calls")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Combined contextualize+classify failed (%s) — falling back to two calls", exc)

        standalone_question = self.contextualize(question, history)
        return standalone_question, self.classify(standalone_question)

    # ------------------------------------------------------------------
    # Step 2: per-engine retrieval
    # ------------------------------------------------------------------
    def query_structured(self, question: str, ba_code: Optional[str] = None) -> Tuple[str, List[str]]:
        schema = self.structured.schema_description()
        if not schema:
            return "(no structured tables loaded yet)", []

        known_tables = {t.lower() for t in self.structured.list_tables()}
        scope_instruction = _scope_instruction(ba_code)

        prompt = _SQL_PROMPT.format(
            scope_instruction=scope_instruction,
            glossary=_GLOSSARY,
            schema=schema,
            question=question,
        )

        # A local/smaller model will sometimes invent a plausible-looking
        # column or table name instead of sticking to the real schema
        # (e.g. "TotalBrokerage" instead of "brokerageamount") — DuckDB
        # rejects that at execution time with a Binder/Catalog error, which
        # used to be dumped straight into the evidence text and then often
        # just parroted back verbatim as "the query failed". Since DuckDB
        # is the actual authority on what columns exist, the fix is to feed
        # its exact error back to the model and let it self-correct, rather
        # than failing on the first hallucinated name. Capped at one retry
        # — a model that gets it wrong twice in a row with the real schema
        # and the exact error in front of it is unlikely to get it right on
        # a third attempt, and every extra attempt adds real latency.
        max_attempts = SQL_RETRY_ATTEMPTS
        last_schema_error: Optional[Exception] = None
        last_sql = ""

        for attempt in range(1, max_attempts + 1):
            try:
                sql_resp = self.llm.invoke(prompt)
                sql = _extract_text(sql_resp.content).strip()
                sql = re.sub(r"^```sql|```$|^```", "", sql, flags=re.I | re.M).strip()
                last_sql = sql

                if sql.upper().startswith("NO_QUERY"):
                    return "I can only look up and report on data — I'm not able to add, change, or delete anything in the system.", []

                # Optionally-quoted identifier after FROM/JOIN — `FROM "table"`,
                # `` FROM `table` ``, or a bare `FROM table` all resolve to the
                # same captured name. This is only used for the hallucinated-
                # table check and citation labeling below; the security-critical
                # BA-scoping check in _sql_is_properly_scoped does NOT depend on
                # this extraction (see its docstring for why).
                referenced = {
                    m.group(1)
                    for m in re.finditer(r"\b(?:from|join)\s+[`\"]?([a-zA-Z0-9_]+)[`\"]?", sql, re.I)
                }
                valid_referenced = {t for t in referenced if t.lower() in known_tables}
                if referenced and not valid_referenced:
                    # The model named at least one table, and NONE of them are
                    # real — this is a fabricated/hallucinated query, not a
                    # legitimate one that just happens to fail. Don't run it;
                    # DuckDB's own error message here would be confusing rather
                    # than helpful ("Catalog Error: Table X does not exist").
                    logger.warning("Generated SQL referenced only unknown table(s): %s", sql)
                    return "I couldn't find a way to look that up in the current data.", []

                # Guardrail 5 enforcement: fail closed if a BA-scoped session's
                # query doesn't demonstrably filter to that BA's own rows. See
                # _sql_is_properly_scoped's module-level comment for why this
                # can't be "fixed" after the fact instead of refused outright.
                if not _sql_is_properly_scoped(sql, ba_code):
                    logger.warning("Refusing improperly-scoped query for BA %s: %s", ba_code, sql)
                    return _SCOPE_DENIED_MESSAGE, []

                df = self.structured.run_sql(sql)
                citations = [f"[Source: CSV::{t}.csv | Table: {t}]" for t in sorted(valid_referenced or {"unknown"})]

                # Cap what's actually printed into the evidence text — NOT the
                # query itself (run_sql already returned up to its own 200-row
                # cap). A row-listing query (e.g. "list all clients") over a
                # 100-row table used to get silently truncated to the first 50
                # rows with no indication anything was cut, which is exactly
                # what "I have 100 rows but it only used 50" looks like from the
                # outside. An aggregate query (SUM/COUNT/...) always returns 1
                # row, so this cap never affects those.
                DISPLAY_CAP = 200
                total_rows = len(df)
                text = df.head(DISPLAY_CAP).to_string(index=False)
                if total_rows > DISPLAY_CAP:
                    text += (
                        f"\n... ({total_rows - DISPLAY_CAP} more row(s) not shown here. If a total/count is "
                        "needed, note that it should be computed with SUM/COUNT in the query rather than by "
                        "counting this partial listing.)"
                    )
                return text, citations
            except ValueError as exc:
                # run_sql's own read-only/write-keyword guard rejected the query —
                # give a clear, friendly answer instead of surfacing the raw error.
                logger.warning("Structured query blocked by write-guard: %s", exc)
                return "I can only look up and report on data — I'm not able to add, change, or delete anything in the system.", []
            except Exception as exc:  # noqa: BLE001
                if _looks_like_rate_limit_error(exc):
                    # Retrying a rate-limit failure with another LLM call
                    # would just wait for the rate limiter again and likely
                    # hit the same wall — no point spending a retry here.
                    logger.warning("Structured query hit a provider rate limit: %s", exc)
                    return _llm_unavailable_message("(no evidence retrieved — rate limited)", exc), []
                if _looks_like_schema_error(exc):
                    last_schema_error = exc
                    if attempt < max_attempts:
                        logger.info(
                            "Structured query referenced a non-existent column/table "
                            "(attempt %d/%d) — retrying with the error fed back to the model: %s",
                            attempt, max_attempts, exc,
                        )
                        prompt = _SQL_RETRY_PROMPT.format(
                            previous_sql=last_sql,
                            error=str(exc),
                            scope_instruction=scope_instruction,
                            glossary=_GLOSSARY,
                            schema=schema,
                            question=question,
                        )
                        continue
                    break  # retries exhausted — fall through to the message after the loop
                logger.warning("Structured query failed: %s", exc)
                return f"(structured query could not be executed: {exc})", []

        # Exhausted retries — every attempt hit a schema error. Surface this
        # plainly rather than the raw DuckDB exception text, since by this
        # point it's clear the requested field genuinely isn't available in
        # the loaded data (or the model can't find the right name for it).
        logger.warning(
            "Structured query gave up after %d attempts, last error: %s", max_attempts, last_schema_error
        )
        return (
            "I couldn't find a column matching that in the current data after checking the "
            f"schema — the field you're asking about may not exist in the loaded CSVs. "
            f"(Last database error: {last_schema_error})"
        ), []

    def query_unstructured(self, question: str) -> Tuple[str, List[str]]:
        hits = self.unstructured.query(question)
        if not hits:
            return "(no relevant product leaflet content found)", []
        evidence_lines, citations = [], []
        for h in hits:
            snippet = h["text"][:500].replace("\n", " ")
            evidence_lines.append(f"- ({h['source_file']} p.{h['page']}) {snippet}")
            citations.append(f"[Source: PDF::{h['source_file']}, Page {h['page']}]")
        return "\n".join(evidence_lines), citations

    def query_external(self, symbol: str) -> Tuple[str, List[str]]:
        quote = get_quote(symbol)
        citation = f"[Source: API::NSE | Timestamp: {quote['timestamp_utc']}]"
        return json.dumps(quote), [citation]

    # ------------------------------------------------------------------
    # Step 3: top-level entry point
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        ba_code: Optional[str] = None,
    ) -> RouterResult:
        """ba_code=None means administrator/unrestricted access. Any other
        value scopes every structured query to that Business Associate's own
        rows only — see Guardrail 5 above."""
        history = history or []

        # Guardrail 1: abusive language never reaches the LLM at all.
        if _contains_profanity(question):
            logger.info("Blocked a message containing flagged language.")
            return RouterResult(answer=_PROFANITY_MESSAGE, intent="BLOCKED", citations=[])

        # Guardrail 3: only let history influence anything (contextualization
        # AND the synthesis step later) when this question actually looks
        # like it depends on it. See _looks_like_followup's docstring for why
        # this exists — it's the fix for "a completely different question
        # still gets answered like the previous one".
        is_followup = bool(history) and _looks_like_followup(question)
        if is_followup:
            # See contextualize_and_classify's docstring: one call instead
            # of two for the common "follow-up question" case.
            standalone_question, intent = self.contextualize_and_classify(question, history)
        else:
            standalone_question = question
            intent = None

        # Guardrail 2: stay in scope. Deterministic check first (doesn't
        # depend on the LLM getting a 5-way classification right); the LLM's
        # own OUT_OF_SCOPE label is a second layer for subtler cases.
        if _looks_out_of_scope(standalone_question):
            return RouterResult(answer=_OUT_OF_SCOPE_MESSAGE, intent="OUT_OF_SCOPE", citations=[])

        if intent is None:
            intent = self.classify(standalone_question)

        if intent == "OUT_OF_SCOPE":
            return RouterResult(answer=_OUT_OF_SCOPE_MESSAGE, intent=intent, citations=[])

        evidence_parts: List[str] = []
        citations: List[str] = []

        if intent in ("STRUCTURED", "CROSS_DOMAIN"):
            text, cites = self.query_structured(standalone_question, ba_code=ba_code)
            evidence_parts.append(f"[Structured data]\n{text}")
            citations += cites

        if intent in ("UNSTRUCTURED", "CROSS_DOMAIN"):
            text, cites = self.query_unstructured(standalone_question)
            evidence_parts.append(f"[Product leaflets]\n{text}")
            citations += cites

        if intent == "EXTERNAL" or _TICKER_HINT.search(standalone_question):
            symbol_match = re.search(r"\b([A-Z]{3,15})\b", standalone_question)
            if symbol_match:
                text, cites = self.query_external(symbol_match.group(1))
                evidence_parts.append(f"[Live market data]\n{text}")
                citations += cites

        evidence = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence retrieved)"

        try:
            final = self.llm.invoke(
                _SYNTHESIS_PROMPT.format(
                    question=standalone_question,
                    evidence=evidence,
                    style=_style_hint(intent),
                    history_block=_history_block(history, is_followup),
                )
            )
            answer_text = _extract_text(final.content).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM synthesis failed: %s", exc)
            answer_text = _llm_unavailable_message(evidence, exc)

        # Citations are returned only via the structured `citations` field —
        # the frontend renders those as the "Source ledger" block. Appending
        # them into answer_text too (as this used to do) made every citation
        # show up twice: once inline in the answer, once in the ledger below it.
        deduped = sorted(set(citations))
        return RouterResult(answer=answer_text, intent=intent, citations=deduped)

    # ------------------------------------------------------------------
    # Streaming variant of answer() — same guardrails/classification/
    # retrieval as answer(), but two things are different:
    #
    # 1. Every step now also yields a {"type": "status", "text": "..."}
    #    event before it runs. On a VRAM-constrained/CPU-offloaded local
    #    model, classify()/contextualize()/the SQL-generation call can
    #    together take several seconds BEFORE the final answer even starts
    #    generating — without these, the UI just sits on a static spinner
    #    that whole time, which is indistinguishable from "it's stuck".
    #    These make each in-progress step visible instead.
    # 2. For CROSS_DOMAIN, structured retrieval (an LLM call, for SQL
    #    generation) and unstructured retrieval (no LLM — local vector
    #    search only) run concurrently via a thread pool instead of one
    #    after the other, since they don't depend on each other. This is a
    #    real, safe latency win specifically because only one of the two
    #    actually uses the GPU/LLM — running two LLM calls "in parallel"
    #    on a single local GPU would NOT be faster, since they'd just queue
    #    for the same model anyway; that's why classify/contextualize/SQL-
    #    gen/synthesis stay sequential rather than being thrown at threads
    #    too.
    #
    # Yields: {"type": "status", "text": "..."} for progress updates,
    # {"type": "chunk", "text": "..."} for each piece of the answer as it's
    # produced, then exactly one {"type": "done", "intent": ..., "citations":
    # [...], "answer": full_text} at the end so the caller can persist the
    # complete turn (chat history) once it's fully known.
    # ------------------------------------------------------------------
    def answer_stream(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        ba_code: Optional[str] = None,
    ):
        history = history or []

        if _contains_profanity(question):
            logger.info("Blocked a message containing flagged language.")
            yield {"type": "chunk", "text": _PROFANITY_MESSAGE}
            yield {"type": "done", "intent": "BLOCKED", "citations": [], "answer": _PROFANITY_MESSAGE}
            return

        is_followup = bool(history) and _looks_like_followup(question)
        if is_followup:
            # One call does both the standalone rewrite and its
            # classification (see contextualize_and_classify's docstring
            # for why this saves a request over the old two-call sequence).
            yield {"type": "status", "text": "Reviewing the conversation and understanding what you're asking…"}
            standalone_question, intent = self.contextualize_and_classify(question, history)
        else:
            standalone_question = question
            intent = None  # not yet known — classified below after the deterministic out-of-scope check

        if _looks_out_of_scope(standalone_question):
            yield {"type": "chunk", "text": _OUT_OF_SCOPE_MESSAGE}
            yield {"type": "done", "intent": "OUT_OF_SCOPE", "citations": [], "answer": _OUT_OF_SCOPE_MESSAGE}
            return

        if intent is None:
            yield {"type": "status", "text": "Understanding what you're asking…"}
            intent = self.classify(standalone_question)

        if intent == "OUT_OF_SCOPE":
            yield {"type": "chunk", "text": _OUT_OF_SCOPE_MESSAGE}
            yield {"type": "done", "intent": intent, "citations": [], "answer": _OUT_OF_SCOPE_MESSAGE}
            return

        evidence_parts: List[str] = []
        citations: List[str] = []

        if intent == "CROSS_DOMAIN":
            yield {"type": "status", "text": "Looking up client/portfolio records and product leaflets…"}
            with ThreadPoolExecutor(max_workers=2) as pool:
                structured_future = pool.submit(self.query_structured, standalone_question, ba_code=ba_code)
                unstructured_future = pool.submit(self.query_unstructured, standalone_question)
                text, cites = structured_future.result()
                evidence_parts.append(f"[Structured data]\n{text}")
                citations += cites
                text, cites = unstructured_future.result()
                evidence_parts.append(f"[Product leaflets]\n{text}")
                citations += cites
        elif intent == "STRUCTURED":
            yield {"type": "status", "text": "Looking up client/portfolio records…"}
            text, cites = self.query_structured(standalone_question, ba_code=ba_code)
            evidence_parts.append(f"[Structured data]\n{text}")
            citations += cites
        elif intent == "UNSTRUCTURED":
            yield {"type": "status", "text": "Searching product leaflets…"}
            text, cites = self.query_unstructured(standalone_question)
            evidence_parts.append(f"[Product leaflets]\n{text}")
            citations += cites

        if intent == "EXTERNAL" or _TICKER_HINT.search(standalone_question):
            symbol_match = re.search(r"\b([A-Z]{3,15})\b", standalone_question)
            if symbol_match:
                yield {"type": "status", "text": "Fetching live market data…"}
                text, cites = self.query_external(symbol_match.group(1))
                evidence_parts.append(f"[Live market data]\n{text}")
                citations += cites

        evidence = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence retrieved)"
        yield {"type": "status", "text": "Writing the answer…"}
        prompt = _SYNTHESIS_PROMPT.format(
            question=standalone_question,
            evidence=evidence,
            style=_style_hint(intent),
            history_block=_history_block(history, is_followup),
        )

        pieces: List[str] = []
        try:
            for chunk in self.llm.stream(prompt):
                piece = _extract_text(chunk.content)
                if piece:
                    pieces.append(piece)
                    yield {"type": "chunk", "text": piece}
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM streaming synthesis failed: %s", exc)
            fallback = _llm_unavailable_message(evidence, exc)
            pieces = [fallback]
            yield {"type": "chunk", "text": fallback}

        answer_text = "".join(pieces).strip()

        # Some models (Qwen3-family "thinking" models included) wrap internal
        # deliberation in <think>...</think> before the real answer — that's
        # never something this app wants shown verbatim. Stripped here
        # (rather than mid-stream) so the live-typed text can keep flowing in
        # real time; this only cleans up the final version sent in the "done"
        # event below, which is what actually replaces the streamed text once
        # generation finishes (see index.html's renderAnswer).
        cleaned = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.S)
        # An UNCLOSED <think> — generation ran out of context/output budget
        # mid-reasoning, before ever reaching a real answer — leaves a
        # dangling opening tag with nothing reliable after it. Everything
        # from that point on is unfinished internal reasoning, not an
        # answer, so it's dropped rather than shown as if it were one.
        cleaned = re.split(r"<think>", cleaned, maxsplit=1)[0].strip()
        if cleaned != answer_text.strip() and cleaned:
            answer_text = cleaned
        elif not cleaned:
            # Either genuinely empty, or 100% consumed by an UNTERMINATED
            # <think> block that ran out of context/output budget before the
            # model ever reached a real answer — either way, showing a blank
            # answer bubble with nothing but citation chips is worse than
            # being explicit about what happened. Falling back to the raw
            # evidence at least still gives something useful to read.
            logger.warning(
                "Synthesis produced no visible answer text (got %d raw chars) — "
                "falling back to raw evidence. This usually means the model spent its "
                "whole output budget on internal reasoning before reaching an answer; "
                "consider a model/config with reasoning disabled, or a larger OLLAMA_NUM_CTX.",
                len(answer_text),
            )
            answer_text = (
                "The model didn't produce a readable answer for this question — it may have "
                "run out of room reasoning internally before writing one. Here is the raw "
                f"evidence I retrieved instead:\n\n{evidence}"
            )

        deduped = sorted(set(citations))
        yield {"type": "done", "intent": intent, "citations": deduped, "answer": answer_text}
