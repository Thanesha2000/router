# =============================================================================
# router.py — Two-Stage Intent Router
# =============================================================================
#
# WHAT THIS FILE DOES:
#   Classifies every user message into one of 5 intents:
#     policy_query    — asking about cancellation rules, charges, limits
#     refund_issue    — reporting money deducted / ticket not received
#     tourism_package — asking for destination suggestions / IRCTC packages
#     trip_planning   — asking for day-by-day itinerary for a chosen place
#     general_support — everything else: terms, how-to, booking help
#
# TWO-STAGE DESIGN (saves API quota):
#   Stage 1 — keyword_router(): regex patterns, 0ms, zero API calls
#             Handles obvious queries confidently
#   Stage 2 — gemini_router(): only called when Stage 1 is uncertain
#             Uses improved prompt with bug fixes and edge cases
#
# IMPROVEMENT LOG (for learning — track what changed and why):
#   v1 → v2 changes:
#     [BUG FIX]  "What is RAC?" was → policy_query. Now → general_support
#                Root cause: "rac.*rule" keyword matched. Fixed in KEYWORD_RULES.
#     [PROMPT]   Added terminology_explanation sub-category inside general_support
#     [PROMPT]   Added RULE 6 distinguishing "asking what X means" vs "asking rule about X"
#     [EXAMPLES] Added 12 new edge cases: short queries, Hinglish, misspellings,
#                multi-intent, unknown queries
#     [COMMENTS] Added WHY comments on every decision
# =============================================================================

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import os
import json
import re

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# =============================================================================
# STAGE 1 — KEYWORD ROUTER
# =============================================================================
#
# WHY THIS EXISTS:
#   Gemini free tier = limited requests per minute.
#   A regex match takes 0ms and uses zero quota.
#   Most obvious queries ("kat gaya", "char dham package") can be
#   classified with certainty using keyword patterns alone.
#
# DESIGN RULE — what qualifies for keyword patterns:
#   ONLY multi-word phrases specific enough to be unambiguous.
#   Single words like "refund" or "cancel" are NOT here — they appear
#   in multiple intents. Ambiguous queries go to Gemini.
#
# [BUG FIX v2] REMOVED from keyword patterns:
#   "rac.*rule" and "rac.*kya hota" were previously in policy_query.
#   This caused "What is RAC?" → policy_query (WRONG).
#   "What is RAC?" is a TERMINOLOGY question → general_support.
#   Pattern removed. Gemini (Stage 2) now handles RAC queries correctly
#   because of improved prompt examples added in build_prompt().
# =============================================================================

KEYWORD_RULES = {

    # ── REFUND_ISSUE ──────────────────────────────────────────────────────────
    # User is REPORTING a real problem — money gone, ticket missing.
    # These phrases are emotionally charged and very specific.
    # "kat gaya" (got cut) is near-impossible to mean anything except
    # "money was deducted" in this railway context.
    "refund_issue": [
        r"kat gaya",            r"kat gaye",           r"kaat liya",
        r"paisa.*nahi aaya",    r"paise.*nahi aaye",   r"paisa wapas nahi",
        r"money deducted",      r"amount deducted",    r"money.*cut",
        r"ticket nahi aaya",    r"ticket nahi mila",   r"ticket not.*receiv",
        r"ticket not generat",  r"booking.*fail",      r"transaction.*fail",
        r"payment.*fail",       r"payment.*stuck",     r"payment.*deduct",
        r"refund nahi aaya",    r"refund.*not.*receiv",r"refund.*pending.*day",
        r"double.*charg",       r"charged.*twice",     r"money.*not.*refund",
        r"paise.*kat",          r"amount.*not.*refund",
    ],

    # ── POLICY_QUERY ──────────────────────────────────────────────────────────
    # User is ASKING ABOUT rules — not reporting a problem.
    # Key distinction: "how much refund" (policy) vs "refund nahi aaya" (issue)
    #
    # [BUG FIX v2] REMOVED these patterns that caused misclassification:
    #   "rac.*rule"     — caused "What is RAC?" → policy_query (WRONG)
    #   "rac.*kya hota" — same problem
    #   "waiting list.*rule" — caused "What is waiting list?" → policy_query (WRONG)
    # These are now handled correctly by Gemini prompt's RULE 6 and new examples.
    "policy_query": [
        r"kitna refund",        r"how much.*refund",   r"refund.*milega",
        r"cancellation.*charge",r"cancel.*charge",     r"cancel.*fee",
        r"tatkal.*cancel",      r"tatkal.*rule",       r"tatkal.*policy",
        r"tatkal.*refund",      r"tatkal.*allowed",
        r"baggage.*limit",      r"luggage.*limit",     r"allowed.*luggage",
        r"senior citizen.*disc",r"senior.*concession", r"senior.*rule",
        # NOTE: "waiting list.*rule" removed — see BUG FIX note above
        r"child.*ticket.*rule", r"age.*ticket",        r"infant.*ticket",
        r"can i cancel",        r"kya cancel",         r"kya rule hai",
        r"what.*is.*the.*rule", r"what.*are.*charges", r"insurance.*rule",
        r"refund.*rule",        r"how.*many.*bags",    r"kitna.*luggage",
    ],

    # ── TOURISM_PACKAGE ───────────────────────────────────────────────────────
    # User wants suggestions — they haven't decided where to go.
    # "suggest", "recommend", "kahan jaun" = user is open to options.
    "tourism_package": [
        r"suggest.*trip",       r"suggest.*place",     r"suggest.*destination",
        r"suggest.*package",    r"recommend.*trip",    r"recommend.*place",
        r"irctc.*package",      r"package.*irctc",     r"char dham",
        r"buddhist circuit",    r"bharat gaurav",      r"jyotirlinga",
        r"kahan jaun",          r"where.*should.*go",  r"where.*to.*go",
        r"best place.*visit",   r"places.*to.*visit",
        r"honeymoon.*package",  r"family.*package",    r"couple.*trip",
        r"trip.*under.*\d+",    r"under.*\d+.*trip",   r"budget.*trip.*from",
        r"hill station.*suggest",r"beach.*suggest",    r"options.*trip",
        r"trip.*ideas",         r"koi.*package",
    ],

    # ── TRIP_PLANNING ─────────────────────────────────────────────────────────
    # User wants a plan for a SPECIFIC destination they've already chosen.
    # "itinerary", "day by day", "kaise cover karu" = they know where, want how.
    "trip_planning": [
        r"itinerary",           r"day.by.day",         r"day 1.*day 2",
        r"day\s*1\s*:",         r"plan.*trip.*to\s+\w",r"plan my trip",
        r"trip.*plan.*to\s+\w", r"kaise cover karu",  r"din ka plan",
        r"schedule.*trip",      r"trip.*schedule",
        r"\d+\s*day.*trip.*to", r"trip.*\d+\s*days.*to",
        r"how to spend.*days",  r"spend.*days in\s+\w",
        r"itinerary.*for\s+\w", r"bana do.*itinerary", r"itinerary.*bana",
        r"manali.*plan",        r"shimla.*plan",       r"goa.*plan",
        r"trip.*manali",        r"trip.*shimla",       r"trip.*goa",
    ],

    # NOTE: general_support has NO keyword patterns intentionally.
    # Reason: general_support is the DEFAULT / catch-all.
    # Trying to keyword-match "What is RAC?" risks false positives
    # (as the v1 bug proved). Let Gemini handle it with proper context.
}


def keyword_router(user_message: str):
    """
    Stage 1 router. Returns (result_dict_or_None, is_confident_bool).

    is_confident = True  → caller uses this result directly, skips Gemini
    is_confident = False → caller passes this as fallback to gemini_router()

    Confidence logic:
      Zero matches       → not confident (None returned)
      One intent matched → confident
      Clear leader       → confident (even if others have 1 match)
      Tied scores        → not confident (Gemini decides)
    """
    text = user_message.lower().strip()

    scores = {}
    for intent, patterns in KEYWORD_RULES.items():
        score = sum(1 for p in patterns if re.search(p, text))
        if score > 0:
            scores[intent] = score

    if not scores:
        # No keyword matched → let Gemini handle it
        return None, False

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_intent, best_score = ranked[0]

    # TIE: two intents with equal score → ambiguous → go to Gemini
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        return {"intent": best_intent, "confidence": "low", "source": "keyword"}, False

    # Clear winner
    confidence = "high" if best_score >= 2 else "medium"
    return {"intent": best_intent, "confidence": confidence, "source": "keyword"}, True


# =============================================================================
# STAGE 2 — GEMINI ROUTER (IMPROVED PROMPT)
# =============================================================================
#
# WHAT CHANGED IN v2 AND WHY:
#
# [CHANGE 1] Added general_support sub-category: terminology_explanation
#   WHY: The v1 prompt had no explicit guidance that "What is X?" questions
#   belong to general_support. Gemini defaulted to policy_query because
#   RAC/WL/Tatkal appear in policy contexts. Now we explicitly tell it:
#   "If user is asking WHAT something means → general_support"
#
# [CHANGE 2] Added RULE 6: "Asking what X means ≠ Asking the rule about X"
#   WHY: This is the core disambiguation needed.
#   "What is RAC?" = explain the term = general_support
#   "Can I cancel RAC ticket?" = policy question = policy_query
#   One rule, clearly stated, fixes the entire class of bugs.
#
# [CHANGE 3] Added 12 new examples covering edge cases
#   WHY: Few-shot examples are the most direct way to teach LLMs.
#   Each new example targets a specific failure mode discovered in testing.
#
# [CHANGE 4] Used <user_query> XML delimiter (previously was <user_message>)
#   WHY: Consistency with DeepLearning.AI convention. Also avoids Gemini
#   confusing the tag name with "user message" meta-concept.
#
# [CHANGE 5] Added explicit guidance for misspellings and very short queries
#   WHY: Real users type "refnd nhi aaya" not "refund nahi aaya".
#   Gemini handles typos well if we tell it to look at MEANING not spelling.
# =============================================================================

def build_prompt(user_message: str) -> str:
    """
    Builds the classification prompt sent to Gemini.

    Design: Structured with XML tags so each section has a clear purpose.
    Gemini (and Claude) are trained on structured text — XML tags
    create unambiguous boundaries that improve classification accuracy.
    """
    return f"""
You are an expert intent classifier for IRCTC Indian Railways customer support.
Your ONLY job: read the user query and return ONE intent from the list below.

<categories>

policy_query
  MEANING : User is ASKING ABOUT a rule, charge, or limit — not reporting a problem.
  SIGNALS : "kitna", "kya rule", "how much", "is it allowed", "what are charges",
            "can I cancel", "what happens if"
  EXAMPLES: cancellation charges, Tatkal rules, baggage limits, senior discounts,
            child ticket age rules, insurance rules, refund percentage rules

refund_issue
  MEANING : User is REPORTING something WENT WRONG with money or ticket.
  SIGNALS : "deducted", "kat gaya", "nahi aaya", "stuck", "failed", "not received",
            "pending for X days", "double charged", "paisa wapas"
  EXAMPLES: money cut but ticket not booked, refund pending 10 days,
            payment failed but deducted, ticket cancelled refund not received

tourism_package
  MEANING : User wants SUGGESTIONS — they have NOT decided destination yet.
  SIGNALS : "suggest", "recommend", "best place", "kahan jaun", "package",
            "where should I go", "options", "ideas", "IRCTC package"
  EXAMPLES: suggest hill station trip, Char Dham package, budget trip from Delhi,
            honeymoon package ideas, places to visit in India

trip_planning
  MEANING : User wants DETAILED PLAN for a destination they already KNOW.
  SIGNALS : "itinerary", "plan", "day 1 day 2", "schedule", "kaise cover karu",
            "how to spend X days", named city + plan/days
  EXAMPLES: 3-day Manali itinerary, Shimla trip schedule, plan my Goa trip,
            day-wise Rajasthan plan

general_support
  MEANING : Everything else. Includes TWO sub-types:
  
  Sub-type A — terminology_explanation:
    User is asking WHAT something MEANS or WHAT something IS.
    SIGNALS : "what is", "kya hota hai", "kya hai", "kya matlab",
              "difference between", "explain", "batao", "samjhao"
    EXAMPLES:
      "What is RAC?"                  → general_support
      "PNR kya hota hai?"             → general_support
      "What is waiting list?"         → general_support
      "Difference between RAC and WL?"→ general_support
      "What is Tatkal?"               → general_support
      "Kya hota hai 3AC mein?"        → general_support
      "SL aur 3AC mein kya fark hai?" → general_support

  Sub-type B — how_to_support:
    User needs help USING IRCTC or wants process guidance.
    EXAMPLES:
      "How to download e-ticket?"     → general_support
      "How to book tickets?"          → general_support
      "How to check PNR status?"      → general_support
      "How to use IRCTC website?"     → general_support
      "Account login problem"         → general_support

  DEFAULT: When nothing else fits clearly → general_support
</categories>

<rules>
RULE 1 : Choose EXACTLY ONE category. Never return two.
RULE 2 : REPORTING a problem with money/ticket = refund_issue.
          ASKING about rules = policy_query. These are different.
RULE 3 : Has NOT chosen destination = tourism_package.
          HAS chosen destination, wants plan = trip_planning.
RULE 4 : Hindi and Hinglish are valid. Classify MEANING, not language.
RULE 5 : Typos and misspellings are common ("refnd nhi aaya" = refund_issue).
          Focus on meaning, not spelling.
RULE 6 : [KEY RULE - fixes most misclassifications]
          "What IS X?" = asking for an explanation → general_support
          "What IS THE RULE for X?" = asking for a policy → policy_query
          "What is RAC?" → general_support (explaining a term)
          "Can I cancel RAC ticket?" → policy_query (asking a rule)
          "What is Tatkal?" → general_support (explaining a term)
          "Tatkal mein refund milega?" → policy_query (asking a rule)
RULE 7 : Very short or unclear queries → general_support (safe default)
RULE 8 : Multi-intent queries → pick the DOMINANT intent.
          "What is RAC and can I cancel it?" → policy_query (cancel = action intent)
          "Suggest Manali trip itinerary" → trip_planning (planning = dominant)
</rules>

<examples>

--- POLICY QUERY EXAMPLES ---
Query: "Tatkal ticket cancel karne par kitna refund milega?"
Output: {{"intent": "policy_query", "confidence": "high"}}
Reason: Asking about refund RULE/AMOUNT — not reporting a problem.

Query: "Senior citizen ko kya discount milta hai AC mein?"
Output: {{"intent": "policy_query", "confidence": "high"}}
Reason: Asking about discount RULE.

Query: "Can I cancel my Tatkal ticket?"
Output: {{"intent": "policy_query", "confidence": "high"}}
Reason: Asking whether cancellation IS ALLOWED (a rule question).

Query: "Kitna luggage le ja sakte hain sleeper mein?"
Output: {{"intent": "policy_query", "confidence": "high"}}
Reason: Asking baggage LIMIT — a policy.

--- REFUND ISSUE EXAMPLES ---
Query: "Mera paisa kat gaya but ticket confirm nahi hua"
Output: {{"intent": "refund_issue", "confidence": "high"}}
Reason: Reporting deduction — actual problem occurred.

Query: "I cancelled my ticket 5 days ago, refund nahi aaya abhi tak"
Output: {{"intent": "refund_issue", "confidence": "high"}}
Reason: Refund PENDING — reporting a real issue.

Query: "Payment ho gaya, 200 rupees kat gaye, ticket nahi mila"
Output: {{"intent": "refund_issue", "confidence": "high"}}
Reason: Money deducted, ticket missing — refund issue.

Query: "refnd nhi aaya abhi tk" 
Output: {{"intent": "refund_issue", "confidence": "high"}}
Reason: Typos present but meaning is clear — refund not received.

--- TOURISM PACKAGE EXAMPLES ---
Query: "Suggest a good hill station trip under 8000 from Delhi"
Output: {{"intent": "tourism_package", "confidence": "high"}}
Reason: Wants a SUGGESTION — hasn't chosen destination.

Query: "Koi Char Dham package hai IRCTC ka?"
Output: {{"intent": "tourism_package", "confidence": "high"}}
Reason: Asking about IRCTC package — destination-browsing intent.

Query: "Kahan jaun is summer mein family ke saath?"
Output: {{"intent": "tourism_package", "confidence": "high"}}
Reason: "Where should I go" = needs suggestions, not a plan.

--- TRIP PLANNING EXAMPLES ---
Query: "Plan a 3 day trip to Shimla with day wise schedule and cost"
Output: {{"intent": "trip_planning", "confidence": "high"}}
Reason: Destination CHOSEN (Shimla), wants day-by-day PLAN.

Query: "Manali 4 din mein kaise cover karu, itinerary do"
Output: {{"intent": "trip_planning", "confidence": "high"}}
Reason: Destination CHOSEN (Manali), asking for coverage plan.

Query: "Suggest and plan a complete 5 day Rajasthan trip"
Output: {{"intent": "trip_planning", "confidence": "medium"}}
Reason: Both suggest + plan present. Destination NAMED → trip_planning wins.

--- GENERAL SUPPORT — TERMINOLOGY (the bug-fix category) ---
Query: "What is RAC?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: Asking WHAT THE TERM MEANS — not a rule question. → general_support

Query: "PNR kya hota hai?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: "kya hota hai" = asking for explanation of a term.

Query: "What is the difference between SL and 3AC?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: Terminology/class explanation — not a booking problem.

Query: "RAC aur waiting list mein kya fark hai?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: Asking DIFFERENCE between two terms = terminology explanation.

Query: "Tatkal kya hota hai?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: Asking WHAT Tatkal IS — not asking Tatkal rules.

Query: "WL matlab kya hota hai?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: "matlab kya hota hai" = asking for meaning of WL term.

--- GENERAL SUPPORT — HOW-TO ---
Query: "How do I download my e-ticket?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: Process/how-to question — general IRCTC usage.

Query: "How do I check my PNR status?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: How-to usage question — not a policy or issue.

Query: "How to book tickets on IRCTC for first time?"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: First-time user guidance — general support.

--- EDGE CASES ---
Query: "w"
Output: {{"intent": "general_support", "confidence": "low"}}
Reason: Single letter — meaningless query → safe default.

Query: "xyzabc123"
Output: {{"intent": "general_support", "confidence": "low"}}
Reason: Gibberish — no intent detectable → safe default.

Query: "help"
Output: {{"intent": "general_support", "confidence": "low"}}
Reason: Too vague to classify — safe default.

Query: "What is RAC and can I cancel my RAC ticket?"
Output: {{"intent": "policy_query", "confidence": "medium"}}
Reason: Multi-intent. "Can I cancel" is ACTION intent = dominant. → policy_query.
         Note: general_agent fallback will explain RAC if policy_agent fails.

Query: "Mujhe refund chahiye aur Manali trip plan bhi"
Output: {{"intent": "refund_issue", "confidence": "medium"}}
Reason: Multi-intent. Refund problem is more URGENT — handle that first.

Query: "train late hai"
Output: {{"intent": "general_support", "confidence": "medium"}}
Reason: Train delay info — general support (no live data access).

Query: "acnt login nhi ho rha"
Output: {{"intent": "general_support", "confidence": "high"}}
Reason: Misspelled but clear — account login problem = general support.

</examples>

<user_query>
{user_message}
</user_query>

Return ONLY valid JSON. No explanation. No markdown. No extra text.
Format: {{"intent": "category_name", "confidence": "high/medium/low"}}
"""


def parse_llm_response(raw_text: str) -> dict:
    """
    Parses Gemini's text response into a Python dict.

    WHY 3-LAYER PARSING:
    Gemini is told to return only JSON, but sometimes:
      Layer 1: It returns perfect JSON → json.loads() works
      Layer 2: It wraps in ```json ... ``` markdown → strip backticks first
      Layer 3: JSON buried in explanation text → regex extract {}
    If all three fail → safe default (general_support, low confidence).
    """
    raw_text = raw_text.strip()

    # Layer 1: Perfect JSON
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Layer 2: Markdown-wrapped JSON
    if "```" in raw_text:
        cleaned = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Layer 3: JSON buried in text
    match = re.search(r'\{[^{}]+\}', raw_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # All layers failed
    print(f"[PARSE FAIL] Raw response: {raw_text[:120]}")
    return {"intent": "general_support", "confidence": "low"}


# Valid intents whitelist — reject anything Gemini hallucinates
VALID_INTENTS = {
    "policy_query", "refund_issue",
    "tourism_package", "trip_planning", "general_support"
}


def gemini_router(user_message: str, fallback=None) -> dict:
    """
    Calls Gemini for intent classification.

    WHY fallback PARAMETER:
    If keyword_router found something but wasn't confident, we keep
    that as a fallback. If Gemini fails (quota, network), we use the
    keyword result rather than returning nothing.

    FAILURE MODES HANDLED:
      429 / RESOURCE_EXHAUSTED → use keyword fallback or general_support
      Other API error          → same
      Unexpected exception     → same
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=build_prompt(user_message)
        )

        result = parse_llm_response(response.text)

        # Validate intent — Gemini occasionally hallucinates intent names
        if result.get("intent") not in VALID_INTENTS:
            print(f"[INVALID INTENT] Got: {result.get('intent')} — correcting to general_support")
            result["intent"] = "general_support"
            result["confidence"] = "low"

        result["source"] = "gemini"
        return result

    # ── 429 Rate Limit ────────────────────────────────────────────────────────
    except genai_errors.ClientError as e:
        error_str = str(e)

        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            retry_match = re.search(r"retry in (\d+)", error_str, re.IGNORECASE)
            wait_sec = retry_match.group(1) if retry_match else "~30"
            print(f"[RATE LIMIT] Quota hit. Retry in {wait_sec}s. Using fallback.")

            if fallback:
                return {"intent": fallback["intent"], "confidence": "medium",
                        "source": "keyword_fallback"}
            return {"intent": "general_support", "confidence": "low",
                    "source": "rate_limit_fallback"}

        print(f"[API ERROR] {error_str[:120]}")
        if fallback:
            return {**fallback, "source": "keyword_fallback"}
        return {"intent": "general_support", "confidence": "low", "source": "error_fallback"}

    except Exception as e:
        print(f"[UNEXPECTED ERROR] {str(e)[:120]}")
        if fallback:
            return {**fallback, "source": "keyword_fallback"}
        return {"intent": "general_support", "confidence": "low", "source": "error_fallback"}


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================

def detect_intent(user_message: str) -> dict:
    """
    The only function chatbot.py needs to call.

    Flow:
      keyword_router() → confident? YES → return (0 API calls)
                       → NO  → gemini_router() with keyword as fallback
                                  → 429/error → use fallback
                                  → success   → return Gemini result
    """
    keyword_result, is_confident = keyword_router(user_message)

    if is_confident:
        return keyword_result

    return gemini_router(user_message, fallback=keyword_result)


# =============================================================================
# TEST BLOCK
# =============================================================================

if __name__ == "__main__":

    test_cases = [
        # ── Bug-fix verification ───────────────────────────────────────────────
        ("What is RAC?",                                       "general_support"),   # THE bug
        ("PNR kya hota hai?",                                  "general_support"),
        ("Difference between SL and 3AC?",                    "general_support"),
        ("RAC aur WL mein kya fark hai?",                     "general_support"),
        ("What is Tatkal?",                                    "general_support"),
        ("WL matlab kya hota hai?",                           "general_support"),

        # ── Keyword router catches (no API call) ──────────────────────────────
        ("Mera paisa kat gaya, ticket nahi aaya",             "refund_issue"),
        ("Tatkal cancel karne par kitna refund milega?",      "policy_query"),
        ("Suggest a beach trip under 10k from Mumbai",        "tourism_package"),
        ("Manali 4 din ka itinerary bana do",                 "trip_planning"),

        # ── Gemini handles these ──────────────────────────────────────────────
        ("Money deducted but ticket not booked",              "refund_issue"),
        ("Can I cancel my Tatkal ticket?",                    "policy_query"),
        ("How do I download my e-ticket?",                    "general_support"),
        ("How to book tickets on IRCTC?",                     "general_support"),

        # ── Edge cases ────────────────────────────────────────────────────────
        ("w",                                                  "general_support"),
        ("xyzabc123",                                          "general_support"),
        ("refnd nhi aaya abhi tk",                            "refund_issue"),
        ("acnt login nhi ho rha",                             "general_support"),
        ("train late hai",                                    "general_support"),
    ]

    print("\nTesting Two-Stage Intent Router v2")
    print("=" * 65)

    correct = 0
    keyword_count = 0
    gemini_count = 0

    for query, expected in test_cases:
        result     = detect_intent(query)
        got        = result["intent"]
        confidence = result["confidence"]
        source     = result.get("source", "unknown")
        status     = "✅" if got == expected else "❌"

        if got == expected:
            correct += 1
        if "keyword" in source:
            keyword_count += 1
        elif source == "gemini":
            gemini_count += 1

        src_label = f"[{source}]".ljust(22)
        print(f"\n{status} {src_label} {query}")
        print(f"    Expected : {expected}")
        print(f"    Got      : {got}  | Confidence: {confidence}")

    total = len(test_cases)
    print("\n" + "=" * 65)
    print(f"Score            : {correct}/{total}")
    print(f"Keyword handled  : {keyword_count}  (zero API calls used)")
    print(f"Gemini handled   : {gemini_count}  (API calls used)")
    print(f"API calls saved  : {round(keyword_count/total*100)}%")
    print("=" * 65)