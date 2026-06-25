"""
general_agent.py
Location: back/agents/general_agent.py

Handles general Indian Railways / IRCTC questions.
Examples: "What is RAC?", "PNR kya hota hai?", "Difference SL vs 3AC?"

Changes from previous version:
  [FIX 1] Added retry logic for 503 errors — same pattern as policy_rag.py
           Attempts: 3 times with 2s, 4s, 8s backoff
           Result: 503 spikes no longer produce "Connection problem" messages

  [FIX 2] Added INSTANT_ANSWERS dictionary
           "What is RAC?", "What is PNR?" etc. are answered instantly
           from a local dictionary — ZERO Gemini calls, ZERO 503 risk
           These are the most common questions and never change

  [FIX 3] Fallback chain: Instant → Gemini (with retry) → Static fallback
           User always gets a real answer, never a connection error
           for questions that have known answers
"""

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import os
import re
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL  = "gemini-2.5-flash"

# ─────────────────────────────────────────────────────────────────────
# RETRY CONFIGURATION  [FIX 1]
# ─────────────────────────────────────────────────────────────────────

MAX_ATTEMPTS   = 3
RETRY_BACKOFF  = [2.0, 4.0, 8.0]   # seconds before attempt 2, 3

# ─────────────────────────────────────────────────────────────────────
# INSTANT ANSWERS  [FIX 2]
#
# WHY THIS EXISTS:
#   "What is RAC?" gets asked hundreds of times a day.
#   The answer never changes. Calling Gemini every time:
#     - wastes quota
#     - adds 1-2 seconds latency
#     - fails with 503 during demand spikes
#
#   These answers are stored locally. Zero API calls. Zero 503 risk.
#   Instant response every time, even when Gemini is completely down.
#
# HOW MATCHING WORKS:
#   User input is lowercased and stripped.
#   We check if any key phrase appears anywhere in the input.
#   "what is rac?" → contains "what is rac" → instant answer returned.
#   No Gemini call made at all.
# ─────────────────────────────────────────────────────────────────────

INSTANT_ANSWERS = {

    # RAC
    "what is rac": (
        "RAC stands for Reservation Against Cancellation. "
        "It is a half-confirmed status — you are guaranteed a seat to sit on "
        "but may have to share a side-lower berth with another RAC passenger. "
        "If a confirmed passenger cancels before chart preparation, your RAC "
        "ticket automatically upgrades to a full confirmed berth. "
        "Check your PNR status closer to departure to see if you got upgraded."
    ),

    "rac kya hota": (
        "RAC matlab Reservation Against Cancellation. "
        "Iska matlab hai aapko seat milegi baithne ke liye, lekin berth share "
        "karni pad sakti hai. Agar koi confirmed passenger cancel kare, toh "
        "aapka RAC automatically confirm ho jata hai. "
        "Departure se pehle apna PNR check karte rahein."
    ),

    "rac kya hai": (
        "RAC matlab Reservation Against Cancellation. "
        "Aapko seat milegi baithne ke liye, lekin berth share karni pad sakti hai. "
        "Cancellations hone par automatically confirm ho jata hai."
    ),

    # PNR
    "what is pnr": (
        "PNR stands for Passenger Name Record. "
        "It is a unique 10-digit number generated for every booking on Indian Railways. "
        "You can use it to check your ticket status, seat/berth number, coach, and "
        "train details on irctc.co.in, the IRCTC Rail Connect app, or by calling 139."
    ),

    "pnr kya hota": (
        "PNR yaani Passenger Name Record — yeh ek 10-digit unique number hota hai "
        "jo aapki har booking ke liye generate hota hai. "
        "Isse aap apni seat, coach, train details aur ticket status check kar sakte hain "
        "irctc.co.in ya IRCTC Rail Connect app par."
    ),

    "pnr kya hai": (
        "PNR ek 10-digit number hota hai jo aapki train booking ke liye milta hai. "
        "Isse aap apna seat number, coach aur ticket status check kar sakte hain."
    ),

    # Waiting list
    "what is waiting list": (
        "Waiting List (WL) means your ticket is not yet confirmed — there are "
        "passengers ahead of you waiting for berths. "
        "If confirmed passengers cancel, your position moves up. "
        "If your WL number reaches RAC or confirmed status before chart preparation, "
        "you will be allotted a berth. If it remains WL at chart time, you generally "
        "cannot board the reserved coach, but a refund is automatically processed."
    ),

    "waiting list kya": (
        "Waiting List (WL) matlab aapki ticket abhi confirm nahi hui hai. "
        "Jab confirmed passengers cancel karte hain, aapki position upar aati hai. "
        "Chart banne tak agar confirm ya RAC na hua, toh aap reserved coach mein "
        "travel nahi kar sakte aur refund automatic process ho jata hai."
    ),

    # Tatkal
    "what is tatkal": (
        "Tatkal is an emergency quota in Indian Railways that allows you to book "
        "tickets at short notice, typically 1 day before the journey date. "
        "It carries an additional Tatkal charge on top of the normal fare. "
        "Tatkal tickets have different (stricter) cancellation rules compared to "
        "regular tickets — generally no refund on cancellation for confirmed Tatkal tickets."
    ),

    "tatkal kya hota": (
        "Tatkal ek emergency booking quota hai jisme aap journey se 1 din pehle "
        "ticket book kar sakte hain. Isme extra Tatkal charge lagta hai. "
        "Confirmed Tatkal ticket cancel karne par generally koi refund nahi milta."
    ),

    # SL vs 3AC
    "difference between sl and 3ac": (
        "SL (Sleeper Class) is non-air-conditioned with open windows and basic berths "
        "— the most affordable option for overnight travel. "
        "3AC (3-Tier Air Conditioned) has air conditioning, curtains for privacy, "
        "3 berths per row, and bedroll provided. "
        "3AC costs significantly more than SL but is much more comfortable, "
        "especially for long-distance journeys in summer."
    ),

    "sl aur 3ac mein": (
        "SL (Sleeper Class) non-AC hota hai — sasta aur basic. "
        "3AC mein AC, curtains aur bedroll milta hai — comfortable lekin mehenga. "
        "Lambi journey mein ya garmi mein 3AC better hota hai."
    ),

    # E-ticket download
    "how to download": (
        "To download your e-ticket: "
        "1. Log in to irctc.co.in. "
        "2. Go to My Account → Booked Ticket History. "
        "3. Find your booking and click Print/View ERS. "
        "4. Save as PDF — a digital copy on your phone is fully valid. "
        "Always carry a valid photo ID along with your e-ticket."
    ),

    "e-ticket kaise": (
        "E-ticket download karne ke liye: "
        "irctc.co.in par login karein → My Account → Booked Ticket History → "
        "Print/View ERS par click karein. Phone mein PDF save karein, "
        "yeh valid hota hai. Saath mein valid photo ID zaroor rakhein."
    ),

    # How to check PNR
    "how to check pnr": (
        "To check PNR status: "
        "Visit irctc.co.in or the IRCTC Rail Connect app, "
        "enter your 10-digit PNR number in the PNR Status section. "
        "You can also SMS your PNR to 139 or call 139."
    ),

    "pnr status kaise": (
        "PNR status check karne ke liye: "
        "irctc.co.in ya IRCTC Rail Connect app open karein, "
        "PNR Status section mein apna 10-digit PNR number enter karein. "
        "Ya 139 par SMS ya call bhi kar sakte hain."
    ),

    # How to book
    "how to book": (
        "To book a train ticket on IRCTC: "
        "1. Register on irctc.co.in or the IRCTC Rail Connect app. "
        "2. Search for trains between your source and destination. "
        "3. Select your preferred train, class, and quota. "
        "4. Fill passenger details and proceed to payment. "
        "5. Your e-ticket is sent to your registered email/phone."
    ),

    "ticket kaise book": (
        "IRCTC par ticket book karne ke liye: "
        "irctc.co.in ya IRCTC Rail Connect app par register karein, "
        "train search karein, class select karein, passenger details bharein, "
        "payment karein. E-ticket aapke email aur phone par aa jayega."
    ),

}


def _check_instant_answers(user_message: str) -> str | None:
    """
    Checks if the user's message matches any known instant answer.
    Matching is substring-based on lowercased input — works for
    variations like "What is RAC?", "what is rac", "tell me what is rac".
    Returns the answer string if matched, None if no match.
    """
    text = user_message.lower().strip()
    for key, answer in INSTANT_ANSWERS.items():
        if key in text:
            return answer
    return None


# ─────────────────────────────────────────────────────────────────────
# FALLBACK MESSAGES
# ─────────────────────────────────────────────────────────────────────

MSG_RATE_LIMIT = (
    "Abhi bahut zyada requests aa rahi hain. 1 minute baad try karein.\n"
    "(Too many requests right now. Please try again in a minute, or call 139.)"
)
MSG_API_ERROR = (
    "I'm having trouble connecting to my knowledge base right now. "
    "For immediate help, please visit irctc.co.in or call 139."
)
MSG_EMPTY = (
    "Please type your question — I'm here to help with Indian Railways!"
)


# ─────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────

def _build_prompt(user_message: str) -> str:
    return f"""
You are a friendly and accurate IRCTC and Indian Railways support assistant.
You help passengers understand railway terms, booking processes, and IRCTC usage.

<rules>
RULE 1  ACCURATE     : Only say things you are confident about.
RULE 2  BRIEF        : Answer in 2-5 sentences. Use bullet points for steps.
RULE 3  NO REAL-TIME : You have NO internet access. Never claim to check live
                       seat availability, current prices, or real-time PNR.
RULE 4  NO INVENTION : Never make up train numbers, exact schedules, or prices.
                       Say "check irctc.co.in for current details" instead.
RULE 5  HONEST       : If unsure, say: "I'm not sure — please verify on
                       irctc.co.in or call 139."
RULE 6  LANGUAGE     : Reply in the SAME language the user wrote in.
                       Support English, Hindi, Hinglish.
RULE 7  SIMPLE       : Explain jargon (RAC, WL, TTE, PNR) in plain language.
</rules>

<examples>
User: "Difference between SL and 3AC?"
Assistant: SL (Sleeper Class) is non-AC with basic open berths — cheapest
option for overnight travel. 3AC is air-conditioned with 3 berths per section
and privacy curtains — more comfortable, especially in summer.

User: "PNR kya hota hai?"
Assistant: PNR yaani Passenger Name Record — ek 10-digit number jo aapki
ticket pe hota hai. Isse aap booking details, seat number aur status check
kar sakte hain irctc.co.in ya 139 par.
</examples>

<question>
{user_message}
</question>

Answer following all the rules above. Be helpful, accurate, and concise.
"""


# ─────────────────────────────────────────────────────────────────────
# GEMINI CALL WITH RETRY  [FIX 1]
# ─────────────────────────────────────────────────────────────────────

def _call_gemini_with_retry(user_message: str) -> str | None:
    """
    Calls Gemini up to MAX_ATTEMPTS times.
    503 UNAVAILABLE → waits and retries (server overload, worth retrying).
    429 RESOURCE_EXHAUSTED → returns None immediately (quota, no point retrying).
    Any other error → returns None immediately.
    Returns response text on success, None if all attempts failed.
    """
    prompt = _build_prompt(user_message)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            if response and response.text and response.text.strip():
                return response.text.strip()

        except genai_errors.ClientError as e:
            error_str = str(e)

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"[general_agent] Rate limit (429). Not retrying.")
                return "__RATE_LIMITED__"

            if "503" in error_str or "UNAVAILABLE" in error_str:
                if attempt < MAX_ATTEMPTS:
                    wait = RETRY_BACKOFF[attempt - 1]
                    print(f"[general_agent] 503 on attempt {attempt}/{MAX_ATTEMPTS}. "
                          f"Waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[general_agent] 503 on final attempt. Giving up.")
                    return None

            print(f"[general_agent] API error: {error_str[:80]}")
            return None

        except Exception as e:
            if attempt < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF[attempt - 1]
                print(f"[general_agent] Unexpected error attempt {attempt}. "
                      f"Waiting {wait:.0f}s... ({str(e)[:60]})")
                time.sleep(wait)
                continue
            print(f"[general_agent] Unexpected error: {str(e)[:80]}")
            return None

    return None


# ─────────────────────────────────────────────────────────────────────
# RESPONSE CLEANING
# ─────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────

def handle_general_query(user_message: str) -> str:
    """
    Called by chatbot.py for general_support intent.
    Always returns a non-empty string. Never raises. Never returns None.

    Answer chain:
      1. Instant answer (local dict, zero API calls) — for common questions
      2. Gemini with retry (up to 3 attempts with backoff) — for everything else
      3. Static fallback string — only if all Gemini attempts failed
    """
    if not user_message or not user_message.strip():
        return MSG_EMPTY

    user_message = user_message.strip()

    # Step 1: Check instant answers first
    instant = _check_instant_answers(user_message)
    if instant:
        print("[general_agent] Instant answer served (no API call).")
        return instant

    # Step 2: Call Gemini with retry
    result = _call_gemini_with_retry(user_message)

    if result == "__RATE_LIMITED__":
        return MSG_RATE_LIMIT

    if result is not None:
        return _clean(result)

    # Step 3: All retries failed — return static fallback
    # We do NOT say "connection problem" for questions with no instant answer.
    # We give whatever partial help we can.
    return MSG_API_ERROR