"""
chatbot.py
Location: back/agents/chatbot.py

Run as:
    cd back/agents
    python chatbot.py --chat     # interactive mode
    python chatbot.py            # single test query
"""

from router       import detect_intent
from general_agent import handle_general_query
from policy_rag   import answer_policy_query

# ─────────────────────────────────────────────────────────────────────
# STUBS — return None until real agents are built
# None from a stub triggers the general_agent fallback automatically.
# Replacement: delete stub, add real import above.
# ─────────────────────────────────────────────────────────────────────

def _handle_refund_issue(msg: str):   return None   # Step 5: pnr_agent.py
def _handle_tourism(msg: str):        return None   # Step 8: tourism_agent.py
def _handle_trip_planning(msg: str):  return None   # Step 9: trip_planner.py


# ─────────────────────────────────────────────────────────────────────
# AGENT MAP
# Maps intent → function.
# "general_support" is intentionally absent — it is the fallback,
# not a specialist destination.
# ─────────────────────────────────────────────────────────────────────

_AGENT_MAP: dict = {
    "policy_query":    answer_policy_query,
    "refund_issue":    _handle_refund_issue,
    "tourism_package": _handle_tourism,
    "trip_planning":   _handle_trip_planning,
}

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

_LAST_RESORT = (
    "I'm having trouble right now. "
    "Please visit irctc.co.in or call IRCTC at 139."
)

_MIN_REPLY_LEN = 10


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _is_usable(reply) -> bool:
    """True only when reply is a non-empty string of meaningful length."""
    return (
        isinstance(reply, str)
        and len(reply.strip()) >= _MIN_REPLY_LEN
    )


def _safe_call(fn, message: str, label: str) -> str | None:
    """
    Calls fn(message). Returns the result if usable, None otherwise.
    Catches all exceptions — a crashing specialist must never crash the app.
    """
    try:
        reply = fn(message)
        if _is_usable(reply):
            return reply
        return None
    except Exception as exc:
        print(f"  [{label}] exception: {str(exc)[:120]}")
        return None


# ─────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────

def get_response(user_message: str) -> str:
    """
    Routing pipeline. Always returns a non-empty string. Never raises.

    Step 1  Empty guard
    Step 2  Classify intent via router.py
    Step 3  Call specialist from _AGENT_MAP (if intent is mapped)
            policy_rag returns None when score < 0.35 → goes to Step 4
    Step 4  Fallback: general_agent handles it
            (also handles general_support intent directly)
    Step 5  Last resort hardcoded string if everything above failed
    """

    # Step 1
    if not user_message or not user_message.strip():
        return "Please type your question — I'm here to help with Indian Railways!"
    user_message = user_message.strip()

    # Step 2
    intent = "general_support"
    try:
        result     = detect_intent(user_message)
        intent     = result.get("intent",     "general_support")
        confidence = result.get("confidence", "low")
        source     = result.get("source",     "unknown")
        print(f"\n[Router] {intent} | {confidence} | {source}")
    except Exception as exc:
        print(f"\n[Router] failed ({str(exc)[:60]}) — using general_support")

    # Step 3 — specialist
    reply: str | None = None
    if intent in _AGENT_MAP:
        fn    = _AGENT_MAP[intent]
        label = fn.__name__ if hasattr(fn, "__name__") else intent
        print(f"  [{label}] calling...")
        reply = _safe_call(fn, user_message, label)

        if reply is None:
            print(f"  [{label}] returned None — falling back to general_agent")

    # Step 4 — general_agent
    # Runs when: (a) intent == general_support, or (b) specialist returned None
    if reply is None:
        print("  [general_agent] calling...")
        reply = _safe_call(handle_general_query, user_message, "general_agent")

    # Step 5 — absolute last resort
    if reply is None:
        print("  [last_resort] all agents failed")
        reply = _LAST_RESORT

    return reply


# ─────────────────────────────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────────────

def run_chat() -> None:
    print("\n" + "=" * 50)
    print("  IRCTC AI Assistant  |  type 'quit' to exit")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("Goodbye! Safe journey.")
            break

        print(f"\nAssistant: {get_response(user_input)}\n")


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--chat" in sys.argv:
        run_chat()
    else:
        # Minimal smoke test — one policy question, one general question
        for q in [
            "Can I get a refund on my Tatkal ticket?",
            "What is RAC?",
        ]:
            print(f"\nQ: {q}")
            print(f"A: {get_response(q)}")
        print("\nRun 'python chatbot.py --chat' to start interactive mode.")