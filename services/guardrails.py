from functools import lru_cache
from typing import Dict, Any
import re

from services.data_loader import load_cars
from services.llm_agent import MAKE_ALIASES


ALLOWED_INTENTS = {
    "greeting",
    "car_search",
    "car_details",
    "compare_listings",
    "booking",
    "lead_capture",
    "returning_user_memory",
    "general_car_advice",
}


REFUSAL_MESSAGE = (
    "I can’t help with that request, but I can help with car searches, listing "
    "details, comparisons, and viewing bookings."
)

@lru_cache(maxsize=1)
def get_known_car_makes() -> list[str]:
    """Return dataset makes plus the deliberately small normalization aliases."""

    makes = set(MAKE_ALIASES) | set(MAKE_ALIASES.values())

    try:
        df = load_cars()
        dataset_makes = df["make"].dropna().astype(str).str.lower().str.strip()
        makes.update(make for make in dataset_makes if make)
    except (FileNotFoundError, KeyError):
        pass

    makes.update(make.replace("-", " ") for make in list(makes))
    return sorted(makes, key=len, reverse=True)


def contains_phrase(message: str, phrases: list[str]) -> bool:
    """
    Checks if any phrase exists in the message.
    Good for multi-word phrases like 'test drive' or 'write code'.
    """
    return any(phrase in message for phrase in phrases)


def contains_word(message: str, words: list[str]) -> bool:
    """
    Checks if a word exists as a full word.
    This prevents 'it' matching inside 'write' or 'capital'.
    """
    return any(re.search(rf"\b{re.escape(word)}\b", message) for word in words)


def is_greeting_message(message: str) -> bool:
    """Recognize standalone conversational greetings, not mixed requests."""

    normalized = re.sub(r"[^\w\s]", " ", message.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    greeting_patterns = [
        r"(?:hi|hello|hey|salam)",
        r"(?:hi|hello|hey|salam) (?:there|assistant|dubizzle)",
        r"(?:good morning|good afternoon|good evening)",
        r"(?:thanks|thank you)",
    ]
    return any(re.fullmatch(pattern, normalized) for pattern in greeting_patterns)


def detect_intent(message: str) -> str:
    message_lower = message.lower().strip()

    if not message_lower:
        return "blocked_non_automotive"

    # 1. Greetings
    if is_greeting_message(message_lower):
        return "greeting"

    # 2. Block competitor-related requests first
    competitor_keywords = [
        "yallamotor",
        "carswitch",
        "opensooq",
        "autotrader",
        "facebook marketplace",
        "sellanycar",
        "carvana",
        "cars24",
        "competitor",
        "compare dubizzle",
    ]

    if contains_phrase(message_lower, competitor_keywords):
        return "blocked_competitor"

    # 3. Block coding requests before car-detail checks
    coding_keywords = [
        "write code",
        "python code",
        "python script",
        "java code",
        "javascript",
        "html",
        "css",
        "debug this",
        "sql query",
        "build me an app",
        "create a function",
        "programming",
    ]

    if contains_phrase(message_lower, coding_keywords):
        return "blocked_coding"

    # 4. Block common non-automotive topics
    history_keywords = [
        "world war",
        "history of",
        "ancient",
        "empire",
        "civilization",
        "capital of",
    ]

    politics_keywords = [
        "election",
        "president",
        "government policy",
        "political",
        "vote",
        "war in",
    ]

    medical_legal_keywords = [
        "medical advice",
        "diagnose",
        "medicine",
        "legal advice",
        "lawsuit",
        "contract law",
    ]

    homework_keywords = [
        "solve my homework",
        "write my essay",
        "assignment answer",
        "do my report",
        "math problem",
    ]

    if contains_phrase(message_lower, history_keywords):
        return "blocked_history"

    if contains_phrase(message_lower, politics_keywords):
        return "blocked_politics"

    if contains_phrase(message_lower, medical_legal_keywords):
        return "blocked_medical_legal"

    if contains_phrase(message_lower, homework_keywords):
        return "blocked_random_homework"

    # 5. Compare listings before generic car search
    compare_keywords = [
        "compare these cars",
        "compare the cars",
        "compare the first",
        "first vs second",
        "which one is better",
        "better option",
        "difference between",
    ]

    if contains_phrase(message_lower, compare_keywords):
        return "compare_listings"

    # 6. Booking
    booking_keywords = [
        "book",
        "booking",
        "viewing",
        "test drive",
        "appointment",
        "schedule",
        "slot",
        "visit",
    ]

    if contains_word(message_lower, booking_keywords) or contains_phrase(message_lower, ["test drive"]):
        return "booking"

    # 7. Lead capture / preferences
    lead_keywords = [
        "my budget",
        "budget",
        "i want",
        "i need",
        "looking for",
        "prefer",
        "under",
        "range",
        "family car",
        "daily use",
        "for work",
    ]

    if contains_phrase(message_lower, lead_keywords):
        return "lead_capture"

    # 8. Car details / follow-up
    detail_phrases = [
        "tell me about",
        "first one",
        "second one",
        "third one",
        "that one",
        "that car",
        "this car",
        "selected car",
        "service history",
    ]

    detail_words = [
        "mileage",
        "warranty",
        "gcc",
        "specs",
        "accident",
        "description",
        "details",
        "it",
    ]

    has_inventory_scope = bool(
        contains_word(
            message_lower,
            ["cars", "listings", "inventory", "options", "vehicles"],
        )
        or contains_word(
            message_lower,
            ["show", "find", "search", "rank", "sort"],
        )
        or contains_word(message_lower, get_known_car_makes())
    )

    if (
        contains_phrase(message_lower, detail_phrases)
        or contains_word(message_lower, detail_words)
    ) and not has_inventory_scope:
        return "car_details"

    # 9. Car search
    car_search_phrases = [
        "show me",
        "find",
        "search",
        "available",
        "listings",
    ]

    car_words = [
        "cars",
        "car",
        "suv",
        "sedan",
        "coupe",
        "hatchback",
        "convertible",
        "truck",
        "pickup",
        "tesla",
    ]

    if (
        contains_phrase(message_lower, car_search_phrases)
        or contains_word(message_lower, car_words)
        or contains_word(message_lower, get_known_car_makes())
    ):
        return "car_search"

    # 10. General car advice
    general_car_keywords = [
        "reliable",
        "fuel efficient",
        "maintenance",
        "resale value",
        "family",
        "off-road",
        "luxury",
        "sports car",
    ]

    if contains_phrase(message_lower, general_car_keywords):
        return "general_car_advice"

    return "blocked_non_automotive"


def is_allowed_intent(intent: str) -> bool:
    return intent in ALLOWED_INTENTS


def apply_guardrails(message: str) -> Dict[str, Any]:
    intent = detect_intent(message)
    allowed = is_allowed_intent(intent)

    if allowed:
        return {
            "allowed": True,
            "intent": intent,
            "refusal_message": "",
        }

    return {
        "allowed": False,
        "intent": intent,
        "refusal_message": REFUSAL_MESSAGE,
    }


def log_intent(username: str, intent: str, results_count: int = 0) -> None:
    print(f"intent={intent} | username={username} | results={results_count}")


if __name__ == "__main__":
    test_messages = [
        "Hi",
        "Show me Mercedes cars",
        "I want an SUV with warranty under AED 120,000",
        "Tell me about the first one",
        "Can I book a viewing on Friday?",
        "Compare the first and second car",
        "Write me Python code",
        "Tell me about World War 2",
        "Compare dubizzle with YallaMotor",
        "Give me legal advice",
        "What is the capital of France?",
    ]

    username = ""

    for message in test_messages:
        result = apply_guardrails(message)
        log_intent(username=username, intent=result["intent"])

        print("\nUser message:")
        print(message)

        print("Guardrail result:")
        print(result)

        if not result["allowed"]:
            print("Assistant refusal:")
            print(result["refusal_message"])

        print("-" * 60)
