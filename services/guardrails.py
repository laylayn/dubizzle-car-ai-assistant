from functools import lru_cache
from typing import Dict, Any
import re

from services.data_loader import load_cars
from services.llm_agent import MAKE_ALIASES


ALLOWED_INTENTS = {
    "greeting",
    "chitchat",
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

COMPETITOR_KEYWORDS = [
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
CODING_KEYWORDS = [
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
HISTORY_KEYWORDS = [
    "world war",
    "history of",
    "ancient",
    "empire",
    "civilization",
    "capital of",
]
POLITICS_KEYWORDS = [
    "election",
    "president",
    "government policy",
    "political",
    "vote",
    "war in",
]
MEDICAL_LEGAL_KEYWORDS = [
    "medical advice",
    "diagnose",
    "medicine",
    "legal advice",
    "lawsuit",
    "contract law",
]
HOMEWORK_KEYWORDS = [
    "solve my homework",
    "write my essay",
    "assignment answer",
    "do my report",
    "math problem",
]
COMPARE_KEYWORDS = [
    "compare these cars",
    "compare the cars",
    "compare the first",
    "first vs second",
    "which one is better",
    "better option",
    "difference between",
]
BOOKING_KEYWORDS = [
    "book",
    "booking",
    "viewing",
    "test drive",
    "appointment",
    "schedule",
    "slot",
    "visit",
]
TEST_DRIVE_PHRASES = ["test drive"]
LEAD_KEYWORDS = [
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
DETAIL_PHRASES = [
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
DETAIL_WORDS = [
    "mileage",
    "warranty",
    "gcc",
    "specs",
    "accident",
    "description",
    "details",
]
INVENTORY_SCOPE_WORDS = [
    "cars",
    "listings",
    "inventory",
    "options",
    "vehicles",
]
SEARCH_ACTION_WORDS = ["show", "find", "search", "rank", "sort"]
CAR_SEARCH_PHRASES = [
    "show me",
    "find",
    "search",
    "available",
    "listings",
]
CAR_WORDS = [
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
GENERAL_CAR_KEYWORDS = [
    "reliable",
    "fuel efficient",
    "maintenance",
    "resale value",
    "family",
    "off-road",
    "luxury",
    "sports car",
]


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
    if contains_phrase(message_lower, COMPETITOR_KEYWORDS):
        return "blocked_competitor"

    # 3. Block coding requests before car-detail checks
    if contains_phrase(message_lower, CODING_KEYWORDS):
        return "blocked_coding"

    # 4. Block common non-automotive topics
    if contains_phrase(message_lower, HISTORY_KEYWORDS):
        return "blocked_history"

    if contains_phrase(message_lower, POLITICS_KEYWORDS):
        return "blocked_politics"

    if contains_phrase(message_lower, MEDICAL_LEGAL_KEYWORDS):
        return "blocked_medical_legal"

    if contains_phrase(message_lower, HOMEWORK_KEYWORDS):
        return "blocked_random_homework"

    # 5. Compare listings before generic car search
    if contains_phrase(message_lower, COMPARE_KEYWORDS):
        return "compare_listings"

    # 6. Booking
    if contains_word(message_lower, BOOKING_KEYWORDS) or contains_phrase(
        message_lower,
        TEST_DRIVE_PHRASES,
    ):
        return "booking"

    # 7. Lead capture / preferences
    if contains_phrase(message_lower, LEAD_KEYWORDS):
        return "lead_capture"

    # 8. Car details / follow-up
    has_inventory_scope = bool(
        contains_word(message_lower, INVENTORY_SCOPE_WORDS)
        or contains_word(message_lower, SEARCH_ACTION_WORDS)
        or contains_word(message_lower, get_known_car_makes())
    )

    if (
        contains_phrase(message_lower, DETAIL_PHRASES)
        or contains_word(message_lower, DETAIL_WORDS)
    ) and not has_inventory_scope:
        return "car_details"

    # 9. Car search
    if (
        contains_phrase(message_lower, CAR_SEARCH_PHRASES)
        or contains_word(message_lower, CAR_WORDS)
        or contains_word(message_lower, get_known_car_makes())
    ):
        return "car_search"

    # 10. General car advice
    if contains_phrase(message_lower, GENERAL_CAR_KEYWORDS):
        return "general_car_advice"

    # Explicitly unsafe/out-of-scope categories were handled above. Let the
    # conversational layer answer harmless small talk naturally.
    return "chitchat"


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
