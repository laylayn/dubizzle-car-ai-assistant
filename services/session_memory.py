import re
from typing import Dict, Any, List, Optional


SESSIONS: Dict[str, Dict[str, Any]] = {}

POSITION_WORDS = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "sixth": 5,
    "seventh": 6,
    "eighth": 7,
    "ninth": 8,
    "tenth": 9,
}


def get_session(session_id: str) -> Dict[str, Any]:
    """
    Create or retrieve short-term memory for one chat session.
    """

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "messages": [],
            "last_results": [],
            "selected_car": None,
            "preferences": {}
        }

    return SESSIONS[session_id]


def add_message(session_id: str, role: str, content: str) -> None:
    """
    Save a user or assistant message in the current session.
    """

    session = get_session(session_id)
    session["messages"].append({
        "role": role,
        "content": content
    })


def save_last_results(session_id: str, cars: List[Dict[str, Any]]) -> None:
    """
    Save the latest search results shown to the user.
    """

    session = get_session(session_id)
    session["last_results"] = cars


def get_last_results(session_id: str) -> List[Dict[str, Any]]:
    """
    Get the latest cars shown to the user.
    """

    session = get_session(session_id)
    return session.get("last_results", [])


def save_selected_car(session_id: str, car: Dict[str, Any]) -> None:
    """
    Save the car currently being discussed.
    """

    session = get_session(session_id)
    session["selected_car"] = car


def get_selected_car(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the currently selected car.
    """

    session = get_session(session_id)
    return session.get("selected_car")


def update_preferences(session_id: str, preferences: Dict[str, Any]) -> None:
    """
    Store temporary preferences from the current session.
    Example:
    {
        "make": "honda",
        "keyword": "warranty"
    }
    """

    session = get_session(session_id)
    session["preferences"].update(preferences)


def get_preferences(session_id: str) -> Dict[str, Any]:
    """
    Return the current session preferences.
    """

    session = get_session(session_id)
    return session.get("preferences", {})


def get_car_by_position(session_id: str, position: int) -> Optional[Dict[str, Any]]:
    """
    Get a car from the last results.

    position:
    0 = first car
    1 = second car
    2 = third car
    """

    last_results = get_last_results(session_id)

    if 0 <= position < len(last_results):
        selected_car = last_results[position]
        save_selected_car(session_id, selected_car)
        return selected_car

    return None


def extract_car_positions(user_message: str) -> List[int]:
    """Extract every ordinal result position in the order the user stated it."""

    word_pattern = "|".join(POSITION_WORDS)
    position_pattern = re.compile(
        rf"\b(?P<word>{word_pattern})\b|"
        r"\b(?P<number>\d+)(?:st|nd|rd|th)\b",
        re.IGNORECASE,
    )
    positions = []

    for match in position_pattern.finditer(user_message):
        if match.group("word"):
            position = POSITION_WORDS[match.group("word").lower()]
        else:
            position = int(match.group("number")) - 1

        if position >= 0 and position not in positions:
            positions.append(position)

    return positions


def resolve_car_reference(session_id: str, user_message: str) -> Optional[Dict[str, Any]]:
    """
    Resolve references like:
    - first one
    - second car
    - third listing
    - it
    - that car
    """

    message = user_message.lower().strip()

    positions = extract_car_positions(message)
    if positions:
        return get_car_by_position(session_id, positions[0])

    pronoun_pattern = (
        r"\b(?:it|that car|this car|the car|selected car|that one|this one)\b"
    )

    if re.search(pronoun_pattern, message):
        return get_selected_car(session_id)

    contextual_demonstrative_pattern = (
        r"\b(?:would|could|can|does|do|is|was|what about|how about)\s+"
        r"(?:this|that)\b"
    )
    if re.search(contextual_demonstrative_pattern, message):
        return get_selected_car(session_id)

    return None


def is_follow_up_about_selected_car(message: str) -> bool:
    """
    Detect a natural attribute question that should use the selected car.

    Explicit references are handled by resolve_car_reference. This helper also
    supports short follow-ups such as "what is the price?" after a car has
    already been selected.
    """

    message_lower = message.lower().strip()

    reference_pattern = (
        r"\b(?:it|that car|this car|the car|selected car|that one|this one|"
        r"first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)\b"
    )
    attribute_pattern = (
        r"(?:\b(?:price|cost|amount|aed|dhs?|dirhams?|color|colour|"
        r"mileage|kilomet(?:er|re)s?|km|"
        r"warranty|gcc|year|make|model|trim|feature|features|option|options)\b"
        r"|how much)"
    )
    question_pattern = (
        r"^(?:what|which|does|do|is|are|how|can|could|tell me|show me|"
        r"price|cost|amount|aed)\b"
    )

    if re.search(reference_pattern, message_lower):
        return True

    if re.search(
        r"\b(?:would|could|can|does|do|is|was|what about|how about)\s+"
        r"(?:this|that)\b",
        message_lower,
    ):
        return True

    if re.search(r"\b(?:tell me more|more details|what else)\b", message_lower):
        return True

    return bool(
        re.search(attribute_pattern, message_lower)
        and re.search(question_pattern, message_lower)
    )


def reset_session(session_id: str) -> None:
    """
    Clear one session's short-term memory.
    """

    if session_id in SESSIONS:
        del SESSIONS[session_id]
