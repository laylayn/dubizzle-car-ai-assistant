from typing import Dict, Any, List, Optional


SESSIONS: Dict[str, Dict[str, Any]] = {}


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


def resolve_car_reference(session_id: str, user_message: str) -> Optional[Dict[str, Any]]:
    """
    Resolve references like:
    - first one
    - second car
    - third listing
    - it
    - that car
    """

    message = user_message.lower()

    position_words = {
        "first": 0,
        "1st": 0,
        "one": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4,
    }

    for word, position in position_words.items():
        if word in message:
            return get_car_by_position(session_id, position)

    pronouns = ["it", "that", "that car", "this car", "the car", "selected car"]

    if any(pronoun in message for pronoun in pronouns):
        return get_selected_car(session_id)

    return None


def reset_session(session_id: str) -> None:
    """
    Clear one session's short-term memory.
    """

    if session_id in SESSIONS:
        del SESSIONS[session_id]