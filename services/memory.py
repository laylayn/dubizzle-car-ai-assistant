from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import sqlite3
import json
import re
import uuid


DB_PATH = Path("storage/users.db")
MEANINGFUL_EVENT_TYPES = {
    "inventory_search",
    "preference_search",
    "booking",
    "booking_intent",
    "liked_car",
    "comparison",
    "memory_recall",
}


def get_connection():
    """
    Create a SQLite connection.
    The storage folder is created automatically if it does not exist.
    """

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the users table if it does not already exist.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            last_budget TEXT,
            preferred_make TEXT,
            preferred_model TEXT,
            preferred_body_type TEXT,
            last_seen_car TEXT,
            liked_cars TEXT,
            last_query TEXT,
            memory_summary TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    existing_columns = {
        row["name"]
        for row in cursor.execute("PRAGMA table_info(users)").fetchall()
    }
    if "memory_summary" not in existing_columns:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN memory_summary TEXT DEFAULT ''"
        )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_interactions (
            interaction_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            query TEXT,
            make TEXT,
            model TEXT,
            budget TEXT,
            listing_id INTEGER,
            car_title TEXT,
            lead_quality TEXT,
            notes TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_interactions_username_timestamp
        ON user_interactions (username, timestamp DESC)
        """
    )

    conn.commit()
    conn.close()


def row_to_dict(row) -> Optional[Dict[str, Any]]:
    """
    Convert a SQLite row into a normal Python dictionary.
    """

    if row is None:
        return None

    user = dict(row)

    try:
        user["liked_cars"] = json.loads(user.get("liked_cars") or "[]")
    except json.JSONDecodeError:
        user["liked_cars"] = []

    return user


def get_user(username: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user profile by username.
    Returns None if the user does not exist.
    """

    init_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    )

    row = cursor.fetchone()
    conn.close()

    return row_to_dict(row)


def create_user(username: str) -> Dict[str, Any]:
    """
    Create a new user profile.
    """

    init_db()

    now = datetime.utcnow().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO users (
            username,
            last_budget,
            preferred_make,
            preferred_model,
            preferred_body_type,
            last_seen_car,
            liked_cars,
            last_query,
            memory_summary,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            "",
            "",
            "",
            "",
            "",
            json.dumps([]),
            "",
            "",
            now,
            now
        )
    )

    conn.commit()
    conn.close()

    return get_user(username)


def get_or_create_user(username: str) -> Dict[str, Any]:
    """
    Get an existing user, or create one if they are new.
    """

    existing_user = get_user(username)

    if existing_user:
        return existing_user

    return create_user(username=username)


def update_user_memory(
    username: str,
    last_budget: Optional[str] = None,
    preferred_make: Optional[str] = None,
    preferred_model: Optional[str] = None,
    preferred_body_type: Optional[str] = None,
    last_seen_car: Optional[str] = None,
    last_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update long-term user memory.
    Only fields provided will be updated.
    """

    init_db()

    user = get_or_create_user(username=username)
    updated_user = user.copy()

    if last_budget is not None:
        updated_user["last_budget"] = last_budget

    if preferred_make is not None:
        updated_user["preferred_make"] = preferred_make

    if preferred_model is not None:
        updated_user["preferred_model"] = preferred_model

    if preferred_body_type is not None:
        updated_user["preferred_body_type"] = preferred_body_type

    if last_seen_car is not None:
        updated_user["last_seen_car"] = last_seen_car

    if last_query is not None:
        updated_user["last_query"] = last_query

    updated_user["updated_at"] = datetime.utcnow().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET
            last_budget = ?,
            preferred_make = ?,
            preferred_model = ?,
            preferred_body_type = ?,
            last_seen_car = ?,
            last_query = ?,
            updated_at = ?
        WHERE username = ?
        """,
        (
            updated_user.get("last_budget", ""),
            updated_user.get("preferred_make", ""),
            updated_user.get("preferred_model", ""),
            updated_user.get("preferred_body_type", ""),
            updated_user.get("last_seen_car", ""),
            updated_user.get("last_query", ""),
            updated_user.get("updated_at", ""),
            username
        )
    )

    conn.commit()
    conn.close()

    return get_user(username)


def add_liked_car(
    username: str,
    listing_id: int,
    make: str = "",
    model: str = "",
    car_title: str = "",
    refresh_summary: bool = True,
) -> Dict[str, Any]:
    """
    Add a liked car listing ID to the user's long-term memory.
    """

    init_db()

    user = get_or_create_user(username)
    liked_cars: List[int] = user.get("liked_cars", [])

    was_added = listing_id not in liked_cars
    if was_added:
        liked_cars.append(listing_id)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET liked_cars = ?, updated_at = ?
        WHERE username = ?
        """,
        (
            json.dumps(liked_cars),
            datetime.utcnow().isoformat(),
            username
        )
    )

    conn.commit()
    conn.close()

    if was_added:
        record_user_interaction(
            username=username,
            event_type="liked_car",
            make=make,
            model=model,
            listing_id=listing_id,
            car_title=car_title,
            notes="User saved this car as a liked listing.",
        )
        if refresh_summary:
            update_memory_summary(username)

    return get_user(username)


def record_user_interaction(
    username: str,
    event_type: str,
    query: str = "",
    make: str = "",
    model: str = "",
    budget: str = "",
    listing_id: Optional[int] = None,
    car_title: str = "",
    lead_quality: str = "",
    notes: str = "",
) -> Optional[Dict[str, Any]]:
    """Persist one meaningful car-shopping event, never general chat."""

    normalized_event = str(event_type or "").lower().strip()
    if normalized_event not in MEANINGFUL_EVENT_TYPES:
        return None
    if query and not is_meaningful_memory_query(query):
        query = ""

    interaction = {
        "interaction_id": str(uuid.uuid4()),
        "username": username,
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": normalized_event,
        "query": str(query or "").strip(),
        "make": str(make or "").strip(),
        "model": str(model or "").strip(),
        "budget": str(budget or "").strip(),
        "listing_id": listing_id,
        "car_title": str(car_title or "").strip(),
        "lead_quality": str(lead_quality or "").strip(),
        "notes": str(notes or "").strip(),
    }

    init_db()
    get_or_create_user(username)
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO user_interactions (
            interaction_id,
            username,
            timestamp,
            event_type,
            query,
            make,
            model,
            budget,
            listing_id,
            car_title,
            lead_quality,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuple(interaction.values()),
    )
    conn.commit()
    conn.close()
    return interaction


def get_recent_user_interactions(
    username: str,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """Load recent meaningful events for grounded summary generation."""

    init_db()
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT *
        FROM user_interactions
        WHERE username = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (username, max(1, int(limit))),
    ).fetchall()
    conn.close()
    return [dict(row) for row in reversed(rows)]


def build_memory_summary_fallback(
    user: Dict[str, Any],
    interactions: List[Dict[str, Any]],
) -> str:
    """Build a concise grounded summary when the LLM is unavailable."""

    interest_parts = [
        str(user.get("preferred_make") or "").strip(),
        str(user.get("preferred_model") or "").strip(),
    ]
    interest = " ".join(part for part in interest_parts if part).strip()
    body_type = str(user.get("preferred_body_type") or "").strip()
    budget = str(user.get("last_budget") or "").strip()
    sentences = []

    if interest:
        sentences.append(f"User has recently shown interest in {interest} cars.")
    elif body_type:
        sentences.append(f"User has recently shown interest in {body_type} cars.")

    if budget:
        sentences.append(f"Their saved budget is {budget}.")

    comparison_events = [
        interaction
        for interaction in interactions
        if interaction.get("event_type") == "comparison"
    ]
    if comparison_events:
        latest = comparison_events[-1]
        compared = latest.get("notes") or latest.get("query")
        if compared:
            sentences.append(f"They recently compared {compared}.")

    booking_events = [
        interaction
        for interaction in interactions
        if interaction.get("event_type") in {"booking", "booking_intent"}
    ]
    if booking_events:
        car_title = booking_events[-1].get("car_title")
        if car_title:
            sentences.append(f"They asked about booking {car_title}.")

    if not sentences and interactions:
        latest_query = next(
            (
                interaction.get("query")
                for interaction in reversed(interactions)
                if interaction.get("query")
            ),
            "",
        )
        if latest_query:
            sentences.append(
                f"User recently searched for cars using: {latest_query}."
            )

    return " ".join(sentences[:3])


def update_memory_summary(username: str) -> str:
    """Generate and save a short summary from structured memory and events."""

    user = get_or_create_user(username)
    interactions = get_recent_user_interactions(username)
    fallback = build_memory_summary_fallback(user, interactions)

    structured_memory = {
        "preferred_make": user.get("preferred_make", ""),
        "preferred_model": user.get("preferred_model", ""),
        "preferred_body_type": user.get("preferred_body_type", ""),
        "last_budget": user.get("last_budget", ""),
        "last_seen_car": user.get("last_seen_car", ""),
        "liked_cars": user.get("liked_cars", []),
    }
    summary_interactions = [
        {
            key: interaction.get(key)
            for key in [
                "event_type",
                "query",
                "make",
                "model",
                "budget",
                "listing_id",
                "car_title",
                "lead_quality",
                "notes",
            ]
        }
        for interaction in interactions
    ]
    from services.llm_agent import generate_memory_summary_text

    summary = generate_memory_summary_text(
        structured_memory=structured_memory,
        interactions=summary_interactions,
        fallback_summary=fallback,
    )
    summary = re.sub(r"\s+", " ", str(summary or "")).strip()

    conn = get_connection()
    conn.execute(
        """
        UPDATE users
        SET memory_summary = ?, updated_at = ?
        WHERE username = ?
        """,
        (summary, datetime.utcnow().isoformat(), username),
    )
    conn.commit()
    conn.close()
    return summary


def is_meaningful_memory_query(query: str) -> bool:
    """Exclude greeting-only text from long-term memory summaries."""

    normalized = re.sub(r"[^\w\s]", " ", str(query).lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return False

    greeting_patterns = [
        r"(?:hi|hello|hey|salam)",
        r"(?:hi|hello|hey|salam) (?:there|assistant|dubizzle)",
        r"good (?:morning|afternoon|evening)",
        r"(?:thanks|thank you)",
        r"(?:ok|okay|alright|sure|cool)",
        r"(?:got it|sounds good|understood)",
    ]
    return not any(
        re.fullmatch(pattern, normalized)
        for pattern in greeting_patterns
    )


def build_welcome_message(user: Dict[str, Any]) -> str:
    """
    Build a friendly returning-user message.
    """

    username = user.get("username") or "there"
    preferred_make = user.get("preferred_make")
    preferred_model = user.get("preferred_model")
    last_budget = user.get("last_budget")
    last_seen_car = user.get("last_seen_car")
    last_query = user.get("last_query")
    memory_summary = str(user.get("memory_summary") or "").strip()

    if memory_summary:
        return (
            f"Welcome back, {username}. "
            f"{format_memory_summary_for_user(memory_summary)}"
        )

    memory_parts = []

    if preferred_make:
        memory_parts.append(f"you were interested in {preferred_make}")

    if preferred_model:
        memory_parts.append(f"specifically {preferred_model}")

    if last_budget:
        memory_parts.append(f"with a budget of {last_budget}")

    if last_seen_car:
        memory_parts.append(f"and last viewed {last_seen_car}")

    if memory_parts:
        return f"Welcome back, {username}. Last time, " + ", ".join(memory_parts) + "."

    if last_query and is_meaningful_memory_query(last_query):
        return f"Welcome back, {username}. Last time, you asked about: {last_query}"

    return f"Welcome back, {username}. I do not have saved car preferences for you yet."


def format_memory_summary_for_user(summary: str) -> str:
    """Convert a stored third-person summary into natural user-facing wording."""

    text = str(summary or "").strip()
    replacements = [
        (r"^user has\b", "You have"),
        (r"^user is\b", "You are"),
        (r"^user recently\b", "You recently"),
        (r"^the user has\b", "You have"),
        (r"^the user is\b", "You are"),
        (r"^the user recently\b", "You recently"),
    ]
    for pattern, replacement in replacements:
        if re.search(pattern, text, re.IGNORECASE):
            return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
    return text


def get_memory_summary(username: str) -> Dict[str, Any]:
    """
    Return a clean summary for the future Streamlit sidebar memory panel.
    """

    user = get_user(username)

    if not user:
        return {
            "returning_user": False,
            "message": (
                f"Hello, {username}. I created a new profile for you. "
                "I do not have saved preferences yet."
            ),
            "preferred_make": "",
            "preferred_model": "",
            "last_budget": "",
            "last_seen_car": "",
            "liked_cars": [],
            "memory_summary": "",
        }

    return {
        "returning_user": True,
        "message": build_welcome_message(user),
        "preferred_make": user.get("preferred_make", ""),
        "preferred_model": user.get("preferred_model", ""),
        "last_budget": user.get("last_budget", ""),
        "last_seen_car": user.get("last_seen_car", ""),
        "liked_cars": user.get("liked_cars", []),
        "memory_summary": user.get("memory_summary", ""),
    }


if __name__ == "__main__":
    print("\nInitializing database...")
    init_db()

    print("\nSESSION 1: Creating/updating user memory")

    username = ""

    user = get_or_create_user(username=username)
    print("Initial user:")
    print(user)

    updated_user = update_user_memory(
        username=username,
        last_budget="AED 120,000",
        preferred_make="Mercedes-Benz",
        preferred_model="C-Class",
        preferred_body_type="Sedan",
        last_seen_car="2019 Mercedes-Benz C-Class",
        last_query="I want a Mercedes with warranty"
    )

    add_liked_car(username=username, listing_id=2)

    print("\nUpdated user:")
    print(updated_user)

    print("\nSESSION 2: Simulating returning user")

    returning_user = get_user(username)
    print(build_welcome_message(returning_user))

    print("\nMemory summary for Streamlit sidebar:")
    print(get_memory_summary(username))
