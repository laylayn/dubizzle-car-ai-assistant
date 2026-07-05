from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import sqlite3
import json


DB_PATH = Path("storage/users.db")


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
            user_id TEXT PRIMARY KEY,
            name TEXT,
            last_budget TEXT,
            preferred_make TEXT,
            preferred_model TEXT,
            preferred_body_type TEXT,
            last_seen_car TEXT,
            liked_cars TEXT,
            last_query TEXT,
            created_at TEXT,
            updated_at TEXT
        )
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


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user profile by user_id.
    Returns None if the user does not exist.
    """

    init_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE user_id = ?",
        (user_id,)
    )

    row = cursor.fetchone()
    conn.close()

    return row_to_dict(row)


def create_user(user_id: str, name: str = "") -> Dict[str, Any]:
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
            user_id,
            name,
            last_budget,
            preferred_make,
            preferred_model,
            preferred_body_type,
            last_seen_car,
            liked_cars,
            last_query,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            name,
            "",
            "",
            "",
            "",
            "",
            json.dumps([]),
            "",
            now,
            now
        )
    )

    conn.commit()
    conn.close()

    return get_user(user_id)


def get_or_create_user(user_id: str, name: str = "") -> Dict[str, Any]:
    """
    Get an existing user, or create one if they are new.
    """

    existing_user = get_user(user_id)

    if existing_user:
        return existing_user

    return create_user(user_id=user_id, name=name)


def update_user_memory(
    user_id: str,
    name: Optional[str] = None,
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

    user = get_or_create_user(user_id=user_id, name=name or "")
    updated_user = user.copy()

    if name is not None:
        updated_user["name"] = name

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
            name = ?,
            last_budget = ?,
            preferred_make = ?,
            preferred_model = ?,
            preferred_body_type = ?,
            last_seen_car = ?,
            last_query = ?,
            updated_at = ?
        WHERE user_id = ?
        """,
        (
            updated_user.get("name", ""),
            updated_user.get("last_budget", ""),
            updated_user.get("preferred_make", ""),
            updated_user.get("preferred_model", ""),
            updated_user.get("preferred_body_type", ""),
            updated_user.get("last_seen_car", ""),
            updated_user.get("last_query", ""),
            updated_user.get("updated_at", ""),
            user_id
        )
    )

    conn.commit()
    conn.close()

    return get_user(user_id)


def add_liked_car(user_id: str, listing_id: int) -> Dict[str, Any]:
    """
    Add a liked car listing ID to the user's long-term memory.
    """

    init_db()

    user = get_or_create_user(user_id)
    liked_cars: List[int] = user.get("liked_cars", [])

    if listing_id not in liked_cars:
        liked_cars.append(listing_id)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET liked_cars = ?, updated_at = ?
        WHERE user_id = ?
        """,
        (
            json.dumps(liked_cars),
            datetime.utcnow().isoformat(),
            user_id
        )
    )

    conn.commit()
    conn.close()

    return get_user(user_id)


def build_welcome_message(user: Dict[str, Any]) -> str:
    """
    Build a friendly returning-user message.
    """

    name = user.get("name") or "there"
    preferred_make = user.get("preferred_make")
    preferred_model = user.get("preferred_model")
    last_budget = user.get("last_budget")
    last_seen_car = user.get("last_seen_car")
    last_query = user.get("last_query")

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
        return f"Welcome back, {name}. Last time, " + ", ".join(memory_parts) + "."

    if last_query:
        return f"Welcome back, {name}. Last time, you asked about: {last_query}"

    return f"Welcome back, {name}. I do not have saved car preferences for you yet."


def get_memory_summary(user_id: str) -> Dict[str, Any]:
    """
    Return a clean summary for the future Streamlit sidebar memory panel.
    """

    user = get_user(user_id)

    if not user:
        return {
            "returning_user": False,
            "message": "New user. No saved memory yet.",
            "preferred_make": "",
            "preferred_model": "",
            "last_budget": "",
            "last_seen_car": "",
            "liked_cars": []
        }

    return {
        "returning_user": True,
        "message": build_welcome_message(user),
        "preferred_make": user.get("preferred_make", ""),
        "preferred_model": user.get("preferred_model", ""),
        "last_budget": user.get("last_budget", ""),
        "last_seen_car": user.get("last_seen_car", ""),
        "liked_cars": user.get("liked_cars", [])
    }


if __name__ == "__main__":
    print("\nInitializing database...")
    init_db()

    print("\nSESSION 1: Creating/updating user memory")

    user_id = "layan123"

    user = get_or_create_user(user_id=user_id, name="Layan")
    print("Initial user:")
    print(user)

    updated_user = update_user_memory(
        user_id=user_id,
        name="Layan",
        last_budget="AED 120,000",
        preferred_make="Mercedes-Benz",
        preferred_model="C-Class",
        preferred_body_type="Sedan",
        last_seen_car="2019 Mercedes-Benz C-Class",
        last_query="I want a Mercedes with warranty"
    )

    add_liked_car(user_id=user_id, listing_id=2)

    print("\nUpdated user:")
    print(updated_user)

    print("\nSESSION 2: Simulating returning user")

    returning_user = get_user(user_id)
    print(build_welcome_message(returning_user))

    print("\nMemory summary for Streamlit sidebar:")
    print(get_memory_summary(user_id))
