from pathlib import Path
from datetime import datetime, date, time
from typing import Optional, Dict, Any, Tuple, Sequence
import csv
import uuid


LEADS_CSV_PATH = Path("storage/leads.csv")
BOOKINGS_CSV_PATH = Path("storage/bookings.csv")


LEAD_COLUMNS = [
    "lead_id",
    "timestamp",
    "username",
    "budget",
    "needs",
    "preferred_make",
    "preferred_model",
    "selected_listing_id",
    "selected_car_title",
    "booking_date",
    "booking_time",
    "lead_status",
    "notes",
]


BOOKING_COLUMNS = [
    "booking_id",
    "timestamp",
    "username",
    "selected_listing_id",
    "selected_car_title",
    "booking_date",
    "booking_time",
    "booking_status",
    "notes",
]


def ensure_csv_exists(path: Path, columns: list) -> None:
    """
    Create a CSV file with headers if it does not already exist.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with open(path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()


def ensure_storage_files_exist() -> None:
    """
    Make sure both leads.csv and bookings.csv exist.
    """

    ensure_csv_exists(LEADS_CSV_PATH, LEAD_COLUMNS)
    ensure_csv_exists(BOOKINGS_CSV_PATH, BOOKING_COLUMNS)


def parse_booking_date(booking_date: str) -> date:
    """
    Parse booking date from YYYY-MM-DD format.
    """

    return datetime.strptime(booking_date, "%Y-%m-%d").date()


def parse_booking_time(booking_time: str) -> time:
    """
    Parse booking time.

    Supports:
    - 15:00
    - 15:30
    - 3:00 PM
    - 3 PM
    """

    booking_time = booking_time.strip()
    formats = ["%H:%M", "%I:%M %p", "%I %p"]

    for fmt in formats:
        try:
            return datetime.strptime(booking_time, fmt).time()
        except ValueError:
            continue

    raise ValueError("Invalid time format. Use HH:MM, like 15:00, or 3 PM.")


def validate_booking_slot(booking_date: str, booking_time: str) -> Tuple[bool, str]:
    """
    Validate if a viewing/test-drive slot is allowed.

    Rules:
    - Monday to Saturday only
    - 8 AM to 8 PM only
    """

    try:
        parsed_date = parse_booking_date(booking_date)
        parsed_time = parse_booking_time(booking_time)
    except ValueError as error:
        return False, str(error)

    # Monday = 0, Sunday = 6
    if parsed_date.weekday() == 6:
        return False, "Bookings are not available on Sundays. Please choose Monday to Saturday."

    opening_time = time(8, 0)
    closing_time = time(20, 0)

    if parsed_time < opening_time:
        return False, "Bookings are only available from 8:00 AM onwards."

    if parsed_time > closing_time:
        return False, "Bookings are only available until 8:00 PM."

    return True, "Booking slot is valid."


def qualify_lead(
    budget: str = "",
    selected_listing_id: Optional[int] = None,
    booking_date: str = "",
    booking_time: str = "",
    preferred_make: str = "",
    preferred_model: str = "",
    needs: str = "",
    source_intent: str = "",
) -> str:
    """
    Classify the lead as Cold, Warm, or Hot.

    Cold:
    - user is browsing only

    Warm:
    - user gave budget, preferences, or needs

    Hot:
    - user selected a car and booked a viewing
    """

    has_budget = bool(budget.strip())
    has_selected_car = selected_listing_id is not None
    has_booking = bool(booking_date.strip() and booking_time.strip())
    has_preferences = any([
        preferred_make.strip(),
        preferred_model.strip(),
        needs.strip(),
    ])

    if has_selected_car and has_booking:
        return "Hot"

    if source_intent in {"booking", "compare_listings"}:
        return "Warm"

    if has_budget or has_preferences:
        return "Warm"

    return "Cold"


def should_create_lead(
    source_intent: str = "",
    is_follow_up: bool = False,
    budget: str = "",
    preferred_make: str = "",
    preferred_model: str = "",
    desired_features: Optional[Sequence[str]] = None,
    needs: str = "",
) -> bool:
    """Decide whether an interaction represents new commercial intent."""

    normalized_intent = source_intent.strip().lower()

    if is_follow_up or normalized_intent == "car_details":
        return False

    if normalized_intent in {"booking", "compare_listings"}:
        return True

    # Keep direct/manual save_lead calls backward compatible.
    if not normalized_intent:
        return True

    if normalized_intent not in {
        "car_search",
        "lead_capture",
        "general_car_advice",
    }:
        return False

    has_budget = bool(budget.strip())
    has_vehicle_preference = bool(
        preferred_make.strip() or preferred_model.strip()
    )
    has_desired_feature = any(
        str(feature).strip()
        for feature in (desired_features or [])
        if feature
    )
    has_explicit_buying_intent = (
        normalized_intent == "lead_capture" and bool(needs.strip())
    )

    return bool(
        has_budget
        or has_vehicle_preference
        or has_desired_feature
        or has_explicit_buying_intent
    )


def save_lead(
    username: str,
    budget: str = "",
    needs: str = "",
    preferred_make: str = "",
    preferred_model: str = "",
    selected_listing_id: Optional[int] = None,
    selected_car_title: str = "",
    booking_date: str = "",
    booking_time: str = "",
    notes: str = "",
    source_intent: str = "",
    desired_features: Optional[Sequence[str]] = None,
    is_follow_up: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Save a lead to leads.csv.

    This can be used even if the user has not booked yet.
    """

    if not should_create_lead(
        source_intent=source_intent,
        is_follow_up=is_follow_up,
        budget=budget,
        preferred_make=preferred_make,
        preferred_model=preferred_model,
        desired_features=desired_features,
        needs=needs,
    ):
        return None

    ensure_storage_files_exist()

    lead_status = qualify_lead(
        budget=budget,
        selected_listing_id=selected_listing_id,
        booking_date=booking_date,
        booking_time=booking_time,
        preferred_make=preferred_make,
        preferred_model=preferred_model,
        needs=needs,
        source_intent=source_intent,
    )

    lead = {
        "lead_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "username": username,
        "budget": budget,
        "needs": needs,
        "preferred_make": preferred_make,
        "preferred_model": preferred_model,
        "selected_listing_id": selected_listing_id if selected_listing_id is not None else "",
        "selected_car_title": selected_car_title,
        "booking_date": booking_date,
        "booking_time": booking_time,
        "lead_status": lead_status,
        "notes": notes,
    }

    with open(LEADS_CSV_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LEAD_COLUMNS)
        writer.writerow(lead)

    return lead


def save_booking(
    username: str,
    selected_listing_id: int,
    selected_car_title: str,
    booking_date: str,
    booking_time: str,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Save an official confirmed booking to bookings.csv.
    """

    ensure_storage_files_exist()

    booking = {
        "booking_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "username": username,
        "selected_listing_id": selected_listing_id,
        "selected_car_title": selected_car_title,
        "booking_date": booking_date,
        "booking_time": booking_time,
        "booking_status": "Confirmed",
        "notes": notes,
    }

    with open(BOOKINGS_CSV_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BOOKING_COLUMNS)
        writer.writerow(booking)

    return booking


def book_viewing(
    username: str,
    selected_listing_id: int,
    selected_car_title: str,
    booking_date: str,
    booking_time: str,
    budget: str = "",
    needs: str = "",
    preferred_make: str = "",
    preferred_model: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """
    Validate a viewing slot.

    If valid:
    - save an official booking to bookings.csv
    - save/update lead as Hot in leads.csv
    """

    is_valid, message = validate_booking_slot(booking_date, booking_time)

    if not is_valid:
        return {
            "success": False,
            "message": message,
            "booking": None,
            "lead": None,
        }

    booking = save_booking(
        username=username,
        selected_listing_id=selected_listing_id,
        selected_car_title=selected_car_title,
        booking_date=booking_date,
        booking_time=booking_time,
        notes=notes,
    )

    lead = save_lead(
        username=username,
        budget=budget,
        needs=needs,
        preferred_make=preferred_make,
        preferred_model=preferred_model,
        selected_listing_id=selected_listing_id,
        selected_car_title=selected_car_title,
        booking_date=booking_date,
        booking_time=booking_time,
        notes="Converted to Hot lead after confirmed booking.",
        source_intent="booking",
        desired_features=[needs] if needs else [],
    )

    return {
        "success": True,
        "message": "Viewing booked successfully.",
        "booking": booking,
        "lead": lead,
    }


if __name__ == "__main__":
    print("\nTEST 1: Cold lead → browsing only")
    cold_lead = save_lead(
        username="layan123",
        notes="User is browsing cars only."
    )
    print(cold_lead)

    print("\nTEST 2: Warm lead → has budget/preferences but no booking")
    warm_lead = save_lead(
        username="layan123",
        budget="AED 120,000",
        needs="Mercedes with warranty",
        preferred_make="Mercedes-Benz",
        preferred_model="C-Class",
        notes="User gave preferences but has not booked yet."
    )
    print(warm_lead)

    print("\nTEST 3: Friday at 3 PM → confirmed booking + Hot lead")
    result = book_viewing(
        username="layan123",
        selected_listing_id=2,
        selected_car_title="2019 Mercedes-Benz C-Class",
        booking_date="2026-07-10",
        booking_time="3 PM",
        budget="AED 120,000",
        needs="Mercedes with warranty",
        preferred_make="Mercedes-Benz",
        preferred_model="C-Class",
    )
    print(result)

    print("\nTEST 4: Sunday at 2 PM → rejected booking")
    result = book_viewing(
        username="layan123",
        selected_listing_id=2,
        selected_car_title="2019 Mercedes-Benz C-Class",
        booking_date="2026-07-05",
        booking_time="2 PM",
        budget="AED 120,000",
    )
    print(result)

    print("\nTEST 5: Monday at 7 AM → rejected booking")
    result = book_viewing(
        username="layan123",
        selected_listing_id=2,
        selected_car_title="2019 Mercedes-Benz C-Class",
        booking_date="2026-07-06",
        booking_time="7 AM",
        budget="AED 120,000",
    )
    print(result)

    print("\nTEST 6: Monday at 9 PM → rejected booking")
    result = book_viewing(
        username="layan123",
        selected_listing_id=2,
        selected_car_title="2019 Mercedes-Benz C-Class",
        booking_date="2026-07-06",
        booking_time="9 PM",
        budget="AED 120,000",
    )
    print(result)

    print(f"\nLeads saved here: {LEADS_CSV_PATH}")
    print(f"Bookings saved here: {BOOKINGS_CSV_PATH}")
