from pathlib import Path
from datetime import datetime, date, time
from typing import Optional, Dict, Any, Tuple, Sequence
import csv
import re
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

WEEKDAY_ALIASES = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


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


def extract_booking_slot(
    message: str,
    reference_date: Optional[date] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract an ISO date and 24-hour time from natural booking text."""

    text = str(message or "").lower().strip()
    today = reference_date or date.today()
    booking_date = None
    booking_time = None

    iso_date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if iso_date_match:
        candidate = iso_date_match.group(1)
        try:
            parse_booking_date(candidate)
            booking_date = candidate
        except ValueError:
            booking_date = None
    elif re.search(r"\btoday\b", text):
        booking_date = today.isoformat()
    elif re.search(r"\btomorrow\b", text):
        booking_date = date.fromordinal(today.toordinal() + 1).isoformat()
    else:
        weekday_pattern = "|".join(
            sorted(WEEKDAY_ALIASES, key=len, reverse=True)
        )
        weekday_match = re.search(
            rf"\b({weekday_pattern})\b",
            text,
        )
        if weekday_match:
            requested_weekday = WEEKDAY_ALIASES[weekday_match.group(1)]
            days_ahead = (requested_weekday - today.weekday()) % 7
            if (
                days_ahead == 0
                and re.search(
                    rf"\bnext\s+{re.escape(weekday_match.group(1))}\b",
                    text,
                )
            ):
                days_ahead = 7
            booking_date = date.fromordinal(
                today.toordinal() + days_ahead
            ).isoformat()

    twelve_hour_match = re.search(
        r"\b(1[0-2]|0?[1-9])"
        r"(?::([0-5]\d))?\s*([ap])\.?m\.?\b",
        text,
    )
    if twelve_hour_match:
        hour = int(twelve_hour_match.group(1))
        minute = int(twelve_hour_match.group(2) or 0)
        meridiem = twelve_hour_match.group(3)
        if meridiem == "a" and hour == 12:
            hour = 0
        elif meridiem == "p" and hour != 12:
            hour += 12
        booking_time = f"{hour:02d}:{minute:02d}"
    else:
        twenty_four_hour_match = re.search(
            r"\b([01]?\d|2[0-3]):([0-5]\d)\b",
            text,
        )
        if twenty_four_hour_match:
            booking_time = (
                f"{int(twenty_four_hour_match.group(1)):02d}:"
                f"{int(twenty_four_hour_match.group(2)):02d}"
            )

    return booking_date, booking_time


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
