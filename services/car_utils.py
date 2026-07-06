import re
from typing import Any, Dict, Optional

from services.data_loader import load_cars
from services.listing_attributes import COLOR_TERMS, OPTIONAL_LISTING_FIELDS


def row_to_car(row) -> Dict[str, Any]:
    """Convert a dataframe row into a clean car dictionary."""

    car = {
        "listing_id": int(row.get("listing_id", 0)),
        "year": int(row.get("year", 0)),
        "make": row.get("make", ""),
        "model": row.get("model", ""),
        "trim": row.get("trim", ""),
        "title": row.get("title", ""),
        "description": row.get("description", ""),
        "photo_url": row.get("photo_url", ""),
        "match_reason": "Matched by listing ID.",
    }
    for optional_field in OPTIONAL_LISTING_FIELDS:
        value = row.get(optional_field, "")
        if str(value).strip():
            car[optional_field] = value
    return car


def get_car_by_listing_id(listing_id: int) -> Optional[Dict[str, Any]]:
    """Find one car by listing ID from the dataset."""

    df = load_cars()
    match = df[df["listing_id"] == int(listing_id)]
    if match.empty:
        return None
    return row_to_car(match.iloc[0])


def get_car_from_message(message: str) -> Optional[Dict[str, Any]]:
    """Resolve an explicit listing reference included by a frontend car card."""

    match = re.search(
        r"\blisting\s*id\s*[:#-]?\s*(\d+)\b",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    return get_car_by_listing_id(int(match.group(1)))


def build_car_title(car: Dict[str, Any]) -> str:
    """Build a readable car title for memory and bookings."""

    return (
        f"{car.get('year', '')} "
        f"{car.get('make', '')} "
        f"{car.get('model', '')} "
        f"{car.get('trim', '')}"
    ).strip()


def format_filter_value(value: Any) -> str:
    """Format an extracted filter for a user-facing response."""

    text = str(value).strip()
    initialisms = {"bmw": "BMW", "gcc": "GCC"}
    return initialisms.get(text.lower(), text.title())


def get_requested_color(message: str) -> Optional[str]:
    """Return an explicitly requested common color, if present."""

    message_lower = message.lower()
    for color in COLOR_TERMS:
        if re.search(rf"\b{re.escape(color)}\b", message_lower):
            return color
    return None
