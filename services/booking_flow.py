from typing import Any, Dict

from services.car_utils import build_car_title
from services.memory import (
    add_liked_car,
    record_user_interaction,
    update_memory_summary,
    update_user_memory,
)


def persist_confirmed_booking_memory(
    username: str,
    car: Dict[str, Any],
    booking_date: str,
    booking_time: str,
    budget: str = "",
    needs: str = "",
    preferred_make: str = "",
    preferred_model: str = "",
) -> None:
    """Apply the same long-term memory updates for UI and chat bookings."""

    selected_car_title = build_car_title(car)
    update_user_memory(
        username=username,
        last_budget=budget,
        preferred_make=preferred_make or str(car.get("make") or ""),
        preferred_model=preferred_model or str(car.get("model") or ""),
        last_seen_car=selected_car_title,
        last_query=f"Booked viewing for {selected_car_title}",
    )
    add_liked_car(
        username,
        int(car.get("listing_id")),
        make=str(car.get("make") or ""),
        model=str(car.get("model") or ""),
        car_title=selected_car_title,
        refresh_summary=False,
    )
    record_user_interaction(
        username=username,
        event_type="booking",
        query=f"Booked viewing for {selected_car_title}",
        make=str(car.get("make") or ""),
        model=str(car.get("model") or ""),
        budget=budget,
        listing_id=int(car.get("listing_id")),
        car_title=selected_car_title,
        lead_quality="Hot",
        notes=f"Viewing booked for {booking_date} at {booking_time}. {needs}".strip(),
    )
    update_memory_summary(username)
