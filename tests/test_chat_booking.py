import csv
from datetime import date

import main
import services.booking as booking
import services.llm_agent as llm_agent
import services.memory as memory
import services.session_memory as session_memory


def selected_car(listing_id=501):
    return {
        "listing_id": listing_id,
        "year": 2022,
        "make": "Toyota",
        "model": "Camry",
        "trim": "SE",
        "title": "2022 Toyota Camry SE",
        "description": "Test listing",
        "photo_url": "",
        "match_reason": "Previously selected",
    }


def configure_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(booking, "LEADS_CSV_PATH", tmp_path / "leads.csv")
    monkeypatch.setattr(booking, "BOOKINGS_CSV_PATH", tmp_path / "bookings.csv")
    monkeypatch.setattr(llm_agent, "call_gemini", lambda prompt: None)
    session_memory.SESSIONS.clear()


def upcoming_saturday():
    today = date.today()
    days_ahead = (5 - today.weekday()) % 7
    return date.fromordinal(today.toordinal() + days_ahead).isoformat()


def test_chat_confirms_natural_language_booking(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch)
    session_id = "chat-booking"
    car = selected_car()
    session_memory.save_selected_car(session_id, car)

    response = main.chat(
        main.ChatRequest(
            username="booking-user",
            session_id=session_id,
            message="booking for 3pm on sat",
        )
    )

    assert response.intent == "booking"
    assert "confirmed" in response.reply.lower()
    assert upcoming_saturday() in response.reply
    assert "15:00" in response.reply

    with booking.BOOKINGS_CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        saved_booking = list(csv.DictReader(handle))[-1]
    assert saved_booking["username"] == "booking-user"
    assert saved_booking["selected_listing_id"] == str(car["listing_id"])
    assert saved_booking["booking_date"] == upcoming_saturday()
    assert saved_booking["booking_time"] == "15:00"

    with booking.LEADS_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        saved_lead = list(csv.DictReader(handle))[-1]
    assert saved_lead["lead_status"] == "Hot"
    assert session_memory.get_preferences(session_id)["pending_booking"] is False
    assert memory.get_recent_user_interactions("booking-user")[-1][
        "event_type"
    ] == "booking"


def test_chat_collects_booking_date_and_time_across_turns(
    tmp_path,
    monkeypatch,
):
    configure_storage(tmp_path, monkeypatch)
    session_id = "multi-turn-booking"
    session_memory.save_selected_car(session_id, selected_car(502))

    first = main.chat(
        main.ChatRequest(
            username="booking-user",
            session_id=session_id,
            message="book this car",
        )
    )
    assert "day or date and time" in first.reply.lower()

    second = main.chat(
        main.ChatRequest(
            username="booking-user",
            session_id=session_id,
            message="Saturday",
        )
    )
    assert "provide the time" in second.reply.lower()

    third = main.chat(
        main.ChatRequest(
            username="booking-user",
            session_id=session_id,
            message="3pm",
        )
    )
    assert "confirmed" in third.reply.lower()
    assert upcoming_saturday() in third.reply


def test_chat_booking_requires_selected_car(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch)

    response = main.chat(
        main.ChatRequest(
            username="booking-user",
            session_id="no-selected-car",
            message="booking for 3pm on sat",
        )
    )

    assert response.cars == []
    assert "select a car first" in response.reply.lower()
    with booking.BOOKINGS_CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert list(csv.DictReader(handle)) == []


def test_returning_user_booking_resolves_previous_car_with_one_llm_call(
    tmp_path,
    monkeypatch,
):
    configure_storage(tmp_path, monkeypatch)
    car = main.get_car_by_listing_id(1)
    memory.record_user_interaction(
        username="maya",
        event_type="preference_search",
        query="show me this car",
        make=str(car.get("make") or ""),
        model=str(car.get("model") or ""),
        listing_id=car["listing_id"],
        car_title=main.build_car_title(car),
    )
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return (
            "I found the car you discussed before. What day and time would "
            "you like to book?"
        )

    monkeypatch.setattr(llm_agent, "call_gemini", fake_llm)
    response = main.chat(
        main.ChatRequest(
            username="maya",
            session_id="maya-new-session",
            message=(
                "maya again, i want to book the car i was asking about before"
            ),
        )
    )

    assert response.intent == "booking"
    assert [result["listing_id"] for result in response.cars] == [
        car["listing_id"]
    ]
    assert "found the car" in response.reply.lower()
    assert session_memory.get_selected_car(
        "maya-new-session"
    )["listing_id"] == car["listing_id"]
    assert len(prompts) == 1
