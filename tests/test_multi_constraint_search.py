import csv
import sqlite3

import main
import services.booking as booking
import services.llm_agent as llm_agent
import services.memory as memory
import services.session_memory as session_memory


MESSAGE = "I want a reliable SUV under AED 120,000 with warranty"


def test_fallback_extracts_every_constraint():
    request = llm_agent.fallback_extract_user_request(MESSAGE)

    assert request["budget"] == "AED 120,000"
    assert request["needs"] == "reliable SUV with warranty"
    assert request["body_type"] == "suv"
    assert set(request["required_keywords"]) == {"suv", "warranty"}
    assert request["soft_preferences"] == ["reliable"]


def test_search_requires_suv_and_warranty_and_applies_budget():
    request = llm_agent.fallback_extract_user_request(MESSAGE)
    cars = main.search_from_extracted_request(request)

    assert cars
    assert any("within AED 120,000" in car["match_reason"] for car in cars)
    for car in cars:
        listing_text = f"{car['title']} {car['description']}".lower()
        assert "suv" in listing_text
        assert "warranty" in listing_text


def test_chat_merges_missed_llm_constraints_and_persists_lead_and_memory(
    tmp_path,
    monkeypatch,
):
    incomplete_llm_extraction = """
    {
      "intent": "lead_capture",
      "make": null,
      "model": null,
      "year": null,
      "keyword": "warranty",
      "keywords": ["warranty"],
      "required_keywords": ["warranty"],
      "soft_preferences": [],
      "body_type": null,
      "budget": "AED 120,000",
      "needs": "warranty",
      "selected_position": null
    }
    """

    def fake_llm(prompt):
        if "intent extraction layer" in prompt:
            return incomplete_llm_extraction
        return None

    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(booking, "LEADS_CSV_PATH", tmp_path / "leads.csv")
    monkeypatch.setattr(booking, "BOOKINGS_CSV_PATH", tmp_path / "bookings.csv")
    monkeypatch.setattr(llm_agent, "call_gemini", fake_llm)
    session_memory.SESSIONS.clear()

    response = main.chat(
        main.ChatRequest(
            username="test_user_005",
            session_id="test_user_005_session",
            message=MESSAGE,
        )
    )

    assert response.intent == "lead_capture"
    assert response.extracted_request["budget"] == "AED 120,000"
    assert response.extracted_request["needs"] == "reliable SUV with warranty"
    assert set(response.extracted_request["required_keywords"]) == {
        "suv",
        "warranty",
    }
    assert response.extracted_request["soft_preferences"] == ["reliable"]

    for car in response.cars:
        listing_text = f"{car['title']} {car['description']}".lower()
        assert "suv" in listing_text
        assert "warranty" in listing_text

    with booking.LEADS_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        lead = list(csv.DictReader(handle))[-1]
    assert lead["lead_status"] == "Warm"
    assert lead["budget"] == "AED 120,000"
    assert lead["needs"] == "reliable SUV with warranty"

    with sqlite3.connect(memory.DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        user = dict(
            connection.execute(
                "SELECT * FROM users WHERE username = ?",
                ("test_user_005",),
            ).fetchone()
        )
    assert user["last_budget"] == "AED 120,000"
    assert user["preferred_body_type"] == "suv"
