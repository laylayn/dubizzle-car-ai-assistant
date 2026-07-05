import main
import services.booking as booking
import services.llm_agent as llm_agent
import services.memory as memory
import services.session_memory as session_memory


def make_car(listing_id):
    return {
        "listing_id": listing_id,
        "year": 2020 + listing_id,
        "make": f"Make {listing_id}",
        "model": f"Model {listing_id}",
        "trim": "",
        "title": f"Car {listing_id}",
        "description": f"Description for car {listing_id}",
        "photo_url": "",
        "match_reason": "Test result",
    }


def test_extracts_all_comparison_positions_in_user_order():
    assert session_memory.extract_car_positions(
        "compare the third and first one"
    ) == [2, 0]
    assert session_memory.extract_car_positions(
        "compare the 2nd with the 5th"
    ) == [1, 4]


def test_chat_compares_first_and_third_without_substitution(
    tmp_path,
    monkeypatch,
):
    def fake_llm(prompt):
        if "intent extraction layer" in prompt:
            return """
            {
              "intent": "compare_listings",
              "make": null,
              "model": null,
              "year": null,
              "keyword": null,
              "keywords": [],
              "selected_position": 0
            }
            """
        return None

    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(booking, "LEADS_CSV_PATH", tmp_path / "leads.csv")
    monkeypatch.setattr(booking, "BOOKINGS_CSV_PATH", tmp_path / "bookings.csv")
    monkeypatch.setattr(llm_agent, "call_gemini", fake_llm)
    session_memory.SESSIONS.clear()

    session_id = "comparison-session"
    session_memory.save_last_results(
        session_id,
        [make_car(1), make_car(2), make_car(3)],
    )

    response = main.chat(
        main.ChatRequest(
            username="comparison-user",
            session_id=session_id,
            message="compare the first and third one",
        )
    )

    assert [car["listing_id"] for car in response.cars] == [1, 3]
    assert response.extracted_request["selected_positions"] == [0, 2]
    assert "Result 1" in response.reply
    assert "Listing ID: 1" in response.reply
    assert "Result 3" in response.reply
    assert "Listing ID: 3" in response.reply
    assert "Listing ID: 2" not in response.reply
