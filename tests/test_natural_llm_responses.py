import main
import services.booking as booking
import services.llm_agent as llm_agent
import services.memory as memory
import services.session_memory as session_memory


def sample_car():
    return {
        "listing_id": 701,
        "year": 2021,
        "make": "Toyota",
        "model": "Camry",
        "trim": "SE",
        "title": "2021 Toyota Camry SE",
        "description": "A provided test listing.",
        "photo_url": "",
        "match_reason": "Previously selected",
    }


def configure_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(booking, "LEADS_CSV_PATH", tmp_path / "leads.csv")
    monkeypatch.setattr(booking, "BOOKINGS_CSV_PATH", tmp_path / "bookings.csv")
    session_memory.SESSIONS.clear()


def test_selected_car_deferral_uses_grounded_llm(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch)
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return "No problem—we can come back to this car whenever you’re ready."

    monkeypatch.setattr(llm_agent, "call_gemini", fake_llm)
    session_id = "natural-selected-car"
    session_memory.save_selected_car(session_id, sample_car())

    response = main.chat(
        main.ChatRequest(
            username="natural-user",
            session_id=session_id,
            message="I’ll come back to it later.",
        )
    )

    assert response.intent == "car_details"
    assert response.reply == (
        "No problem—we can come back to this car whenever you’re ready."
    )
    assert prompts
    assert "selected_car" in prompts[-1]
    assert "I’ll come back to it later." in prompts[-1]


def test_selected_car_deferral_has_natural_offline_fallback(
    tmp_path,
    monkeypatch,
):
    configure_storage(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_agent, "call_gemini", lambda prompt: None)
    session_id = "offline-selected-car"
    session_memory.save_selected_car(session_id, sample_car())

    response = main.chat(
        main.ChatRequest(
            username="natural-user",
            session_id=session_id,
            message="I will come back to it later.",
        )
    )

    assert response.intent == "car_details"
    assert response.reply.startswith("Of course.")
    assert "Description:" not in response.reply
