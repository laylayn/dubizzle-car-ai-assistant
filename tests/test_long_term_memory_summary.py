import sqlite3

import main
import services.booking as booking
import services.llm_agent as llm_agent
import services.memory as memory
import services.session_memory as session_memory


def configure_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(booking, "LEADS_CSV_PATH", tmp_path / "leads.csv")
    monkeypatch.setattr(booking, "BOOKINGS_CSV_PATH", tmp_path / "bookings.csv")
    session_memory.SESSIONS.clear()


def test_init_db_migrates_users_and_creates_interactions(tmp_path, monkeypatch):
    database_path = tmp_path / "users.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE users (
                username TEXT PRIMARY KEY,
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

    monkeypatch.setattr(memory, "DB_PATH", database_path)
    memory.init_db()

    with sqlite3.connect(database_path) as connection:
        user_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(users)")
        }
        interaction_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(user_interactions)"
            )
        }

    assert "memory_summary" in user_columns
    assert interaction_columns == {
        "interaction_id",
        "username",
        "timestamp",
        "event_type",
        "query",
        "make",
        "model",
        "budget",
        "listing_id",
        "car_title",
        "lead_quality",
        "notes",
    }


def test_update_memory_summary_saves_llm_grounded_text(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch)
    memory.update_user_memory(
        "summary-user",
        preferred_make="Mercedes-Benz",
        preferred_model="C-Class",
    )
    memory.record_user_interaction(
        username="summary-user",
        event_type="preference_search",
        query="show me Mercedes cars",
        make="Mercedes-Benz",
        model="C-Class",
    )

    generated_summary = (
        "User has recently shown interest in Mercedes-Benz C-Class cars."
    )

    def fake_llm(prompt):
        assert "Mercedes-Benz" in prompt
        assert "show me Mercedes cars" in prompt
        return generated_summary

    monkeypatch.setattr(llm_agent, "call_gemini", fake_llm)
    assert memory.update_memory_summary("summary-user") == generated_summary
    assert memory.get_user("summary-user")["memory_summary"] == generated_summary
    assert (
        memory.get_memory_summary("summary-user")["memory_summary"]
        == generated_summary
    )

    monkeypatch.setattr(
        llm_agent,
        "call_gemini",
        lambda prompt: "User prefers Ferrari SUVs with a budget of AED 900,000.",
    )
    safe_summary = memory.update_memory_summary("summary-user")
    assert "Ferrari" not in safe_summary
    assert "900,000" not in safe_summary
    assert "Mercedes-Benz" in safe_summary


def test_acknowledgements_are_not_saved_as_interaction_queries(
    tmp_path,
    monkeypatch,
):
    configure_storage(tmp_path, monkeypatch)

    for query in ["hi", "thanks", "okay", "got it"]:
        interaction = memory.record_user_interaction(
            username="ack-user",
            event_type="inventory_search",
            query=query,
        )
        assert interaction["query"] == ""

    assert all(
        not interaction["query"]
        for interaction in memory.get_recent_user_interactions("ack-user")
    )


def test_search_comparison_followup_and_returning_user_flow(
    tmp_path,
    monkeypatch,
):
    configure_storage(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_agent, "call_gemini", lambda prompt: None)

    username = "memory-flow-user"
    first_session = "memory-flow-search"
    search_response = main.chat(
        main.ChatRequest(
            username=username,
            session_id=first_session,
            message="show me mercedes cars",
        )
    )
    assert len(search_response.cars) >= 3

    comparison_response = main.chat(
        main.ChatRequest(
            username=username,
            session_id=first_session,
            message="compare the first and third one",
        )
    )
    assert len(comparison_response.cars) == 2

    user_after_comparison = memory.get_user(username)
    summary_before_followup = user_after_comparison["memory_summary"]
    interactions_before_followup = memory.get_recent_user_interactions(username)
    assert "mercedes-benz" in summary_before_followup.lower()
    assert "compared" in summary_before_followup.lower()
    assert [event["event_type"] for event in interactions_before_followup] == [
        "preference_search",
        "comparison",
    ]

    followup_response = main.chat(
        main.ChatRequest(
            username=username,
            session_id=first_session,
            message="what is the price of it?",
        )
    )
    assert followup_response.intent == "car_details"
    assert memory.get_user(username)["memory_summary"] == summary_before_followup
    assert (
        memory.get_recent_user_interactions(username)
        == interactions_before_followup
    )

    greeting_response = main.chat(
        main.ChatRequest(
            username=username,
            session_id="memory-flow-new-chat",
            message="hi",
        )
    )
    assert "mercedes-benz" in greeting_response.reply.lower()
    assert "compared" in greeting_response.reply.lower()

    recall_response = main.chat(
        main.ChatRequest(
            username=username,
            session_id="memory-flow-recall",
            message="continue from last time",
        )
    )
    assert recall_response.intent == "returning_user_memory"
    assert "mercedes-benz" in recall_response.reply.lower()
    assert memory.get_recent_user_interactions(username)[-1][
        "event_type"
    ] == "memory_recall"
