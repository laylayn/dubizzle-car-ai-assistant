import main
import services.booking as booking
import services.llm_agent as llm_agent
import services.memory as memory
import services.session_memory as session_memory
from services.guardrails import apply_guardrails
from services.listing_attributes import (
    extract_color_from_car,
    extract_mileage_from_car,
    extract_price_from_car,
)
from services.query_planner import build_query_plan


def fallback_plan(message):
    extracted = llm_agent.fallback_extract_user_request(message)
    return extracted, build_query_plan(message, extracted)


def test_planner_translates_ranking_language_to_operations():
    cases = [
        ("what cars do you have with low mileage?", "mileage", "ascending"),
        ("show me cars with lowest km", "mileage", "ascending"),
        ("cheapest cars", "price", "ascending"),
        ("most expensive cars", "price", "descending"),
        ("newest cars", "year", "descending"),
    ]

    for message, sort_by, sort_order in cases:
        _, plan = fallback_plan(message)
        assert plan["sort_by"] == sort_by
        assert plan["sort_order"] == sort_order
        assert all(
            sort_by not in feature
            for feature in plan["filters"]["features"]
        )

    plan = build_query_plan(
        "what cars do you have with low mileage?",
        {
            "required_keywords": ["low mileage"],
            "keyword": "low mileage",
        },
    )
    assert plan["filters"]["features"] == []


def test_planner_preserves_filters_while_adding_ranking():
    _, plan = fallback_plan("low mileage mercedes with warranty")

    assert plan["filters"]["make"] == "mercedes-benz"
    assert plan["filters"]["features"] == ["warranty"]
    assert plan["sort_by"] == "mileage"
    assert plan["sort_order"] == "ascending"


def test_reusable_attribute_extractors_cover_listing_text_formats():
    assert extract_mileage_from_car(
        {"title": "One owner 56,000 KM", "description": ""}
    )["value"] == 56_000
    assert extract_mileage_from_car(
        {"title": "", "description": "Mileage: 169859"}
    )["value"] == 169_859
    assert extract_mileage_from_car(
        {"title": "", "description": "Driven 56000"}
    )["value"] == 56_000
    assert extract_color_from_car(
        {"title": "White Mini Cooper", "description": ""}
    )["value"] == "white"

    financed_car = {
        "title": "AED 1,400 monthly",
        "description": "AED 70,000 cash",
    }
    assert extract_price_from_car(financed_car)["value"] == 70_000


def test_query_executor_ranks_dataset_attributes_without_literal_search():
    extracted, plan = fallback_plan("what cars do you have with low mileage?")
    cars, execution = main.execute_query_plan(plan, extracted)

    assert execution["status"] == "ranked"
    assert execution["partial_attribute_coverage"] is True
    assert [car["ranked_value"] for car in cars] == sorted(
        car["ranked_value"] for car in cars
    )

    extracted, plan = fallback_plan("low mileage mercedes")
    cars, _ = main.execute_query_plan(plan, extracted)
    assert cars
    assert all(car["make"].lower() == "mercedes-benz" for car in cars)

    extracted, plan = fallback_plan("cheapest cars")
    cars, _ = main.execute_query_plan(plan, extracted)
    assert [car["ranked_value"] for car in cars] == sorted(
        car["ranked_value"] for car in cars
    )

    extracted, plan = fallback_plan("newest cars")
    cars, _ = main.execute_query_plan(plan, extracted)
    assert [car["year"] for car in cars] == sorted(
        (car["year"] for car in cars),
        reverse=True,
    )


def test_missing_ranking_attribute_returns_no_random_cars(monkeypatch):
    monkeypatch.setattr(
        main,
        "search_cars",
        lambda **kwargs: [
            {
                "listing_id": 1,
                "year": 2020,
                "make": "Example",
                "model": "Car",
                "title": "No odometer information",
                "description": "",
                "match_reason": "Candidate",
            }
        ],
    )
    extracted, plan = fallback_plan("cars with the lowest mileage")
    cars, execution = main.execute_query_plan(plan, extracted)

    assert cars == []
    assert execution["status"] == "attribute_unavailable"
    assert (
        main.build_unavailable_attribute_reply("mileage")
        == "The provided inventory does not include enough mileage information "
        "for me to rank cars by mileage."
    )


def test_selected_car_attribute_questions_stay_memory_first(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(booking, "LEADS_CSV_PATH", tmp_path / "leads.csv")
    monkeypatch.setattr(booking, "BOOKINGS_CSV_PATH", tmp_path / "bookings.csv")
    monkeypatch.setattr(llm_agent, "call_gemini", lambda prompt: None)
    monkeypatch.setattr(
        main,
        "search_cars",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("selected-car follow-up triggered inventory search")
        ),
    )
    session_memory.SESSIONS.clear()

    selected_car = {
        "listing_id": 10,
        "year": 2021,
        "make": "Example",
        "model": "Selected",
        "trim": "",
        "title": "Example Selected",
        "description": "Driven 56,000 km. Listed for AED 80,000.",
        "photo_url": "",
        "match_reason": "Previously selected",
    }
    session_id = "attribute-follow-up"
    session_memory.save_selected_car(session_id, selected_car)

    mileage_response = main.chat(
        main.ChatRequest(
            username="attribute-user",
            session_id=session_id,
            message="what is the mileage of it?",
        )
    )
    assert "56,000 km" in mileage_response.reply
    assert mileage_response.intent == "car_details"

    price_response = main.chat(
        main.ChatRequest(
            username="attribute-user",
            session_id=session_id,
            message="what is the price of it?",
        )
    )
    assert "AED 80,000" in price_response.reply
    assert price_response.intent == "car_details"


def test_guardrails_allow_ranking_but_still_block_non_automotive():
    assert apply_guardrails("what cars have low mileage?")["allowed"] is True
    assert apply_guardrails("cheapest cars")["allowed"] is True
    assert apply_guardrails("write me Python code")["allowed"] is False
