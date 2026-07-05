from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.data_loader import load_cars
from services.retrieval import search_cars
from services.guardrails import apply_guardrails, log_intent
from services.llm_agent import (
    extract_user_request,
    generate_inventory_reply,
    generate_car_detail_reply,
    generate_refusal_reply,
)
from services.session_memory import (
    add_message,
    save_last_results,
    save_selected_car,
    get_selected_car,
    get_car_by_position,
    resolve_car_reference,
    update_preferences,
    get_last_results,
)
from services.memory import (
    init_db,
    get_or_create_user,
    update_user_memory,
    get_memory_summary,
    add_liked_car,
)
from services.booking import book_viewing, save_lead


app = FastAPI(
    title="dubizzle Cars AI Assistant",
    description="FastAPI backend for an AI assistant that searches car inventory, remembers users, and books viewings.",
    version="1.0.0",
)


# Allows Streamlit frontend to call this backend locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Pydantic request/response models
# -----------------------------

class ChatRequest(BaseModel):
    user_id: str = Field(..., example="layan123")
    session_id: str = Field(..., example="session_abc")
    message: str = Field(..., example="Show me Mercedes-Benz cars from 2019")
    name: Optional[str] = Field(default="", example="Layan")


class ChatResponse(BaseModel):
    reply: str
    intent: str
    cars: List[Dict[str, Any]] = []
    extracted_request: Dict[str, Any] = {}
    memory_update: Dict[str, Any] = {}


class BookingRequest(BaseModel):
    user_id: str = Field(..., example="layan123")
    name: str = Field(default="", example="Layan")
    listing_id: int = Field(..., example=4)
    date: str = Field(..., example="2026-07-10")
    time: str = Field(..., example="15:00")
    budget: str = Field(default="", example="AED 100,000")
    needs: str = Field(default="", example="SUV with warranty")
    preferred_make: str = Field(default="", example="Mercedes-Benz")
    preferred_model: str = Field(default="", example="C-Class")
    notes: str = Field(default="", example="User requested a test drive.")


# -----------------------------
# Startup
# -----------------------------

@app.on_event("startup")
def startup_event():
    """
    Initialize database and check dataset at startup.
    """

    init_db()
    load_cars()
    print("[STARTUP] Backend ready. Dataset loaded and user memory database initialized.")


# -----------------------------
# Helper functions
# -----------------------------

def row_to_car(row) -> Dict[str, Any]:
    """
    Convert a dataframe row into a clean car dictionary.
    """

    return {
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


def get_car_by_listing_id(listing_id: int) -> Optional[Dict[str, Any]]:
    """
    Find one car by listing ID from the dataset.
    """

    df = load_cars()
    match = df[df["listing_id"] == int(listing_id)]

    if match.empty:
        return None

    return row_to_car(match.iloc[0])


def build_car_title(car: Dict[str, Any]) -> str:
    """
    Build readable car title for memory/bookings.
    """

    return (
        f"{car.get('year', '')} "
        f"{car.get('make', '')} "
        f"{car.get('model', '')} "
        f"{car.get('trim', '')}"
    ).strip()


def is_llm_blocked_intent(intent: str) -> bool:
    """
    Second guardrail layer:
    even if rule-based guardrails miss something,
    the LLM can still classify it as blocked.
    """

    return intent in {
        "blocked_non_automotive",
        "blocked_competitor",
        "blocked_coding",
        "blocked_history",
        "blocked_politics",
        "blocked_medical_legal",
        "blocked_random_homework",
    }


def search_from_extracted_request(
    extracted_request: Dict[str, Any],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search cars using extracted make/model/year/keyword.

    Tries:
    1. keyword field
    2. each keyword from keywords list
    3. no keyword fallback, if make/model/year exists
    """

    make = extracted_request.get("make")
    model = extracted_request.get("model")
    year = extracted_request.get("year")
    keyword = extracted_request.get("keyword")
    keywords = extracted_request.get("keywords") or []

    keyword_attempts = []

    if keyword:
        keyword_attempts.append(str(keyword))

    for item in keywords:
        if item and str(item) not in keyword_attempts:
            keyword_attempts.append(str(item))

    if not keyword_attempts:
        keyword_attempts.append(None)

    for keyword_attempt in keyword_attempts:
        cars = search_cars(
            make=make,
            model=model,
            year=year,
            keyword=keyword_attempt,
            limit=limit,
        )

        if cars:
            return cars

    # If keyword search was too strict, try structured filters only
    if make or model or year:
        cars = search_cars(
            make=make,
            model=model,
            year=year,
            keyword=None,
            limit=limit,
        )
        return cars

    return []


def maybe_save_interest_lead(
    user_id: str,
    name: str,
    extracted_request: Dict[str, Any],
    cars: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Save Cold/Warm interest leads from chat.

    Does not save just because user opened the app.
    Saves only if user actually searched or gave preferences.
    """

    intent = extracted_request.get("intent", "")
    message_needs = extracted_request.get("needs") or ""
    budget = extracted_request.get("budget") or ""
    preferred_make = extracted_request.get("make") or ""
    preferred_model = extracted_request.get("model") or ""

    if intent not in {"car_search", "lead_capture"}:
        return None

    selected_listing_id = None
    selected_car_title = ""

    if cars:
        first_car = cars[0]
        selected_listing_id = first_car.get("listing_id")
        selected_car_title = build_car_title(first_car)

    # If no buying info and no results, do not create junk lead
    if not any([message_needs, budget, preferred_make, preferred_model, cars]):
        return None

    lead = save_lead(
        user_id=user_id,
        name=name,
        budget=budget,
        needs=message_needs,
        preferred_make=preferred_make,
        preferred_model=preferred_model,
        selected_listing_id=selected_listing_id,
        selected_car_title=selected_car_title,
        notes="Lead captured from chat interaction.",
    )

    return lead


def compare_two_cars(car_one: Dict[str, Any], car_two: Dict[str, Any]) -> str:
    """
    Simple grounded comparison between two cars.
    """

    return f"""
Here is a quick comparison based only on the provided inventory data:

1. {build_car_title(car_one)}
- Listing ID: {car_one.get("listing_id")}
- Title: {car_one.get("title")}
- Description preview: {car_one.get("description", "")[:350]}...

2. {build_car_title(car_two)}
- Listing ID: {car_two.get("listing_id")}
- Title: {car_two.get("title")}
- Description preview: {car_two.get("description", "")[:350]}...

I can only compare details that are available in the listings. If you want, I can also help you book a viewing for either one.
""".strip()


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
def root():
    return {
        "message": "dubizzle Cars AI Assistant API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
    }


@app.get("/inventory/search")
def inventory_search(
    make: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    year: Optional[int] = Query(default=None),
    keyword: Optional[str] = Query(default=None),
    limit: int = Query(default=5, ge=1, le=20),
):
    """
    Direct inventory search endpoint.
    Useful for testing retrieval without the LLM.
    """

    cars = search_cars(
        make=make,
        model=model,
        year=year,
        keyword=keyword,
        limit=limit,
    )

    return {
        "count": len(cars),
        "cars": cars,
    }


@app.get("/user/{user_id}")
def get_user_profile(user_id: str, name: str = ""):
    """
    Return long-term user memory summary.
    If user does not exist, create a basic profile.
    """

    get_or_create_user(user_id=user_id, name=name)
    summary = get_memory_summary(user_id)

    return summary


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Save user message in short-term memory.
    2. Apply rule-based guardrails.
    3. Use LLM to extract structured request.
    4. Apply second LLM-based guardrail.
    5. Route to search/details/compare/booking guidance.
    6. Save memory and return reply.
    """

    user_id = request.user_id
    session_id = request.session_id
    user_message = request.message
    name = request.name or ""

    get_or_create_user(user_id=user_id, name=name)
    add_message(session_id, "user", user_message)

    # Layer 1: rule-based guardrails
    guardrail = apply_guardrails(user_message)

    if not guardrail["allowed"]:
        reply = guardrail["refusal_message"]
        add_message(session_id, "assistant", reply)
        log_intent(user_id=user_id, intent=guardrail["intent"], results_count=0)

        return ChatResponse(
            reply=reply,
            intent=guardrail["intent"],
            cars=[],
            extracted_request={},
            memory_update=get_memory_summary(user_id),
        )

    # LLM extraction
    extracted_request = extract_user_request(user_message)
    intent = extracted_request.get("intent", guardrail["intent"])

    # Layer 2: LLM-based guardrails
    if is_llm_blocked_intent(intent):
        reply = generate_refusal_reply()
        add_message(session_id, "assistant", reply)
        log_intent(user_id=user_id, intent=intent, results_count=0)

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=[],
            extracted_request=extracted_request,
            memory_update=get_memory_summary(user_id),
        )

    # Greeting / returning user
    if intent == "greeting":
        memory_summary = get_memory_summary(user_id)
        reply = memory_summary.get(
            "message",
            f"Hi {name or 'there'}! I can help you search car listings, compare options, or book a viewing.",
        )

        add_message(session_id, "assistant", reply)
        log_intent(user_id=user_id, intent=intent, results_count=0)

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=[],
            extracted_request=extracted_request,
            memory_update=memory_summary,
        )

    # Search / lead capture
    if intent in {"car_search", "lead_capture", "general_car_advice"}:
        cars = search_from_extracted_request(extracted_request, limit=5)

        save_last_results(session_id, cars)

        if cars:
            save_selected_car(session_id, cars[0])

        session_preferences = {
            "make": extracted_request.get("make"),
            "model": extracted_request.get("model"),
            "year": extracted_request.get("year"),
            "keyword": extracted_request.get("keyword"),
            "budget": extracted_request.get("budget"),
            "needs": extracted_request.get("needs"),
        }

        update_preferences(session_id, session_preferences)

        first_car_title = build_car_title(cars[0]) if cars else ""

        update_user_memory(
            user_id=user_id,
            name=name,
            last_budget=extracted_request.get("budget"),
            preferred_make=extracted_request.get("make"),
            preferred_model=extracted_request.get("model"),
            last_seen_car=first_car_title,
            last_query=user_message,
        )

        lead = maybe_save_interest_lead(
            user_id=user_id,
            name=name,
            extracted_request=extracted_request,
            cars=cars,
        )

        reply = generate_inventory_reply(
            user_message=user_message,
            cars=cars,
            extracted_request=extracted_request,
        )

        add_message(session_id, "assistant", reply)
        log_intent(user_id=user_id, intent=intent, results_count=len(cars))

        memory_summary = get_memory_summary(user_id)
        memory_summary["lead_saved"] = lead

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=cars,
            extracted_request=extracted_request,
            memory_update=memory_summary,
        )

    # Details about selected/previous car
    if intent == "car_details":
        selected_car = None

        selected_position = extracted_request.get("selected_position")

        if selected_position is not None:
            selected_car = get_car_by_position(session_id, int(selected_position))

        if selected_car is None:
            selected_car = resolve_car_reference(session_id, user_message)

        if selected_car is None:
            selected_car = get_selected_car(session_id)

        if selected_car is None:
            reply = "I do not have a selected car yet. Please search for cars first, then ask about a specific result."

            add_message(session_id, "assistant", reply)
            log_intent(user_id=user_id, intent=intent, results_count=0)

            return ChatResponse(
                reply=reply,
                intent=intent,
                cars=[],
                extracted_request=extracted_request,
                memory_update=get_memory_summary(user_id),
            )

        save_selected_car(session_id, selected_car)

        update_user_memory(
            user_id=user_id,
            name=name,
            last_seen_car=build_car_title(selected_car),
            last_query=user_message,
        )

        reply = generate_car_detail_reply(user_message, selected_car)

        add_message(session_id, "assistant", reply)
        log_intent(user_id=user_id, intent=intent, results_count=1)

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=[selected_car],
            extracted_request=extracted_request,
            memory_update=get_memory_summary(user_id),
        )

    # Compare two current listings
    if intent == "compare_listings":
        last_results = get_last_results(session_id)

        if len(last_results) < 2:
            reply = "Please search for cars first so I can compare two listings from the current results."
        else:
            reply = compare_two_cars(last_results[0], last_results[1])

        add_message(session_id, "assistant", reply)
        log_intent(user_id=user_id, intent=intent, results_count=len(last_results))

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=last_results[:2],
            extracted_request=extracted_request,
            memory_update=get_memory_summary(user_id),
        )

    # Booking intent inside chat gives guidance.
    # Official booking is handled by /book-viewing.
    if intent == "booking":
        selected_car = get_selected_car(session_id)

        if selected_car:
            reply = (
                f"I can help book a viewing for {build_car_title(selected_car)}. "
                "Viewing slots are available Monday to Saturday, from 8 AM to 8 PM. "
                "Please provide a date and time, or use the booking form."
            )
        else:
            reply = (
                "I can help book a viewing, but please select a car first. "
                "Search for cars, choose a result, then provide a date and time. "
                "Viewing slots are Monday to Saturday, 8 AM to 8 PM."
            )

        add_message(session_id, "assistant", reply)
        log_intent(user_id=user_id, intent=intent, results_count=1 if selected_car else 0)

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=[selected_car] if selected_car else [],
            extracted_request=extracted_request,
            memory_update=get_memory_summary(user_id),
        )

    # Fallback for any allowed but unsupported intent
    reply = (
        "I can help with car searches, listing details, comparisons, and viewing bookings. "
        "Try asking me to show cars by make, model, year, or feature."
    )

    add_message(session_id, "assistant", reply)
    log_intent(user_id=user_id, intent=intent, results_count=0)

    return ChatResponse(
        reply=reply,
        intent=intent,
        cars=[],
        extracted_request=extracted_request,
        memory_update=get_memory_summary(user_id),
    )


@app.post("/book-viewing")
def book_viewing_endpoint(request: BookingRequest):
    """
    Official booking endpoint.

    Validates:
    - Monday to Saturday
    - 8 AM to 8 PM

    If valid:
    - saves official booking to bookings.csv
    - saves Hot lead to leads.csv
    - updates long-term user memory
    """

    car = get_car_by_listing_id(request.listing_id)

    if not car:
        raise HTTPException(
            status_code=404,
            detail=f"Listing ID {request.listing_id} was not found in the inventory.",
        )

    selected_car_title = build_car_title(car)

    result = book_viewing(
        user_id=request.user_id,
        name=request.name,
        selected_listing_id=request.listing_id,
        selected_car_title=selected_car_title,
        booking_date=request.date,
        booking_time=request.time,
        budget=request.budget,
        needs=request.needs,
        preferred_make=request.preferred_make,
        preferred_model=request.preferred_model,
        notes=request.notes,
    )

    if result["success"]:
        update_user_memory(
            user_id=request.user_id,
            name=request.name,
            last_budget=request.budget,
            preferred_make=request.preferred_make,
            preferred_model=request.preferred_model,
            last_seen_car=selected_car_title,
            last_query=f"Booked viewing for {selected_car_title}",
        )

        add_liked_car(request.user_id, request.listing_id)

    return result