from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.booking import book_viewing
from services.booking_flow import persist_confirmed_booking_memory
from services.car_utils import (
    build_car_title,
    get_car_by_listing_id,
)
from services.chat_service import handle_chat
from services.data_loader import load_cars
from services.memory import (
    get_memory_summary,
    get_or_create_user,
    get_user,
    init_db,
)
from services.retrieval import search_cars

# Compatibility imports for existing callers and tests. Implementations now
# live in focused service modules rather than this FastAPI entry point.
from services import search_flow
from services.search_flow import (  # noqa: F401
    build_unavailable_attribute_reply,
    search_from_extracted_request,
)


app = FastAPI(
    title="dubizzle Cars AI Assistant",
    description=(
        "FastAPI backend for an AI assistant that searches car inventory, "
        "remembers users, and books viewings."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    username: str = Field(..., example="layan123")
    session_id: str = Field(..., example="session_abc")
    message: str = Field(
        ...,
        example="Show me Mercedes-Benz cars from 2019",
    )


class ChatResponse(BaseModel):
    reply: str
    intent: str
    cars: List[Dict[str, Any]] = []
    extracted_request: Dict[str, Any] = {}
    memory_update: Dict[str, Any] = {}


class BookingRequest(BaseModel):
    username: str = Field(..., example="layan123")
    listing_id: int = Field(..., example=4)
    date: str = Field(..., example="2026-07-10")
    time: str = Field(..., example="15:00")
    budget: str = Field(default="", example="AED 100,000")
    needs: str = Field(default="", example="SUV with warranty")
    preferred_make: str = Field(default="", example="Mercedes-Benz")
    preferred_model: str = Field(default="", example="C-Class")
    notes: str = Field(default="", example="User requested a test drive.")


def execute_query_plan(query_plan, extracted_request, limit=5):
    """Compatibility proxy for callers that patch ``main.search_cars``."""

    return search_flow.execute_query_plan(
        query_plan,
        extracted_request,
        limit=limit,
        search_fn=search_cars,
    )


@app.on_event("startup")
def startup_event():
    """Initialize the database and inventory at startup."""

    init_db()
    load_cars()
    print(
        "[STARTUP] Backend ready. Dataset loaded and user memory database "
        "initialized."
    )


@app.get("/")
def root():
    return {
        "message": "dubizzle Cars AI Assistant API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/inventory/search")
def inventory_search(
    make: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    year: Optional[int] = Query(default=None),
    keyword: Optional[str] = Query(default=None),
    limit: int = Query(default=5, ge=1, le=20),
):
    """Search inventory directly without chat extraction."""

    cars = search_cars(
        make=make,
        model=model,
        year=year,
        keyword=keyword,
        limit=limit,
    )
    return {"count": len(cars), "cars": cars}


@app.get("/user/{username}")
def get_user_profile(username: str):
    """Return long-term user memory, creating a profile when needed."""

    existing_user = get_user(username)
    if not existing_user:
        get_or_create_user(username=username)
        return {
            "returning_user": False,
            "message": (
                f"Hello, {username}. I created a new profile for you. "
                "I do not have saved car preferences yet."
            ),
            "preferred_make": "",
            "preferred_model": "",
            "last_budget": "",
            "last_seen_car": "",
            "liked_cars": [],
            "memory_summary": "",
        }
    summary = get_memory_summary(username)
    summary["returning_user"] = True
    return summary


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return handle_chat(request, ChatResponse)


@app.post("/book-viewing")
def book_viewing_endpoint(request: BookingRequest):
    """Validate and persist an official inventory viewing booking."""

    car = get_car_by_listing_id(request.listing_id)
    if not car:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Listing ID {request.listing_id} was not found in the "
                "inventory."
            ),
        )
    result = book_viewing(
        username=request.username,
        selected_listing_id=request.listing_id,
        selected_car_title=build_car_title(car),
        booking_date=request.date,
        booking_time=request.time,
        budget=request.budget,
        needs=request.needs,
        preferred_make=request.preferred_make,
        preferred_model=request.preferred_model,
        notes=request.notes,
    )
    if result["success"]:
        persist_confirmed_booking_memory(
            username=request.username,
            car=car,
            booking_date=request.date,
            booking_time=request.time,
            budget=request.budget,
            needs=request.needs,
            preferred_make=request.preferred_make,
            preferred_model=request.preferred_model,
        )
    return result
