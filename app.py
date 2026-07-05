import html
import os
import uuid
from datetime import date, time

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
WELCOME_MESSAGE = (
    "Hi! I’m your dubizzle Cars assistant. Tell me what kind of car you’re "
    "looking for, your budget, or any must-have features, and I’ll help you find a match."
)


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="dubizzle Cars AI Assistant",
    page_icon="🚗",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 24px;
    }
    .small-label {
        font-size: 13px;
        color: #777;
        font-weight: 600;
    }
    .memory-box {
        padding: 12px;
        border-radius: 12px;
        background-color: #f7f7f7;
        border: 1px solid #e5e5e5;
        color: #1f2937;
        margin-bottom: 12px;
    }
    .memory-box * {
        color: #1f2937;
    }
    .car-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .car-meta {
        font-size: 14px;
        color: #555;
        margin-bottom: 8px;
    }
    .match-reason {
        font-size: 13px;
        color: #0f7b4f;
        font-weight: 600;
    }
    .selected-car-banner {
        padding: 12px 16px;
        border-radius: 12px;
        background-color: #e8f5ee;
        border: 2px solid #0f7b4f;
        color: #164e3b;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .selected-car-banner * {
        color: #164e3b;
    }
    div[class*="st-key-matched_cars_"] [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap;
        overflow-x: auto;
        padding: 0.25rem 0 1rem;
        scrollbar-width: thin;
    }
    div[class*="st-key-matched_cars_"] [data-testid="stColumn"] {
        flex: 0 0 285px;
        min-width: 285px;
    }
    div[class*="st-key-car_card_"] {
        min-height: 100%;
        position: relative;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }
    div[class*="st-key-car_card_"]:hover {
        box-shadow: 0 8px 24px rgba(15, 123, 79, 0.16);
        transform: translateY(-2px);
    }
    div[class*="st-key-car_card_"]:hover::after {
        content: "Interested in this? Ask more.";
        position: absolute;
        top: 8px;
        right: 8px;
        z-index: 10;
        padding: 5px 8px;
        border-radius: 6px;
        background: #1f2937;
        color: white;
        font-size: 12px;
        pointer-events: none;
    }
    div[class*="st-key-car_card_"] img {
        display: block;
        width: 100%;
        height: 155px;
        object-fit: cover;
        border-radius: 8px;
    }
    div[class*="st-key-reference_"] {
        position: absolute;
        inset: 0;
        z-index: 2;
        margin: 0 !important;
    }
    div[class*="st-key-reference_"] button {
        width: 100%;
        height: 100%;
        min-height: 100%;
        padding: 0;
        border: 0;
        opacity: 0;
        cursor: pointer;
    }
    div[class*="st-key-book_"] {
        position: relative;
        z-index: 3;
    }
    .car-reference {
        display: inline-block;
        padding: 7px 10px;
        margin-bottom: 8px;
        border: 1px solid #0f7b4f;
        border-radius: 8px;
        background: #e8f5ee;
        color: #164e3b;
        font-size: 13px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]

if "last_cars" not in st.session_state:
    st.session_state.last_cars = []

if "selected_car" not in st.session_state:
    st.session_state.selected_car = None

if "memory_summary" not in st.session_state:
    st.session_state.memory_summary = None

if "loaded_username" not in st.session_state:
    st.session_state.loaded_username = None

if "active_username" not in st.session_state:
    st.session_state.active_username = None

if "referenced_car" not in st.session_state:
    st.session_state.referenced_car = None

if "booking_car" not in st.session_state:
    st.session_state.booking_car = None


# -----------------------------
# API helpers
# -----------------------------

def api_get(endpoint: str, params: dict | None = None):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        st.error(f"Backend connection error: {error}")
        return None


def api_post(endpoint: str, payload: dict):
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        st.error(f"Backend connection error: {error}")
        return None


def load_user_memory(username: str):
    if not username:
        return None

    result = api_get(f"/user/{username}")
    st.session_state.memory_summary = result
    st.session_state.loaded_username = username
    return result


def reset_current_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.last_cars = []
    st.session_state.selected_car = None
    st.session_state.referenced_car = None
    st.session_state.booking_car = None


def switch_user(username: str):
    """Start a fresh short-term chat when the active identity changes."""

    clean_username = username.strip()
    if st.session_state.active_username == clean_username:
        return

    reset_current_session()
    st.session_state.memory_summary = None
    st.session_state.loaded_username = None
    st.session_state.active_username = clean_username


def car_display_name(car: dict) -> str:
    return (
        f"{car.get('year', '')} "
        f"{car.get('make', '').title()} "
        f"{car.get('model', '').title()}"
    ).strip()


def submit_chat_message(username: str, message: str, display_message: str | None = None):
    clean_username = username.strip()

    if not clean_username:
        st.warning("Enter a username before sending a message.")
        return None

    if st.session_state.active_username != clean_username:
        switch_user(clean_username)
        load_user_memory(clean_username)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": display_message or message,
        }
    )

    response = api_post(
        "/chat",
        {
            "username": clean_username,
            "session_id": st.session_state.session_id,
            "message": message,
        },
    )

    if not response:
        return None

    intent = response.get("intent", "")
    cars = response.get("cars", [])
    matched_cars = (
        cars
        if intent in {
            "car_search",
            "lead_capture",
            "general_car_advice",
            "returning_user_memory",
            "memory_recall",
            "similar_to_previous",
        }
        else []
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.get("reply", ""),
            "cars": matched_cars,
        }
    )
    st.session_state.last_cars = matched_cars

    memory_update = response.get("memory_update")
    if memory_update:
        st.session_state.memory_summary = memory_update
        st.session_state.loaded_username = clean_username

    return response


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("User")

    username = st.text_input("Username", value="")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Continue", use_container_width=True):
            if username.strip():
                clean_username = username.strip()
                switch_user(clean_username)
                load_user_memory(clean_username)
                st.rerun()
            else:
                st.warning("Enter a username first.")

    with col_b:
        if st.button("New Chat", use_container_width=True):
            reset_current_session()
            st.success("New chat started.")

    st.divider()

    st.subheader("Memory Panel")

    memory = (
        st.session_state.memory_summary
        if st.session_state.loaded_username == username.strip()
        else None
    )

    if memory:
        returning_user = memory.get("returning_user", False)
        status = "Returning user" if returning_user else "New user"

        st.markdown(
            f"""
            <div class="memory-box">
                <div><b>Status:</b> {status}</div>
                <div><b>Last budget:</b> {memory.get("last_budget") or "Not saved yet"}</div>
                <div><b>Preferred make:</b> {memory.get("preferred_make") or "Not saved yet"}</div>
                <div><b>Preferred model:</b> {memory.get("preferred_model") or "Not saved yet"}</div>
                <div><b>Last viewed car:</b> {memory.get("last_seen_car") or "Not saved yet"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(memory.get("message", ""))
    else:
        st.caption("Enter a username and select Continue to load saved memory.")

    st.divider()

    st.caption(f"Session ID: {st.session_state.session_id[:8]}...")


# -----------------------------
# Main header
# -----------------------------

st.markdown('<div class="main-title">🚗 dubizzle Cars AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Search inventory, ask follow-up questions, remember preferences, and book viewing slots.</div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Backend health check
# -----------------------------

health = api_get("/health")

if health and health.get("status") == "ok":
    st.success("Backend connected.")
else:
    st.warning("Backend is not connected. Start FastAPI with: uv run uvicorn main:app --reload")


# -----------------------------
# Chat and matched cars
# -----------------------------

st.subheader("Chat Assistant")


def render_matched_cars(cars: list[dict], message_index: int) -> None:
    count = len(cars)
    st.markdown(f"**{count} matched {'car' if count == 1 else 'cars'}**")

    with st.container(key=f"matched_cars_{message_index}"):
        columns = st.columns(count, gap="small")

        for car_index, (column, car) in enumerate(zip(columns, cars), start=1):
            listing_id = car.get("listing_id", "")
            card_key = f"car_card_{message_index}_{listing_id}_{car_index}"

            with column:
                with st.container(border=True, key=card_key):
                    if car.get("photo_url"):
                        photo_url = html.escape(str(car["photo_url"]), quote=True)
                        alt_text = html.escape(car_display_name(car), quote=True)
                        st.markdown(
                            f'<img src="{photo_url}" alt="{alt_text}">',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No image available")

                    st.markdown(f"**{car_index}. {car_display_name(car)}**")

                    if st.button(
                        f"Ask more about {car_display_name(car)}",
                        key=f"reference_{message_index}_{listing_id}_{car_index}",
                        help="Interested in this? Ask more.",
                    ):
                        st.session_state.referenced_car = car
                        st.rerun()

                    st.caption(
                        f"{car.get('trim') or 'Trim not listed'} · "
                        f"Listing ID {listing_id}"
                    )

                    description = car.get("description", "")
                    if description:
                        preview = description[:145]
                        st.caption(preview + ("…" if len(description) > 145 else ""))

                    st.markdown(
                        f'<div class="match-reason">{car.get("match_reason", "")}</div>',
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "Book Now",
                        key=f"book_{message_index}_{listing_id}_{car_index}",
                        type="primary",
                        use_container_width=True,
                    ):
                        booking_message = (
                            f"I would like to book the {car_display_name(car)} "
                            f"(Listing ID: {listing_id})."
                        )
                        response = submit_chat_message(
                            username,
                            booking_message,
                        )
                        if response:
                            st.session_state.booking_car = car
                            st.session_state.referenced_car = None
                            st.rerun()


for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        cars = message.get("cars") or []
        if cars:
            render_matched_cars(cars, message_index)


# -----------------------------
# Inline booking confirmation
# -----------------------------

booking_car = st.session_state.booking_car

if booking_car:
    with st.chat_message("assistant"):
        st.markdown(
            f"#### Confirm viewing for {car_display_name(booking_car)}\n"
            "Choose a date and time. Viewing slots are Monday to Saturday, "
            "from 8 AM to 8 PM."
        )

        with st.form(
            f"booking_form_{booking_car.get('listing_id')}_{st.session_state.session_id}"
        ):
            booking_date = st.date_input("Viewing date", value=date.today())
            booking_time = st.time_input("Viewing time", value=time(15, 0))
            booking_needs = st.text_area(
                "Needs / notes",
                placeholder="e.g. Interested in warranty, low mileage, family use...",
            )
            submitted = st.form_submit_button(
                "Confirm booking",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            clean_username = username.strip()

            if not clean_username:
                st.warning("Enter a username before confirming the booking.")
            else:
                saved_memory = (
                    st.session_state.memory_summary
                    if st.session_state.loaded_username == clean_username
                    else {}
                ) or {}

                result = api_post(
                    "/book-viewing",
                    {
                        "username": clean_username,
                        "listing_id": int(booking_car.get("listing_id")),
                        "date": booking_date.isoformat(),
                        "time": booking_time.strftime("%H:%M"),
                        "budget": saved_memory.get("last_budget", ""),
                        "needs": booking_needs,
                        "preferred_make": (
                            saved_memory.get("preferred_make")
                            or booking_car.get("make", "")
                        ),
                        "preferred_model": (
                            saved_memory.get("preferred_model")
                            or booking_car.get("model", "")
                        ),
                        "notes": "Booking submitted from the inline chat form.",
                    },
                )

                if result:
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Viewing date: {booking_date.isoformat()}\n\n"
                                f"Viewing time: {booking_time.strftime('%H:%M')}"
                            ),
                        }
                    )

                    if result.get("success"):
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"✅ {result.get('message', 'Viewing booked successfully.')} "
                                    f"Your viewing for {car_display_name(booking_car)} is "
                                    f"confirmed for {booking_date.isoformat()} at "
                                    f"{booking_time.strftime('%H:%M')}."
                                ),
                            }
                        )
                        st.session_state.booking_car = None
                        load_user_memory(clean_username)
                    else:
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result.get(
                                    "message",
                                    "The booking could not be completed.",
                                ),
                            }
                        )

                    st.rerun()


# -----------------------------
# Referenced car and chat input
# -----------------------------

referenced_car = st.session_state.referenced_car

if referenced_car:
    ref_col, clear_col = st.columns([5, 1])
    with ref_col:
        st.markdown(
            f'<div class="car-reference">↳ Asking about '
            f'{car_display_name(referenced_car)} · Listing ID '
            f'{referenced_car.get("listing_id", "")}</div>',
            unsafe_allow_html=True,
        )
    with clear_col:
        if st.button("Clear", key="clear_car_reference", use_container_width=True):
            st.session_state.referenced_car = None
            st.rerun()

user_message = st.chat_input(
    "Ask me about cars, listings, warranty, mileage, or bookings..."
)

if user_message:
    backend_message = user_message
    display_message = user_message

    if referenced_car:
        car_name = car_display_name(referenced_car)
        listing_id = referenced_car.get("listing_id", "")
        backend_message = (
            f"Tell me about {car_name} (Listing ID: {listing_id}). "
            f"My question is: {user_message}"
        )
        display_message = f"↳ **{car_name} · Listing ID {listing_id}**\n\n{user_message}"

    response = submit_chat_message(
        username,
        backend_message,
        display_message=display_message,
    )

    if response:
        st.session_state.referenced_car = None
        st.rerun()
