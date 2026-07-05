from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from services.data_loader import load_cars


load_dotenv()

try:
    from google import genai
except ImportError:
    genai = None


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_client():
    """
    Create a Gemini client if the API key and package are available.
    If not available, return None so the app can use fallback mode.
    """

    if not GEMINI_API_KEY or genai is None:
        return None

    return genai.Client(api_key=GEMINI_API_KEY)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON from an LLM response.

    The LLM should return JSON only, but this helper also tries to recover JSON
    if the response includes extra text accidentally.
    """

    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON object from the text
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def call_gemini(prompt: str) -> Optional[str]:
    """
    Send a prompt to Gemini.

    If Gemini fails for any reason, return None.
    This keeps the app from crashing.
    """

    client = get_gemini_client()

    if client is None:
        return None

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        return response.text

    except Exception as error:
        print(f"[LLM ERROR] Gemini call failed: {error}")
        return None


def get_known_values() -> Dict[str, List[str]]:
    """
    Load known makes/models from the dataset.
    This helps fallback extraction avoid random invented makes/models.
    """

    df = load_cars()

    makes = sorted(set(df["make"].dropna().astype(str).str.lower().str.strip()))
    models = sorted(set(df["model"].dropna().astype(str).str.lower().str.strip()))

    return {
        "makes": [make for make in makes if make],
        "models": [model for model in models if model],
    }


def fallback_extract_user_request(message: str) -> Dict[str, Any]:
    """
    Rule-based fallback if the LLM is unavailable.

    This is not as smart as the LLM, but it keeps the project functional.
    """

    message_lower = message.lower().strip()
    known_values = get_known_values()

    detected_make = None
    detected_model = None
    detected_year = None
    keywords = []

    # Detect year
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", message_lower)
    if year_match:
        detected_year = int(year_match.group(0))

    # Detect make
    for make in known_values["makes"]:
        if make in message_lower:
            detected_make = make
            break

    # Detect model
    for model in known_values["models"]:
        if model in message_lower:
            detected_model = model
            break

    keyword_candidates = [
        "warranty",
        "gcc",
        "low mileage",
        "service history",
        "accident",
        "clean",
        "japanese",
        "american",
        "suv",
        "sedan",
        "coupe",
        "luxury",
        "family",
        "reliable",
        "automatic",
        "electric",
        "hybrid",
    ]

    for keyword in keyword_candidates:
        if keyword in message_lower:
            keywords.append(keyword)

    # Very simple intent guess
    if any(word in message_lower for word in ["book", "booking", "viewing", "test drive", "appointment"]):
        intent = "booking"
    elif any(word in message_lower for word in ["first one", "second one", "third one", "it", "that car", "details", "warranty", "mileage"]):
        intent = "car_details"
    elif any(word in message_lower for word in ["budget", "i want", "i need", "looking for", "prefer", "under"]):
        intent = "lead_capture"
    else:
        intent = "car_search"

    # For our current retrieval.py, we use one keyword string.
    keyword = " ".join(keywords) if len(keywords) > 1 else (keywords[0] if keywords else None)

    return {
        "intent": intent,
        "make": detected_make,
        "model": detected_model,
        "year": detected_year,
        "keyword": keyword,
        "keywords": keywords,
        "budget": extract_budget_text(message_lower),
        "needs": message,
        "selected_position": extract_selected_position(message_lower),
        "llm_used": False,
    }


def extract_budget_text(message: str) -> Optional[str]:
    """
    Extract a rough budget phrase from the message.
    Example:
    'under AED 120,000' → 'AED 120,000'
    """

    budget_patterns = [
        r"(?:aed|dh|dhs)\s?[0-9,]+",
        r"[0-9,]+\s?(?:aed|dh|dhs)",
        r"under\s?[0-9,]+",
    ]

    for pattern in budget_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(0)

    return None


def extract_selected_position(message: str) -> Optional[int]:
    """
    Extract references like first/second/third as zero-based positions.
    """

    position_map = {
        "first": 0,
        "1st": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4,
    }

    for word, position in position_map.items():
        if re.search(rf"\b{re.escape(word)}\b", message):
            return position

    return None


def extract_user_request(message: str) -> Dict[str, Any]:
    """
    Use Gemini to extract a structured request from natural language.

    If Gemini fails or returns invalid JSON, use fallback extraction.
    """

    prompt = f"""
You are an intent extraction layer for a dubizzle used-cars AI assistant.

Return ONLY valid JSON. No markdown. No explanation.

Allowed intents:
- greeting
- car_search
- car_details
- compare_listings
- booking
- lead_capture
- returning_user_memory
- general_car_advice
- blocked_non_automotive
- blocked_competitor

Extract these fields:
{{
  "intent": "car_search | car_details | booking | lead_capture | compare_listings | greeting | general_car_advice | blocked_non_automotive | blocked_competitor",
  "make": string or null,
  "model": string or null,
  "year": integer or null,
  "keyword": string or null,
  "keywords": list of strings,
  "budget": string or null,
  "needs": string or null,
  "selected_position": integer or null
}}

Rules:
- If user asks about a make/model/year/features, intent is car_search.
- If user gives budget/preferences/needs, intent is lead_capture.
- If user asks about "first one", "second one", "it", or a previously shown car, intent is car_details.
- selected_position must be zero-based: first=0, second=1, third=2.
- If user asks to book/view/test drive, intent is booking.
- If user compares two inventory cars, intent is compare_listings.
- If user asks about competitors or other car platforms, intent is blocked_competitor.
- If the request is not about cars, inventory, booking, or car advice, intent is blocked_non_automotive.
- Do not invent a make or model if the user did not mention one.

User message:
{message}
""".strip()

    response_text = call_gemini(prompt)

    if response_text is None:
        return fallback_extract_user_request(message)

    parsed = safe_json_loads(response_text)

    if parsed is None:
        return fallback_extract_user_request(message)

    # Normalize keys so backend can trust the shape
    return {
        "intent": parsed.get("intent") or "car_search",
        "make": parsed.get("make"),
        "model": parsed.get("model"),
        "year": parsed.get("year"),
        "keyword": parsed.get("keyword"),
        "keywords": parsed.get("keywords") or [],
        "budget": parsed.get("budget"),
        "needs": parsed.get("needs") or message,
        "selected_position": parsed.get("selected_position"),
        "llm_used": True,
    }


def format_cars_for_prompt(cars: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved cars into compact text for the LLM.

    Important:
    Only these cars are allowed to be mentioned in the final response.
    """

    if not cars:
        return "No matching inventory results were found."

    formatted = []

    for index, car in enumerate(cars, start=1):
        formatted.append(
            f"""
Result {index}
Listing ID: {car.get("listing_id", "")}
Year: {car.get("year", "")}
Make: {car.get("make", "")}
Model: {car.get("model", "")}
Trim: {car.get("trim", "")}
Title: {car.get("title", "")}
Description: {car.get("description", "")}
Match reason: {car.get("match_reason", "")}
""".strip()
        )

    return "\n\n".join(formatted)


def generate_inventory_reply(
    user_message: str,
    cars: List[Dict[str, Any]],
    extracted_request: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a friendly reply for inventory search results.

    The LLM can only use the retrieved cars.
    If Gemini fails, return a fallback response.
    """

    if not cars:
        return (
            "I could not find matching cars in the provided inventory. "
            "Try searching by another make, model, year, or feature."
        )

    inventory_context = format_cars_for_prompt(cars)

    prompt = f"""
You are a dubizzle cars assistant.

User message:
{user_message}

Structured request:
{json.dumps(extracted_request or {}, ensure_ascii=False)}

Retrieved inventory results:
{inventory_context}

Rules:
- Only mention cars from the retrieved inventory results above.
- Do not invent price, mileage, warranty, availability, or service history.
- If a detail is missing, say it is not available in the listing.
- Keep the response concise and helpful.
- Number the cars clearly.
- Mention why the cars matched when useful.
- Encourage the user to ask about a specific result or book a viewing.
""".strip()

    response_text = call_gemini(prompt)

    if response_text:
        return response_text.strip()

    return fallback_inventory_reply(cars)


def fallback_inventory_reply(cars: List[Dict[str, Any]]) -> str:
    """
    Basic response if Gemini is unavailable.
    """

    lines = ["I found these matching cars in the inventory:"]

    for index, car in enumerate(cars, start=1):
        lines.append(
            f"{index}. {car.get('year', '')} {car.get('make', '')} "
            f"{car.get('model', '')} {car.get('trim', '')} "
            f"(Listing ID: {car.get('listing_id', '')})"
        )
        lines.append(f"   {car.get('match_reason', '')}")

    lines.append("\nYou can ask me about the first one, second one, or book a viewing.")
    return "\n".join(lines)


def generate_car_detail_reply(user_message: str, car: Dict[str, Any]) -> str:
    """
    Generate a reply about one selected car.

    The LLM can only use this selected car's listing data.
    """

    prompt = f"""
You are a dubizzle cars assistant.

User asked:
{user_message}

Selected car listing:
Listing ID: {car.get("listing_id", "")}
Year: {car.get("year", "")}
Make: {car.get("make", "")}
Model: {car.get("model", "")}
Trim: {car.get("trim", "")}
Title: {car.get("title", "")}
Description: {car.get("description", "")}

Rules:
- Only answer using the selected car listing above.
- Do not invent price, mileage, warranty, service history, accident history, or availability.
- If the listing does not mention something, say it is not available in the listing.
- Keep the answer concise.
""".strip()

    response_text = call_gemini(prompt)

    if response_text:
        return response_text.strip()

    return fallback_car_detail_reply(car)


def fallback_car_detail_reply(car: Dict[str, Any]) -> str:
    """
    Basic selected-car response if Gemini is unavailable.
    """

    return (
        f"{car.get('year', '')} {car.get('make', '')} {car.get('model', '')} "
        f"{car.get('trim', '')}\n\n"
        f"Title: {car.get('title', '')}\n\n"
        f"Description: {car.get('description', '')}\n\n"
        f"Listing ID: {car.get('listing_id', '')}"
    )


def generate_refusal_reply() -> str:
    """
    Standard refusal for out-of-scope requests.
    """

    return (
        "I’m here to help with dubizzle car listings, vehicle details, and booking viewings. "
        "I can help you find cars by make, model, year, features, or preferences."
    )


if __name__ == "__main__":
    test_messages = [
        "I want a reliable SUV with warranty.",
        "Show me Mercedes-Benz cars from 2019.",
        "Tell me about the first one.",
        "Can I book a viewing for Friday at 3 PM?",
        "Write me Python code.",
    ]

    for message in test_messages:
        print("\nUser message:")
        print(message)

        extracted = extract_user_request(message)

        print("Extracted request:")
        print(json.dumps(extracted, indent=2))
        print("-" * 60)