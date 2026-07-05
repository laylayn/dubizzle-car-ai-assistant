import re
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.data_loader import load_cars
from services.retrieval import search_cars
from services.listing_attributes import (
    COLOR_TERMS,
    PRICE_FIELD_NAMES,
    extract_color_from_car,
    extract_mileage_from_car,
    extract_mileage_from_text,
    extract_price_from_car,
    extract_price_from_text,
    format_price_amount,
    get_car_listing_text,
    text_mentions_feature,
)
from services.query_planner import build_query_plan
from services.guardrails import apply_guardrails, log_intent
from services.llm_agent import (
    extract_user_request,
    is_memory_recall_request,
    generate_grounded_reply,
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
    extract_car_positions,
    resolve_car_reference,
    is_follow_up_about_selected_car,
    update_preferences,
    get_preferences,
    get_last_results,
)
from services.memory import (
    init_db,
    get_user,
    get_or_create_user,
    update_user_memory,
    get_memory_summary,
    add_liked_car,
    is_meaningful_memory_query,
    format_memory_summary_for_user,
    get_recent_user_interactions,
    record_user_interaction,
    update_memory_summary,
)
from services.booking import book_viewing, extract_booking_slot, save_lead


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
    username: str = Field(..., example="layan123")
    session_id: str = Field(..., example="session_abc")
    message: str = Field(..., example="Show me Mercedes-Benz cars from 2019")


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

    car = {
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

    for optional_field in [
        "price",
        "amount",
        "listing_price",
        "sale_price",
        "price_aed",
        "asking_price",
        "color",
        "colour",
        "mileage",
        "kilometers",
        "kilometres",
        "km",
        "odometer",
        "exterior_color",
        "exterior_colour",
        "features",
    ]:
        value = row.get(optional_field, "")
        if str(value).strip():
            car[optional_field] = value

    return car


def get_car_by_listing_id(listing_id: int) -> Optional[Dict[str, Any]]:
    """
    Find one car by listing ID from the dataset.
    """

    df = load_cars()
    match = df[df["listing_id"] == int(listing_id)]

    if match.empty:
        return None

    return row_to_car(match.iloc[0])


def get_car_from_message(message: str) -> Optional[Dict[str, Any]]:
    """Resolve an explicit listing reference included by a frontend car card."""

    match = re.search(
        r"\blisting\s*id\s*[:#-]?\s*(\d+)\b",
        message,
        re.IGNORECASE,
    )

    if not match:
        return None

    return get_car_by_listing_id(int(match.group(1)))


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


def get_requested_color(message: str) -> Optional[str]:
    """Return an explicitly requested common color, if present."""

    message_lower = message.lower()
    for color in COLOR_TERMS:
        if re.search(rf"\b{re.escape(color)}\b", message_lower):
            return color
    return None


def get_car_reference_label(car: Dict[str, Any], message: str) -> str:
    """Build a natural label such as 'the first Mini Cooper'."""

    make_model = " ".join(
        part
        for part in [
            str(car.get("make", "")).title(),
            str(car.get("model", "")).title(),
        ]
        if part
    ).strip() or "selected car"

    ordinal_patterns = [
        (r"\b(?:first|1st)\b", "first"),
        (r"\b(?:second|2nd)\b", "second"),
        (r"\b(?:third|3rd)\b", "third"),
        (r"\b(?:fourth|4th)\b", "fourth"),
        (r"\b(?:fifth|5th)\b", "fifth"),
    ]
    for pattern, ordinal in ordinal_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return f"the {ordinal} {make_model}"

    return f"this {make_model}"


def extract_asked_feature(message: str) -> Optional[str]:
    """Extract a short natural feature phrase from questions like 'does it have X?'."""

    match = re.search(
        r"\b(?:have|include|mention|with)\s+"
        r"(?:a|an|any|the)?\s*([^?.!,]{2,60})",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None

    feature = match.group(1).strip()
    if len(feature.split()) > 6:
        return None
    return feature


def answer_car_question(car: Dict[str, Any], user_message: str) -> Optional[str]:
    """Build a deterministic fallback for attribute-specific follow-ups."""

    message_lower = user_message.lower()
    car_label = get_car_reference_label(car, user_message)
    listing_text = get_car_listing_text(car)

    if re.search(
        r"\b(?:price|cost|how much|aed|dhs?|dirhams?|amount)\b",
        message_lower,
    ):
        price_result = extract_price_from_car(car)
        if price_result and price_result["source"] == "structured":
            return f"The listed price is AED {price_result['amount']}."

        if price_result:
            if price_result.get("currency_explicit"):
                return (
                    "The price shown in the listing is "
                    f"AED {price_result['amount']}."
                )
            return (
                f"The listed price appears to be AED {price_result['amount']}."
            )

        return "I couldn’t find a price in the provided listing."

    requested_color = get_requested_color(user_message)
    if requested_color:
        if text_mentions_feature(car, requested_color):
            return (
                f"Yes. The provided listing text explicitly mentions "
                f"{requested_color} for {car_label}."
            )
        return (
            f"I can’t confirm that {car_label} is {requested_color} from the "
            "provided listing text. The title and description do not mention "
            "the color."
        )

    if re.search(r"\b(?:mileage|kilomet(?:er|re)s?|km)\b", message_lower):
        mileage_result = extract_mileage_from_car(car)
        if mileage_result:
            return (
                f"The listed mileage for {car_label} is "
                f"{mileage_result['display']}."
            )
        return f"The listing does not provide a clear mileage for {car_label}."

    if re.search(r"\bwarranty\b", message_lower):
        if text_mentions_feature(car, "warranty"):
            return f"Yes. The listing text mentions warranty for {car_label}."
        return f"The listing text does not mention warranty for {car_label}."

    if re.search(r"\bgcc\b", message_lower):
        if text_mentions_feature(car, "gcc"):
            return f"Yes. The listing text identifies {car_label} as GCC."
        return f"The listing text does not mention GCC for {car_label}."

    direct_fields = {
        "year": "year",
        "make": "make",
        "model": "model",
        "trim": "trim",
    }
    for attribute, field_name in direct_fields.items():
        if re.search(rf"\b{attribute}\b", message_lower):
            value = car.get(field_name)
            if str(value or "").strip():
                return f"The {attribute} for {car_label} is {value}."
            return f"The listing does not provide a {attribute} for {car_label}."

    asked_feature = extract_asked_feature(user_message)
    if asked_feature:
        if text_mentions_feature(car, asked_feature):
            return f"Yes. The listing text mentions {asked_feature} for {car_label}."
        return (
            f"I can’t confirm {asked_feature} for {car_label}; it is not "
            "mentioned in the provided title or description."
        )

    if re.search(r"\b(?:feature|features|option|options)\b", message_lower):
        description = str(car.get("description", "")).strip()
        if description and description != ".":
            preview = description[:400]
            suffix = "…" if len(description) > 400 else ""
            return f"The listing describes these details for {car_label}: {preview}{suffix}"
        return f"The listing does not provide a usable features list for {car_label}."

    return None


def build_selected_car_evidence(
    car: Dict[str, Any],
    user_message: str,
) -> Dict[str, Any]:
    """Derive difficult facts while leaving question interpretation to the LLM."""

    requested_color = get_requested_color(user_message)
    asked_feature = extract_asked_feature(user_message)
    structured_values = {
        field_name: car.get(field_name)
        for field_name in [
            "year",
            "make",
            "model",
            "trim",
            *PRICE_FIELD_NAMES,
            "mileage",
            "kilometers",
            "kilometres",
            "km",
            "color",
            "colour",
            "features",
        ]
        if str(car.get(field_name, "")).strip()
    }

    evidence: Dict[str, Any] = {
        "structured_values": structured_values,
        "price_extraction": extract_price_from_car(car),
        "mileage_extraction": extract_mileage_from_car(car),
        "color_extraction": extract_color_from_car(car),
    }

    if requested_color:
        evidence["requested_color"] = requested_color
        evidence["requested_color_explicitly_mentioned"] = text_mentions_feature(
            car,
            requested_color,
        )

    if asked_feature:
        evidence["asked_feature"] = asked_feature
        evidence["asked_feature_explicitly_mentioned"] = text_mentions_feature(
            car,
            asked_feature,
        )

    return evidence


def build_search_label(extracted_request: Dict[str, Any]) -> str:
    """Build a readable make/model label for partial feature matches."""

    values = [
        extracted_request.get("make"),
        extracted_request.get("model"),
    ]
    return " ".join(format_filter_value(value) for value in values if value).strip()


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
    """Search all objective constraints together, then relax transparently."""

    make = str(extracted_request.get("make") or "").strip() or None
    model = str(extracted_request.get("model") or "").strip() or None
    year = extracted_request.get("year")
    keyword = str(extracted_request.get("keyword") or "").strip() or None
    keyword_values = [
        *(extracted_request.get("required_keywords") or []),
        *(extracted_request.get("keywords") or []),
    ]
    body_type = str(extracted_request.get("body_type") or "").strip()
    keywords = []

    # A singular keyword is the legacy representation. When the extractor
    # supplies the complete list, that list is authoritative and prevents an
    # LLM-created phrase such as "reliable SUV with warranty" becoming one
    # impossible literal filter.
    keyword_inputs = keyword_values or ([keyword] if keyword else [])
    if body_type:
        keyword_inputs.append(body_type)

    for value in keyword_inputs:
        clean_value = str(value or "").lower().strip()
        if clean_value and clean_value not in keywords:
            keywords.append(clean_value)

    soft_preferences = [
        str(value).strip()
        for value in extracted_request.get("soft_preferences") or []
        if str(value).strip()
    ]

    has_structured_filter = any([make, model, year])
    has_keyword_filter = bool(keywords)
    budget_text = extracted_request.get("budget")
    # Budget filtering happens after textual retrieval because some datasets
    # store price in listing text. Pull enough candidates first so an early
    # over-budget row cannot hide a later in-budget match.
    candidate_limit = max(limit * 20, 100) if budget_text else limit

    if not has_structured_filter and not has_keyword_filter:
        return []

    cars = search_cars(
        make=make,
        model=model,
        year=year,
        keywords=keywords,
        require_all_keywords=True,
        limit=candidate_limit,
    )

    if not cars and keywords:
        cars = search_cars(
            make=make,
            model=model,
            year=year,
            keywords=keywords,
            require_all_keywords=False,
            limit=candidate_limit,
        )

    if not cars and has_structured_filter:
        cars = search_cars(
            make=make,
            model=model,
            year=year,
            limit=candidate_limit,
        )
        if keywords:
            missing_terms = ", ".join(keywords)
            cars = [
                {
                    **car,
                    "match_reason": (
                        f"{car.get('match_reason', '')}; listing text does not "
                        f"confirm {missing_terms}"
                    ),
                }
                for car in cars
            ]

    if soft_preferences:
        preference_text = ", ".join(soft_preferences)
        cars = [
            {
                **car,
                "match_reason": (
                    f"{car.get('match_reason', '')}; {preference_text} is a "
                    "requested preference but is not independently confirmed"
                ),
            }
            for car in cars
        ]

    budget_amount = format_price_amount(budget_text)
    if budget_amount and cars:
        budget_limit = int(float(budget_amount.replace(",", "")))
        within_budget = []
        unknown_budget = []

        for car in cars:
            price_result = extract_price_from_car(car)
            price_is_comparable = bool(
                price_result
                and (
                    price_result.get("source") in {"structured", "title"}
                    or (
                        price_result.get("source") == "description"
                        and price_result.get("currency_explicit")
                    )
                )
            )

            if price_is_comparable:
                car_price = int(
                    float(price_result["amount"].replace(",", ""))
                )
                if car_price <= budget_limit:
                    within_budget.append(
                        {
                            **car,
                            "match_reason": (
                                f"{car.get('match_reason', '')}; listed price "
                                f"appears within {budget_text}"
                            ),
                        }
                    )
            else:
                unknown_budget.append(
                    {
                        **car,
                        "match_reason": (
                            f"{car.get('match_reason', '')}; budget fit could "
                            "not be confirmed from the listing"
                        ),
                    }
                )

        cars = (within_budget + unknown_budget)[:limit]

    return cars[:limit]


def execute_query_plan(
    query_plan: Dict[str, Any],
    extracted_request: Dict[str, Any],
    limit: int = 5,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute normalized filters and attribute ranking against inventory."""

    sort_by = query_plan.get("sort_by")
    sort_order = query_plan.get("sort_order")
    filters = query_plan.get("filters") or {}

    if not sort_by:
        cars = search_from_extracted_request(extracted_request, limit=limit)
        return cars, {
            "status": "matched" if cars else "no_matches",
            "sort_by": None,
            "sort_order": None,
            "candidate_count": len(cars),
            "ranked_count": 0,
            "missing_attribute_count": 0,
        }

    attribute_extractors = {
        "mileage": extract_mileage_from_car,
        "price": extract_price_from_car,
        "year": lambda car: (
            {
                "value": int(car.get("year")),
                "display": str(car.get("year")),
                "source": "structured",
            }
            if str(car.get("year") or "").isdigit()
            and int(car.get("year")) > 0
            else None
        ),
    }
    extractor = attribute_extractors.get(sort_by)
    if extractor is None or sort_order not in {"ascending", "descending"}:
        return [], {
            "status": "unexecutable",
            "sort_by": sort_by,
            "sort_order": sort_order,
            "candidate_count": 0,
            "ranked_count": 0,
            "missing_attribute_count": 0,
        }

    candidates = search_cars(
        make=filters.get("make"),
        model=filters.get("model"),
        year=filters.get("year"),
        keywords=list(filters.get("features") or []),
        require_all_keywords=True,
        limit=100_000,
    )

    budget_text = filters.get("budget")
    budget_amount = format_price_amount(budget_text)
    if budget_amount:
        budget_limit = int(float(budget_amount.replace(",", "")))
        budget_filtered = []
        for car in candidates:
            price_result = extract_price_from_car(car)
            if price_result and price_result["value"] > budget_limit:
                continue
            if not price_result:
                car = {
                    **car,
                    "match_reason": (
                        f"{car.get('match_reason', '')}; budget fit could not "
                        "be confirmed from the listing"
                    ),
                }
            budget_filtered.append(car)
        candidates = budget_filtered

    ranked_cars = []
    for car in candidates:
        attribute_result = extractor(car)
        if not attribute_result:
            continue
        ranked_cars.append(
            {
                **car,
                "ranked_attribute": sort_by,
                "ranked_value": attribute_result["value"],
                "ranked_value_display": attribute_result["display"],
                "match_reason": (
                    f"{car.get('match_reason', '')}; {sort_by} available as "
                    f"{attribute_result['display']}"
                ),
            }
        )

    ranked_cars.sort(
        key=lambda car: car["ranked_value"],
        reverse=sort_order == "descending",
    )
    missing_count = len(candidates) - len(ranked_cars)
    if ranked_cars:
        status = "ranked"
    elif candidates:
        status = "attribute_unavailable"
    else:
        status = "no_matches"
    return ranked_cars[:limit], {
        "status": status,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "candidate_count": len(candidates),
        "ranked_count": len(ranked_cars),
        "missing_attribute_count": missing_count,
        "partial_attribute_coverage": bool(ranked_cars and missing_count),
    }


def build_ranked_inventory_fallback(
    cars: List[Dict[str, Any]],
    execution: Dict[str, Any],
) -> str:
    """Build a factual fallback from query execution metadata."""

    sort_by = execution.get("sort_by")
    sort_order = execution.get("sort_order")
    introductions = {
        ("mileage", "ascending"): (
            "I found these listings with the lowest mileage information available."
        ),
        ("mileage", "descending"): (
            "I found these listings with the highest mileage information available."
        ),
        ("price", "ascending"): (
            "I found these listings with the lowest prices available."
        ),
        ("price", "descending"): (
            "I found these listings with the highest prices available."
        ),
        ("year", "ascending"): "I found the oldest listings available.",
        ("year", "descending"): "I found the newest listings available.",
    }
    lines = [
        introductions.get(
            (sort_by, sort_order),
            f"I ranked these listings by {sort_by}.",
        )
    ]
    for index, car in enumerate(cars, start=1):
        lines.append(
            f"{index}. {build_car_title(car)} — "
            f"{car.get('ranked_value_display')}"
        )

    if execution.get("partial_attribute_coverage"):
        if sort_by == "mileage":
            lines.append(
                "I only ranked listings where mileage was available in the "
                "listing text."
            )
        elif sort_by == "price":
            lines.append(
                "I only ranked listings where a clear price was available."
            )

    return "\n".join(lines)


def build_unavailable_attribute_reply(attribute: str) -> str:
    """Explain why an attribute-ranking plan could not return results."""

    messages = {
        "mileage": (
            "The provided inventory does not include enough mileage information "
            "for me to rank cars by mileage."
        ),
        "price": (
            "The provided inventory does not include enough clear price "
            "information for me to rank cars by price."
        ),
        "year": (
            "The provided inventory does not include enough year information "
            "for me to rank these cars."
        ),
    }
    return messages.get(
        attribute,
        "The provided inventory does not include enough information to rank those cars.",
    )


def get_listing_similarity_tokens(car: Dict[str, Any]) -> set[str]:
    """Build general text tokens for ranking similar listings."""

    text = " ".join(
        str(car.get(field, ""))
        for field in ["make", "model", "trim", "title", "description"]
    ).lower()
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", text)
        if not token.isdigit()
    }


def find_similar_to_liked_cars(
    liked_cars: List[Dict[str, Any]],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Find alternatives sharing make/model and listing-text similarity."""

    liked_ids = {
        int(car.get("listing_id"))
        for car in liked_cars
        if car.get("listing_id") is not None
    }
    candidates_by_id: Dict[int, Dict[str, Any]] = {}

    for liked_car in liked_cars:
        make = str(liked_car.get("make") or "").strip()
        model = str(liked_car.get("model") or "").strip()

        searches = []
        if make and model:
            searches.extend(search_cars(make=make, model=model, limit=100))
        if make:
            searches.extend(search_cars(make=make, limit=100))

        for candidate in searches:
            listing_id = int(candidate.get("listing_id", 0))
            if listing_id and listing_id not in liked_ids:
                candidates_by_id[listing_id] = candidate

    if not candidates_by_id:
        return [
            {
                **car,
                "match_reason": (
                    "This is your saved liked listing; no alternative from the "
                    "same make or model was found."
                ),
            }
            for car in liked_cars[:limit]
        ]

    liked_token_sets = [
        get_listing_similarity_tokens(car)
        for car in liked_cars
    ]
    liked_models = {
        str(car.get("model") or "").lower().strip()
        for car in liked_cars
        if car.get("model")
    }

    def candidate_score(candidate: Dict[str, Any]) -> tuple[int, int]:
        candidate_model = str(candidate.get("model") or "").lower().strip()
        candidate_tokens = get_listing_similarity_tokens(candidate)
        text_overlap = max(
            (
                len(candidate_tokens & liked_tokens)
                for liked_tokens in liked_token_sets
            ),
            default=0,
        )
        return (int(candidate_model in liked_models), text_overlap)

    ranked_candidates = sorted(
        candidates_by_id.values(),
        key=candidate_score,
        reverse=True,
    )[:limit]

    basis = ", ".join(
        dict.fromkeys(
            " ".join(
                format_filter_value(value)
                for value in [
                    car.get("make"),
                    car.get("model"),
                ]
                if value
            )
            for car in liked_cars
        )
    )

    return [
        {
            **car,
            "match_reason": f"Similar to your saved interest in {basis}.",
        }
        for car in ranked_candidates
    ]


def has_search_filters(extracted_request: Dict[str, Any]) -> bool:
    """Return whether an extracted historical request can safely drive search."""

    return bool(
        extracted_request.get("make")
        or extracted_request.get("model")
        or extracted_request.get("year")
        or extracted_request.get("keyword")
        or any(extracted_request.get("keywords") or [])
    )


def search_from_user_memory(
    user: Dict[str, Any],
    limit: int = 5,
) -> tuple[List[Dict[str, Any]], str, str, bool]:
    """Search inventory from long-term memory in explicit priority order."""

    liked_cars = []
    for listing_id in user.get("liked_cars") or []:
        try:
            liked_car = get_car_by_listing_id(int(listing_id))
        except (TypeError, ValueError):
            liked_car = None
        if liked_car:
            liked_cars.append(liked_car)

    if liked_cars:
        basis = ", ".join(
            dict.fromkeys(
                " ".join(
                    format_filter_value(value)
                    for value in [
                        car.get("make"),
                        car.get("model"),
                    ]
                    if value
                )
                for car in liked_cars
            )
        )
        return (
            find_similar_to_liked_cars(liked_cars, limit=limit),
            basis,
            "liked_cars",
            True,
        )

    preferred_make = str(user.get("preferred_make") or "").strip()
    preferred_model = str(user.get("preferred_model") or "").strip()
    if preferred_make or preferred_model:
        cars = search_cars(
            make=preferred_make or None,
            model=preferred_model or None,
            limit=limit,
        )
        basis = " ".join(
            format_filter_value(value)
            for value in [preferred_make, preferred_model]
            if value
        )
        cars = [
            {
                **car,
                "match_reason": f"Matched your saved preference for {basis}.",
            }
            for car in cars
        ]
        return cars, basis, "saved_preferences", True

    historical_text_fields = [
        ("last_seen_car", user.get("last_seen_car")),
        ("last_query", user.get("last_query")),
    ]
    for source_field, historical_text in historical_text_fields:
        text = str(historical_text or "").strip()
        if not text:
            continue
        if source_field == "last_query" and (
            not is_meaningful_memory_query(text)
            or is_memory_recall_request(text)
        ):
            continue

        extracted_request = extract_user_request(text)
        if not has_search_filters(extracted_request):
            continue

        cars = search_from_extracted_request(extracted_request, limit=limit)
        basis_values = [
            extracted_request.get("make"),
            extracted_request.get("model"),
            extracted_request.get("year"),
            extracted_request.get("keyword"),
        ]
        basis = " ".join(
            format_filter_value(value)
            for value in basis_values
            if value
        ) or "your previous search"
        cars = [
            {
                **car,
                "match_reason": (
                    f"Matched details remembered from your {source_field.replace('_', ' ')}."
                ),
            }
            for car in cars
        ]
        return cars, basis, source_field, True

    return [], "", "", False


def resolve_car_from_long_term_memory(username: str) -> Optional[Dict[str, Any]]:
    """Resolve the most recently discussed concrete listing for a user."""

    user = get_user(username) or {}
    last_seen_car = re.sub(
        r"\s+",
        " ",
        str(user.get("last_seen_car") or "").lower(),
    ).strip()
    if last_seen_car:
        for _, row in load_cars().iterrows():
            car = row_to_car(row)
            remembered_title = re.sub(
                r"\s+",
                " ",
                build_car_title(car).lower(),
            ).strip()
            if remembered_title == last_seen_car:
                return car

    interactions = get_recent_user_interactions(username, limit=20)
    for interaction in reversed(interactions):
        listing_id = interaction.get("listing_id")
        if listing_id is None or str(listing_id).strip() == "":
            continue
        try:
            car = get_car_by_listing_id(int(listing_id))
        except (TypeError, ValueError):
            car = None
        if car:
            return car

    for listing_id in reversed(user.get("liked_cars") or []):
        try:
            car = get_car_by_listing_id(int(listing_id))
        except (TypeError, ValueError):
            car = None
        if car:
            return car
    return None


def format_filter_value(value: Any) -> str:
    """Format an extracted filter for a user-facing no-results message."""

    text = str(value).strip()
    initialisms = {
        "bmw": "BMW",
        "gcc": "GCC",
    }
    return initialisms.get(text.lower(), text.title())


def build_no_matching_listings_reply(extracted_request: Dict[str, Any]) -> str:
    """Build a precise no-results reply from the strongest extracted filter."""

    make = extracted_request.get("make")
    model = extracted_request.get("model")
    year = extracted_request.get("year")
    keyword = extracted_request.get("keyword")

    if make:
        return (
            f"I couldn’t find any {format_filter_value(make)} listings "
            "in the provided inventory."
        )

    if model:
        return (
            f"I couldn’t find any {format_filter_value(model)} listings "
            "in the provided inventory."
        )

    if year:
        return f"I couldn’t find any {year} listings in the provided inventory."

    if keyword:
        return (
            f"I couldn’t find any listings matching “{keyword}” "
            "in the provided inventory."
        )

    return "I couldn’t find any matching listings in the provided inventory."


def maybe_save_interest_lead(
    username: str,
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
    desired_features = [
        *(extracted_request.get("required_keywords") or []),
        *(extracted_request.get("keywords") or []),
        *(extracted_request.get("soft_preferences") or []),
    ]
    body_type = extracted_request.get("body_type")
    keyword = extracted_request.get("keyword")

    if body_type and body_type not in desired_features:
        desired_features.append(body_type)
    if keyword and keyword not in desired_features:
        desired_features.append(keyword)

    desired_features = list(dict.fromkeys(
        str(feature).strip()
        for feature in desired_features
        if str(feature).strip()
    ))

    if intent not in {"car_search", "lead_capture", "general_car_advice"}:
        return None

    selected_listing_id = None
    selected_car_title = ""

    if cars:
        first_car = cars[0]
        selected_listing_id = first_car.get("listing_id")
        selected_car_title = build_car_title(first_car)

    lead = save_lead(
        username=username,
        budget=budget,
        needs=message_needs,
        preferred_make=preferred_make,
        preferred_model=preferred_model,
        selected_listing_id=selected_listing_id,
        selected_car_title=selected_car_title,
        notes="Lead captured from chat interaction.",
        source_intent=intent,
        desired_features=desired_features,
        is_follow_up=is_follow_up_about_selected_car(message_needs),
    )

    return lead


def compare_cars(
    cars: List[Dict[str, Any]],
    positions: List[int],
) -> str:
    """Build a grounded comparison for the exact result positions requested."""

    lines = ["Here is a comparison based on the provided inventory data:"]
    for car, position in zip(cars, positions):
        lines.extend(
            [
                "",
                f"Result {position + 1}: {build_car_title(car)}",
                f"- Listing ID: {car.get('listing_id')}",
                f"- Title: {car.get('title')}",
                (
                    "- Description preview: "
                    f"{car.get('description', '')[:350]}..."
                ),
            ]
        )

    return "\n".join(lines)


def respond_about_selected_car(
    username: str,
    session_id: str,
    user_message: str,
    selected_car: Dict[str, Any],
    extracted_request: Optional[Dict[str, Any]] = None,
) -> ChatResponse:
    """Return a grounded response and persist the selected car in session memory."""

    save_selected_car(session_id, selected_car)
    update_user_memory(
        username=username,
        last_seen_car=build_car_title(selected_car),
    )

    fallback_reply = answer_car_question(selected_car, user_message)
    reply = generate_car_detail_reply(
        user_message=user_message,
        car=selected_car,
        derived_evidence=build_selected_car_evidence(
            selected_car,
            user_message,
        ),
        fallback_reply=fallback_reply,
    )

    add_message(session_id, "assistant", reply)
    log_intent(username=username, intent="car_details", results_count=1)

    return ChatResponse(
        reply=reply,
        intent="car_details",
        cars=[selected_car],
        extracted_request=extracted_request or {"intent": "car_details"},
        memory_update=get_memory_summary(username),
    )


def persist_confirmed_booking_memory(
    username: str,
    car: Dict[str, Any],
    booking_date: str,
    booking_time: str,
    budget: str = "",
    needs: str = "",
    preferred_make: str = "",
    preferred_model: str = "",
) -> None:
    """Apply the same long-term memory updates for UI and chat bookings."""

    selected_car_title = build_car_title(car)
    update_user_memory(
        username=username,
        last_budget=budget,
        preferred_make=preferred_make or str(car.get("make") or ""),
        preferred_model=preferred_model or str(car.get("model") or ""),
        last_seen_car=selected_car_title,
        last_query=f"Booked viewing for {selected_car_title}",
    )
    add_liked_car(
        username,
        int(car.get("listing_id")),
        make=str(car.get("make") or ""),
        model=str(car.get("model") or ""),
        car_title=selected_car_title,
        refresh_summary=False,
    )
    record_user_interaction(
        username=username,
        event_type="booking",
        query=f"Booked viewing for {selected_car_title}",
        make=str(car.get("make") or ""),
        model=str(car.get("model") or ""),
        budget=budget,
        listing_id=int(car.get("listing_id")),
        car_title=selected_car_title,
        lead_quality="Hot",
        notes=f"Viewing booked for {booking_date} at {booking_time}. {needs}".strip(),
    )
    update_memory_summary(username)


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


@app.get("/user/{username}")
def get_user_profile(username: str):
    """
    Return long-term user memory summary.
    If user does not exist, create a basic profile.
    """

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

    username = request.username
    session_id = request.session_id
    user_message = request.message

    get_or_create_user(username=username)
    add_message(session_id, "user", user_message)
    session_preferences = get_preferences(session_id)
    booking_date_hint, booking_time_hint = extract_booking_slot(user_message)
    booking_followup = bool(
        session_preferences.get("pending_booking")
        and (booking_date_hint or booking_time_hint)
    )

    # Layer 1: rule-based guardrails
    guardrail = apply_guardrails(user_message)
    memory_recall_requested = is_memory_recall_request(user_message)
    is_comparison_request = bool(
        re.search(
            r"\b(?:compare|versus|vs\.?|difference between)\b",
            user_message,
            re.IGNORECASE,
        )
    )

    referenced_car = get_car_from_message(user_message)
    if referenced_car is None and not is_comparison_request:
        referenced_car = resolve_car_reference(session_id, user_message)

    if (
        referenced_car is None
        and is_follow_up_about_selected_car(user_message)
    ):
        referenced_car = get_selected_car(session_id)

    memory_detail_intents = {
        "car_details",
        "car_search",
        "lead_capture",
        "general_car_advice",
        "blocked_non_automotive",
    }
    if (
        referenced_car
        and not is_comparison_request
        and guardrail["intent"] in memory_detail_intents
    ):
        return respond_about_selected_car(
            username=username,
            session_id=session_id,
            user_message=user_message,
            selected_car=referenced_car,
        )

    memory_scope_override = (
        memory_recall_requested
        and guardrail["intent"] == "blocked_non_automotive"
    )
    if (
        not guardrail["allowed"]
        and not memory_scope_override
        and not booking_followup
    ):
        reply = generate_refusal_reply(
            user_message=user_message,
            blocked_intent=guardrail["intent"],
        )
        add_message(session_id, "assistant", reply)
        log_intent(username=username, intent=guardrail["intent"], results_count=0)

        return ChatResponse(
            reply=reply,
            intent=guardrail["intent"],
            cars=[],
            extracted_request={},
            memory_update=get_memory_summary(username),
        )

    # LLM extraction
    extracted_request = extract_user_request(user_message)
    intent = extracted_request.get("intent", guardrail["intent"])

    if (
        booking_followup
        or (
            memory_recall_requested
            and guardrail["intent"] == "booking"
        )
    ):
        intent = "booking"
        extracted_request["intent"] = intent
    elif memory_recall_requested:
        intent = "returning_user_memory"
        extracted_request["intent"] = intent
    elif is_comparison_request:
        intent = "compare_listings"
        extracted_request["intent"] = intent
    elif guardrail["intent"] == "greeting":
        intent = "greeting"
        extracted_request["intent"] = intent
    elif guardrail["intent"] == "car_search" and referenced_car is None:
        intent = "car_search"
        extracted_request["intent"] = intent
    elif guardrail["intent"] in {"booking", "compare_listings"}:
        intent = guardrail["intent"]
        extracted_request["intent"] = intent

    # Layer 2: LLM-based guardrails
    if is_llm_blocked_intent(intent):
        reply = generate_refusal_reply(
            user_message=user_message,
            blocked_intent=intent,
        )
        add_message(session_id, "assistant", reply)
        log_intent(username=username, intent=intent, results_count=0)

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=[],
            extracted_request=extracted_request,
            memory_update=get_memory_summary(username),
        )

    # Greeting / returning user
    if intent == "greeting":
        memory_summary = get_memory_summary(username)
        fallback_reply = memory_summary.get(
            "message",
            f"Hi {username}! I can help you search car listings, compare options, or book a viewing.",
        )
        reply = generate_grounded_reply(
            user_message=user_message,
            response_task=(
                "Respond warmly to the greeting. Use saved user memory only "
                "when it is present, and offer concise car-shopping help."
            ),
            grounded_context={
                "username": username,
                "memory": memory_summary,
            },
            fallback_reply=fallback_reply,
        )

        add_message(session_id, "assistant", reply)
        log_intent(username=username, intent=intent, results_count=0)

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=[],
            extracted_request=extracted_request,
            memory_update=memory_summary,
        )

    # Long-term memory recall / similar-to-previous search
    if intent in {"returning_user_memory", "memory_recall", "similar_to_previous"}:
        user_memory = get_user(username) or {}
        saved_memory_summary = str(
            user_memory.get("memory_summary") or ""
        ).strip()
        cars, memory_basis, memory_source, has_useful_memory = search_from_user_memory(
            user_memory,
            limit=5,
        )

        save_last_results(session_id, cars)
        if cars:
            save_selected_car(session_id, cars[0])

        if cars:
            remembered_context = (
                f"{format_memory_summary_for_user(saved_memory_summary)} "
                if saved_memory_summary
                else ""
            )
            fallback_reply = (
                f"{remembered_context}Based on your previous interest in "
                f"{memory_basis}, here are similar listings I found."
            )
            response_task = (
                "Explain naturally that these listings were selected from the "
                "user's saved car interests. Mention the remembered basis and "
                "briefly introduce only the provided matching listings."
            )
        elif has_useful_memory:
            fallback_reply = (
                f"I found your saved interest in {memory_basis}, but I couldn’t "
                "find similar listings in the provided inventory."
            )
            response_task = (
                "Explain naturally that useful saved preferences were found, "
                "but no provided inventory listings matched them."
            )
        else:
            fallback_reply = (
                "I don’t have enough saved car preferences for you yet. Tell me "
                "what kind of car you’re looking for, and I’ll remember it for "
                "next time."
            )
            response_task = (
                "Explain warmly that this user has no useful saved car "
                "preferences yet and ask what kind of car they want."
            )

        compact_cars = [
            {
                "listing_id": car.get("listing_id"),
                "year": car.get("year"),
                "make": car.get("make"),
                "model": car.get("model"),
                "trim": car.get("trim"),
                "title": car.get("title"),
                "match_reason": car.get("match_reason"),
            }
            for car in cars
        ]
        reply = generate_grounded_reply(
            user_message=user_message,
            response_task=response_task,
            grounded_context={
                "memory_source": memory_source,
                "memory_basis": memory_basis,
                "memory_summary": saved_memory_summary,
                "has_useful_memory": has_useful_memory,
                "matching_cars": compact_cars,
                "verified_finding": fallback_reply,
            },
            fallback_reply=fallback_reply,
        )

        add_message(session_id, "assistant", reply)
        log_intent(
            username=username,
            intent="returning_user_memory",
            results_count=len(cars),
        )
        record_user_interaction(
            username=username,
            event_type="memory_recall",
            query=user_message,
            notes=(
                f"Used structured memory source {memory_source} "
                f"with basis {memory_basis}."
            ),
        )
        update_memory_summary(username)

        return ChatResponse(
            reply=reply,
            intent="returning_user_memory",
            cars=cars,
            extracted_request=extracted_request,
            memory_update=get_memory_summary(username),
        )

    # Search / lead capture
    if intent in {"car_search", "lead_capture", "general_car_advice"}:
        query_plan = build_query_plan(user_message, extracted_request)
        cars, query_execution = execute_query_plan(
            query_plan,
            extracted_request,
            limit=5,
        )
        extracted_request["query_plan"] = query_plan
        extracted_request["query_execution"] = query_execution
        requested_color = get_requested_color(user_message)
        partial_color_match = bool(
            cars
            and requested_color
            and (extracted_request.get("make") or extracted_request.get("model"))
            and not any(
                text_mentions_feature(car, requested_color)
                for car in cars
            )
        )

        if partial_color_match:
            search_label = build_search_label(extracted_request) or "matching car"
            cars = [
                {
                    **car,
                    "match_reason": (
                        f"Matched the {search_label} part of your request, but "
                        "color was not confirmed in the listing text."
                    ),
                }
                for car in cars
            ]

        save_last_results(session_id, cars)

        if cars:
            save_selected_car(session_id, cars[0])

        session_preferences = {
            "make": extracted_request.get("make"),
            "model": extracted_request.get("model"),
            "year": extracted_request.get("year"),
            "keyword": extracted_request.get("keyword"),
            "required_keywords": extracted_request.get("required_keywords"),
            "soft_preferences": extracted_request.get("soft_preferences"),
            "body_type": extracted_request.get("body_type"),
            "budget": extracted_request.get("budget"),
            "needs": extracted_request.get("needs"),
            "query_plan": query_plan,
            "pending_booking": False,
            "pending_booking_date": None,
            "pending_booking_time": None,
        }

        update_preferences(session_id, session_preferences)

        first_car_title = build_car_title(cars[0]) if cars else ""

        update_user_memory(
            username=username,
            last_budget=extracted_request.get("budget"),
            preferred_make=extracted_request.get("make"),
            preferred_model=extracted_request.get("model"),
            preferred_body_type=extracted_request.get("body_type"),
            last_seen_car=first_car_title,
            last_query=user_message,
        )

        lead = maybe_save_interest_lead(
            username=username,
            extracted_request=extracted_request,
            cars=cars,
        )
        search_features = [
            *(extracted_request.get("required_keywords") or []),
            *(extracted_request.get("soft_preferences") or []),
        ]
        search_notes = []
        if search_features:
            search_notes.append(
                "Requested preferences: " + ", ".join(search_features)
            )
        if query_plan.get("sort_by"):
            search_notes.append(
                f"Ranked by {query_plan['sort_by']} "
                f"{query_plan.get('sort_order') or ''}".strip()
            )
        has_preferences = any(
            [
                extracted_request.get("make"),
                extracted_request.get("model"),
                extracted_request.get("body_type"),
                extracted_request.get("budget"),
                search_features,
                query_plan.get("sort_by"),
            ]
        )
        first_car = cars[0] if cars else {}
        record_user_interaction(
            username=username,
            event_type=(
                "preference_search" if has_preferences else "inventory_search"
            ),
            query=user_message,
            make=extracted_request.get("make") or "",
            model=extracted_request.get("model") or "",
            budget=extracted_request.get("budget") or "",
            listing_id=first_car.get("listing_id"),
            car_title=build_car_title(first_car) if first_car else "",
            lead_quality=(lead or {}).get("lead_status", ""),
            notes="; ".join(search_notes),
        )
        update_memory_summary(username)

        if (
            query_plan.get("sort_by")
            and query_execution.get("status") == "attribute_unavailable"
        ):
            fallback_reply = build_unavailable_attribute_reply(
                query_plan["sort_by"]
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain that the requested inventory ranking could not be "
                    "performed because the required attribute was unavailable. "
                    "Do not return unrelated cars."
                ),
                grounded_context={
                    "query_plan": query_plan,
                    "execution": query_execution,
                    "verified_finding": fallback_reply,
                },
                fallback_reply=fallback_reply,
            )
        elif query_plan.get("sort_by") and cars:
            fallback_reply = build_ranked_inventory_fallback(
                cars,
                query_execution,
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Introduce and summarize the ranked inventory results. "
                    "Preserve their order and extracted attribute values. "
                    "Mention partial attribute coverage when stated."
                ),
                grounded_context={
                    "query_plan": query_plan,
                    "execution": query_execution,
                    "ranked_cars": cars,
                    "verified_finding": fallback_reply,
                },
                fallback_reply=fallback_reply,
            )
        elif query_plan.get("sort_by"):
            fallback_reply = build_no_matching_listings_reply(
                extracted_request
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain that no cars satisfied the filters required before "
                    "the requested ranking. Do not return unrelated cars."
                ),
                grounded_context={
                    "query_plan": query_plan,
                    "execution": query_execution,
                    "verified_finding": fallback_reply,
                },
                fallback_reply=fallback_reply,
            )
        elif partial_color_match:
            search_label = build_search_label(extracted_request) or "requested"
            fallback_reply = (
                f"I found {search_label} listings, but none of the provided "
                f"listing text explicitly mentions {requested_color}."
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain this partial inventory match naturally. Make clear "
                    "that make/model matched but the requested color was not "
                    "confirmed in any provided listing text."
                ),
                grounded_context={
                    "request": extracted_request,
                    "requested_color": requested_color,
                    "matched_cars": cars,
                    "verified_finding": fallback_reply,
                },
                fallback_reply=fallback_reply,
            )
        elif cars:
            reply = generate_inventory_reply(
                user_message=user_message,
                cars=cars,
                extracted_request=extracted_request,
            )
        else:
            fallback_reply = build_no_matching_listings_reply(extracted_request)
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain naturally that no provided inventory listings "
                    "matched the extracted request. Do not suggest that other "
                    "cars matched."
                ),
                grounded_context={
                    "request": extracted_request,
                    "matching_count": 0,
                    "verified_finding": fallback_reply,
                },
                fallback_reply=fallback_reply,
            )

        add_message(session_id, "assistant", reply)
        log_intent(username=username, intent=intent, results_count=len(cars))

        memory_summary = get_memory_summary(username)
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
        selected_car = get_car_from_message(user_message)

        selected_position = extracted_request.get("selected_position")

        if selected_position is not None:
            selected_car = get_car_by_position(session_id, int(selected_position))

        if selected_car is None:
            selected_car = resolve_car_reference(session_id, user_message)

        if selected_car is None:
            selected_car = get_selected_car(session_id)

        if selected_car is None:
            fallback_reply = (
                "I do not have a selected car yet. Please search for cars "
                "first, then ask about a specific result."
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain that no car is selected in this chat session and "
                    "ask the user to search for or select a listing first."
                ),
                grounded_context={
                    "selected_car": None,
                    "last_results_count": len(get_last_results(session_id)),
                },
                fallback_reply=fallback_reply,
            )

            add_message(session_id, "assistant", reply)
            log_intent(username=username, intent=intent, results_count=0)

            return ChatResponse(
                reply=reply,
                intent=intent,
                cars=[],
                extracted_request=extracted_request,
                memory_update=get_memory_summary(username),
            )

        return respond_about_selected_car(
            username=username,
            session_id=session_id,
            user_message=user_message,
            selected_car=selected_car,
            extracted_request=extracted_request,
        )

    # Compare two current listings
    if intent == "compare_listings":
        last_results = get_last_results(session_id)
        requested_positions = extract_car_positions(user_message)
        extracted_request["selected_positions"] = requested_positions

        # A general "compare these cars" request keeps the existing default,
        # while explicit ordinals are always honored exactly.
        positions_to_compare = (
            requested_positions
            if requested_positions
            else [0, 1]
        )
        invalid_positions = [
            position
            for position in positions_to_compare
            if position < 0 or position >= len(last_results)
        ]
        comparison_cars = (
            [
                last_results[position]
                for position in positions_to_compare
            ]
            if not invalid_positions
            else []
        )
        selected_car = comparison_cars[0] if comparison_cars else None
        comparison_lead = save_lead(
            username=username,
            needs=user_message,
            selected_listing_id=(
                selected_car.get("listing_id") if selected_car else None
            ),
            selected_car_title=(
                build_car_title(selected_car) if selected_car else ""
            ),
            notes="Lead captured from a comparison request.",
            source_intent="compare_listings",
        )

        if invalid_positions:
            unavailable = ", ".join(
                f"result {position + 1}"
                for position in invalid_positions
            )
            fallback_reply = (
                f"I can’t compare {unavailable} because the current search "
                f"contains {len(last_results)} results."
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain which requested result positions are unavailable "
                    "and state how many current results exist. Do not substitute "
                    "different listings."
                ),
                grounded_context={
                    "requested_positions": [
                        position + 1 for position in positions_to_compare
                    ],
                    "unavailable_positions": [
                        position + 1 for position in invalid_positions
                    ],
                    "available_results_count": len(last_results),
                },
                fallback_reply=fallback_reply,
            )
        elif len(positions_to_compare) < 2:
            fallback_reply = (
                "Please specify at least two result positions to compare."
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Ask the user to specify at least two result positions. "
                    "Do not choose another listing for them."
                ),
                grounded_context={
                    "requested_positions": [
                        position + 1 for position in positions_to_compare
                    ],
                    "available_results_count": len(last_results),
                },
                fallback_reply=fallback_reply,
            )
        elif len(last_results) < 2:
            fallback_reply = (
                "Please search for cars first so I can compare two listings "
                "from the current results."
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain that at least two current inventory results are "
                    "needed before a comparison can be made."
                ),
                grounded_context={
                    "available_results_count": len(last_results),
                },
                fallback_reply=fallback_reply,
            )
        else:
            positioned_cars = [
                {
                    "requested_result_position": position + 1,
                    "car": car,
                }
                for position, car in zip(
                    positions_to_compare,
                    comparison_cars,
                )
            ]
            fallback_reply = compare_cars(
                comparison_cars,
                positions_to_compare,
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Compare exactly the requested listings in their requested "
                    "order, using only their provided fields. Do not substitute "
                    "another result or declare a winner without evidence."
                ),
                grounded_context={
                    "requested_cars": positioned_cars,
                    "verified_comparison": fallback_reply,
                },
                fallback_reply=fallback_reply,
            )

        add_message(session_id, "assistant", reply)
        log_intent(
            username=username,
            intent=intent,
            results_count=len(comparison_cars),
        )
        if len(comparison_cars) >= 2 and not invalid_positions:
            compared_titles = " and ".join(
                build_car_title(car)
                for car in comparison_cars
            )
            record_user_interaction(
                username=username,
                event_type="comparison",
                query=user_message,
                make=str(comparison_cars[0].get("make") or ""),
                model=str(comparison_cars[0].get("model") or ""),
                listing_id=comparison_cars[0].get("listing_id"),
                car_title=build_car_title(comparison_cars[0]),
                lead_quality=(comparison_lead or {}).get("lead_status", ""),
                notes=compared_titles,
            )
            update_memory_summary(username)

        memory_summary = get_memory_summary(username)
        memory_summary["lead_saved"] = comparison_lead

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=comparison_cars,
            extracted_request=extracted_request,
            memory_update=memory_summary,
        )

    # Booking intent inside chat collects missing slot details or confirms
    # through the same booking service used by /book-viewing.
    if intent == "booking":
        selected_car = get_car_from_message(user_message)

        if selected_car:
            save_selected_car(session_id, selected_car)
        else:
            selected_car = get_selected_car(session_id)
        if selected_car is None and memory_recall_requested:
            selected_car = resolve_car_from_long_term_memory(username)
            if selected_car:
                save_selected_car(session_id, selected_car)

        current_preferences = get_preferences(session_id)
        booking_date = (
            booking_date_hint
            or current_preferences.get("pending_booking_date")
        )
        booking_time = (
            booking_time_hint
            or current_preferences.get("pending_booking_time")
        )
        update_preferences(
            session_id,
            {
                "pending_booking": bool(selected_car),
                "pending_booking_date": booking_date,
                "pending_booking_time": booking_time,
            },
        )

        booking_result = None
        booking_lead = None
        if selected_car and booking_date and booking_time:
            booking_result = book_viewing(
                username=username,
                selected_listing_id=int(selected_car.get("listing_id")),
                selected_car_title=build_car_title(selected_car),
                booking_date=booking_date,
                booking_time=booking_time,
                budget=str(current_preferences.get("budget") or ""),
                needs=str(current_preferences.get("needs") or user_message),
                preferred_make=str(
                    current_preferences.get("make")
                    or selected_car.get("make")
                    or ""
                ),
                preferred_model=str(
                    current_preferences.get("model")
                    or selected_car.get("model")
                    or ""
                ),
                notes="Booking confirmed through chat.",
            )
            booking_lead = booking_result.get("lead")

        if booking_result and booking_result["success"]:
            update_preferences(
                session_id,
                {
                    "pending_booking": False,
                    "pending_booking_date": None,
                    "pending_booking_time": None,
                },
            )
            persist_confirmed_booking_memory(
                username=username,
                car=selected_car,
                booking_date=booking_date,
                booking_time=booking_time,
                budget=str(current_preferences.get("budget") or ""),
                needs=str(current_preferences.get("needs") or user_message),
                preferred_make=str(current_preferences.get("make") or ""),
                preferred_model=str(current_preferences.get("model") or ""),
            )
            fallback_reply = (
                f"Your viewing for {build_car_title(selected_car)} is confirmed "
                f"for {booking_date} at {booking_time}."
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Confirm the completed viewing booking naturally. Preserve "
                    "the exact car, date, and time."
                ),
                grounded_context={
                    "selected_car": selected_car,
                    "booking": booking_result.get("booking"),
                    "verified_confirmation": fallback_reply,
                },
                fallback_reply=fallback_reply,
            )
        elif booking_result:
            if not current_preferences.get("pending_booking"):
                booking_lead = save_lead(
                    username=username,
                    needs=user_message,
                    selected_listing_id=selected_car.get("listing_id"),
                    selected_car_title=build_car_title(selected_car),
                    notes="Lead captured from an invalid booking request.",
                    source_intent="booking",
                )
            fallback_reply = booking_result["message"]
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain why this requested viewing slot is invalid and ask "
                    "for a valid Monday-to-Saturday time from 8 AM to 8 PM."
                ),
                grounded_context={
                    "selected_car": selected_car,
                    "requested_date": booking_date,
                    "requested_time": booking_time,
                    "validation_message": fallback_reply,
                },
                fallback_reply=fallback_reply,
            )
        elif selected_car:
            if not current_preferences.get("pending_booking"):
                booking_lead = save_lead(
                    username=username,
                    needs=user_message,
                    selected_listing_id=selected_car.get("listing_id"),
                    selected_car_title=build_car_title(selected_car),
                    notes="Lead captured from a booking request.",
                    source_intent="booking",
                )
            missing_details = []
            if not booking_date:
                missing_details.append("day or date")
            if not booking_time:
                missing_details.append("time")
            missing_text = " and ".join(missing_details)
            fallback_reply = (
                f"I can help book a viewing for {build_car_title(selected_car)}. "
                "Viewing slots are available Monday to Saturday, from 8 AM to 8 PM. "
                f"Please provide the {missing_text}."
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Acknowledge the selected car and ask only for the missing "
                    "booking details."
                ),
                grounded_context={
                    "selected_car": selected_car,
                    "known_booking_date": booking_date,
                    "known_booking_time": booking_time,
                    "missing_details": missing_details,
                    "viewing_schedule": {
                        "days": "Monday to Saturday",
                        "hours": "8 AM to 8 PM",
                    },
                },
                fallback_reply=fallback_reply,
            )
        else:
            update_preferences(
                session_id,
                {
                    "pending_booking": False,
                    "pending_booking_date": None,
                    "pending_booking_time": None,
                },
            )
            booking_lead = save_lead(
                username=username,
                needs=user_message,
                notes="Booking requested without a selected car.",
                source_intent="booking",
            )
            fallback_reply = (
                "I can help book a viewing, but please select a car first. "
                "Search for cars, choose a result, then provide a date and time. "
                "Viewing slots are Monday to Saturday, 8 AM to 8 PM."
            )
            reply = generate_grounded_reply(
                user_message=user_message,
                response_task=(
                    "Explain that booking requires a selected inventory car, "
                    "then ask the user to search and choose one."
                ),
                grounded_context={
                    "selected_car": None,
                    "viewing_schedule": {
                        "days": "Monday to Saturday",
                        "hours": "8 AM to 8 PM",
                    },
                },
                fallback_reply=fallback_reply,
            )

        add_message(session_id, "assistant", reply)
        log_intent(username=username, intent=intent, results_count=1 if selected_car else 0)
        if not (booking_result and booking_result["success"]):
            if not current_preferences.get("pending_booking"):
                record_user_interaction(
                    username=username,
                    event_type="booking_intent",
                    query=user_message,
                    make=str((selected_car or {}).get("make") or ""),
                    model=str((selected_car or {}).get("model") or ""),
                    listing_id=(selected_car or {}).get("listing_id"),
                    car_title=(
                        build_car_title(selected_car) if selected_car else ""
                    ),
                    lead_quality=(booking_lead or {}).get("lead_status", ""),
                    notes="User asked to arrange a viewing.",
                )

        memory_summary = get_memory_summary(username)
        memory_summary["lead_saved"] = booking_lead

        return ChatResponse(
            reply=reply,
            intent=intent,
            cars=[selected_car] if selected_car else [],
            extracted_request=extracted_request,
            memory_update=memory_summary,
        )

    # Fallback for any allowed but unsupported intent
    fallback_reply = (
        "I can help with car searches, listing details, comparisons, and viewing bookings. "
        "Try asking me to show cars by make, model, year, or feature."
    )
    reply = generate_grounded_reply(
        user_message=user_message,
        response_task=(
            "Explain the assistant's supported car-shopping capabilities and "
            "invite a concrete inventory, listing, comparison, or booking question."
        ),
        grounded_context={
            "supported_scope": [
                "inventory search",
                "listing details",
                "listing comparison",
                "viewing booking",
            ],
        },
        fallback_reply=fallback_reply,
    )

    add_message(session_id, "assistant", reply)
    log_intent(username=username, intent=intent, results_count=0)

    return ChatResponse(
        reply=reply,
        intent=intent,
        cars=[],
        extracted_request=extracted_request,
        memory_update=get_memory_summary(username),
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
        username=request.username,
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
