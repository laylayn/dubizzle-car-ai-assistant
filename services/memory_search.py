import re
from typing import Any, Dict, List, Optional

from services.car_utils import (
    build_car_title,
    format_filter_value,
    get_car_by_listing_id,
    row_to_car,
)
from services.data_loader import load_cars
from services.llm_agent import extract_user_request, is_memory_recall_request
from services.memory import (
    get_recent_user_interactions,
    get_user,
    is_meaningful_memory_query,
)
from services.retrieval import search_cars
from services.search_flow import search_from_extracted_request


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

    liked_token_sets = [get_listing_similarity_tokens(car) for car in liked_cars]
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
                for value in [car.get("make"), car.get("model")]
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
                    for value in [car.get("make"), car.get("model")]
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

    for source_field, historical_text in [
        ("last_seen_car", user.get("last_seen_car")),
        ("last_query", user.get("last_query")),
    ]:
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
                    f"Matched details remembered from your "
                    f"{source_field.replace('_', ' ')}."
                ),
            }
            for car in cars
        ]
        return cars, basis, source_field, True
    return [], "", "", False


def resolve_car_from_long_term_memory(
    username: str,
) -> Optional[Dict[str, Any]]:
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
