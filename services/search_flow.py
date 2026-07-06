from typing import Any, Dict, List, Optional

from services.booking import save_lead
from services.car_utils import (
    build_car_title,
    format_filter_value,
)
from services.listing_attributes import (
    extract_mileage_from_car,
    extract_price_from_car,
    format_price_amount,
    text_mentions_feature,
)
from services.retrieval import search_cars
from services.session_memory import is_follow_up_about_selected_car


def search_from_extracted_request(
    extracted_request: Dict[str, Any],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search objective constraints together, then relax transparently."""

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
    preference_terms = list(
        dict.fromkeys(
            str(value).lower().strip()
            for value in extracted_request.get("preference_terms") or []
            if str(value).strip()
        )
    )
    has_structured_filter = any([make, model, year])
    has_keyword_filter = bool(keywords)
    has_preference_filter = bool(preference_terms)
    budget_text = extracted_request.get("budget")
    candidate_limit = (
        max(limit * 20, 100)
        if budget_text or has_preference_filter
        else limit
    )
    if (
        not has_structured_filter
        and not has_keyword_filter
        and not has_preference_filter
    ):
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
    if preference_terms and cars:
        ranked_preference_cars = []
        for car in cars:
            matched_terms = [
                term
                for term in preference_terms
                if text_mentions_feature(car, term)
            ]
            if matched_terms:
                ranked_preference_cars.append(
                    {
                        **car,
                        "_preference_score": len(matched_terms),
                        "match_reason": (
                            f"{car.get('match_reason', '')}; listing text "
                            "supports preference concepts: "
                            f"{', '.join(matched_terms)}"
                        ),
                    }
                )
        if ranked_preference_cars:
            cars = sorted(
                ranked_preference_cars,
                key=lambda car: car["_preference_score"],
                reverse=True,
            )
            cars = [
                {key: value for key, value in car.items() if key != "_preference_score"}
                for car in cars
            ]
        elif not has_structured_filter and not has_keyword_filter:
            cars = []

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
                car_price = int(float(price_result["amount"].replace(",", "")))
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
    search_fn=None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute normalized filters and attribute ranking against inventory."""

    inventory_search = search_fn or search_cars
    sort_by = query_plan.get("sort_by")
    sort_order = query_plan.get("sort_order")
    filters = query_plan.get("filters") or {}
    has_concrete_filter = bool(
        filters.get("make")
        or filters.get("model")
        or filters.get("year")
        or filters.get("features")
        or filters.get("body_type")
        or filters.get("budget")
        or query_plan.get("preference_terms")
        or sort_by
    )
    if extracted_request.get("soft_preferences") and not has_concrete_filter:
        return [], {
            "status": "needs_preference_clarification",
            "sort_by": None,
            "sort_order": None,
            "candidate_count": 0,
            "ranked_count": 0,
            "missing_attribute_count": 0,
        }
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
            if str(car.get("year") or "").isdigit() and int(car.get("year")) > 0
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
    candidates = inventory_search(
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
    status = (
        "ranked"
        if ranked_cars
        else "attribute_unavailable" if candidates else "no_matches"
    )
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
        ("price", "ascending"): "I found these listings with the lowest prices available.",
        ("price", "descending"): "I found these listings with the highest prices available.",
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


def build_no_matching_listings_reply(
    extracted_request: Dict[str, Any],
) -> str:
    """Build a precise no-results reply from the strongest filter."""

    for field_name, noun in [
        ("make", "listings"),
        ("model", "listings"),
        ("year", "listings"),
    ]:
        value = extracted_request.get(field_name)
        if value:
            display = (
                str(value)
                if field_name == "year"
                else format_filter_value(value)
            )
            return (
                f"I couldn’t find any {display} {noun} "
                "in the provided inventory."
            )
    keyword = extracted_request.get("keyword")
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
    """Save Cold/Warm interest leads only after meaningful chat activity."""

    intent = extracted_request.get("intent", "")
    message_needs = extracted_request.get("needs") or ""
    budget = extracted_request.get("budget") or ""
    preferred_make = extracted_request.get("make") or ""
    preferred_model = extracted_request.get("model") or ""
    desired_features = [
        *(extracted_request.get("required_keywords") or []),
        *(extracted_request.get("keywords") or []),
        *(extracted_request.get("soft_preferences") or []),
        *(extracted_request.get("preference_terms") or []),
    ]
    body_type = extracted_request.get("body_type")
    keyword = extracted_request.get("keyword")
    if body_type and body_type not in desired_features:
        desired_features.append(body_type)
    if keyword and keyword not in desired_features:
        desired_features.append(keyword)
    desired_features = list(
        dict.fromkeys(
            str(feature).strip()
            for feature in desired_features
            if str(feature).strip()
        )
    )
    if intent not in {"car_search", "lead_capture", "general_car_advice"}:
        return None
    first_car = cars[0] if cars else {}
    return save_lead(
        username=username,
        budget=budget,
        needs=message_needs,
        preferred_make=preferred_make,
        preferred_model=preferred_model,
        selected_listing_id=first_car.get("listing_id"),
        selected_car_title=build_car_title(first_car) if first_car else "",
        notes="Lead captured from chat interaction.",
        source_intent=intent,
        desired_features=desired_features,
        is_follow_up=is_follow_up_about_selected_car(message_needs),
    )
