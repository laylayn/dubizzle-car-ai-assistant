import re
from typing import Any, Dict, Optional

from services.listing_attributes import COLOR_TERMS


SUPPORTED_SORT_ATTRIBUTES = {"mileage", "price", "year"}
SUPPORTED_SORT_ORDERS = {"ascending", "descending"}

ATTRIBUTE_TERMS = {
    "mileage": {
        "mileage",
        "km",
        "kilometer",
        "kilometers",
        "kilometre",
        "kilometres",
        "driven",
        "odometer",
    },
    "price": {
        "price",
        "prices",
        "cost",
        "amount",
        "aed",
        "dirham",
        "dirhams",
    },
    "year": {
        "year",
        "newest",
        "latest",
        "newer",
        "oldest",
        "older",
    },
}

ASCENDING_TERMS = {
    "low",
    "lowest",
    "least",
    "cheapest",
    "cheap",
    "affordable",
}
DESCENDING_TERMS = {
    "high",
    "highest",
    "most",
    "expensive",
    "priciest",
    "newest",
    "latest",
    "newer",
}


def message_terms(message: str) -> set[str]:
    """Return normalized word tokens used by the deterministic planner."""

    return set(re.findall(r"[a-z0-9]+", message.lower()))


def infer_query_operations(message: str) -> Dict[str, Optional[str]]:
    """Infer attribute/sort operations independently from sentence wording."""

    terms = message_terms(message)
    mentioned_attributes = [
        attribute
        for attribute, vocabulary in ATTRIBUTE_TERMS.items()
        if terms.intersection(vocabulary)
    ]
    has_ascending_signal = bool(terms.intersection(ASCENDING_TERMS))
    has_descending_signal = bool(terms.intersection(DESCENDING_TERMS))

    sort_by = None
    sort_order = None
    for attribute in mentioned_attributes:
        if has_ascending_signal:
            sort_by = attribute
            sort_order = "ascending"
            break
        if has_descending_signal:
            sort_by = attribute
            sort_order = "descending"
            break

    # These common ranking adjectives imply their attribute even when the
    # user does not repeat words such as "price" or "year".
    if sort_by is None and terms.intersection({"cheapest", "cheap", "affordable"}):
        sort_by, sort_order = "price", "ascending"
    elif sort_by is None and terms.intersection({"expensive", "priciest"}):
        sort_by, sort_order = "price", "descending"
    elif sort_by is None and terms.intersection({"newest", "latest", "newer"}):
        sort_by, sort_order = "year", "descending"
    elif sort_by is None and terms.intersection({"oldest", "older"}):
        sort_by, sort_order = "year", "ascending"

    attribute_request = (
        mentioned_attributes[0]
        if mentioned_attributes and sort_by is None
        else None
    )
    return {
        "attribute_request": attribute_request,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }


def build_query_plan(
    message: str,
    extracted_request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize LLM/rule extraction into one executable inventory plan."""

    extracted = extracted_request or {}
    inferred = infer_query_operations(message)

    parsed_sort_by = str(extracted.get("sort_by") or "").lower().strip()
    parsed_sort_order = str(extracted.get("sort_order") or "").lower().strip()
    sort_by = (
        inferred["sort_by"]
        or (
            parsed_sort_by
            if parsed_sort_by in SUPPORTED_SORT_ATTRIBUTES
            else None
        )
    )
    sort_order = (
        inferred["sort_order"]
        or (
            parsed_sort_order
            if parsed_sort_order in SUPPORTED_SORT_ORDERS
            else None
        )
    )
    if sort_by and not sort_order:
        sort_order = "ascending"

    features = []
    raw_features = [
        *(extracted.get("required_keywords") or []),
        *(extracted.get("keywords") or []),
    ]
    if extracted.get("keyword"):
        raw_features.append(extracted["keyword"])

    attribute_vocabulary = ATTRIBUTE_TERMS.get(sort_by, set())
    for feature in raw_features:
        normalized = str(feature or "").lower().strip()
        feature_terms = message_terms(normalized)
        is_sort_expression = bool(
            sort_by and feature_terms.intersection(attribute_vocabulary)
        )
        if normalized and not is_sort_expression and normalized not in features:
            features.append(normalized)

    body_type = str(extracted.get("body_type") or "").lower().strip() or None
    if body_type and body_type not in features:
        features.append(body_type)

    color = next(
        (
            feature
            for feature in features
            if feature in COLOR_TERMS
        ),
        None,
    )

    return {
        "intent": "inventory_search",
        "filters": {
            "make": extracted.get("make"),
            "model": extracted.get("model"),
            "year": extracted.get("year"),
            "features": features,
            "color": color,
            "body_type": body_type,
            "budget": extracted.get("budget"),
        },
        "attribute_request": (
            inferred["attribute_request"]
            or extracted.get("attribute_request")
        ),
        "sort_by": sort_by,
        "sort_order": sort_order,
        "comparison": extracted.get("selected_positions"),
        "preference_terms": list(
            dict.fromkeys(
                str(term).lower().strip()
                for term in extracted.get("preference_terms") or []
                if str(term).strip()
            )
        ),
    }
