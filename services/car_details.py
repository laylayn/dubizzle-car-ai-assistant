import re
from typing import Any, Dict, Optional, Type

from services.car_utils import (
    build_car_title,
    get_requested_color,
)
from services.guardrails import log_intent
from services.listing_attributes import (
    PRICE_FIELD_NAMES,
    extract_color_from_car,
    extract_mileage_from_car,
    extract_price_from_car,
    text_mentions_feature,
)
from services.llm_agent import generate_car_detail_reply
from services.memory import get_memory_summary, update_user_memory
from services.session_memory import add_message, save_selected_car


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
    """Extract a short feature phrase from questions such as 'does it have X?'."""

    match = re.search(
        r"\b(?:have|include|mention|with)\s+"
        r"(?:a|an|any|the)?\s*([^?.!,]{2,60})",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    feature = match.group(1).strip()
    return None if len(feature.split()) > 6 else feature


def answer_car_question(
    car: Dict[str, Any],
    user_message: str,
) -> Optional[str]:
    """Build a deterministic fallback for attribute-specific follow-ups."""

    message_lower = user_message.lower()
    car_label = get_car_reference_label(car, user_message)

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
            return f"The listed price appears to be AED {price_result['amount']}."
        return "I couldn’t find a price in the provided listing."

    requested_color = get_requested_color(user_message)
    if requested_color:
        if text_mentions_feature(car, requested_color):
            return (
                "Yes. The provided listing text explicitly mentions "
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

    for attribute in ["year", "make", "model", "trim"]:
        if re.search(rf"\b{attribute}\b", message_lower):
            value = car.get(attribute)
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
            return (
                f"The listing describes these details for {car_label}: "
                f"{preview}{suffix}"
            )
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


def respond_about_selected_car(
    username: str,
    session_id: str,
    user_message: str,
    selected_car: Dict[str, Any],
    response_model: Type,
    extracted_request: Optional[Dict[str, Any]] = None,
):
    """Return a grounded response and persist the selected car."""

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
    return response_model(
        reply=reply,
        intent="car_details",
        cars=[selected_car],
        extracted_request=extracted_request or {"intent": "car_details"},
        memory_update=get_memory_summary(username),
    )
