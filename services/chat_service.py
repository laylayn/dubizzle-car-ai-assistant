import re
from typing import Any, Dict, Optional, Type

from services.booking import book_viewing, extract_booking_slot, save_lead
from services.booking_flow import persist_confirmed_booking_memory
from services.car_details import respond_about_selected_car
from services.car_utils import (
    build_car_title,
    format_filter_value,
    get_car_from_message,
    get_requested_color,
)
from services.comparison import compare_cars
from services.guardrails import apply_guardrails, log_intent
from services.listing_attributes import text_mentions_feature
from services.llm_agent import (
    extract_user_request,
    generate_grounded_reply,
    generate_inventory_reply,
    generate_refusal_reply,
    is_memory_recall_request,
)
from services.memory import (
    format_memory_summary_for_user,
    get_memory_summary,
    get_or_create_user,
    get_user,
    record_user_interaction,
    update_memory_summary,
    update_user_memory,
)
from services.memory_search import (
    resolve_car_from_long_term_memory,
    search_from_user_memory,
)
from services.query_planner import build_query_plan
from services.search_flow import (
    build_no_matching_listings_reply,
    build_ranked_inventory_fallback,
    build_unavailable_attribute_reply,
    execute_query_plan,
    maybe_save_interest_lead,
)
from services.session_memory import (
    add_message,
    extract_car_positions,
    get_car_by_position,
    get_last_results,
    get_messages,
    get_preferences,
    get_selected_car,
    is_follow_up_about_selected_car,
    resolve_car_reference,
    save_last_results,
    save_selected_car,
    update_preferences,
)


BLOCKED_LLM_INTENTS = {
    "blocked_non_automotive",
    "blocked_competitor",
    "blocked_coding",
    "blocked_history",
    "blocked_politics",
    "blocked_medical_legal",
    "blocked_random_homework",
}


def _response(
    response_model: Type,
    *,
    reply: str,
    intent: str,
    cars=None,
    extracted_request=None,
    memory_update=None,
):
    return response_model(
        reply=reply,
        intent=intent,
        cars=cars or [],
        extracted_request=extracted_request or {},
        memory_update=memory_update or {},
    )


def _build_search_label(extracted_request: Dict[str, Any]) -> str:
    values = [extracted_request.get("make"), extracted_request.get("model")]
    return " ".join(format_filter_value(value) for value in values if value).strip()


def _handle_conversation(
    username: str,
    session_id: str,
    user_message: str,
    intent: str,
    extracted_request: Dict[str, Any],
    response_model: Type,
):
    memory_summary = get_memory_summary(username)
    if intent == "greeting":
        fallback_reply = memory_summary.get(
            "message",
            (
                f"Hi {username}! I can help you search car listings, "
                "compare options, or book a viewing."
            ),
        )
        response_task = (
            "Respond warmly to the greeting. Use saved user memory only "
            "when it is present, and offer concise car-shopping help."
        )
    else:
        fallback_reply = (
            "I’m here with you. We can keep chatting, or continue whenever "
            "you’re ready to look at cars."
        )
        response_task = (
            "Respond naturally to this harmless conversational message. "
            "Use recent conversation only for continuity, keep it brief, "
            "and do not invent user preferences or vehicle facts. If the "
            "message actually asks for information or work unrelated to "
            "cars, do not answer that content; briefly redirect to the "
            "car-assistant scope."
        )
    reply = generate_grounded_reply(
        user_message=user_message,
        response_task=response_task,
        grounded_context={
            "username": username,
            "memory": memory_summary,
            "recent_messages": get_messages(session_id, limit=6),
        },
        fallback_reply=fallback_reply,
    )
    add_message(session_id, "assistant", reply)
    log_intent(username=username, intent=intent, results_count=0)
    return _response(
        response_model,
        reply=reply,
        intent=intent,
        extracted_request=extracted_request,
        memory_update=memory_summary,
    )


def _handle_memory_recall(
    username: str,
    session_id: str,
    user_message: str,
    extracted_request: Dict[str, Any],
    response_model: Type,
):
    user_memory = get_user(username) or {}
    saved_memory_summary = str(user_memory.get("memory_summary") or "").strip()
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
    return _response(
        response_model,
        reply=reply,
        intent="returning_user_memory",
        cars=cars,
        extracted_request=extracted_request,
        memory_update=get_memory_summary(username),
    )


def _build_search_reply(
    user_message: str,
    extracted_request: Dict[str, Any],
    query_plan: Dict[str, Any],
    query_execution: Dict[str, Any],
    cars,
    partial_color_match: bool,
    requested_color: Optional[str],
) -> str:
    if query_execution.get("status") == "needs_preference_clarification":
        preferences = ", ".join(extracted_request.get("soft_preferences") or [])
        fallback_reply = (
            f"I understand you’re looking for {preferences or 'a suitable car'}. "
            "Could you share one or two concrete preferences, such as body "
            "style, seating needs, budget, or must-have features?"
        )
        return generate_grounded_reply(
            user_message=user_message,
            response_task=(
                "Acknowledge the subjective car preference and ask one "
                "concise clarifying question for concrete searchable needs. "
                "Do not claim that the inventory has no matches."
            ),
            grounded_context={
                "soft_preferences": extracted_request.get("soft_preferences"),
                "query_execution": query_execution,
                "verified_finding": (
                    "There is not yet enough concrete listing evidence to "
                    "rank inventory safely."
                ),
            },
            fallback_reply=fallback_reply,
        )
    if (
        query_plan.get("sort_by")
        and query_execution.get("status") == "attribute_unavailable"
    ):
        fallback_reply = build_unavailable_attribute_reply(query_plan["sort_by"])
        return generate_grounded_reply(
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
    if query_plan.get("sort_by") and cars:
        fallback_reply = build_ranked_inventory_fallback(cars, query_execution)
        return generate_grounded_reply(
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
    if query_plan.get("sort_by"):
        fallback_reply = build_no_matching_listings_reply(extracted_request)
        return generate_grounded_reply(
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
    if partial_color_match:
        search_label = _build_search_label(extracted_request) or "requested"
        fallback_reply = (
            f"I found {search_label} listings, but none of the provided "
            f"listing text explicitly mentions {requested_color}."
        )
        return generate_grounded_reply(
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
    if cars:
        return generate_inventory_reply(
            user_message=user_message,
            cars=cars,
            extracted_request=extracted_request,
        )
    fallback_reply = build_no_matching_listings_reply(extracted_request)
    return generate_grounded_reply(
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


def _handle_search(
    username: str,
    session_id: str,
    user_message: str,
    intent: str,
    extracted_request: Dict[str, Any],
    response_model: Type,
):
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
        and not any(text_mentions_feature(car, requested_color) for car in cars)
    )
    if partial_color_match:
        search_label = _build_search_label(extracted_request) or "matching car"
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
    update_preferences(
        session_id,
        {
            "make": extracted_request.get("make"),
            "model": extracted_request.get("model"),
            "year": extracted_request.get("year"),
            "keyword": extracted_request.get("keyword"),
            "required_keywords": extracted_request.get("required_keywords"),
            "soft_preferences": extracted_request.get("soft_preferences"),
            "preference_terms": extracted_request.get("preference_terms"),
            "body_type": extracted_request.get("body_type"),
            "budget": extracted_request.get("budget"),
            "needs": extracted_request.get("needs"),
            "query_plan": query_plan,
            "pending_booking": False,
            "pending_booking_date": None,
            "pending_booking_time": None,
        },
    )
    update_user_memory(
        username=username,
        last_budget=extracted_request.get("budget"),
        preferred_make=extracted_request.get("make"),
        preferred_model=extracted_request.get("model"),
        preferred_body_type=extracted_request.get("body_type"),
        last_seen_car=build_car_title(cars[0]) if cars else "",
        last_query=user_message,
    )
    lead = maybe_save_interest_lead(username, extracted_request, cars)
    search_features = [
        *(extracted_request.get("required_keywords") or []),
        *(extracted_request.get("soft_preferences") or []),
        *(extracted_request.get("preference_terms") or []),
    ]
    search_notes = []
    if search_features:
        search_notes.append("Requested preferences: " + ", ".join(search_features))
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
        event_type="preference_search" if has_preferences else "inventory_search",
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
    reply = _build_search_reply(
        user_message,
        extracted_request,
        query_plan,
        query_execution,
        cars,
        partial_color_match,
        requested_color,
    )
    add_message(session_id, "assistant", reply)
    log_intent(username=username, intent=intent, results_count=len(cars))
    memory_summary = get_memory_summary(username)
    memory_summary["lead_saved"] = lead
    return _response(
        response_model,
        reply=reply,
        intent=intent,
        cars=cars,
        extracted_request=extracted_request,
        memory_update=memory_summary,
    )


def _handle_car_details(
    username: str,
    session_id: str,
    user_message: str,
    intent: str,
    extracted_request: Dict[str, Any],
    response_model: Type,
):
    selected_car = get_car_from_message(user_message)
    selected_position = extracted_request.get("selected_position")
    if selected_position is not None:
        selected_car = get_car_by_position(session_id, int(selected_position))
    if selected_car is None:
        selected_car = resolve_car_reference(session_id, user_message)
    if selected_car is None:
        selected_car = get_selected_car(session_id)
    if selected_car is not None:
        return respond_about_selected_car(
            username=username,
            session_id=session_id,
            user_message=user_message,
            selected_car=selected_car,
            response_model=response_model,
            extracted_request=extracted_request,
        )

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
    return _response(
        response_model,
        reply=reply,
        intent=intent,
        extracted_request=extracted_request,
        memory_update=get_memory_summary(username),
    )


def _handle_comparison(
    username: str,
    session_id: str,
    user_message: str,
    intent: str,
    extracted_request: Dict[str, Any],
    response_model: Type,
):
    last_results = get_last_results(session_id)
    requested_positions = extract_car_positions(user_message)
    extracted_request["selected_positions"] = requested_positions
    positions_to_compare = requested_positions if requested_positions else [0, 1]
    invalid_positions = [
        position
        for position in positions_to_compare
        if position < 0 or position >= len(last_results)
    ]
    comparison_cars = (
        [last_results[position] for position in positions_to_compare]
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
        selected_car_title=build_car_title(selected_car) if selected_car else "",
        notes="Lead captured from a comparison request.",
        source_intent="compare_listings",
    )

    if invalid_positions:
        unavailable = ", ".join(
            f"result {position + 1}" for position in invalid_positions
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
        fallback_reply = "Please specify at least two result positions to compare."
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
            grounded_context={"available_results_count": len(last_results)},
            fallback_reply=fallback_reply,
        )
    else:
        positioned_cars = [
            {"requested_result_position": position + 1, "car": car}
            for position, car in zip(positions_to_compare, comparison_cars)
        ]
        fallback_reply = compare_cars(comparison_cars, positions_to_compare)
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
            build_car_title(car) for car in comparison_cars
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
    return _response(
        response_model,
        reply=reply,
        intent=intent,
        cars=comparison_cars,
        extracted_request=extracted_request,
        memory_update=memory_summary,
    )


def _handle_booking(
    username: str,
    session_id: str,
    user_message: str,
    extracted_request: Dict[str, Any],
    response_model: Type,
    booking_date_hint: Optional[str],
    booking_time_hint: Optional[str],
    memory_recall_requested: bool,
):
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
        booking_date_hint or current_preferences.get("pending_booking_date")
    )
    booking_time = (
        booking_time_hint or current_preferences.get("pending_booking_time")
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
    log_intent(
        username=username,
        intent="booking",
        results_count=1 if selected_car else 0,
    )
    if not (booking_result and booking_result["success"]):
        if not current_preferences.get("pending_booking"):
            record_user_interaction(
                username=username,
                event_type="booking_intent",
                query=user_message,
                make=str((selected_car or {}).get("make") or ""),
                model=str((selected_car or {}).get("model") or ""),
                listing_id=(selected_car or {}).get("listing_id"),
                car_title=build_car_title(selected_car) if selected_car else "",
                lead_quality=(booking_lead or {}).get("lead_status", ""),
                notes="User asked to arrange a viewing.",
            )
    memory_summary = get_memory_summary(username)
    memory_summary["lead_saved"] = booking_lead
    return _response(
        response_model,
        reply=reply,
        intent="booking",
        cars=[selected_car] if selected_car else [],
        extracted_request=extracted_request,
        memory_update=memory_summary,
    )


def handle_chat(request, response_model: Type):
    """Run the full chat workflow for the FastAPI endpoint."""

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

    # Rule-based guardrails intentionally run before LLM extraction.
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
    if referenced_car is None and is_follow_up_about_selected_car(user_message):
        referenced_car = get_selected_car(session_id)

    memory_detail_intents = {
        "car_details",
        "car_search",
        "lead_capture",
        "general_car_advice",
        "chitchat",
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
            response_model=response_model,
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
        log_intent(
            username=username,
            intent=guardrail["intent"],
            results_count=0,
        )
        return _response(
            response_model,
            reply=reply,
            intent=guardrail["intent"],
            memory_update=get_memory_summary(username),
        )

    extracted_request = extract_user_request(user_message)
    intent = extracted_request.get("intent", guardrail["intent"])
    if booking_followup or (
        memory_recall_requested and guardrail["intent"] == "booking"
    ):
        intent = "booking"
        extracted_request["intent"] = intent
    elif memory_recall_requested:
        intent = "returning_user_memory"
        extracted_request["intent"] = intent
    elif is_comparison_request:
        intent = "compare_listings"
        extracted_request["intent"] = intent
    elif guardrail["intent"] in {"greeting", "chitchat"} and referenced_car is None:
        intent = guardrail["intent"]
        extracted_request["intent"] = intent
    elif guardrail["intent"] == "car_search" and referenced_car is None:
        intent = "car_search"
        extracted_request["intent"] = intent
    elif guardrail["intent"] in {"booking", "compare_listings"}:
        intent = guardrail["intent"]
        extracted_request["intent"] = intent

    # The extracted LLM intent is the second guardrail layer.
    if intent in BLOCKED_LLM_INTENTS:
        reply = generate_refusal_reply(
            user_message=user_message,
            blocked_intent=intent,
        )
        add_message(session_id, "assistant", reply)
        log_intent(username=username, intent=intent, results_count=0)
        return _response(
            response_model,
            reply=reply,
            intent=intent,
            extracted_request=extracted_request,
            memory_update=get_memory_summary(username),
        )

    if intent in {"greeting", "chitchat"}:
        return _handle_conversation(
            username,
            session_id,
            user_message,
            intent,
            extracted_request,
            response_model,
        )
    if intent in {
        "returning_user_memory",
        "memory_recall",
        "similar_to_previous",
    }:
        return _handle_memory_recall(
            username,
            session_id,
            user_message,
            extracted_request,
            response_model,
        )
    if intent in {"car_search", "lead_capture", "general_car_advice"}:
        return _handle_search(
            username,
            session_id,
            user_message,
            intent,
            extracted_request,
            response_model,
        )
    if intent == "car_details":
        return _handle_car_details(
            username,
            session_id,
            user_message,
            intent,
            extracted_request,
            response_model,
        )
    if intent == "compare_listings":
        return _handle_comparison(
            username,
            session_id,
            user_message,
            intent,
            extracted_request,
            response_model,
        )
    if intent == "booking":
        return _handle_booking(
            username,
            session_id,
            user_message,
            extracted_request,
            response_model,
            booking_date_hint,
            booking_time_hint,
            memory_recall_requested,
        )

    fallback_reply = (
        "I can help with car searches, listing details, comparisons, and "
        "viewing bookings. Try asking me to show cars by make, model, year, "
        "or feature."
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
    return _response(
        response_model,
        reply=reply,
        intent=intent,
        extracted_request=extracted_request,
        memory_update=get_memory_summary(username),
    )
