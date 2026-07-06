from typing import Any, Dict, List

from services.car_utils import build_car_title


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
                f"- Description preview: {car.get('description', '')[:350]}...",
            ]
        )
    return "\n".join(lines)
