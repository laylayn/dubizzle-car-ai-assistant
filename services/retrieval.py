from typing import Optional, List, Dict, Any
from services.data_loader import load_cars


def search_cars(
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    keyword: Optional[str] = None,
    limit: int = 5,
    keywords: Optional[List[str]] = None,
    require_all_keywords: bool = True,
) -> List[Dict[str, Any]]:
    """
    Searches the car inventory using simple pandas filtering.

    Supports:
    - make search
    - model search
    - year search
    - keyword search inside title/description/combined text

    Returns:
    - list of matching car dictionaries
    - each result includes match_reason
    """

    df = load_cars()

    # Start with all cars
    results = df.copy()

    # Store match reasons globally
    applied_filters = []

    if make:
        make_clean = make.lower().strip()
        results = results[
            results["make_search"].str.contains(make_clean, na=False, regex=False)
        ]
        applied_filters.append(f'make contains "{make}"')

    if model:
        model_clean = model.lower().strip()
        results = results[
            results["model_search"].str.contains(model_clean, na=False, regex=False)
        ]
        applied_filters.append(f'model contains "{model}"')

    if year:
        results = results[results["year"] == int(year)]
        applied_filters.append(f"year is {year}")

    keyword_terms = []
    for value in [keyword, *(keywords or [])]:
        clean_value = str(value or "").lower().strip()
        if clean_value and clean_value not in keyword_terms:
            keyword_terms.append(clean_value)

    if keyword_terms:
        keyword_matches = [
            results["combined_search"].str.contains(
                term,
                na=False,
                regex=False,
            )
            for term in keyword_terms
        ]
        match_count = sum(match.astype(int) for match in keyword_matches)
        results = results.assign(_keyword_match_count=match_count)

        if require_all_keywords:
            results = results[
                results["_keyword_match_count"] == len(keyword_terms)
            ]
        elif not results.empty:
            best_match_count = int(results["_keyword_match_count"].max())
            if best_match_count > 0:
                results = results[
                    results["_keyword_match_count"] == best_match_count
                ]
            else:
                results = results.iloc[0:0]

        results = results.sort_values(
            "_keyword_match_count",
            ascending=False,
            kind="stable",
        )

    # Limit results
    results = results.head(limit)

    cars = []

    for _, row in results.iterrows():
        matched_keywords = [
            term
            for term in keyword_terms
            if term in str(row.get("combined_search", "")).lower()
        ]
        missing_keywords = [
            term
            for term in keyword_terms
            if term not in matched_keywords
        ]
        reason_parts = list(applied_filters)

        if matched_keywords:
            reason_parts.append(
                "listing text mentions " + ", ".join(matched_keywords)
            )
        if missing_keywords:
            reason_parts.append(
                "listing text does not confirm " + ", ".join(missing_keywords)
            )

        match_reason = (
            "Matched because " + "; ".join(reason_parts)
            if reason_parts
            else "Matched because it is part of the available inventory."
        )

        car = {
            "listing_id": int(row.get("listing_id", 0)),
            "year": int(row.get("year", 0)),
            "make": row.get("make", ""),
            "model": row.get("model", ""),
            "trim": row.get("trim", ""),
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "photo_url": row.get("photo_url", ""),
            "match_reason": match_reason
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

        cars.append(car)

    return cars


def print_results(cars: List[Dict[str, Any]]) -> None:
    """
     helper function to print search results nicely in terminal
    """

    if not cars:
        print("No matching cars found.")
        return

    for index, car in enumerate(cars, start=1):
        print(f"\nResult {index}")
        print("-" * 40)
        print(f"Listing ID: {car['listing_id']}")
        print(f"Car: {car['year']} {car['make']} {car['model']} {car['trim']}")
        print(f"Title: {car['title']}")
        print(f"Reason: {car['match_reason']}")
        print(f"Description: {car['description'][:250]}...")


if __name__ == "__main__":
    print("\nTEST 1: Search by make = ford")
    ford_results = search_cars(make="ford")
    print_results(ford_results)

    print("\n\nTEST 2: Search by make = mercedes-benz and year = 2019")
    mercedes_results = search_cars(make="mercedes-benz", year=2019)
    print_results(mercedes_results)

    print("\n\nTEST 3: Search by keyword = warranty")
    warranty_results = search_cars(keyword="warranty")
    print_results(warranty_results)

    print("\n\nTEST 4: Search by keyword = gcc")
    gcc_results = search_cars(keyword="gcc")
    print_results(gcc_results)
