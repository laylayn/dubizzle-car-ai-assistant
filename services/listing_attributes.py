import re
from typing import Any, Dict, Optional


PRICE_FIELD_NAMES = [
    "price",
    "amount",
    "listing_price",
    "sale_price",
    "price_aed",
    "asking_price",
]
MILEAGE_FIELD_NAMES = [
    "mileage",
    "kilometers",
    "kilometres",
    "km",
    "odometer",
]
COLOR_FIELD_NAMES = ["color", "colour", "exterior_color", "exterior_colour"]
OPTIONAL_LISTING_FIELDS = [
    *PRICE_FIELD_NAMES,
    *COLOR_FIELD_NAMES[:2],
    *MILEAGE_FIELD_NAMES,
    *COLOR_FIELD_NAMES[2:],
    "features",
]
COLOR_TERMS = [
    "white",
    "black",
    "silver",
    "grey",
    "gray",
    "blue",
    "red",
    "green",
    "beige",
    "brown",
    "gold",
    "orange",
    "yellow",
    "purple",
]

PRICE_NUMBER_PATTERN = (
    r"(\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*([kK]?)"
)
PRICE_CURRENCY_PATTERN = r"(?:AED|dhs?|dirhams?)"
MILEAGE_NUMBER_PATTERN = r"(\d{1,3}(?:[,\s]\d{3})+|\d{4,7})"


def get_car_listing_text(car: Dict[str, Any]) -> str:
    """Combine only listing-provided text used for grounded extraction."""

    text_fields = [
        car.get("title", ""),
        car.get("description", ""),
        car.get("color", ""),
        car.get("colour", ""),
        car.get("features", ""),
    ]
    return " ".join(str(value) for value in text_fields if value).strip()


def text_mentions_feature(car: Dict[str, Any], feature: str) -> bool:
    """Check whether listing-provided text explicitly mentions a feature."""

    normalized_feature = str(feature or "").lower().strip()
    if not normalized_feature:
        return False

    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized_feature)}(?!\w)",
            get_car_listing_text(car).lower(),
        )
    )


def parse_numeric_value(raw_value: Any) -> Optional[int]:
    """Parse a positive integer from a structured or extracted value."""

    match = re.search(
        r"\d{1,3}(?:[,\s]\d{3})+|\d+(?:\.\d+)?",
        str(raw_value or ""),
    )
    if not match:
        return None

    try:
        value = int(float(re.sub(r"[,\s]", "", match.group(0))))
    except ValueError:
        return None
    return value if value >= 0 else None


def extract_mileage_from_text(
    text: str,
    source: str = "listing_text",
) -> Optional[Dict[str, Any]]:
    """Extract mileage from units or mileage/driven labels."""

    patterns = [
        rf"\b{MILEAGE_NUMBER_PATTERN}\s*(?:km|kilomet(?:er|re)s?)\b",
        rf"\b(?:mileage|odometer)\s*(?:is|:|-)?\s*{MILEAGE_NUMBER_PATTERN}\b",
        rf"\bdriven\s*(?:for|:|-)?\s*{MILEAGE_NUMBER_PATTERN}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        value = parse_numeric_value(match.group(1))
        if value is None or value > 2_000_000:
            continue

        return {
            "value": value,
            "display": f"{value:,} km",
            "source": source,
        }

    return None


def extract_mileage_from_car(car: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read structured mileage first, then inspect title and description."""

    for field_name in MILEAGE_FIELD_NAMES:
        raw_value = car.get(field_name)
        if not str(raw_value or "").strip():
            continue
        value = parse_numeric_value(raw_value)
        if value is not None and value <= 2_000_000:
            return {
                "value": value,
                "display": f"{value:,} km",
                "source": "structured",
                "field": field_name,
            }

    for field_name in ["title", "description"]:
        result = extract_mileage_from_text(
            str(car.get(field_name, "")),
            source=field_name,
        )
        if result:
            return result

    return None


def extract_color_from_car(car: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Read a structured exterior color or find an explicit color in text."""

    for field_name in COLOR_FIELD_NAMES:
        value = str(car.get(field_name, "")).strip()
        if value:
            return {
                "value": value.lower(),
                "display": value,
                "source": "structured",
                "field": field_name,
            }

    listing_text = get_car_listing_text(car).lower()
    for color in COLOR_TERMS:
        if re.search(rf"(?<!\w){re.escape(color)}(?!\w)", listing_text):
            return {
                "value": color,
                "display": color.title(),
                "source": "listing_text",
            }

    return None


def format_price_amount(raw_value: Any) -> Optional[str]:
    """Normalize a numeric price value for a natural AED response."""

    text = str(raw_value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    match = re.search(PRICE_NUMBER_PATTERN, text)
    if not match:
        return None

    number_text = re.sub(r"[,\s]", "", match.group(1))
    try:
        numeric_value = float(number_text)
    except ValueError:
        return None

    if match.group(2).lower() == "k":
        numeric_value *= 1_000
    if numeric_value <= 0:
        return None

    if numeric_value.is_integer():
        return f"{int(numeric_value):,}"
    return f"{numeric_value:,.2f}".rstrip("0").rstrip(".")


def is_non_price_number_context(
    text: str,
    match: re.Match,
    amount: str,
) -> bool:
    """Reject years, mileage, engine size, cylinders, and phone-like values."""

    numeric_value = int(float(amount.replace(",", "")))
    before = text[max(0, match.start() - 20):match.start()].lower()
    after = text[match.end():match.end() + 22].lower()
    if 1900 <= numeric_value <= 2100 or numeric_value > 10_000_000:
        return True
    if re.match(
        r"\s*(?:km|kilomet(?:er|re)s?|cc|cylinder|cylinders|litre|liter)\b",
        after,
    ):
        return True
    if re.search(r"\b(?:mileage|odometer|engine)\s*[:\-]?\s*$", before):
        return True
    if re.search(r"(?:\+?\d[\d\s-]{7,})$", before + match.group(0)):
        return True
    return False


def extract_currency_price_from_text(
    text: str,
    source: str,
) -> Optional[Dict[str, Any]]:
    """Extract a sale price explicitly linked to AED/dirham wording."""

    patterns = [
        rf"\b{PRICE_CURRENCY_PATTERN}\b\s*[:\-]?\s*{PRICE_NUMBER_PATTERN}",
        rf"{PRICE_NUMBER_PATTERN}\s*\b{PRICE_CURRENCY_PATTERN}\b",
    ]
    candidates = []
    seen = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            amount = format_price_amount(f"{match.group(1)}{match.group(2)}")
            if not amount or is_non_price_number_context(text, match, amount):
                continue

            candidate_key = (match.start(), match.end(), amount)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)

            before = text[max(0, match.start() - 30):match.start()].lower()
            after = text[match.end():match.end() + 30].lower()
            local_context = f"{before} {match.group(0).lower()} {after}"
            if re.match(
                r"\s*(?:per\s+month|month(?:ly|y)?|/\s*(?:month|mo)|"
                r"p\.?\s*m\.?)\b",
                after,
            ):
                continue

            is_cash_price = bool(
                re.search(r"\b(?:cash|full\s+price|sale\s+price)\b", local_context)
            )
            candidates.append(
                {
                    "amount": amount,
                    "value": int(float(amount.replace(",", ""))),
                    "display": f"AED {amount}",
                    "source": source,
                    "currency_explicit": True,
                    "price_kind": "cash" if is_cash_price else "listed",
                    "_rank": 2 if is_cash_price else 1,
                    "_position": match.start(),
                }
            )

    if not candidates:
        return None

    best = sorted(
        candidates,
        key=lambda candidate: (-candidate["_rank"], candidate["_position"]),
    )[0]
    return {
        key: value
        for key, value in best.items()
        if not key.startswith("_")
    }


def extract_textual_price_candidate(
    text: str,
    source: str,
    allow_trailing_number: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract high-confidence price text without accepting arbitrary numbers."""

    price_cue_pattern = (
        rf"\b(?:asking\s+price|sale\s+price|price|amount|only)\b"
        rf"\s*(?:is|:|\-)?\s*{PRICE_NUMBER_PATTERN}"
    )
    cue_match = re.search(price_cue_pattern, text, re.IGNORECASE)
    if cue_match:
        amount = format_price_amount(
            f"{cue_match.group(1)}{cue_match.group(2)}"
        )
        if amount and not is_non_price_number_context(text, cue_match, amount):
            return {
                "amount": amount,
                "value": int(float(amount.replace(",", ""))),
                "display": f"AED {amount}",
                "source": source,
                "currency_explicit": False,
            }

    if not allow_trailing_number:
        return None

    for match in re.finditer(PRICE_NUMBER_PATTERN, text):
        amount = format_price_amount(f"{match.group(1)}{match.group(2)}")
        if not amount or is_non_price_number_context(text, match, amount):
            continue

        numeric_value = int(float(amount.replace(",", "")))
        trailing_text = text[match.end():].strip(" \t\r\n.,;:|-/")
        if not trailing_text and 5_000 <= numeric_value <= 10_000_000:
            return {
                "amount": amount,
                "value": numeric_value,
                "display": f"AED {amount}",
                "source": source,
                "currency_explicit": False,
            }

    return None


def extract_price_from_car(car: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read structured price first, then extract a high-confidence text price."""

    for field_name in PRICE_FIELD_NAMES:
        if field_name not in car:
            continue
        amount = format_price_amount(car.get(field_name))
        if amount:
            return {
                "amount": amount,
                "value": int(float(amount.replace(",", ""))),
                "display": f"AED {amount}",
                "source": "structured",
                "field": field_name,
                "currency_explicit": True,
            }

    title = str(car.get("title", ""))
    description = str(car.get("description", ""))
    currency_candidates = []
    for text, source in [(title, "title"), (description, "description")]:
        currency_price = extract_currency_price_from_text(text, source)
        if currency_price:
            currency_candidates.append(currency_price)

    if currency_candidates:
        return sorted(
            currency_candidates,
            key=lambda candidate: (
                candidate.get("price_kind") != "cash",
                candidate.get("source") != "title",
            ),
        )[0]

    title_candidate = extract_textual_price_candidate(
        title,
        source="title",
        allow_trailing_number=True,
    )
    if title_candidate:
        return title_candidate

    return extract_textual_price_candidate(
        description,
        source="description",
        allow_trailing_number=False,
    )


def extract_price_from_text(text: str) -> Optional[str]:
    """Backward-compatible price helper returning only the amount."""

    result = extract_price_from_car({"title": text, "description": ""})
    return result.get("amount") if result else None
