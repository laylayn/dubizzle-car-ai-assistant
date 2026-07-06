from functools import lru_cache
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/Copy_of_sample_cars_dataset.xlsx")
SHEET_NAME = "cleaned dataset"


@lru_cache(maxsize=2)
def _load_cars_cached(
    resolved_data_path: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    """Load and prepare one immutable cached inventory snapshot."""

    del modified_time_ns
    df = pd.read_excel(resolved_data_path, sheet_name=SHEET_NAME)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Fill missing values
    df = df.fillna("")

    # Make sure year/listing_id are clean
    if "year" in df.columns:
        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce",
        ).fillna(0).astype(int)

    if "listing_id" in df.columns:
        df["listing_id"] = pd.to_numeric(
            df["listing_id"],
            errors="coerce",
        ).fillna(0).astype(int)

    # Create lowercase searchable versions of text columns
    text_columns = ["make", "model", "trim", "title", "description"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[f"{col}_search"] = df[col].str.lower()

    # Combined text field for keyword searching later
    df["combined_search"] = (
        df.get("make_search", "") + " " +
        df.get("model_search", "") + " " +
        df.get("trim_search", "") + " " +
        df.get("title_search", "") + " " +
        df.get("description_search", "")
    )
    return df


def load_cars() -> pd.DataFrame:
    """
    Loads the cleaned car inventory dataset and prepares it for searching.

    Returns:
        pd.DataFrame: cleaned car listings dataframe
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    resolved_path = DATA_PATH.resolve()
    cached_cars = _load_cars_cached(
        str(resolved_path),
        resolved_path.stat().st_mtime_ns,
    )
    # Preserve the previous contract: callers receive an independent dataframe.
    return cached_cars.copy(deep=True)
