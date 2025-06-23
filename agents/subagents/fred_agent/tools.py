import json
import re
import os
import logging
import time
import datetime
import pandas as pd
import requests
from collections import Counter
from typing import List, Dict, Optional, Any
from pathlib import Path
from functools import reduce

# --- Global Configuration & Setup ---

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.stlouisfed.org/fred/"
DATA_BASES_DIR = "data_bases"
os.makedirs(DATA_BASES_DIR, exist_ok=True)

# Frequency mapping and hierarchy for data normalization
FREQ_CODE_MAP = {
    "Daily": "D",
    "Day": "D",
    "D": "D",
    "Weekly": "W",
    "Week": "W",
    "W": "W",
    "Biweekly": "2W",
    "2W": "2W",
    "Monthly": "MS",
    "Month": "MS",
    "M": "MS",
    "MS": "MS",
    "ME": "MS",
    "Quarterly": "QS",
    "Quarters": "QS",
    "Q": "QS",
    "QS": "QS",
    "QE": "QS",
    "Annual": "AS",
    "Yearly": "AS",
    "Year": "AS",
    "AS": "AS",
    "YS": "AS",
    "YE": "AS",
}
FREQ_HIERARCHY_MAP = {"D": 0, "W": 1, "2W": 2, "MS": 3, "QS": 4, "AS": 5}


# --- Part 1: Category Lookup (from wtools.py) ---


class FredCategoryLookup:
    """
    Finds relevant FRED category IDs from a natural language query.
    It uses local JSON files for keyword-to-category mapping and category searching.
    """

    def __init__(self):
        """Initializes the class and defines file paths."""
        self.base_path = Path(__file__).parent
        self.keyword_files = [
            self.base_path / "data" / "keywords_mapping_H_Z.json",
            self.base_path / "data" / "keywords_mapping_integers_G.json",
        ]
        self.fred_categories_file = self.base_path / "data" / "fred_categories.json"

        # Caching for loaded data to avoid redundant file I/O
        self._keyword_mappings: Optional[Dict] = None
        self._fred_categories: Optional[List] = None

    @property
    def keyword_mappings(self) -> Dict:
        """Lazy loads and caches keyword mappings from JSON files."""
        if self._keyword_mappings is None:
            self._keyword_mappings = self._load_keyword_mappings()
        return self._keyword_mappings

    @property
    def fred_categories(self) -> List:
        """Lazy loads and caches FRED categories from a JSON file."""
        if self._fred_categories is None:
            self._fred_categories = self._load_fred_categories()
        return self._fred_categories

    def _load_keyword_mappings(self) -> Dict:
        """Loads and merges multiple keyword mapping files."""
        keyword_mappings = {}
        for file_path in self.keyword_files:
            try:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        keyword_mappings.update(data)
                    logger.info(
                        f"Successfully loaded keyword mappings from '{file_path}'."
                    )
                else:
                    logger.warning(f"Keyword mapping file not found: '{file_path}'")
            except Exception as e:
                logger.error(f"Error loading '{file_path}': {e}")
        logger.info(f"Loaded {len(keyword_mappings)} total keyword mappings.")
        return keyword_mappings

    def _load_fred_categories(self) -> List:
        """Loads the list of FRED categories."""
        try:
            if self.fred_categories_file.exists():
                with open(self.fred_categories_file, "r", encoding="utf-8") as f:
                    categories = json.load(f)
                logger.info(f"Loaded {len(categories)} FRED categories.")
                return categories
            else:
                logger.error(
                    f"FRED categories file not found: {self.fred_categories_file}"
                )
                return []
        except Exception as e:
            logger.error(f"Error loading FRED categories file: {e}")
            return []

    def _tokenize_query(self, query: str) -> List[str]:
        """Extracts and normalizes tokens (words) from the query string."""
        return re.findall(r"\b\w+\b", query.lower())

    def _keywords_to_categories(self, query: str) -> List[str]:
        """Finds category IDs by matching tokens from the query with stored keywords."""
        tokens = self._tokenize_query(query)
        all_matched_ids = []
        for token in tokens:
            if token in self.keyword_mappings:
                all_matched_ids.extend(self.keyword_mappings[token])

        if not all_matched_ids:
            return []

        # Count occurrences and rank by frequency
        category_counts = Counter(all_matched_ids)
        ranked_ids = [str(cat_id) for cat_id, _ in category_counts.most_common()]
        logger.info(f"Found {len(ranked_ids)} unique categories from keyword matching.")
        return ranked_ids

    def _search_fred_categories(self, query: str) -> List[str]:
        """Performs a direct search on the names and keywords of FRED categories."""
        query_lower = query.lower()
        matching_ids = set()
        for category in self.fred_categories:
            name = category.get("name", "").lower()
            keywords = [k.lower() for k in category.get("keywords", [])]
            category_id = category.get("id")

            if category_id and (
                query_lower in name or any(query_lower in kw for kw in keywords)
            ):
                matching_ids.add(str(category_id))

        logger.info(
            f"Found {len(matching_ids)} categories from direct FRED category search."
        )
        return list(matching_ids)

    def find_categories(self, query: str) -> List[str]:
        """
        Main method to find relevant FRED category IDs by combining keyword and direct search methods.

        Args:
            query: The user's natural language query.

        Returns:
            A list of matching FRED category IDs, ranked by relevance.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided.")
            return []

        logger.info(f"Processing query: '{query}'")
        keyword_results = self._keywords_to_categories(query)
        fred_search_results = self._search_fred_categories(query)
        combined_results = []
        seen_ids = set()
        for cat_id in keyword_results + fred_search_results:
            if cat_id not in seen_ids:
                combined_results.append(cat_id)
                seen_ids.add(cat_id)

        logger.info(f"Total unique categories found: {len(combined_results)}")
        logger.info(f"Top 10 results: {combined_results[:10]}")
        combined_results = combined_results[
            :10
        ]  # just 10 the top 10 results for demonstration purposes
        return combined_results


def get_categories_series_info(category_ids: list, fred_api_key: str) -> dict:
    """Fetches series metadata for a list of FRED categories."""
    all_series_data = {"seriess": []}
    for category_id in category_ids:
        url = f"{BASE_URL}category/series?category_id={category_id}&api_key={fred_api_key}&file_type=json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if "seriess" in data:
                all_series_data["seriess"].extend(data["seriess"])
        except Exception as e:
            logger.error(f"Request failed for category ID {category_id}: {e}")
        time.sleep(0.1)  # Rate limit
    return all_series_data


def get_series_observations_from_fred(
    series_id: str, start_date: str, end_date: str, fred_api_key: str
) -> pd.DataFrame:
    """Fetches time series observations for a single FRED series ID."""
    url = f"{BASE_URL}series/observations?series_id={series_id}&observation_start={start_date}&observation_end={end_date}&api_key={fred_api_key}&file_type=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("observations", [])
        df = pd.DataFrame(data)
        if not df.empty:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df[["date", "value"]]
    except Exception as e:
        logger.error(f"Failed to get observations for series ID {series_id}: {e}")
    return pd.DataFrame(columns=["date", "value"])


def _map_frequency(fred_frequency: str) -> tuple[int, str]:
    """Maps FRED frequency strings (e.g., "Monthly") to Pandas frequency codes (e.g., "MS")."""
    freq_code = FREQ_CODE_MAP.get(fred_frequency, "MS")
    freq_hierarchy = FREQ_HIERARCHY_MAP.get(freq_code, 3)
    return freq_hierarchy, freq_code


def normalize_frequency(
    df_series: pd.DataFrame, source_fred_frequency: str, target_frequency: str
) -> pd.DataFrame:
    """Normalizes a time series to a target frequency using aggregation or interpolation."""
    if df_series.empty:
        return pd.DataFrame(columns=["date", "value"])

    df_series = df_series.copy()
    df_series["date"] = pd.to_datetime(df_series["date"])
    df_series = df_series.drop_duplicates(subset="date").set_index("date")

    _, source_freq_pd = _map_frequency(source_fred_frequency)
    _, target_freq_pd = _map_frequency(target_frequency)
    source_hier = FREQ_HIERARCHY_MAP.get(source_freq_pd, 3)
    target_hier = FREQ_HIERARCHY_MAP.get(target_freq_pd, 3)

    if source_hier == target_hier:
        return df_series.reset_index()
    elif source_hier < target_hier:  # Aggregate (e.g., Daily to Monthly)
        logger.info(f"Aggregating from {source_freq_pd} to {target_freq_pd}.")
        return df_series.resample(target_freq_pd).mean().reset_index()
    else:  # Disaggregate (e.g., Quarterly to Monthly)
        logger.info(f"Disaggregating from {source_freq_pd} to {target_freq_pd}.")
        resampled_df = df_series.resample(target_freq_pd).asfreq()
        return resampled_df.interpolate(method="linear").reset_index()


def create_unique_file_path(base_file_name: str) -> str:
    """Creates a unique file path to avoid overwriting existing files."""
    os.makedirs(DATA_BASES_DIR, exist_ok=True)
    sanitized_base_name = re.sub(
        r'[\\/*?:"<>|]', "", base_file_name
    )  # Remove invalid chars
    file_path = os.path.join(DATA_BASES_DIR, f"{sanitized_base_name}.csv")
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(DATA_BASES_DIR, f"{sanitized_base_name}_{counter}.csv")
        counter += 1
    return file_path


def find_and_process_fred_data(
    query: str,
    start_date: str,
    end_date: str,
    target_frequency: str,
    fred_api_key: str,
) -> str:
    """
    The main pipeline function that integrates all steps.

    1. Finds FRED category IDs based on the query.
    2. Fetches all series within those categories.
    3. Processes each series to a normalized frequency.
    4. Merges all series into a single DataFrame.
    5. Saves the result to a CSV file.

    Args:
        query: The natural language query (e.g., "US inflation and unemployment").
        start_date: Start date for observations (YYYY-MM-DD).
        end_date: End date for observations (YYYY-MM-DD).
        target_frequency: The desired output frequency (e.g., "Monthly", "Quarterly").
        fred_api_key: Your FRED API key.

    Returns:
        The file path to the resulting CSV file or an error message.
    """
    # --- Step 1: Find Categories ---
    fred_api_key = str(os.getenv("FRED_API_KEY"))
    logger.info("--- Step 1: Finding Categories ---")
    lookup = FredCategoryLookup()
    category_ids = lookup.find_categories(query)

    if not category_ids:
        message = "No relevant categories found for the query."
        logger.warning(message)
        return message

    logger.info(
        f"Found {len(category_ids)} categories. Proceeding to fetch series info."
    )

    # --- Step 2: Get Series Info from Categories ---
    logger.info("--- Step 2: Fetching Series Information ---")
    series_info_raw = get_categories_series_info(category_ids, fred_api_key)
    series_to_process = series_info_raw.get("seriess", [])

    if not series_to_process:
        message = "Found categories, but no series were available for them."
        logger.warning(message)
        return message

    logger.info(f"Found {len(series_to_process)} series to process.")

    # --- Step 3: Process Each Series and Combine ---
    logger.info("--- Step 3: Fetching, Normalizing, and Combining Series Data ---")
    all_processed_series = []

    # Sort series by popularity to process the most relevant ones first
    series_to_process.sort(key=lambda x: x.get("popularity", 0), reverse=True)
    series_to_process = series_to_process[
        :50
    ]  # just 50 series for demonstration purposes
    for series in series_to_process:
        series_id = series.get("id")
        series_freq = series.get("frequency")
        logger.info(f"Processing series: {series_id} (Frequency: {series_freq})")

        # Fetch observations
        temp_df = get_series_observations_from_fred(
            series_id, start_date, end_date, fred_api_key
        )
        if temp_df.empty:
            continue

        # Normalize frequency
        processed_df = normalize_frequency(temp_df, series_freq, target_frequency)
        if not processed_df.empty:
            processed_df.rename(
                columns={"value": series_id, "date": "time"}, inplace=True
            )
            all_processed_series.append(processed_df.set_index("time"))

    if not all_processed_series:
        message = "Could not process or find data for any series."
        logger.error(message)
        return message

    # --- Step 4: Merge DataFrames and Save ---
    logger.info("--- Step 4: Merging all data into a final DataFrame ---")
    # Join all dataframes on the time index
    final_df = pd.concat(all_processed_series, axis=1, join="outer")
    final_df.sort_index(inplace=True)
    final_df.reset_index(inplace=True)

    # --- Step 5: Save to CSV ---
    logger.info("--- Step 5: Saving Data to CSV ---")
    output_filename_base = f"fred_data_{query.replace(' ', '_')}"
    fred_output_file_path = create_unique_file_path(output_filename_base)
    final_df.to_csv(fred_output_file_path, index=False)

    success_message = f"File created and saved to: {fred_output_file_path}"
    logger.info(success_message)
    return fred_output_file_path


if __name__ == "__main__":  # for stand alone testing
    print("--- Running Full FRED Data Pipeline ---")
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("\nERROR: FRED_API_KEY environment variable not set.")
        print("Please set the variable and try again.")
    else:
        user_query = "united states inflation"
        start = "2020-01-01"
        end = "2024-12-31"
        frequency = "Monthly"

        print(f"\nQuery: '{user_query}'")
        print(f"Date Range: {start} to {end}")
        print(f"Target Frequency: {frequency}\n")

        result_path = find_and_process_fred_data(
            query=user_query,
            start_date=start,
            end_date=end,
            target_frequency=frequency,
            fred_api_key=api_key,
        )

        print(f"\nPipeline finished. Result: {result_path}")
