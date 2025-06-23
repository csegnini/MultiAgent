import os
import pandas as pd
import io
from serpapi import GoogleSearch
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import logging

load_dotenv()
SERPAPI_KEY = str(os.getenv("SERPAPI_API_KEY"))
DATA_BASES_DIR = "data_bases"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_unique_file_path(base_file_name: str) -> str:
    os.makedirs(DATA_BASES_DIR, exist_ok=True)
    full_file_name = f"{base_file_name}.csv"
    file_path = os.path.join(DATA_BASES_DIR, full_file_name)
    counter = 1
    while os.path.exists(file_path):
        full_file_name = f"{base_file_name}_{counter}.csv"
        file_path = os.path.join(DATA_BASES_DIR, full_file_name)
        counter += 1
    return file_path


def get_google_trends_data(
    keywords: List[str], geo: str, time_frame: str, api_key: Optional[str] = None
) -> Optional[List[str]]:
    if not api_key:
        api_key = SERPAPI_KEY
    if not api_key:
        print("Error: SerpApi API key not found.")
        return None
    if isinstance(keywords, list) and len(keywords) > 5:
        print("Warning: Google Trends supports max 5 keywords. Using the first 5.")
        keywords = keywords[:5]
    query_string = ", ".join(keywords) if isinstance(keywords, list) else keywords
    params = {
        "engine": "google_trends",
        "q": query_string,
        "hl": "en",
        "geo": geo,
        "date": time_frame,
        "data_type": "TIMESERIES",
        "csv": "true",
        "api_key": api_key,
    }
    print(f"Requesting data for keywords: '{query_string}' in geo: '{geo}'...")
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        if "error" in results:
            print(f"SerpApi Error: {results['error']}")
            return None
        return results.get("csv")
    except Exception as e:
        print(f"An unexpected error occurred during API request: {e}")
        return None


def parse_trends_to_dataframe(csv_data: List[str], keywords: List[str]) -> pd.DataFrame:
    if not csv_data or len(csv_data) < 3:
        return pd.DataFrame()
    header_index = next(
        (i for i, row in enumerate(csv_data) if "Category:" not in row and row), -1
    )
    if header_index == -1:
        return pd.DataFrame()
    csv_string = "\n".join(csv_data[header_index:])
    try:
        df = pd.read_csv(io.StringIO(csv_string))
        date_column_name = df.columns[0]
        df.rename(columns={date_column_name: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        column_rename_map = {}
        for col_name in df.columns:
            matching_keyword = next(
                (kw for kw in keywords if kw.lower() in col_name.lower()), None
            )
            if matching_keyword:
                column_rename_map[col_name] = matching_keyword
        df.rename(columns=column_rename_map, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Aggregate to monthly by summing the values for each month
        df_monthly = df.resample("M").sum()
        # Ensure index is DatetimeIndex before converting
        if not isinstance(df_monthly.index, pd.DatetimeIndex):
            df_monthly.index = pd.to_datetime(df_monthly.index)
        # Set index to first day of each month
        df_monthly.index = df_monthly.index.to_period("M").to_timestamp()
        return df_monthly
    except Exception as e:
        print(f"An error occurred while parsing data: {e}")
        return pd.DataFrame()


def main_get_google_trends(
    search_keywords: List[str],
    search_geo: str,
    search_timeframe: str,
    output_filename: str = "trends_processed_data",
) -> str:
    """
    Main tool function that fetches, parses, and saves Google Trends data to a CSV file.

    Args:
        search_keywords (List[str]): List of keywords to search for on Google Trends. Max 5.
        search_geo (str): The two-letter country code for the geographic region (e.g., "US").
        search_timeframe (str): The time frame for the trends data (e.g., "today 12-m").
        output_filename (str, optional): Base name for the output CSV file.

    Returns:
        str: The path to the saved CSV file or an error message.
    """
    raw_csv_data: Optional[List[str]] = get_google_trends_data(
        keywords=search_keywords,
        geo=search_geo,
        time_frame=search_timeframe,
    )
    if raw_csv_data:
        trends_df: pd.DataFrame = parse_trends_to_dataframe(
            raw_csv_data, search_keywords
        )
        if not trends_df.empty:
            output_file_path: str = create_unique_file_path(output_filename)
            trends_df.to_csv(output_file_path, index=True)
            logging.info(f"Data saved to {output_file_path}")
            return output_file_path
    return "Failed to retrieve or process Google Trends data."


if __name__ == "__main__":
    print("--- Running Tool as a Standalone Script for Testing ---")
    test_keywords = ["Inflation", "Interest rates"]
    test_geo = "US"
    test_timeframe = "today 12-m"

    file_path = main_get_google_trends(test_keywords, test_geo, test_timeframe)

    if file_path:
        print(f"\nTest successful. Data written to: {file_path}")
