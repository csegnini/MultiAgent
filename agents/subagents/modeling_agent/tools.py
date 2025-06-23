import pandas as pd
import numpy as np
import os
import json
from typing import List
from collections import defaultdict
import logging
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import json


DATA_BASES_DIR = "data_bases"
os.makedirs(DATA_BASES_DIR, exist_ok=True)


def create_unique_modeling_file_path(base_file_name: str) -> str:
    os.makedirs(DATA_BASES_DIR, exist_ok=True)
    full_file_name = f"{base_file_name}.csv"
    file_path = os.path.join(DATA_BASES_DIR, full_file_name)
    counter = 1
    while os.path.exists(file_path):
        full_file_name = f"{base_file_name}_{counter}.csv"
        file_path = os.path.join(DATA_BASES_DIR, full_file_name)
        counter += 1
    return file_path


def join_dataframes(fred_csv_path: str, trends_csv_path: str) -> str:
    """
    Joins the economic data from FRED and the web data from Google Trends.

    Args:
        fred_csv_path (str): The file path to the CSV from the fred_agent.
        trends_csv_path (str): The file path to the CSV from the trends_agent.

    Returns:
        str: The file path to the new, merged CSV file.
    """
    logging.info(
        f"Modeling Agent: Joining data from {fred_csv_path} and {trends_csv_path}"
    )
    try:
        # Read FRED data
        fred_df = pd.read_csv(fred_csv_path)
        fred_df["time"] = pd.to_datetime(fred_df["time"])

        # Read trends data and handle different date column names
        trends_df = pd.read_csv(trends_csv_path)

        # Check for different possible date column names
        date_columns = ["Date", "date", "time", "Time"]
        date_col = None
        for col in date_columns:
            if col in trends_df.columns:
                date_col = col
                break

        if date_col is None:
            raise ValueError(
                f"No date column found in trends data. Available columns: {trends_df.columns.tolist()}"
            )

        # Rename to standardize
        if date_col != "time":
            trends_df.rename(columns={date_col: "time"}, inplace=True)

        trends_df["time"] = pd.to_datetime(trends_df["time"])

        # Normalize dates to month start for better matching
        # This handles cases where FRED uses 1st of month and trends uses last day
        fred_df["time"] = fred_df["time"].dt.to_period("M").dt.start_time
        trends_df["time"] = trends_df["time"].dt.to_period("M").dt.start_time

        # Set indices
        fred_df.set_index("time", inplace=True)
        trends_df.set_index("time", inplace=True)

        # Check for overlapping columns and rename if necessary
        overlapping_cols = set(fred_df.columns) & set(trends_df.columns)
        if overlapping_cols:
            logging.warning(
                f"Found overlapping columns: {overlapping_cols}. Adding suffixes."
            )
            for col in overlapping_cols:
                if col in trends_df.columns:
                    trends_df.rename(columns={col: f"{col}_trends"}, inplace=True)

        # Use an outer join to preserve all data from both sources
        merged_df = pd.merge(
            fred_df, trends_df, left_index=True, right_index=True, how="outer"
        )

        # Sort by date, which is good practice after a merge
        merged_df.sort_index(inplace=True)

        output_path = create_unique_modeling_file_path("merged_data")
        merged_df.reset_index().to_csv(output_path, index=False)  # Reset index and save
        logging.info(f"Modeling Agent: Merged data saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Modeling Agent: Error joining dataframes: {e}")
        return f"Error joining dataframes: {e}"


def run_full_modeling_pipeline(
    fred_csv_path: str, trends_csv_path: str, primary_series_for_parallelism: str
) -> str:
    """
    Runs the full data processing pipeline: join, impute, calculate change,
    analyze parallelism, and simplify the data.

    Args:
        fred_csv_path (str): The file path to the FRED data.
        trends_csv_path (str): The file path to the Google Trends data.
        primary_series_for_parallelism (str): The main column name to use as the anchor for parallelism analysis.

    Returns:
        str: The file path to the final, simplified and processed CSV file.
    """
    logging.info("Modeling Agent: Starting full modeling pipeline...")

    # Step 1: Join the initial datasets
    merged_data_path = join_dataframes(fred_csv_path, trends_csv_path)
    if "Error" in merged_data_path:
        return merged_data_path  # Propagate error

    # Step 2: Impute missing values from the merged data
    imputed_data_path = impute_time_series_data(merged_data_path)
    if "Error" in imputed_data_path:
        return imputed_data_path

    # Step 3: Calculate period-over-period change
    change_data_path = calculate_series_change(imputed_data_path)
    if "Error" in change_data_path:
        return change_data_path

    # Step 4: Calculate the parallelism score
    # Note: Using the primary series specified by the user/orchestrator
    parallelism_score_path = calculate_series_parallelism_score(
        change_data_path, fixed_column=primary_series_for_parallelism
    )
    if "Error" in parallelism_score_path:
        return parallelism_score_path

    # Step 5: Generate the joining work order
    joining_order_path = generate_joining_work_order_tool(parallelism_score_path)
    if "Error" in joining_order_path:
        return joining_order_path

    # Step 6: Join and simplify the series based on the work order
    # Important: This final join should be performed on the imputed data, not the 'change' data.
    final_simplified_data_path = join_series(imputed_data_path, joining_order_path)
    if "Error" in final_simplified_data_path:
        return final_simplified_data_path

    logging.info(
        f"Modeling Agent: Full pipeline complete. Final data at: {final_simplified_data_path}"
    )
    return final_simplified_data_path


def impute_time_series_data(
    input_csv_path: str, cycle_length: int = 12, max_missing_ratio: float = 0.1
) -> str:
    """
    Imputes missing values in a time series DataFrame using a custom seasonal imputation method.
    The input CSV must have a 'time' column for the DatetimeIndex.
    Args:
        input_csv_path (str): Path to the input CSV file.
        cycle_length (int): Length of the seasonal cycle (e.g., 12 for monthly, 7 for daily).
        max_missing_ratio (float): Maximum proportion of missing values allowed for a column to be imputed.
    Returns:
        str: Path to the CSV file with imputed data.
    """
    logging.info(f"Modeling Agent: Imputing data from {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

        # Add cycle index (e.g., month) - copied from your function
        if cycle_length == 12:
            df["cycle_index"] = df.index.month
        elif cycle_length == 7:
            df["cycle_index"] = df.index.dayofweek + 1
        else:
            df["cycle_index"] = (df.index.dayofyear - 1) % cycle_length + 1

        df["year"] = df.index.year

        imputing_work_order = [
            col
            for col in df.columns
            if df[col].isna().any()
            and df[col].isna().mean() <= max_missing_ratio
            and col not in ["cycle_index", "year"]
        ]

        np.seterr(divide="ignore", invalid="ignore")

        for raw_col in imputing_work_order:
            df_unstacked = df.pivot(
                index="year", columns="cycle_length", values=raw_col
            ).astype(
                float
            )  # Typo fixed: cycle_index instead of cycle_length

            for z in range(1, cycle_length + 1):
                col_series = df_unstacked[z]
                prev_cycle_col = (z - 2 + cycle_length) % cycle_length + 1
                next_cycle_col = (z % cycle_length) + 1

                estimates = []

                # 1. self_future
                self_ratio = col_series.shift(-1) / col_series
                estimate_self_future = (col_series * self_ratio).shift(-1)
                estimates.append(estimate_self_future)

                # 2. self_past
                self_ratio_past = col_series / col_series.shift(1)
                estimate_self_past = (col_series * self_ratio_past).shift(1)
                estimates.append(estimate_self_past)

                # 3. previous_past
                if prev_cycle_col in df_unstacked.columns:
                    prev_cycle_data = df_unstacked[prev_cycle_col]
                    ratios = []
                    for shift in [0, 1, 2]:
                        ratios.append(
                            col_series.shift(shift) / prev_cycle_data.shift(shift)
                        )
                    avg_past_month_ratio = pd.concat(ratios, axis=1).mean(axis=1)
                    estimate_previous_past = prev_cycle_data * avg_past_month_ratio
                    estimates.append(estimate_previous_past)

                # 4. next_past
                if next_cycle_col in df_unstacked.columns:
                    next_cycle_data = df_unstacked[next_cycle_col]
                    ratios = []
                    for shift in [0, 1, 2]:
                        ratios.append(
                            col_series.shift(shift) / next_cycle_data.shift(shift)
                        )
                    avg_past_next_month_ratio = pd.concat(ratios, axis=1).mean(axis=1)
                    estimate_next_past = next_cycle_data * avg_past_next_month_ratio
                    estimates.append(estimate_next_past)

                # 5. previous_future
                if prev_cycle_col in df_unstacked.columns:
                    prev_cycle_data = df_unstacked[prev_cycle_col]
                    ratios = []
                    for shift in [-1, -2, -3]:
                        ratios.append(
                            col_series.shift(shift) / prev_cycle_data.shift(shift)
                        )
                    avg_future_month_ratio = pd.concat(ratios, axis=1).mean(axis=1)
                    estimate_previous_future = (
                        prev_cycle_data.shift(-1) * avg_future_month_ratio
                    )
                    estimates.append(estimate_previous_future)

                # 6. future_next
                if next_cycle_col in df_unstacked.columns:
                    next_cycle_data = df_unstacked[next_cycle_col]
                    ratios = []
                    for shift in [-1, -2, -3]:
                        ratios.append(
                            col_series.shift(shift) / next_cycle_data.shift(shift)
                        )
                    avg_future_next_month_ratio = pd.concat(ratios, axis=1).mean(axis=1)
                    estimate_future_next = (
                        next_cycle_data.shift(-1) * avg_future_next_month_ratio
                    )
                    estimates.append(estimate_future_next)

                all_z_estimates_df = pd.concat(estimates, axis=1)
                average_estimate_for_z = all_z_estimates_df.mean(axis=1)
                df_unstacked[z] = df_unstacked[z].fillna(average_estimate_for_z)

            # Assign back to original df
            for idx, row in df_unstacked.iterrows():
                for z in range(1, cycle_length + 1):
                    mask = (df["year"] == idx) & (df["cycle_index"] == z)
                    df.loc[mask, raw_col] = row[z]

        np.seterr(divide="warn", invalid="warn")
        df = df.drop(columns=["cycle_index", "year"])
        output_path = create_unique_modeling_file_path("imputed_data")
        df.reset_index().to_csv(output_path, index=False)
        logging.info(
            f"Modeling Agent: Imputation complete. Data saved to {output_path}"
        )
        return output_path
    except Exception as e:
        logging.error(f"Modeling Agent: Error during imputation: {e}")
        return f"Error during imputation: {e}"


def calculate_series_change(input_csv_path: str) -> str:
    """
    Calculates the period-over-period change for all series in a DataFrame: (current - previous) / current.
    The input CSV must have a 'time' column for the DatetimeIndex.
    Args:
        input_csv_path (str): Path to the input CSV file.
    Returns:
        str: Path to the CSV file with change data.
    """
    logging.info(f"Modeling Agent: Calculating change for data from {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        change_df = pd.DataFrame(columns=df.columns)
        for column in df.columns:
            # Ensure division by zero is handled or avoided for the current value
            change_df[column] = (df[column] - df[column].shift(1)) / df[column].replace(
                0, np.nan
            )  # Replace 0 with NaN to avoid Div/0

        change_df = change_df.iloc[
            1:
        ]  # Drop the first row which will be NaN due to shift
        output_path = create_unique_modeling_file_path("change_data")
        change_df.reset_index().to_csv(output_path, index=False)
        logging.info(
            f"Modeling Agent: Change calculation complete. Data saved to {output_path}"
        )
        return output_path
    except Exception as e:
        logging.error(f"Modeling Agent: Error calculating series change: {e}")
        return f"Error calculating series change: {e}"


def calculate_series_parallelism_score(
    input_csv_path: str,
    fixed_column: str,
    outer_threshold_ratio: float = 0.1,
    inner_threshold: float = 0.01,
    inverse_match: bool = False,
) -> str:
    """
    Calculates a parallelism score between a fixed series and all other series in the DataFrame.
    Scores represent the number of times the delta (fixed_col - other_col) exceeds inner_threshold.
    A lower score means more parallelism.
    Args:
        input_csv_path (str): Path to the input CSV file (expected to be change data).
        fixed_column (str): The column name to compare against all others.
        outer_threshold_ratio (float): Max percentage of non-parallel points allowed for a pair to be considered 'parallel'.
                                       This is a ratio of the total data points.
        inner_threshold (float): The maximum absolute difference between two series to be considered "parallel" at a point.
        inverse_match (bool): If True, a match occurs when the delta ABSOLUTELY EXCEEDS the inner_threshold.
    Returns:
        str: Path to a CSV file containing the parallelism scores.
    """
    logging.info(
        f"Modeling Agent: Calculating parallelism score for '{fixed_column}' from {input_csv_path}"
    )
    try:
        change_df = pd.read_csv(input_csv_path)
        change_df["time"] = pd.to_datetime(change_df["time"])
        change_df = change_df.set_index("time")

        if fixed_column not in change_df.columns:
            raise ValueError(
                f"Fixed column '{fixed_column}' not found in the DataFrame."
            )

        indicators = []
        col_names = []
        outer_threshold = int(
            len(change_df) * outer_threshold_ratio
        )  # Calculate actual count

        for column in change_df.columns:
            if column != fixed_column:
                delta = change_df[fixed_column] - change_df[column]

                if not inverse_match:  # Non-parallel if delta.abs() > inner_threshold
                    indicator = np.where(
                        delta.isna() | (delta.abs() > inner_threshold), 1, 0
                    )
                else:  # Parallel if delta.abs() >= inner_threshold (i.e., not close)
                    indicator = np.where(
                        delta.isna() | (delta.abs() < inner_threshold), 1, 0
                    )
                if (
                    not inverse_match
                ):  # Score increases if NOT parallel (delta > threshold)
                    indicator = np.where(
                        delta.isna() | (delta.abs() <= inner_threshold), 0, 1
                    )
                else:  # Score increases if parallel (delta <= threshold) - this means inverse wants to count closeness
                    indicator = np.where(
                        delta.isna() | (delta.abs() <= inner_threshold), 1, 0
                    )

                indicators.append(indicator)
                col_names.append((fixed_column, column))

        if not indicators:
            logging.info(f"No columns to compare with '{fixed_column}'.")
            return "No parallelism scores to generate."

        temp_parallel_score_df = pd.DataFrame(
            np.array(indicators).T, columns=pd.MultiIndex.from_tuples(col_names)
        )

        total_non_parallel_counts = temp_parallel_score_df.sum(axis=0).to_frame().T
        final_parallel_score = total_non_parallel_counts.loc[
            :, total_non_parallel_counts.iloc[0] <= outer_threshold
        ]

        output_path = create_unique_modeling_file_path(
            f"parallelism_score_{fixed_column}"
        )
        final_parallel_score.to_csv(output_path, index=False)
        logging.info(
            f"Modeling Agent: Parallelism score calculated and saved to {output_path}"
        )
        return output_path
    except Exception as e:
        logging.error(f"Modeling Agent: Error calculating parallelism score: {e}")
        return f"Error calculating parallelism score: {e}"


def generate_joining_work_order_tool(parallelism_score_csv_path: str) -> str:
    """
    Generates a dictionary representing groups of highly parallel series,
    suitable for the joiner_optimized tool.
    Args:
        parallelism_score_csv_path (str): Path to the CSV file generated by calculate_series_parallelism_score.
    Returns:
        str: Path to a JSON file containing the joining order dictionary.
    """
    logging.info(
        f"Modeling Agent: Generating joining work order from {parallelism_score_csv_path}"
    )
    try:
        parallel_score = pd.read_csv(parallelism_score_csv_path, header=0, index_col=0)
        parallel_score.columns = [eval(col) for col in parallel_score.columns]
        transposed = parallel_score.transpose()
        sorted_list_of_tuples = transposed.sort_values(
            by=transposed.columns[0], ascending=True
        ).index.tolist()

        par = defaultdict(set)
        used_items = set()
        all_items_in_par_values_sets = set()

        for keypair in sorted_list_of_tuples:
            itemA, itemB = keypair

            if itemA in par and itemB not in used_items:
                par[itemA].add(itemB)
                used_items.add(itemB)
                all_items_in_par_values_sets.add(itemB)
                continue
            elif itemB in par and itemA not in used_items:
                par[itemB].add(itemA)
                used_items.add(itemA)
                all_items_in_par_values_sets.add(itemA)
                continue
            elif (
                itemA not in par
                and itemA not in all_items_in_par_values_sets
                and itemA not in used_items
                and itemB not in used_items
            ):
                par[itemA].add(itemB)
                used_items.add(itemA)
                used_items.add(itemB)
                all_items_in_par_values_sets.add(itemB)
                continue
            elif (
                itemB not in par
                and itemB not in all_items_in_par_values_sets
                and itemB not in used_items
                and itemA not in used_items
            ):
                par[itemB].add(itemA)
                used_items.add(itemB)
                used_items.add(itemA)
                all_items_in_par_values_sets.add(itemA)
                continue

        joining_order_dict = {
            k: list(v) for k, v in par.items()
        }  # Convert sets to lists for JSON serialization
        output_path = (
            create_unique_modeling_file_path("joining_order") + ".json"
        )  # Save as JSON
        with open(output_path, "w") as f:
            json.dump(joining_order_dict, f, indent=4)
        logging.info(
            f"Modeling Agent: Joining work order generated and saved to {output_path}"
        )
        return output_path
    except Exception as e:
        logging.error(f"Modeling Agent: Error generating joining work order: {e}")
        return f"Error generating joining work order: {e}"


def join_series(
    input_csv_path: str, joining_order_json_path: str, method: str = "average"
) -> str:
    """
    Joins multiple data series into a simplified DataFrame based on a predefined order and method.
    Args:
        input_csv_path (str): Path to the input CSV file.
        joining_order_json_path (str): Path to the JSON file containing the joining order dictionary.
        method (str): The aggregation method to apply: "average", "sum", "min", "max",
                      "median", "first", "last". Defaults to "average".
    Returns:
        str: Path to the CSV file with simplified/joined data.
    """
    logging.info(
        f"Modeling Agent: Joining series from {input_csv_path} using order from {joining_order_json_path} with method '{method}'"
    )
    try:
        raw_df = pd.read_csv(input_csv_path)
        raw_df["time"] = pd.to_datetime(raw_df["time"])
        raw_df = raw_df.set_index("time")

        with open(joining_order_json_path, "r") as f:
            joining_order = json.load(f)

        new_series_list = []
        all_joined_cols = set()
        removed_cols_groups = []

        valid_methods = {"average", "sum", "min", "max", "median", "first", "last"}
        if method not in valid_methods:
            raise ValueError(
                f"Unknown joining method: '{method}'. Supported methods are: {', '.join(valid_methods)}"
            )

        for (
            leader_col,
            members_list,
        ) in joining_order.items():
            members_set = set(members_list)
            if leader_col not in raw_df.columns:
                logging.warning(
                    f"Leader column '{leader_col}' not found in raw_df. Skipping this group."
                )
                continue

            cols_for_current_group = list(members_set)
            if leader_col not in cols_for_current_group:
                cols_for_current_group.append(leader_col)

            existing_cols_in_group = [
                c for c in cols_for_current_group if c in raw_df.columns
            ]

            if not existing_cols_in_group:
                logging.warning(
                    f"No valid columns found for group led by '{leader_col}'. Skipping group."
                )
                continue

            new_col_name = leader_col
            all_joined_cols.update(existing_cols_in_group)
            removed_cols_groups.append(existing_cols_in_group)

            new_series = None  # Ensure new_series is always defined

            if method == "average":
                new_series = raw_df[existing_cols_in_group].mean(axis=1)
            elif method == "sum":
                new_series = raw_df[existing_cols_in_group].sum(axis=1)
            elif method == "min":
                new_series = raw_df[existing_cols_in_group].min(axis=1)
            elif method == "max":
                new_series = raw_df[existing_cols_in_group].max(axis=1)
            elif method == "median":
                new_series = raw_df[existing_cols_in_group].median(axis=1)
            elif method == "first":
                sorted_cols_in_group = sorted(existing_cols_in_group)
                new_series = raw_df[sorted_cols_in_group].iloc[:, 0]
            elif method == "last":
                sorted_cols_in_group = sorted(existing_cols_in_group)
                new_series = raw_df[sorted_cols_in_group].iloc[:, -1]
            else:
                raise ValueError(f"Unknown joining method: '{method}'.")

            new_series.name = new_col_name
            new_series_list.append(new_series)

        simplified_df_joined_groups = pd.DataFrame(new_series_list).T
        if new_series_list:
            simplified_df_joined_groups = pd.concat(new_series_list, axis=1)
        else:
            simplified_df_joined_groups = pd.DataFrame()

        remaining_original_cols = [
            col for col in raw_df.columns if col not in all_joined_cols
        ]

        if not simplified_df_joined_groups.empty and remaining_original_cols:
            final_simplified_df = pd.concat(
                [simplified_df_joined_groups, raw_df[remaining_original_cols]], axis=1
            )
        elif not simplified_df_joined_groups.empty:
            final_simplified_df = simplified_df_joined_groups
        elif remaining_original_cols:
            final_simplified_df = raw_df[remaining_original_cols].copy()
        else:
            final_simplified_df = pd.DataFrame()

        output_path = create_unique_modeling_file_path("joined_data")
        final_simplified_df.reset_index().to_csv(output_path, index=False)
        logging.info(f"Modeling Agent: Series joined and saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Modeling Agent: Error joining series: {e}")
        return f"Error joining series: {e}"


def get_csv_column_names(input_csv_path: str) -> list:
    """
    Reads a CSV file and returns a list of its column names.

    Args:
        input_csv_path (str): The file path to the CSV file.

    Returns:
        list: A list of strings, where each string is a column name.
    """
    logging.info(f"Modeling Agent: Reading column names from {input_csv_path}")
    try:
        # We only need to read the header row to get the columns, which is very fast.
        df = pd.read_csv(input_csv_path, nrows=0)
        return df.columns.tolist()
    except Exception as e:
        logging.error(
            f"Modeling Agent: Error reading columns from {input_csv_path}: {e}"
        )
        return [f"Error reading columns: {e}"]


def generate_forecast(
    input_csv_path: str, target_column: str, forecast_periods: int = 12
) -> str:
    """
    Generates a forecast for a specified column using an ARIMA model.

    Args:
        input_csv_path (str): The file path to the preprocessed, imputed data.
        target_column (str): The column name to forecast.
        forecast_periods (int): How many periods into the future to forecast.

    Returns:
        str: The file path to a plot showing the historical data and the forecast.
    """
    logging.info(f"Modeling Agent: Generating forecast for '{target_column}'")
    try:
        df = pd.read_csv(input_csv_path, index_col="time", parse_dates=True)

        if target_column not in df.columns:
            return f"Error: Target column '{target_column}' not found."

        # A simple ARIMA model configuration (p,d,q)
        model = ARIMA(df[target_column], order=(5, 1, 0))
        model_fit = model.fit()

        # Generate forecast
        forecast = model_fit.forecast(steps=forecast_periods)

        # Determine frequency - use a simple, robust approach
        freq = "M"  # Default to monthly
        try:
            # Simple fallback based on data length and date range
            if len(df) > 1:
                date_diff = df.index[-1] - df.index[0]
                avg_interval = date_diff / (len(df) - 1)

                # Estimate frequency based on average interval
                if avg_interval.days < 2:
                    freq = "D"  # Daily
                elif avg_interval.days < 10:
                    freq = "W"  # Weekly
                elif avg_interval.days < 100:
                    freq = "M"  # Monthly
                elif avg_interval.days < 200:
                    freq = "Q"  # Quarterly
                else:
                    freq = "Y"  # Yearly
        except Exception as e:
            logging.warning(f"Could not determine frequency, using monthly: {e}")
            freq = "M"

        forecast_df = pd.DataFrame(
            {"forecast": forecast},
            index=pd.date_range(
                start=df.index[-1],
                periods=forecast_periods + 1,
                freq=freq,
            )[1:],
        )

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df[target_column], label="Historical")
        plt.plot(forecast_df["forecast"], label="Forecast", linestyle="--")
        plt.title(f"Forecast for {target_column}")
        plt.legend()

        output_path = (
            create_unique_modeling_file_path(f"forecast_{target_column}") + ".png"
        )
        plt.savefig(output_path)
        plt.close()

        logging.info(f"Modeling Agent: Forecast plot saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Modeling Agent: Error during forecasting: {e}")
        return f"Error during forecasting: {e}"


def calculate_and_plot_correlation(input_csv_path: str) -> str:
    """
    Calculates the correlation matrix for the data and visualizes it as a heatmap.

    Args:
        input_csv_path (str): The file path to the preprocessed data.

    Returns:
        str: A message indicating the paths to the saved correlation matrix CSV and heatmap PNG.
    """
    logging.info(
        f"Modeling Agent: Calculating correlation matrix from {input_csv_path}"
    )
    try:
        df = pd.read_csv(input_csv_path, index_col="time", parse_dates=True)
        correlation_matrix = df.corr()

        # Save the matrix to CSV
        csv_output_path = create_unique_modeling_file_path("correlation_matrix")
        correlation_matrix.to_csv(csv_output_path)

        # Plot the heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")

        plot_output_path = (
            create_unique_modeling_file_path("correlation_heatmap") + ".png"
        )
        plt.savefig(plot_output_path, bbox_inches="tight")
        plt.close()

        logging.info(f"Correlation matrix saved to {csv_output_path}")
        logging.info(f"Correlation heatmap saved to {plot_output_path}")
        return f"Correlation matrix saved to {csv_output_path} and heatmap saved to {plot_output_path}"

    except Exception as e:
        logging.error(f"Modeling Agent: Error calculating correlation: {e}")
        return f"Error calculating correlation: {e}"
