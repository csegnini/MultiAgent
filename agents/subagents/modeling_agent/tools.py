import pandas as pd
import numpy as np
import os
import json
from typing import List
from collections import defaultdict
import logging
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_BASES_DIR = "data_bases"
os.makedirs(DATA_BASES_DIR, exist_ok=True)


def create_unique_modeling_file_path(
    base_file_name: str, extension: str = ".csv"
) -> str:
    """Creates a unique file path to avoid overwriting existing files."""
    os.makedirs(DATA_BASES_DIR, exist_ok=True)
    full_file_name = f"{base_file_name}{extension}"
    file_path = os.path.join(DATA_BASES_DIR, full_file_name)
    counter = 1
    while os.path.exists(file_path):
        full_file_name = f"{base_file_name}_{counter}{extension}"
        file_path = os.path.join(DATA_BASES_DIR, full_file_name)
        counter += 1
    return file_path


def join_dataframes(fred_csv_path: str, trends_csv_path: str) -> str:
    """Joins the economic data from FRED and the web data from Google Trends."""
    logging.info(
        f"Modeling Agent: Joining data from {fred_csv_path} and {trends_csv_path}"
    )
    try:
        fred_df = pd.read_csv(fred_csv_path)
        fred_df["time"] = pd.to_datetime(fred_df["time"])
        trends_df = pd.read_csv(trends_csv_path)
        date_columns = ["Date", "date", "time", "Time"]
        date_col = next((col for col in date_columns if col in trends_df.columns), None)
        if date_col is None:
            raise ValueError(f"No date column found in trends data.")
        if date_col != "time":
            trends_df.rename(columns={date_col: "time"}, inplace=True)
        trends_df["time"] = pd.to_datetime(trends_df["time"])
        fred_df["time"] = fred_df["time"].dt.to_period("M").dt.start_time
        trends_df["time"] = trends_df["time"].dt.to_period("M").dt.start_time
        fred_df.set_index("time", inplace=True)
        trends_df.set_index("time", inplace=True)
        merged_df = pd.merge(
            fred_df, trends_df, left_index=True, right_index=True, how="outer"
        )
        merged_df.sort_index(inplace=True)
        output_path = create_unique_modeling_file_path("merged_data")
        merged_df.to_csv(output_path)
        logging.info(f"Modeling Agent: Merged data saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Modeling Agent: Error joining dataframes: {e}")
        return f"Error joining dataframes: {e}"


def impute_time_series_data(
    input_csv_path: str, cycle_length: int = 12, max_missing: float = 0.1
) -> str:
    """
    Imputes missing values in a time series DataFrame using your custom seasonal imputation method.
    """
    logging.info(
        f"Modeling Agent: Imputing data from {input_csv_path} using custom seasonal imputer."
    )
    try:
        df = pd.read_csv(input_csv_path, index_col="time", parse_dates=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

        if cycle_length == 12:
            df["cycle_index"] = df.index.month
        else:
            df["cycle_index"] = (df.index.dayofyear - 1) % cycle_length + 1
        df["year"] = df.index.year

        imputing_work_order = [
            col
            for col in df.columns
            if df[col].isna().any()
            and df[col].isna().mean() <= max_missing
            and col not in ["cycle_index", "year"]
        ]
        np.seterr(divide="ignore", invalid="ignore")
        for raw_col in imputing_work_order:
            df_unstacked = df.pivot(
                index="year", columns="cycle_index", values=raw_col
            ).astype(float)
            for z in range(1, cycle_length + 1):
                if z not in df_unstacked.columns:
                    continue
                col_series = df_unstacked[z]
                prev_cycle_col = (z - 2 + cycle_length) % cycle_length + 1
                next_cycle_col = (z % cycle_length) + 1
                estimates = []
                # Ratio-based estimates
                if not col_series.empty:
                    # Self-future/past
                    estimates.append(
                        (col_series * (col_series.shift(-1) / col_series)).shift(-1)
                    )
                    estimates.append(
                        (col_series * (col_series / col_series.shift(1))).shift(1)
                    )
                    # Neighboring cycles
                    for neighbor_col_idx in [prev_cycle_col, next_cycle_col]:
                        if neighbor_col_idx in df_unstacked.columns:
                            neighbor_series = df_unstacked[neighbor_col_idx]
                            for shift_val in [0, 1, 2, -1, -2, -3]:
                                ratio = col_series.shift(
                                    shift_val
                                ) / neighbor_series.shift(shift_val)
                                estimate = neighbor_series * ratio.mean()
                                estimates.append(estimate)

                if estimates:
                    all_z_estimates_df = pd.concat(estimates, axis=1)
                    average_estimate_for_z = all_z_estimates_df.mean(axis=1)
                    df_unstacked[z] = df_unstacked[z].fillna(average_estimate_for_z)

            for idx, row in df_unstacked.iterrows():
                for z in range(1, cycle_length + 1):
                    if z in row:
                        mask = (df["year"] == idx) & (df["cycle_index"] == z)
                        df.loc[mask, raw_col] = row[z]

        np.seterr(divide="warn", invalid="warn")
        df_imputed = df.drop(columns=["cycle_index", "year"])
        output_path = create_unique_modeling_file_path("imputed_data")
        df_imputed.to_csv(output_path)
        logging.info(
            f"Modeling Agent: Imputation complete. Data saved to {output_path}"
        )
        return output_path
    except Exception as e:
        logging.error(f"Modeling Agent: Error during imputation: {e}")
        return f"Error during imputation: {e}"


def calculate_series_change(input_csv_path: str) -> str:
    """Calculates the period-over-period percentage change for all series in a DataFrame."""
    logging.info(f"Modeling Agent: Calculating change for data from {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path, index_col="time", parse_dates=True)
        change_df = df.pct_change().iloc[1:]
        output_path = create_unique_modeling_file_path("change_data")
        change_df.to_csv(output_path)
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
) -> str:
    """Calculates a parallelism score between a fixed series and all other series."""
    logging.info(f"Modeling Agent: Calculating parallelism score for '{fixed_column}'")
    try:
        change_df = pd.read_csv(input_csv_path, index_col="time", parse_dates=True)
        if fixed_column not in change_df.columns:
            raise ValueError(f"Fixed column '{fixed_column}' not found.")

        outer_threshold = int(len(change_df) * outer_threshold_ratio)
        scores = {}
        for other_col in change_df.columns:
            if fixed_column == other_col:
                continue
            delta = change_df[fixed_column] - change_df[other_col]
            non_parallel_points = np.sum(np.where(delta.abs() > inner_threshold, 1, 0))
            if non_parallel_points <= outer_threshold:
                scores[(fixed_column, other_col)] = non_parallel_points

        if not scores:
            return f"No series found to be parallel with '{fixed_column}'."

        score_df = pd.DataFrame.from_dict(scores, orient="index", columns=["score"])
        output_path = create_unique_modeling_file_path(
            f"parallelism_score_{fixed_column}"
        )
        score_df.to_csv(output_path)
        return output_path
    except Exception as e:
        return f"Error calculating parallelism score for {fixed_column}: {e}"


def generate_joining_work_order_tool(score_files: List[str]) -> str:
    """Generates a dictionary work order for joining series from multiple score files."""
    logging.info("Modeling Agent: Generating joining work order.")
    try:
        all_scores_df = pd.concat(
            [pd.read_csv(f, index_col=[0, 1]) for f in score_files]
        )
        all_scores_df = all_scores_df.loc[~all_scores_df.index.duplicated(keep="first")]
        sorted_scores = all_scores_df.sort_values(by="score", ascending=True)
        sorted_list_of_tuples = sorted_scores.index.tolist()

        par = defaultdict(set)
        used_items = set()
        for itemA, itemB in sorted_list_of_tuples:
            if itemA in used_items and itemB in used_items:
                continue

            found_group = False
            for leader, members in par.items():
                if itemA == leader or itemA in members:
                    if itemB not in used_items:
                        par[leader].add(itemB)
                        used_items.add(itemB)
                    found_group = True
                    break
                if itemB == leader or itemB in members:
                    if itemA not in used_items:
                        par[leader].add(itemA)
                        used_items.add(itemA)
                    found_group = True
                    break

            if not found_group:
                par[itemA].add(itemB)
                used_items.add(itemA)
                used_items.add(itemB)

        joining_order_dict = {k: list(v) for k, v in par.items()}
        output_path = create_unique_modeling_file_path(
            "joining_order", extension=".json"
        )
        with open(output_path, "w") as f:
            json.dump(joining_order_dict, f, indent=4)
        return output_path
    except Exception as e:
        return f"Error generating joining work order: {e}"


def join_series(
    input_csv_path: str, joining_order_json_path: str, method: str = "average"
) -> str:
    """Joins multiple data series into a simplified DataFrame based on a predefined work order."""
    logging.info(f"Modeling Agent: Joining series from {input_csv_path}")
    try:
        raw_df = pd.read_csv(input_csv_path, index_col="time", parse_dates=True)
        with open(joining_order_json_path, "r") as f:
            joining_order = json.load(f)

        simplified_df = raw_df.copy()
        all_removed_cols = set()
        for leader_col, members_list in joining_order.items():
            cols_to_join = [leader_col] + members_list
            existing_cols = [
                col for col in cols_to_join if col in simplified_df.columns
            ]
            if not existing_cols:
                continue

            simplified_df[leader_col] = simplified_df[existing_cols].mean(axis=1)

            cols_to_remove = [col for col in existing_cols if col != leader_col]
            all_removed_cols.update(cols_to_remove)

        final_df = simplified_df.drop(columns=list(all_removed_cols))
        output_path = create_unique_modeling_file_path("final_simplified_data")
        final_df.to_csv(output_path)
        return output_path
    except Exception as e:
        return f"Error joining series: {e}"


def run_full_modeling_pipeline(fred_csv_path: str, trends_csv_path: str) -> str:
    """Runs the full data processing pipeline to find and group similar time series."""
    logging.info("Modeling Agent: Starting full modeling pipeline...")

    # Step 1: Join
    merged_data_path = join_dataframes(fred_csv_path, trends_csv_path)
    if "Error" in merged_data_path:
        return merged_data_path

    # Step 2: Impute
    imputed_data_path = impute_time_series_data(merged_data_path)
    if "Error" in imputed_data_path:
        return imputed_data_path

    # Step 3: Calculate Change
    change_data_path = calculate_series_change(imputed_data_path)
    if "Error" in change_data_path:
        return change_data_path

    # Step 4: Calculate all-pairs parallelism score by looping
    columns = get_csv_column_names(change_data_path)
    if "time" in columns:
        columns.remove("time")

    score_files = []
    for col in columns:
        score_file = calculate_series_parallelism_score(
            change_data_path, fixed_column=col
        )
        if "Error" not in score_file and "No series found" not in score_file:
            score_files.append(score_file)

    if not score_files:
        logging.warning(
            "No parallel series were found. Returning the imputed data without simplification."
        )
        return imputed_data_path

    # Step 5: Generate Joining Work Order
    joining_order_path = generate_joining_work_order_tool(score_files)
    if "Error" in joining_order_path:
        return joining_order_path

    # Step 6: Join Series
    final_simplified_data_path = join_series(imputed_data_path, joining_order_path)
    if "Error" in final_simplified_data_path:
        return final_simplified_data_path

    logging.info(
        f"Modeling Agent: Full pipeline complete. Final simplified data at: {final_simplified_data_path}"
    )
    return final_simplified_data_path


def get_csv_column_names(input_csv_path: str) -> list:
    """Reads a CSV file and returns a list of its column names."""
    try:
        df = pd.read_csv(input_csv_path, nrows=0)
        return df.columns.tolist()
    except Exception as e:
        return [f"Error reading columns: {e}"]


def calculate_and_plot_correlation(input_csv_path: str) -> str:
    """Calculates the correlation matrix and visualizes it as a heatmap."""
    logging.info(
        f"Modeling Agent: Calculating correlation matrix from {input_csv_path}"
    )
    try:
        df = pd.read_csv(input_csv_path, index_col="time", parse_dates=True)
        correlation_matrix = df.corr()

        csv_output_path = create_unique_modeling_file_path("correlation_matrix")
        correlation_matrix.to_csv(csv_output_path)

        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plot_output_path = create_unique_modeling_file_path(
            "correlation_heatmap", extension=".png"
        )
        plt.savefig(plot_output_path, bbox_inches="tight")
        plt.close()
        return f"Correlation matrix saved to {csv_output_path} and heatmap saved to {plot_output_path}"
    except Exception as e:
        return f"Error calculating correlation: {e}"
