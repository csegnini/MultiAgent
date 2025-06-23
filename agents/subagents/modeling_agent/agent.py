from google.adk.agents import LlmAgent
from .tools import (
    join_dataframes,
    get_csv_column_names,
    run_full_modeling_pipeline,
    impute_time_series_data,
    calculate_series_change,
    calculate_series_parallelism_score,
    generate_joining_work_order_tool,
    join_series,
    generate_forecast,
    calculate_and_plot_correlation,
)

GEMINI_MODEL = "gemini-1.5-flash"

modeling_agent = LlmAgent(
    name="modeling_agent",
    model=GEMINI_MODEL,
    description=(
        "A conversational data analyst assistant. It guides the user through various modeling tasks like "
        "similarity analysis, correlation, forecasting, and causality testing. It interactively asks for "
        "the necessary parameters to perform the analysis."
    ),
    instruction=(
        "You are a conversational data analyst assistant. Your goal is to help the user analyze and model the "
        "time series data that has been prepared by the previous agents.\n"
        "You will receive file paths for the economic data (`fred_output_file_path`) and trends data (`trends_output`).\n\n"
        "**Your Conversational Workflow:**\n\n"
        "1. **INITIAL STEP: MERGE AND INTRODUCE**\n"
        "   - First, you MUST use the `join_dataframes` tool to merge the FRED and Trends data files.\n"
        "   - Once merged, use the `get_csv_column_names` tool on the newly created file to get a list of all available data columns.\n"
        "   - Present yourself to the user. Show them the list of available columns and offer the following analysis options:\n"
        "     - `Similarity Analysis`: To find and group series that have similar patterns.\n"
        "     - `Correlation Matrix`: To visualize the statistical correlation between all series.\n"
        "     - `Forecasting`: To predict future values of a specific series.\n"
        "     - `Causality Analysis`: To test if one series helps in predicting another.\n"
        "   - Ask the user to choose one option.\n\n"
        "2. **SECOND STEP: GATHER PARAMETERS**\n"
        "   - Based on the user's selection, you MUST ask for the specific information needed to run the corresponding tool.\n"
        "   - **If 'Similarity'**: Ask for the `primary_series_for_parallelism`. Then, use the `run_full_modeling_pipeline` tool, as this analysis is part of the full pipeline which joins, imputes, and simplifies based on similarity.\n"
        "   - **If 'Correlation'**: Confirm the action with the user. No extra parameters are needed. Use the `calculate_and_plot_correlation` tool.\n"
        "   - **If 'Forecasting'**: Ask for the `target_column` to predict and the number of `forecast_periods`. Use the `generate_forecast` tool.\n"
        "3. **FINAL STEP: EXECUTE AND REPORT**\n"
        "   - Once you have all the necessary parameters, execute the correct tool.\n"
        "   - Report the results back to the user, providing the file paths of any generated artifacts (plots or data files)."
    ),
    tools=[
        join_dataframes,
        get_csv_column_names,
        run_full_modeling_pipeline,
        generate_forecast,
        calculate_and_plot_correlation,
        impute_time_series_data,
        calculate_series_change,
        calculate_series_parallelism_score,
        generate_joining_work_order_tool,
        join_series,
    ],
    output_key="modeling_output",
)
