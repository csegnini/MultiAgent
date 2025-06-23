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
        "1. **INITIAL STEP: INTRODUCE ANALYSIS OPTIONS**\n"
        "     Start by greeting the user\n"
        "   - Then, you MUST use the `run_full_modeling_pipeline` tool to merge the FRED and Trends data files.\n"
        "   - Once merged, you MUST use the `get_csv_column_names` tool on the newly created file to get a list of all available data columns.\n"
        "   - Show them the list of available columns and offer the following analysis options:\n"
        "     - `Correlation Matrix`: To visualize the statistical correlation between all series.\n"
        "   - Ask the user to choose one option.\n\n"
        "2. **THIRD STEP: EXECUTE THE CHOSEN ANALYSIS**\n"
        "   - Now, using the **final, processed file** from the pipeline, execute the analysis the user originally chose.\n"
        "   - **If 'Correlation'**: Use the `calculate_and_plot_correlation` tool on the final file from the pipeline.\n"
        "   - **If 'Causality Analysis'**: This option seems to be missing a specific tool in your `tools.py`. You would need to implement a tool for this analysis.\n\n"
        "3. **FINAL STEP: REPORT**\n"
        "   - Report the results back to the user, providing the file paths of any generated artifacts (plots or data files)."
    ),
    tools=[
        join_dataframes,
        get_csv_column_names,
        run_full_modeling_pipeline,
        calculate_and_plot_correlation,
        impute_time_series_data,
        calculate_series_change,
        calculate_series_parallelism_score,
        generate_joining_work_order_tool,
        join_series,
    ],
    output_key="modeling_output",
)
