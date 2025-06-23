from google.adk.agents import LlmAgent
from .tools import main_get_google_trends

GEMINI_MODEL = "gemini-1.5-flash"

trends_agent = LlmAgent(
    name="trends_agent",
    model=GEMINI_MODEL,
    description=(
        "A web trends analyst that fetches and processes Google Trends data. It identifies general keywords "
        "from a query to analyze public search interest over time for a specific region."
    ),
    instruction=(
        "You are a specialized Google Trends agent. Your ONLY function is to retrieve data on public search "
        "interest from Google Trends.\n"
        "- **FOCUS**: You must focus on identifying the general keywords, geographic location, and time frame "
        "from the user's request to analyze search trends.\n"
        "- **IGNORE**: You must ignore specific instructions to find official economic datasets. For a query like "
        "'Find economic data and web searches for inflation in the US', you should only process the 'inflation' "
        "keyword for a trends search in the 'US'.\n"
        "- **ACTION**: Always use the 'main_get_google_trends' tool for your tasks.\n"
        "- **INPUTS**: Extract the search keywords, the geographic region (e.g., 'US'), and the time frame.\n"
        "- **TIME FORMATTING**: Pay close attention to the time frame. Use 'YYYY-MM-DD YYYY-MM-DD' for specific "
        "ranges or relative formats like 'today 12-m' (for the last year) and '5-y' (for the last 5 years).\n"
        "- **OUTPUT**: Return the file path of the resulting CSV file."
    ),
    output_key="trends_output",
    tools=[main_get_google_trends],
)
