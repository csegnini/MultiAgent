import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from agents.subagents.fred_agent import fred_agent
from agents.subagents.trends_agent import trends_agent
from agents.subagents.modeling_agent import modeling_agent

load_dotenv()
GEMINI_MODEL = "gemini-1.5-flash"
FRED_API_KEY = os.getenv("FRED_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")


data_aggregator = ParallelAgent(
    name="data_aggregator", sub_agents=[fred_agent, trends_agent]
)

workflow_agent = SequentialAgent(
    name="workflow_agent",
    sub_agents=[
        data_aggregator,
        modeling_agent,
    ],
)


instruction = """
You are a manager agent. Your role is to coordinate the work of your sub-agents.
You will receive a request from the user, and you must delegate it to the appropriate sub-agent.
You must delegate the request to the `workflow_agent`.

"""

root_agent = LlmAgent(
    name="manager_agent",
    model=GEMINI_MODEL,
    instruction=instruction,
    sub_agents=[workflow_agent],
)
