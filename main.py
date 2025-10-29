import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Web search tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

# Use Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# agent = create_deep_agent(
#     tools=[internet_search],
#     system_prompt=research_instructions,
#     model=llm,
# )

# # Invoke the agent
# result = agent.invoke({"messages": [{"role": "user", "content": "What big has happend in Paris recently?"}]})
# print(result["messages"][-1].content)

#Tools: Do one specific acton (what?)
#Middleware: Add behaviour/state around the agent

from langchain.agents.middleware import AgentMiddleware

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

@tool
def get_temperature(city: str) -> str:
    """Get the temperature in a city."""
    return f"The temperature in {city} is 70 degrees Fahrenheit."

class WeatherMiddleware(AgentMiddleware):
  tools = [get_weather, get_temperature]

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions,
    model=llm,
    middleware=[WeatherMiddleware()]
)

# Invoke the agent with weather tools
result1 = agent.invoke({"messages": [{"role": "user", "content": "What big has happend in Paris recently and what is the temperature there?"}]})
print(result1["messages"][-1].content)