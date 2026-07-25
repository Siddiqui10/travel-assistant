"""
tools.py
---------
Custom tools that the CrewAI agents can call. This is the "Tool Calling" part
of the assignment:

- TravelRAGSearchTool -> semantic search over the ingested travel PDFs (Agentic RAG)
- WeatherTool          -> live weather lookup for a city (wttr.in, no API key needed)
- BudgetEstimatorTool  -> rough trip cost calculator based on days/people/style
"""

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag_engine import TravelRAGEngine

# one shared RAG engine instance so we don't reload the embedding model per tool call
_rag_engine = TravelRAGEngine()


# ---------------------------------------------------------------------
# 1. RAG SEARCH TOOL
# ---------------------------------------------------------------------
class RAGSearchInput(BaseModel):
    query: str = Field(..., description="The travel-related question to search for in the uploaded guides")


class TravelRAGSearchTool(BaseTool):
    name: str = "travel_knowledge_search"
    description: str = (
        "Searches the uploaded travel guide PDFs (destinations, attractions, local "
        "guidelines, transport info) using semantic search. Always use this before "
        "answering a factual question about a destination. Returns matching text "
        "along with the source PDF and page number."
    )
    args_schema: type[BaseModel] = RAGSearchInput

    def _run(self, query: str) -> str:
        hits = _rag_engine.search(query)
        context, sources = _rag_engine.format_context(hits)
        if not sources:
            return "No matching information found in the travel guide documents."
        return f"{context}\n\nSOURCES USED: {', '.join(sources)}"


# ---------------------------------------------------------------------
# 2. WEATHER TOOL
# ---------------------------------------------------------------------
class WeatherInput(BaseModel):
    city: str = Field(..., description="City or destination name to check current weather for")


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = (
        "Fetches the current weather (temperature, condition, humidity) for a given city. "
        "Use this whenever the user asks about weather or when it is needed to plan an itinerary."
    )
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, city: str) -> str:
        try:
            # wttr.in - free, no API key, returns plain text weather summary
            url = f"https://wttr.in/{city}?format=%C+%t+humidity:+%h"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200 and resp.text.strip():
                return f"Current weather in {city}: {resp.text.strip()}"
            return f"Could not fetch live weather for {city} right now."
        except requests.exceptions.RequestException as e:
            return f"Weather service unavailable ({e}). Please check manually before travelling."


# ---------------------------------------------------------------------
# 3. BUDGET ESTIMATOR TOOL
# ---------------------------------------------------------------------
class BudgetInput(BaseModel):
    destination: str = Field(..., description="Destination name")
    days: int = Field(..., description="Number of days of the trip")
    people: int = Field(default=1, description="Number of travellers")
    style: str = Field(default="mid-range", description="budget / mid-range / luxury")


# very rough per-day cost table in INR, used only as a starting estimate.
# a real product would pull this from a pricing API - kept simple for the project.
_DAILY_COST_INR = {
    "budget": 2000,
    "mid-range": 5000,
    "luxury": 12000,
}


class BudgetEstimatorTool(BaseTool):
    name: str = "estimate_budget"
    description: str = (
        "Estimates an approximate total trip budget (in INR) given the destination, "
        "number of days, number of people and travel style (budget/mid-range/luxury). "
        "Covers stay + food + local travel, NOT flights/trains to reach the destination."
    )
    args_schema: type[BaseModel] = BudgetInput

    def _run(self, destination: str, days: int, people: int = 1, style: str = "mid-range") -> str:
        style_key = style.lower().strip()
        per_day = _DAILY_COST_INR.get(style_key, _DAILY_COST_INR["mid-range"])

        per_person_total = per_day * days
        grand_total = per_person_total * people

        breakdown = (
            f"Budget estimate for {destination} ({days} days, {people} traveller(s), {style} style):\n"
            f"- Approx. cost per person per day: Rs. {per_day}\n"
            f"- Approx. cost per person for the trip: Rs. {per_person_total}\n"
            f"- Approx. TOTAL for {people} traveller(s): Rs. {grand_total}\n"
            f"(This covers stay, food and local transport only. Add flight/train fare separately.)"
        )
        return breakdown
