"""
agents.py
----------
Defines the CrewAI agents used across the assistant, and helper functions
that assemble the right Task(s)/Crew for a given user intent.

Design:
- Each agent has a narrow role and only the tools it actually needs.
- For a simple factual question we run a SINGLE agent (destination_expert) -> Agentic RAG.
- For an itinerary/compare/budget request we run MULTIPLE agents together as a Crew
  (Multi-Agent Collaboration), where later agents build on earlier agents' output.
"""

from crewai import Agent, Task, Crew, Process, LLM

import config
from tools import TravelRAGSearchTool, WeatherTool, BudgetEstimatorTool

rag_tool = TravelRAGSearchTool()
weather_tool = WeatherTool()
budget_tool = BudgetEstimatorTool()

_llm_kwargs = dict(
    model=config.HF_MODEL,
    temperature=config.LLM_TEMPERATURE,
    api_key=config.HF_API_KEY,
)
# Only pass api_base if the user configured a dedicated endpoint - otherwise
# let litellm route through the shared Hugging Face Inference Providers API.
if config.HF_API_BASE:
    _llm_kwargs["api_base"] = config.HF_API_BASE

llm = LLM(**_llm_kwargs)


# ---------------------------------------------------------------------
# AGENTS
# ---------------------------------------------------------------------

destination_expert = Agent(
    role="Destination Knowledge Expert",
    goal="Answer travel questions accurately using ONLY the uploaded travel guide documents, "
         "and always mention which document/page the answer came from.",
    backstory=(
        "You are a well-travelled guide who has read every travel brochure and guidebook "
        "the user has uploaded. You never make up facts - if the documents don't cover "
        "something, you say so honestly instead of guessing."
    ),
    tools=[rag_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

weather_guidelines_agent = Agent(
    role="Weather and Local Guidelines Advisor",
    goal="Provide up-to-date weather info and any local travel guidelines relevant to the trip.",
    backstory=(
        "You track live weather conditions and also cross-check the uploaded guides for "
        "visa rules, safety notes, and local customs travellers should know about."
    ),
    tools=[weather_tool, rag_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

itinerary_planner = Agent(
    role="Itinerary Planning Specialist",
    goal="Design a practical, day-by-day travel itinerary personalised to the user's trip length, "
         "interests and budget, grounded in real attractions from the travel guides.",
    backstory=(
        "You have planned hundreds of trips. You balance sightseeing with rest time, group "
        "nearby attractions on the same day, and always consider the weather before suggesting "
        "outdoor activities."
    ),
    tools=[rag_tool, weather_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

budget_analyst = Agent(
    role="Trip Budget Analyst",
    goal="Give the user a clear, itemised budget estimate for their trip.",
    backstory="You are a meticulous planner who never lets a traveller be surprised by costs.",
    tools=[budget_tool, rag_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

comparison_agent = Agent(
    role="Destination Comparison Specialist",
    goal="Compare two or more destinations across cost, weather, attractions and suitability "
         "for the traveller's stated preferences, using the uploaded guides as evidence.",
    backstory="You help indecisive travellers choose between destinations with an honest, side-by-side comparison.",
    tools=[rag_tool, weather_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------
# CREW BUILDERS - one function per intent, called by the router
# ---------------------------------------------------------------------

def run_qa(user_query: str) -> str:
    """Single-agent Agentic RAG - for plain factual questions."""
    task = Task(
        description=f"Answer this travel question using the knowledge search tool: '{user_query}'. "
                     f"Cite the source document(s) and page number(s) you used.",
        expected_output="A clear answer followed by a 'Sources:' line listing document + page.",
        agent=destination_expert,
    )
    crew = Crew(agents=[destination_expert], tasks=[task], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


def run_itinerary(user_query: str) -> str:
    """Multi-agent collaboration: knowledge lookup -> weather check -> itinerary building."""
    research_task = Task(
        description=f"Research relevant attractions, transport and local tips for: '{user_query}'. "
                     f"Use the knowledge search tool.",
        expected_output="A bullet list of relevant facts with sources.",
        agent=destination_expert,
    )
    weather_task = Task(
        description=f"Check the current weather for the destination mentioned in: '{user_query}' "
                     f"and note anything that should influence day planning (e.g. avoid outdoor "
                     f"activities on rainy days).",
        expected_output="A short weather summary and planning note.",
        agent=weather_guidelines_agent,
    )
    plan_task = Task(
        description=f"Using the research and weather notes above, build a day-by-day itinerary for: "
                     f"'{user_query}'. Be specific about which attraction goes on which day.",
        expected_output="A day-wise itinerary (Day 1, Day 2, ...) with a short reason for the ordering.",
        agent=itinerary_planner,
        context=[research_task, weather_task],
    )
    crew = Crew(
        agents=[destination_expert, weather_guidelines_agent, itinerary_planner],
        tasks=[research_task, weather_task, plan_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


def run_weather(user_query: str) -> str:
    """Single-agent workflow using the weather + guidelines agent (tool calling)."""
    task = Task(
        description=f"Answer this weather/guidelines question: '{user_query}'. "
                     f"Use the weather tool for live conditions, and the knowledge search tool "
                     f"for any local guidelines mentioned in the travel guides.",
        expected_output="Current weather info plus any relevant local guidelines, with sources where used.",
        agent=weather_guidelines_agent,
    )
    crew = Crew(agents=[weather_guidelines_agent], tasks=[task], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


def run_budget(user_query: str) -> str:
    """Multi-agent: knowledge lookup (for context) + budget calculation tool."""
    research_task = Task(
        description=f"Find any cost-related notes (stay, food, transport) in the guides relevant to: "
                     f"'{user_query}'.",
        expected_output="Bullet notes on cost-relevant info with sources, or 'no cost info found'.",
        agent=destination_expert,
    )
    budget_task = Task(
        description=f"Using the notes above and the budget estimation tool, produce a full trip "
                     f"budget for: '{user_query}'. Extract days/people/style from the request "
                     f"(assume 1 traveller, mid-range style, 3 days if not stated).",
        expected_output="An itemised budget breakdown.",
        agent=budget_analyst,
        context=[research_task],
    )
    crew = Crew(
        agents=[destination_expert, budget_analyst],
        tasks=[research_task, budget_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


def run_comparison(user_query: str) -> str:
    """Multi-agent: knowledge lookup + weather + side-by-side comparison."""
    research_task = Task(
        description=f"Gather key facts (attractions, costs, culture) about each destination "
                     f"mentioned in: '{user_query}'.",
        expected_output="Facts grouped by destination, with sources.",
        agent=destination_expert,
    )
    weather_task = Task(
        description=f"Fetch current weather for each destination mentioned in: '{user_query}'.",
        expected_output="Weather summary per destination.",
        agent=weather_guidelines_agent,
    )
    compare_task = Task(
        description=f"Using the research and weather above, write a side-by-side comparison for: "
                     f"'{user_query}'. End with a one-line recommendation.",
        expected_output="A structured comparison (table-like, in text) plus a final recommendation.",
        agent=comparison_agent,
        context=[research_task, weather_task],
    )
    crew = Crew(
        agents=[destination_expert, weather_guidelines_agent, comparison_agent],
        tasks=[research_task, weather_task, compare_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())
