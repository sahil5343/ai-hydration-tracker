"""FastAPI service for logging water and requesting hydration insights."""

from __future__ import annotations

import os
import json
from datetime import date as Date, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

import database

app = FastAPI(
    title="Water Tracker API",
    description="Log daily water intake and get AI-generated hydration encouragement.",
    version="1.0.0",
)


class IntakeCreate(BaseModel):
    """Fields accepted when adding an intake entry."""

    model_config = ConfigDict(str_strip_whitespace=True)

    amount_ml: int = Field(gt=0, le=10_000, description="Amount of water in milliliters")
    notes: str = Field(default="", max_length=500)
    date: Date = Field(default_factory=Date.today)


class IntakeEntry(BaseModel):
    id: int
    date: Date
    amount_ml: int
    notes: str


class InsightResponse(BaseModel):
    date: Date
    total_ml: int
    target_ml: int
    remaining_ml: int
    feedback: str
    steps: list["AgentStep"]


class AgentStep(BaseModel):
    tool: str
    input: dict[str, Any]
    result: str


@tool
def get_water_history(days: int) -> str:
    """Summarize daily water intake totals over the last 1 to 365 days."""
    if not 1 <= days <= 365:
        return "days must be between 1 and 365."
    entries = database.get_history(days)
    daily_totals: dict[str, int] = {}
    for entry in entries:
        day = entry["date"]
        daily_totals[day] = daily_totals.get(day, 0) + entry["amount_ml"]
    summary = [
        {"date": day, "total_ml": total}
        for day, total in sorted(daily_totals.items(), reverse=True)
    ]
    return json.dumps(summary) if summary else "No water intake was logged in that period."


def make_calculate_streak_tool(target_ml: int):
    """Create a streak tool that uses the target selected for this insight."""

    @tool
    def calculate_streak() -> str:
        """Calculate consecutive days meeting the user's daily water target."""
        totals = database.get_daily_totals(days=3650)
        today = Date.today()
        # Today's streak is still in progress; until the target is reached, count
        # from yesterday instead of prematurely breaking a streak.
        current_day = today
        if totals.get(today.isoformat(), 0) < target_ml:
            current_day -= timedelta(days=1)

        streak = 0
        while totals.get(current_day.isoformat(), 0) >= target_ml:
            streak += 1
            current_day -= timedelta(days=1)
        return f"Current streak: {streak} consecutive day(s) meeting {target_ml} ml."

    return calculate_streak


def _agent_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=30,
        max_retries=1,
    )


def run_hydration_agent(day: Date, target_ml: int) -> tuple[str, list[AgentStep]]:
    """Run the agent and collect visible tool activity (not private reasoning)."""
    history_tool = get_water_history
    streak_tool = make_calculate_streak_tool(target_ml)
    agent = create_agent(
        model=_agent_model(),
        tools=[history_tool, streak_tool],
        system_prompt=(
            "You are a friendly hydration coach. Use the available tools when they help "
            "you review recent history or the current streak before giving a concise, "
            "personalized recommendation. Do not reveal hidden chain-of-thought; the UI "
            "shows tool calls and results as the workflow trace. Keep advice general and "
            "non-medical, never diagnose or urge excessive drinking, and note that needs "
            "vary and clinician guidance takes priority."
        ),
    )
    request = (
        f"Review my water intake for {day.isoformat()}. My daily target is {target_ml} ml. "
        "Use the tools as useful and give me a friendly, short recommendation."
    )
    steps: list[AgentStep] = []
    steps_by_call_id: dict[str, AgentStep] = {}
    feedback = ""
    for update in agent.stream(
        {"messages": [{"role": "user", "content": request}]},
        stream_mode="updates",
    ):
        for node_update in update.values():
            for message in node_update.get("messages", []):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        step = AgentStep(
                            tool=call["name"],
                            input=call.get("args", {}),
                            result="Tool call completed.",
                        )
                        steps.append(step)
                        steps_by_call_id[call["id"]] = step
                    if isinstance(message.content, str) and message.content.strip():
                        feedback = message.content.strip()
                elif isinstance(message, ToolMessage):
                    step = steps_by_call_id.get(message.tool_call_id)
                    if step is not None:
                        step.result = str(message.content)[:800]
    if not feedback:
        raise RuntimeError("The AI agent returned no recommendation.")
    return feedback, steps


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/intake", response_model=IntakeEntry, status_code=status.HTTP_201_CREATED)
def log_intake(payload: IntakeCreate) -> IntakeEntry:
    entry = database.add_intake(payload.date, payload.amount_ml, payload.notes)
    return IntakeEntry(**entry)


@app.get("/history", response_model=list[IntakeEntry])
def read_history(
    days: int = Query(default=30, ge=1, le=365, description="Calendar days to include"),
) -> list[IntakeEntry]:
    return [IntakeEntry(**entry) for entry in database.get_history(days)]


@app.get("/ai-insight", response_model=InsightResponse)
@app.get("/insights", response_model=InsightResponse, include_in_schema=False)
def get_ai_insight(
    day: Date = Query(default_factory=Date.today, description="Day to analyze"),
    target_ml: int = Query(default=2_000, ge=500, le=10_000),
) -> InsightResponse:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI insights need an OPENAI_API_KEY. Set it in the environment and restart the API.",
        )

    total_ml = database.get_daily_total(day)
    remaining_ml = max(target_ml - total_ml, 0)
    try:
        feedback, steps = run_hydration_agent(day, target_ml)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI agent could not generate an insight. Check the API key and try again.",
        ) from exc

    return InsightResponse(
        date=day,
        total_ml=total_ml,
        target_ml=target_ml,
        remaining_ml=remaining_ml,
        feedback=feedback,
        steps=steps,
    )
