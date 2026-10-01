"""FastAPI service for logging water and requesting hydration insights."""

from __future__ import annotations

import os
import json
from datetime import date as Date, timedelta
from io import BytesIO
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Query, Response, status
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import AuthenticationError
from pydantic import BaseModel, ConfigDict, Field
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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


class NaturalIntakeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    phrase: str = Field(min_length=3, max_length=1000)
    date: Date = Field(default_factory=Date.today)


class ParsedIntake(BaseModel):
    amount_ml: int = Field(ge=0, le=10_000)
    notes: str = Field(default="", max_length=500)


class WeatherTargetResponse(BaseModel):
    location: str
    temperature_c: float
    hot_weather: bool
    base_target_ml: int
    adjusted_target_ml: int
    adjustment_ml: int
    message: str


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


def parse_intake_phrase(phrase: str) -> ParsedIntake:
    """Extract a water amount and concise context from a natural phrase."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Natural-language logging needs an OPENAI_API_KEY. You can still log water with the standard form.",
        )
    parser_model = _agent_model().with_structured_output(ParsedIntake)
    prompt = (
        "Extract a water-drinking amount in milliliters and optional short context "
        "from this user phrase. Convert common drink sizes to a reasonable ml estimate "
        "(small glass 200 ml, glass 250 ml, large glass 350 ml, bottle 500 ml). "
        "Only treat it as an intake if the phrase clearly says the user drank water. "
        "If unclear or not water, do not invent an amount; return amount_ml=0 and explain "
        "briefly in notes. Do not infer medical advice. Phrase: {phrase}"
    )
    try:
        parsed = (
            ChatPromptTemplate.from_messages([("human", prompt)]) | parser_model
        ).invoke({"phrase": phrase})
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "OpenAI rejected OPENAI_API_KEY. Set a valid OpenAI API key in the "
                "environment used to start FastAPI, then restart the API."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not parse the drink phrase. Try adding a clear amount such as 250 ml.",
        ) from exc
    if parsed.amount_ml == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=parsed.notes or "Could not identify a water intake amount from that phrase.",
        )
    return parsed


@app.post("/intake/natural", response_model=IntakeEntry, status_code=status.HTTP_201_CREATED)
def log_natural_intake(payload: NaturalIntakeRequest) -> IntakeEntry:
    parsed = parse_intake_phrase(payload.phrase)
    entry = database.add_intake(payload.date, parsed.amount_ml, parsed.notes)
    return IntakeEntry(**entry)


@app.get("/weather-target", response_model=WeatherTargetResponse)
def get_weather_target(
    location: str = Query(min_length=2, max_length=120),
    base_target_ml: int = Query(default=2_000, ge=500, le=10_000),
) -> WeatherTargetResponse:
    """Use Open-Meteo geocoding and current temperature for a conservative adjustment."""
    try:
        geocode_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        geocode_response.raise_for_status()
        results = geocode_response.json().get("results", [])
        if not results:
            raise HTTPException(status_code=404, detail="Location not found. Try a nearby city name.")
        place = results[0]
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": place["latitude"], "longitude": place["longitude"], "current": "temperature_2m"},
            timeout=8,
        )
        weather_response.raise_for_status()
        temperature_c = float(weather_response.json()["current"]["temperature_2m"])
    except HTTPException:
        raise
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Weather service is unavailable. Keep your base target or try again later.",
        ) from exc

    hot_weather = temperature_c >= 30.0
    # A modest, capped encouragement—not a clinical fluid prescription.
    adjustment_ml = 500 if hot_weather else 0
    resolved_location = ", ".join(
        part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
    )
    return WeatherTargetResponse(
        location=resolved_location,
        temperature_c=temperature_c,
        hot_weather=hot_weather,
        base_target_ml=base_target_ml,
        adjusted_target_ml=min(base_target_ml + adjustment_ml, 10_000),
        adjustment_ml=adjustment_ml,
        message=(
            "Warm conditions detected; consider a gentle extra 500 ml target if appropriate. "
            "Hydration needs vary, so follow clinician guidance."
            if hot_weather
            else "No hot-weather adjustment suggested today. Hydration needs vary by person."
        ),
    )


@app.get("/weekly-report.pdf")
def weekly_report_pdf() -> Response:
    """Generate a downloadable seven-day intake summary as a PDF."""
    entries = database.get_history(days=7)
    start_day = Date.today() - timedelta(days=6)
    totals: dict[str, int] = {}
    for entry in entries:
        totals[entry["date"]] = totals.get(entry["date"], 0) + int(entry["amount_ml"])

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Weekly Water Intake Summary",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        textColor=colors.HexColor("#07516b"), spaceAfter=8,
    )
    story = [
        Paragraph("Weekly Water Intake Summary", title_style),
        Paragraph(f"{start_day.strftime('%b %d, %Y')} – {Date.today().strftime('%b %d, %Y')}", styles["Normal"]),
        Spacer(1, 16),
    ]
    data = [["Date", "Total intake", "Entries"]]
    entry_counts: dict[str, int] = {}
    for entry in entries:
        entry_counts[entry["date"]] = entry_counts.get(entry["date"], 0) + 1
    for offset in range(7):
        day = start_day + timedelta(days=offset)
        data.append([day.strftime("%a, %b %d"), f"{totals.get(day.isoformat(), 0):,} ml", str(entry_counts.get(day.isoformat(), 0))])
    average = round(sum(totals.values()) / 7)
    data.extend([["7-day total", f"{sum(totals.values()):,} ml", str(len(entries))], ["Daily average", f"{average:,} ml", ""]])
    table = Table(data, colWidths=[3.2 * inch, 2.0 * inch, 1.2 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087e9b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -3), [colors.white, colors.HexColor("#effaff")]),
        ("BACKGROUND", (0, -2), (-1, -1), colors.HexColor("#dff5f4")),
        ("FONTNAME", (0, -2), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8dce3")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([table, Spacer(1, 14), Paragraph("A supportive tracking summary, not medical advice. Personal hydration needs vary.", styles["Italic"])])
    document.build(story)
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=weekly-water-summary.pdf"},
    )


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
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "OpenAI rejected OPENAI_API_KEY. Set a valid OpenAI API key in the "
                "environment used to start FastAPI, then restart the API. The credential "
                "was not accepted; do not paste it into chat or logs."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The AI agent could not generate an insight because the model service "
                "or agent request failed. Check the API server terminal for details."
            ),
        ) from exc

    return InsightResponse(
        date=day,
        total_ml=total_ml,
        target_ml=target_ml,
        remaining_ml=remaining_ml,
        feedback=feedback,
        steps=steps,
    )
