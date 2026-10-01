"""Streamlit dashboard for the water tracker API."""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("WATER_TRACKER_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 10

st.set_page_config(page_title="Aqua | Water Tracker", page_icon="💧", layout="wide")
st.markdown(
    """
    <style>
        .block-container { max-width: 1100px; padding-top: 2rem; }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #effaff, #f8fcff);
            border: 1px solid #d9eff8; padding: 1rem 1.2rem; border-radius: 16px;
            color: #073b55 !important;
        }
        [data-testid="stMetric"] * { color: #073b55 !important; }
        .hero { padding: 1.6rem 1.8rem; border-radius: 20px; color: #073b55;
            background: linear-gradient(120deg, #def7ff, #effcf8); margin-bottom: 1.2rem; }
        .hero h1 { margin: 0; }
        .hero p { margin: .4rem 0 0; color: #406374; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, **params: Any) -> Any:
    response = requests.get(
        f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(
        f"{API_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def show_api_error(error: requests.RequestException) -> None:
    detail = "The API server is unavailable. Start it in another terminal and refresh this page."
    response = getattr(error, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail", detail)
        except (ValueError, AttributeError):
            pass
    st.error(detail)


st.markdown(
    "<div class='hero'><h1>💧 Your water, in balance</h1>"
    "<p>Build a gentle hydration habit, one glass at a time.</p></div>",
    unsafe_allow_html=True,
)

today = date.today()
target_ml = st.sidebar.number_input(
    "Daily target (ml)", min_value=500, max_value=10_000, value=2_000, step=250
)
history_days = st.sidebar.selectbox("History range", [7, 14, 30, 90], index=2)

try:
    today_entries = api_get("/history", days=1)
    total_today = sum(entry["amount_ml"] for entry in today_entries)
    first, second, third = st.columns(3)
    first.metric("Logged today", f"{total_today:,} ml")
    second.metric("Personal target", f"{int(target_ml):,} ml")
    second.progress(min(total_today / target_ml, 1.0))
    third.metric("To your target", f"{max(int(target_ml) - total_today, 0):,} ml")
except requests.RequestException as exc:
    total_today = 0
    show_api_error(exc)

st.divider()
left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.subheader("Log a drink")
    with st.form("intake_form", clear_on_submit=True):
        amount_ml = st.number_input(
            "Amount (ml)", min_value=1, max_value=10_000, value=250, step=50
        )
        intake_date = st.date_input("Date", value=today, max_value=today)
        notes = st.text_input("Notes (optional)", max_chars=500, placeholder="After a walk")
        submitted = st.form_submit_button("Add water", type="primary", use_container_width=True)

    if submitted:
        try:
            api_post(
                "/intake",
                {"amount_ml": int(amount_ml), "notes": notes, "date": intake_date.isoformat()},
            )
            st.success(f"Added {int(amount_ml):,} ml to your log.")
            st.rerun()
        except requests.RequestException as exc:
            show_api_error(exc)

with right:
    st.subheader("Your hydration trend")
    try:
        entries = api_get("/history", days=int(history_days))
        start_day = today - timedelta(days=int(history_days) - 1)
        dates = pd.date_range(start=start_day, end=today, freq="D")
        chart = pd.DataFrame({"Water (ml)": 0}, index=dates)
        for entry in entries:
            entry_day = pd.Timestamp(entry["date"])
            if entry_day in chart.index:
                chart.loc[entry_day, "Water (ml)"] += entry["amount_ml"]
        st.line_chart(chart, y="Water (ml)", height=260)
        if entries:
            detail = pd.DataFrame(entries)
            detail["date"] = pd.to_datetime(detail["date"]).dt.strftime("%b %d, %Y")
            detail = detail.rename(
                columns={"date": "Date", "amount_ml": "Amount (ml)", "notes": "Notes"}
            )
            st.dataframe(detail[["Date", "Amount (ml)", "Notes"]], hide_index=True, use_container_width=True)
        else:
            st.info("Your log is empty for this range. Add your first drink to get started.")
    except requests.RequestException as exc:
        show_api_error(exc)

st.divider()
st.subheader("Agentic hydration coach")
st.caption("The agent can review your history and streak before making a recommendation. This is general guidance, not medical advice.")
if st.button("Ask the hydration agent", type="secondary"):
    try:
        with st.spinner("The agent is reviewing your hydration log…"):
            insight = api_get("/ai-insight", day=today.isoformat(), target_ml=int(target_ml))
        steps = insight.get("steps", [])
        if steps:
            with st.expander("Agent workflow · tools used", expanded=True):
                for index, step in enumerate(steps, start=1):
                    st.markdown(f"**{index}. `{step['tool']}`**")
                    if step.get("input"):
                        st.caption(f"Input: {step['input']}")
                    st.caption(f"Result: {step.get('result', '')}")
        else:
            st.caption("The agent answered without needing a database tool.")
        st.info(insight["feedback"])
    except requests.RequestException as exc:
        show_api_error(exc)
