# Aqua — Water Tracker

A local-first hydration tracker with a FastAPI + SQLite backend, Streamlit dashboard, and a LangChain agent backed by OpenAI. The agent can choose to inspect water history and calculate a target-based streak before recommending a next step.

## Requirements

- Python 3.10 or newer
- An OpenAI API key for AI insights (logging and history work without one)

## Install (Windows PowerShell)

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-openai-api-key"
```

Set `OPENAI_API_KEY` only in your local shell; do not commit it. Optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`) or `WATER_TRACKER_DB` (defaults to `water_tracker.db` in this folder).

## Run

Open two PowerShell terminals in the project folder and activate the virtual environment in each.

Terminal 1 — API:

```powershell
uvicorn backend:app --reload
```

Terminal 2 — dashboard:

```powershell
streamlit run app.py
```

Open the Streamlit URL printed by the second command (usually `http://localhost:8501`). The API docs are available at `http://127.0.0.1:8000/docs`. Set `OPENAI_API_KEY` in the API terminal before starting Uvicorn to enable AI feedback. If it is unset, the rest of the tracker still works and the agent endpoint returns a clear setup message.

## API

- `POST /intake` — save an amount in ml, optional notes, and a date
- `GET /history?days=30` — list intake entries for the last 1–365 days
- `GET /ai-insight?day=YYYY-MM-DD&target_ml=2000` — run the LangChain agent and return its recommendation plus tool usage trace
- `GET /insights?day=YYYY-MM-DD&target_ml=2000` — backward-compatible alias for `/ai-insight`
- `GET /health` — basic health check

The agent tools include `get_water_history(days)` and `calculate_streak()`. The Streamlit panel shows tool calls and their results, not hidden model reasoning. Additional features:

- Weather lookup via Open-Meteo (no API key required): enter a city in the sidebar. At 30 °C or above, the dashboard applies a modest 500 ml target adjustment for that location and base target. This is general encouragement, not a medical prescription.
- Natural-language logging: use the quick-log sentence field. LangChain and the configured OpenAI model extract an amount and short context; this requires `OPENAI_API_KEY`.
- Weekly report: use **Download weekly PDF** to export the last seven days of totals, entry counts, and daily average.

Related API endpoints are `GET /weather-target?location=Seattle&base_target_ml=2000`, `POST /intake/natural` with a `phrase` and optional `date`, and `GET /weekly-report.pdf`. The SQLite database is created automatically at first startup. Intake is stored with the `id`, `date`, `amount_ml`, and `notes` fields.
