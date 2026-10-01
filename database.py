"""SQLite persistence for water intake entries."""

from __future__ import annotations

import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "water_tracker.db"
DB_PATH = Path(os.getenv("WATER_TRACKER_DB", str(DEFAULT_DB_PATH)))


def get_connection() -> sqlite3.Connection:
    """Create a connection with rows addressable by column name."""
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Create the intake table if it does not already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS water_intake (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount_ml INTEGER NOT NULL CHECK (amount_ml > 0),
                notes TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_water_intake_date ON water_intake(date)"
        )


def add_intake(day: date, amount_ml: int, notes: str = "") -> dict[str, Any]:
    """Save an intake entry and return the inserted record."""
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO water_intake (date, amount_ml, notes) VALUES (?, ?, ?)",
            (day.isoformat(), amount_ml, notes),
        )
        row = connection.execute(
            "SELECT id, date, amount_ml, notes FROM water_intake WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    return dict(row)


def get_history(days: int = 30) -> list[dict[str, Any]]:
    """Return intake entries from the requested number of calendar days."""
    cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, date, amount_ml, notes
            FROM water_intake
            WHERE date >= ?
            ORDER BY date DESC, id DESC
            """,
            (cutoff,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_daily_total(day: date) -> int:
    """Return the total amount logged for one calendar day."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COALESCE(SUM(amount_ml), 0) AS total FROM water_intake WHERE date = ?",
            (day.isoformat(),),
        ).fetchone()
    return int(row["total"])


def get_daily_totals(days: int = 365) -> dict[str, int]:
    """Return intake totals by date for streak calculations."""
    if not 1 <= days <= 3650:
        raise ValueError("days must be between 1 and 3650")
    cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT date, SUM(amount_ml) AS total_ml
            FROM water_intake
            WHERE date >= ?
            GROUP BY date
            ORDER BY date DESC
            """,
            (cutoff,),
        ).fetchall()
    return {str(row["date"]): int(row["total_ml"]) for row in rows}


init_db()
