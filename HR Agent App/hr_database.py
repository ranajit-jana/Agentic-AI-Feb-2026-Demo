"""
HR SQLite Database
Manages employees, leave balances, and leave requests.
Auto-initialises and seeds on first import.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "hr_database.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables and seed initial data (idempotent)."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id  TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            department   TEXT NOT NULL,
            role         TEXT NOT NULL,
            manager_id   TEXT,
            start_date   TEXT NOT NULL,
            email        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS leave_balances (
            employee_id  TEXT PRIMARY KEY,
            annual       INTEGER NOT NULL DEFAULT 15,
            sick         INTEGER NOT NULL DEFAULT 10,
            personal     INTEGER NOT NULL DEFAULT 3,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        );

        CREATE TABLE IF NOT EXISTS leave_requests (
            request_id   TEXT PRIMARY KEY,
            employee_id  TEXT NOT NULL,
            leave_type   TEXT NOT NULL,
            start_date   TEXT NOT NULL,
            end_date     TEXT NOT NULL,
            days         INTEGER NOT NULL,
            reason       TEXT,
            status       TEXT NOT NULL DEFAULT 'pending_approval',
            submitted_at TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        );
    """)

    # ── Seed employees ────────────────────────────────────────────────
    conn.executemany(
        """INSERT OR IGNORE INTO employees
           (employee_id, name, department, role, manager_id, start_date, email)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            ("E001", "Alice Johnson", "Engineering", "Senior Engineer",  "E003", "2022-03-15", "alice.johnson@company.com"),
            ("E002", "Bob Smith",     "Marketing",   "Marketing Manager","E004", "2021-07-01", "bob.smith@company.com"),
            ("E003", "Carol White",   "Engineering", "VP Engineering",   None,   "2019-01-10", "carol.white@company.com"),
            ("E004", "David Brown",   "Marketing",   "CMO",              None,   "2018-05-20", "david.brown@company.com"),
        ],
    )

    # ── Seed leave balances ───────────────────────────────────────────
    conn.executemany(
        """INSERT OR IGNORE INTO leave_balances
           (employee_id, annual, sick, personal) VALUES (?, ?, ?, ?)""",
        [
            ("E001", 15, 10, 3),
            ("E002", 12, 10, 3),
            ("E003", 20, 10, 5),
            ("E004", 20, 10, 5),
        ],
    )

    conn.commit()
    conn.close()


# ── Query helpers ─────────────────────────────────────────────────────────────

def fetch_employee(employee_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
        ).fetchone()
    return dict(row) if row else None


def fetch_leave_balance(employee_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM leave_balances WHERE employee_id = ?", (employee_id,)
        ).fetchone()
    return dict(row) if row else None


def fetch_all_employees() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM employees ORDER BY department, name").fetchall()
    return [dict(r) for r in rows]


def create_leave_request(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    days: int,
    reason: str,
) -> str:
    """Insert a leave request and deduct from balance. Returns request_id."""
    with get_connection() as conn:
        # Count existing requests to generate ID
        count = conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0]
        request_id = f"LR{count + 1:04d}"

        conn.execute(
            """INSERT INTO leave_requests
               (request_id, employee_id, leave_type, start_date, end_date,
                days, reason, status, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending_approval', ?)""",
            (request_id, employee_id, leave_type, start_date, end_date,
             days, reason, datetime.now().isoformat()),
        )
        conn.execute(
            f"UPDATE leave_balances SET {leave_type} = {leave_type} - ? WHERE employee_id = ?",
            (days, employee_id),
        )
        conn.commit()
    return request_id


# Auto-initialise on import
init_db()
