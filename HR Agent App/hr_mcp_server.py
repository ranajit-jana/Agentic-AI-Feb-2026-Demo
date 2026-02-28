"""
HR MCP Server — built with FastMCP
Exposes HR policies (as resources) and employee data (from SQLite) as MCP tools.

Run for Inspector debugging:
    mcp dev hr_mcp_server.py

Run standalone (stdio):
    python hr_mcp_server.py
"""
from __future__ import annotations

import json
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from hr_database import (
    create_leave_request,
    fetch_all_employees,
    fetch_employee,
    fetch_leave_balance,
)

mcp = FastMCP("HR Server")

# ─────────────────────────────────────────────
# HR Policy Data
# ─────────────────────────────────────────────

HR_POLICIES: dict[str, str] = {
    "remote_work": """Remote Work Policy:
- Employees may work remotely up to 3 days per week
- Remote work requires manager approval for the arrangement
- Core collaboration hours: 10 am–3 pm in your local timezone
- Must be reachable via Slack/Teams during core hours
- Home-office equipment allowance: $500 per year
- VPN must be used when accessing company systems remotely""",

    "leave": """Leave & Time-Off Policy:
- Annual leave: 15 days (accrued monthly, max carry-over 5 days)
- Sick leave: 10 days per year (no carry-over)
- Personal leave: 3 days per year
- Parental leave: 16 weeks fully paid (primary caregiver); 4 weeks (secondary)
- Bereavement: 5 days for immediate family, 3 days for extended family
- Leave requests require at least 2 weeks advance notice (except sick/emergency)
- All requests must be submitted through the HR portal""",

    "performance": """Performance Review Policy:
- Annual reviews held every December
- Mid-year check-ins held every June
- Ratings on a 1–5 scale (1 = Below Expectations, 5 = Exceptional)
- 360-degree feedback collected from peers and direct reports
- Performance Improvement Plans (PIP) issued for ratings below 2
- Merit salary increases tied to performance ratings
- Promotion eligibility reviewed annually""",

    "code_of_conduct": """Code of Conduct:
- Treat all colleagues with respect and professionalism
- Zero tolerance for harassment, discrimination, or bullying
- Conflicts of interest must be disclosed to HR immediately
- Confidential information must never be shared outside authorised channels
- Report violations to HR or the anonymous Ethics Hotline
- Retaliation against anyone who reports in good faith is strictly prohibited
- Violations may result in disciplinary action up to and including termination""",

    "compensation": """Compensation & Benefits Policy:
- Salaries reviewed annually following performance reviews
- Equity grants vest over 4 years with a 1-year cliff
- Health insurance: company covers 90% of premium (employee + dependants)
- 401(k): company matches up to 4% of salary
- Annual learning & development budget: $1,500 per employee
- Gym/wellness reimbursement: $50/month""",
}

# ─────────────────────────────────────────────
# Resources — policy documents
# ─────────────────────────────────────────────

@mcp.resource("hr://policies/{topic}")
def get_policy_resource(topic: str) -> str:
    """Get a specific HR policy document by topic.

    Available topics: remote_work, leave, performance, code_of_conduct, compensation
    """
    if topic in HR_POLICIES:
        return HR_POLICIES[topic]
    available = ", ".join(HR_POLICIES.keys())
    return f"Policy '{topic}' not found. Available topics: {available}"


@mcp.resource("hr://policies")
def list_policies_resource() -> str:
    """List all available HR policy topics."""
    lines = ["Available HR Policy Topics:\n"]
    for topic in HR_POLICIES:
        lines.append(f"  • {topic}")
    lines.append("\nAccess any policy at: hr://policies/<topic>")
    return "\n".join(lines)


@mcp.resource("hr://employees/{employee_id}")
def get_employee_resource(employee_id: str) -> str:
    """Get employee profile from the SQLite database."""
    emp = fetch_employee(employee_id.upper())
    if not emp:
        return f"Employee '{employee_id}' not found."
    return json.dumps(emp, indent=2)


@mcp.resource("hr://employees")
def list_employees_resource() -> str:
    """List all employees from the SQLite database."""
    employees = fetch_all_employees()
    lines = [f"{'ID':<6} {'Name':<20} {'Role':<25} {'Department'}", "-" * 65]
    for e in employees:
        lines.append(f"{e['employee_id']:<6} {e['name']:<20} {e['role']:<25} {e['department']}")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# Tools — Policy
# ─────────────────────────────────────────────

@mcp.tool()
def get_hr_policy(topic: str) -> str:
    """Retrieve company HR policy text for a given topic.

    Args:
        topic: Policy topic. One of: remote_work, leave, performance,
               code_of_conduct, compensation. Partial names accepted.
    """
    key = topic.lower().replace(" ", "_")
    if key in HR_POLICIES:
        return HR_POLICIES[key]
    for k, policy in HR_POLICIES.items():
        if key in k or k in key:
            return policy
    available = ", ".join(HR_POLICIES.keys())
    return f"Policy '{topic}' not found. Available topics: {available}"


@mcp.tool()
def list_hr_policies() -> str:
    """List all available HR policy topics with short descriptions."""
    descriptions = {
        "remote_work":    "Remote work days, core hours, equipment allowance",
        "leave":          "Annual, sick, personal, parental, and bereavement leave",
        "performance":    "Review cycle, ratings scale, PIPs, merit increases",
        "code_of_conduct":"Respect, harassment, conflicts of interest, reporting",
        "compensation":   "Salary reviews, equity, health insurance, 401(k), L&D budget",
    }
    lines = ["HR Policy Topics:\n"]
    for topic, desc in descriptions.items():
        lines.append(f"  {topic}: {desc}")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# Tools — Employee (SQLite-backed)
# ─────────────────────────────────────────────

@mcp.tool()
def get_employee_info(employee_id: str) -> str:
    """Get profile information for an employee from the HR database.

    Args:
        employee_id: Employee ID (e.g. E001, E002).
    """
    emp = fetch_employee(employee_id.upper())

    if not emp:
        return f"Employee '{employee_id}' not found in the database."
    return (
        f"Employee:   {emp['name']} ({emp['employee_id']})\n"
        f"Role:       {emp['role']}\n"
        f"Department: {emp['department']}\n"
        f"Manager:    {emp['manager_id'] or 'None (top-level)'}\n"
        f"Start date: {emp['start_date']}\n"
        f"Email:      {emp['email']}"
    )


@mcp.tool()
def list_employees() -> str:
    """List all employees in the HR database with their department and role."""
    employees = fetch_all_employees()
    if not employees:
        return "No employees found in the database."
    lines = [f"{'ID':<6} {'Name':<20} {'Role':<25} {'Department'}", "-" * 65]
    for e in employees:
        lines.append(f"{e['employee_id']:<6} {e['name']:<20} {e['role']:<25} {e['department']}")
    return "\n".join(lines)


@mcp.tool()
def check_leave_balance(employee_id: str) -> str:
    """Return the current leave balance for an employee from the HR database.

    Args:
        employee_id: Employee ID (e.g. E001).
    """
    emp = fetch_employee(employee_id.upper())
    if not emp:
        return f"Employee '{employee_id}' not found."

    bal = fetch_leave_balance(employee_id.upper())
    if not bal:
        return f"No leave balance record found for '{employee_id}'."

    return (
        f"Leave balance for {emp['name']} ({employee_id}):\n"
        f"  Annual:   {bal['annual']} day(s)\n"
        f"  Sick:     {bal['sick']} day(s)\n"
        f"  Personal: {bal['personal']} day(s)"
    )


@mcp.tool()
def submit_leave_request(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    reason: str = "",
) -> str:
    """Submit a leave request for an employee. Updates the SQLite database.

    Args:
        employee_id: Employee ID (e.g. E001).
        leave_type:  Type of leave — annual, sick, or personal.
        start_date:  Start date in YYYY-MM-DD format.
        end_date:    End date in YYYY-MM-DD format.
        reason:      Optional reason for the leave.
    """
    eid = employee_id.upper()

    emp = fetch_employee(eid)
    if not emp:
        return f"Employee '{employee_id}' not found."

    if leave_type not in ("annual", "sick", "personal"):
        return f"Invalid leave type '{leave_type}'. Must be annual, sick, or personal."

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end   = datetime.strptime(end_date,   "%Y-%m-%d").date()
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD."

    days = (end - start).days + 1
    if days <= 0:
        return "End date must be on or after start date."

    bal = fetch_leave_balance(eid)
    available = bal[leave_type] if bal else 0
    if days > available:
        return (
            f"Insufficient {leave_type} leave for {emp['name']}. "
            f"Requested: {days} day(s), Available: {available} day(s)."
        )

    request_id = create_leave_request(eid, leave_type, start_date, end_date, days, reason)
    remaining  = available - days

    return (
        f"Leave request {request_id} submitted successfully.\n"
        f"  Employee:  {emp['name']} ({eid})\n"
        f"  Type:      {leave_type}\n"
        f"  Dates:     {start_date} → {end_date} ({days} day(s))\n"
        f"  Status:    pending_approval\n"
        f"  Remaining {leave_type} balance: {remaining} day(s)"
    )

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
