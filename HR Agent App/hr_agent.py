"""
HR Management Agent — built with LangGraph
Handles: Leave Management, Policy Q&A, Onboarding, Recruitment, Performance Reviews

Employee data (employees, leave balances, leave requests) is stored in SQLite
and served via the MCP server (hr_mcp_server.py).
HR policies are also served via the MCP server as resources.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODEL_NAME  = "gpt-4o-mini"
TEMPERATURE = 0.2

# ─────────────────────────────────────────────
# MCP Client Helper
# All employee & policy data is fetched from hr_mcp_server.py
# ─────────────────────────────────────────────

_MCP_SERVER = str(Path(__file__).parent / "hr_mcp_server.py")


async def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Spawn the MCP server as a subprocess and call a named tool."""
    server_params = StdioServerParameters(command="python3", args=[_MCP_SERVER])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text


def _mcp(tool_name: str, **kwargs) -> str:
    """Synchronous wrapper around _call_mcp_tool for use inside @tool functions."""
    return asyncio.run(_call_mcp_tool(tool_name, kwargs))

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class HRState(TypedDict):
    messages:    Annotated[list, add_messages]
    intent:      str
    employee_id: str | None
    context:     dict[str, Any]

# ─────────────────────────────────────────────
# Structured Output Model
# ─────────────────────────────────────────────

class Intent(BaseModel):
    """Classified intent of an HR request."""

    intent: str = Field(
        description=(
            "One of: leave_management, policy_question, onboarding, "
            "recruitment, performance_review, general"
        )
    )
    employee_id: str | None = Field(
        default=None,
        description="Employee ID if mentioned (format: E001, E002, …)",
    )
    reasoning: str = Field(description="Brief reasoning for the classification")

# ─────────────────────────────────────────────
# HR Tools — Employee (via MCP → SQLite)
# ─────────────────────────────────────────────

@tool
def get_employee_info(employee_id: str) -> str:
    """Get profile information for an employee from the HR database.

    Args:
        employee_id: Employee ID (e.g. E001, E002).
    """
    return _mcp("get_employee_info", employee_id=employee_id)


@tool
def list_employees() -> str:
    """List all employees in the HR database."""
    return _mcp("list_employees")


@tool
def check_leave_balance(employee_id: str) -> str:
    """Return the current leave balance for an employee from the HR database.

    Args:
        employee_id: Employee ID (e.g. E001).
    """
    return _mcp("check_leave_balance", employee_id=employee_id)


@tool
def submit_leave_request(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    reason: str = "",
) -> str:
    """Submit a leave request. Validates balance and persists to SQLite via MCP.

    Args:
        employee_id: Employee ID (e.g. E001).
        leave_type:  Type of leave — annual, sick, or personal.
        start_date:  Start date in YYYY-MM-DD format.
        end_date:    End date in YYYY-MM-DD format.
        reason:      Optional reason for the leave.
    """
    return _mcp(
        "submit_leave_request",
        employee_id=employee_id,
        leave_type=leave_type,
        start_date=start_date,
        end_date=end_date,
        reason=reason,
    )

# ─────────────────────────────────────────────
# HR Tools — Policy (via MCP)
# ─────────────────────────────────────────────

@tool
def get_hr_policy(policy_topic: str) -> str:
    """Retrieve HR policy text from the MCP Policy Server.

    Args:
        policy_topic: One of: remote_work, leave, performance,
                      code_of_conduct, compensation. Partial names accepted.
    """
    return _mcp("get_hr_policy", topic=policy_topic)

# ─────────────────────────────────────────────
# HR Tools — Generative (local, no DB needed)
# ─────────────────────────────────────────────

@tool
def generate_onboarding_checklist(
    employee_id: str,
    start_date: str,
    department: str = "",
) -> dict:
    """Generate a personalised onboarding checklist for a new employee.

    Args:
        employee_id: The new employee's ID.
        start_date:  Employee start date in YYYY-MM-DD format.
        department:  Department name for role-specific items.
    """
    checklist: dict[str, list[str]] = {
        "week_1": [
            "Complete HR paperwork and benefits enrolment",
            "Set up workstation and accounts (email, Slack, HRIS portal)",
            "30-minute orientation call with direct manager",
            "Review company handbook and code of conduct",
            "Complete mandatory compliance training (online)",
            "Collect building-access badge from reception",
        ],
        "week_2": [
            "Shadow team members across key functions",
            "Set up recurring 1-on-1s with manager",
            "Review current projects and team roadmap",
            "Complete role-specific tool and system training",
            "Intro meeting with skip-level manager",
        ],
        "month_1": [
            "30-day check-in with manager",
            "Define 90-day goals aligned to team OKRs",
            "Join relevant Slack channels and recurring meetings",
            "Complete department-specific certification or training",
            "Attend company all-hands or town hall",
        ],
        "month_3": [
            "90-day performance review with manager",
            "Submit onboarding experience feedback survey",
            "Begin leading or owning an independent workstream",
            "Confirm probationary period completion with HR",
        ],
    }

    dept = department.lower()
    if dept in ("engineering", "tech", "software", "data"):
        checklist["week_1"].append("Security training and code-repository access (GitHub/GitLab)")
        checklist["week_2"].append("Review codebase, architecture docs, and local dev setup")
        checklist["week_2"].append("Merge first pull request or complete a starter task")
    elif dept in ("sales", "marketing"):
        checklist["week_1"].append("Complete product and pricing knowledge training")
        checklist["week_2"].append("Shadow live sales calls or active marketing campaigns")
    elif dept in ("hr", "people"):
        checklist["week_1"].append("Review HRIS systems and HR data access procedures")
        checklist["week_2"].append("Attend an HR team standup and any active project calls")

    return {
        "employee_id": employee_id,
        "start_date":  start_date,
        "department":  department or "General",
        "checklist":   checklist,
    }


@tool
def generate_interview_questions(
    role: str,
    level: str = "mid",
    interview_type: str = "behavioral",
) -> dict:
    """Generate tailored interview questions for a role.

    Args:
        role:           Job title (e.g. Software Engineer, Marketing Manager).
        level:          Experience level — junior, mid, senior, or lead.
        interview_type: Interview style — behavioral, technical, or cultural.
    """
    behavioral = [
        "Tell me about a time you had to manage competing priorities. How did you handle it?",
        "Describe a situation where you collaborated with a difficult teammate.",
        "Give an example of a project that didn't go as planned. What did you learn?",
        "How do you handle feedback you disagree with?",
        "Tell me about your most significant professional achievement.",
    ]
    cultural = [
        f"Why are you interested in this {role} role specifically?",
        "How do you prefer to communicate progress to your team?",
        "What kind of work environment helps you do your best work?",
        "How do you approach continuous learning in your field?",
        "What motivates you beyond financial compensation?",
    ]
    technical_map: dict[str, list[str]] = {
        "software engineer": [
            "Walk me through how you debug a production issue under pressure.",
            "How do you ensure code quality and prevent regressions?",
            "Explain a significant technical architecture decision you made.",
            "How do you approach and manage technical debt?",
            "Describe your experience with CI/CD and DevOps practices.",
        ],
        "marketing manager": [
            "How do you measure the success of a marketing campaign?",
            "Walk me through building a go-to-market strategy from scratch.",
            "How do you align marketing goals with sales targets?",
            "Describe your experience with marketing analytics platforms.",
            "How do you prioritise channels for a new product launch?",
        ],
        "default": [
            f"Describe experience most relevant to this {role} position.",
            "What tools and methodologies do you rely on most?",
            "How do you stay current with trends in your field?",
            "Walk me through a typical high-output day in your current role.",
        ],
    }
    senior_extras = [
        "How do you mentor and grow junior team members?",
        "Describe your approach to cross-functional stakeholder management.",
        "How do you balance long-term strategy with short-term delivery?",
    ]

    role_key = role.lower()
    tech_qs  = technical_map.get(role_key, technical_map["default"])

    question_map = {"behavioral": behavioral, "technical": tech_qs, "cultural": cultural}
    questions    = question_map.get(interview_type, behavioral)

    if level in ("senior", "lead"):
        questions = questions + senior_extras[:2]

    return {
        "role":           role,
        "level":          level,
        "interview_type": interview_type,
        "questions":      questions,
        "tip":            "Use STAR (Situation, Task, Action, Result) prompts for follow-up depth.",
    }


HR_TOOLS = [
    # Employee tools (MCP → SQLite)
    get_employee_info,
    list_employees,
    check_leave_balance,
    submit_leave_request,
    # Policy tool (MCP)
    get_hr_policy,
    # Generative tools (local)
    generate_onboarding_checklist,
    generate_interview_questions,
]

# ─────────────────────────────────────────────
# LLM Helpers
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


@lru_cache(maxsize=1)
def get_llm_with_tools() -> ChatOpenAI:
    return get_llm().bind_tools(HR_TOOLS)

# ─────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an HR assistant intent classifier.
Classify the user's HR request into exactly one of these intents:
- leave_management   : leave requests, vacation, PTO, sick days, time off
- policy_question    : questions about HR policies, company rules, procedures
- onboarding         : new-employee onboarding, setup checklists, getting started
- recruitment        : hiring, job descriptions, interview questions, job postings
- performance_review : performance reviews, feedback, ratings, PIPs
- general            : any HR question that doesn't fit the above categories

Also extract the employee ID if mentioned (format: E001, E002, …)."""

INTENT_PROMPTS: dict[str, str] = {
    "leave_management": (
        "You are an HR Leave Management specialist. "
        "Help employees check balances, submit requests, and understand leave policies. "
        "Always verify the employee ID before taking action. Be empathetic and clear."
    ),
    "policy_question": (
        "You are an HR Policy expert. "
        "Answer questions about company HR policies with clarity and accuracy. "
        "Use get_hr_policy to retrieve exact policy text. "
        "If a topic isn't covered, clearly state what you can help with."
    ),
    "onboarding": (
        "You are an HR Onboarding specialist. "
        "Help new hires understand what they need to do to get set up. "
        "Generate personalised onboarding checklists and guide them through their first weeks."
    ),
    "recruitment": (
        "You are an HR Recruitment specialist. "
        "Assist hiring managers with interview questions, job descriptions, and hiring best practices. "
        "Generate tailored questions based on role, level, and interview type."
    ),
    "performance_review": (
        "You are an HR Performance Management specialist. "
        "Help managers and employees navigate performance reviews, feedback frameworks, and PIPs. "
        "Reference company performance policy when relevant."
    ),
    "general": (
        "You are a helpful HR assistant. "
        "Answer general HR questions accurately and direct employees to the right resources. "
        "Use available tools to provide the most accurate information possible."
    ),
}


def classify_intent(state: HRState) -> dict:
    """Classify the intent of the latest user message."""
    print("[HR Agent] Classifying intent…")
    last_message = state["messages"][-1].content

    classifier = get_llm().with_structured_output(Intent)
    result: Intent = classifier.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_message),
    ])

    print(f"[HR Agent] Intent: {result.intent} | Employee ID: {result.employee_id}")
    return {
        "intent":      result.intent,
        "employee_id": result.employee_id,
        "context":     {"reasoning": result.reasoning},
    }


def hr_agent_node(state: HRState) -> dict:
    """Main HR agent node — calls tools and builds a response."""
    intent      = state.get("intent", "general")
    employee_id = state.get("employee_id")
    print(f"[HR Agent] Handling '{intent}' request…")

    system_prompt = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["general"])
    if employee_id:
        system_prompt += f"\n\nThe employee making this request has ID: {employee_id}."

    response = get_llm_with_tools().invoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ])
    return {"messages": [response]}


def should_continue(state: HRState) -> str:
    """Route to tool execution or finish."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        print(f"[HR Agent] Invoking tools: {[tc['name'] for tc in last.tool_calls]}")
        return "tools"
    print("[HR Agent] Response complete.")
    return "end"

# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_hr_graph():
    """Compile and return the HR agent StateGraph."""
    tool_node = ToolNode(HR_TOOLS)

    graph = StateGraph(HRState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("hr_agent",        hr_agent_node)
    graph.add_node("tools",           tool_node)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "hr_agent")
    graph.add_conditional_edges(
        "hr_agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "hr_agent")

    return graph.compile()
