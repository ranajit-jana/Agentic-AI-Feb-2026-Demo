"""
Streamlit chat UI for the HR Management Agent.
Run:  streamlit run ui.py
"""
from __future__ import annotations

import asyncio
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from hr_agent import build_hr_graph
from hr_database import fetch_all_employees

EMPLOYEES = {e["employee_id"]: e for e in fetch_all_employees()}

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="HR Management Agent",
    page_icon="👥",
    layout="wide",
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("👥 HR Agent")
    st.markdown("Powered by **LangGraph + GPT-4o-mini**")
    st.divider()

    st.subheader("Employee Context")
    employee_options = {"None (anonymous)": None} | {
        f"{info['name']} ({eid})": eid
        for eid, info in EMPLOYEES.items()
    }
    selected_label = st.selectbox("Select employee", list(employee_options.keys()))
    employee_id = employee_options[selected_label]

    if employee_id:
        emp = EMPLOYEES[employee_id]
        st.info(
            f"**{emp['name']}**  \n"
            f"Role: {emp['role']}  \n"
            f"Dept: {emp['department']}  \n"
            f"Since: {emp['start_date']}"
        )

    st.divider()
    st.subheader("What can I help with?")
    st.markdown(
        "- **Leave management** — check balances, submit requests\n"
        "- **HR policies** — remote work, leave, code of conduct, compensation\n"
        "- **Onboarding** — new-hire checklists & guides\n"
        "- **Recruitment** — interview questions by role & level\n"
        "- **Performance reviews** — frameworks & policy\n"
        "- **General HR** — any other HR question"
    )

    st.divider()
    st.subheader("Quick prompts")
    quick_prompts = [
        "What is my annual leave balance?",
        "What is the remote work policy?",
        "Generate an onboarding checklist for a new engineer starting 2024-03-01",
        "Give me behavioral interview questions for a Senior Software Engineer",
        "How does the performance review process work?",
        "What are the parental leave entitlements?",
    ]
    for prompt in quick_prompts:
        if st.button(prompt, use_container_width=True):
            st.session_state.pending_prompt = prompt

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────

st.title("HR Management Agent")
st.markdown("Ask me anything about HR policies, leave, onboarding, or recruitment.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("Details", expanded=False):
                st.json(msg["meta"])

# Handle quick-prompt injection
if "pending_prompt" in st.session_state:
    user_input = st.session_state.pop("pending_prompt")
else:
    user_input = st.chat_input("Type your HR question…")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("HR Agent is thinking…"):
            graph = build_hr_graph()
            initial_state: dict[str, Any] = {
                "messages": [HumanMessage(content=user_input)],
                "intent": "",
                "employee_id": employee_id,
                "context": {},
            }
            result = asyncio.run(graph.ainvoke(initial_state))

        # Extract answer
        messages = result["messages"]
        ai_messages = [
            m for m in messages
            if hasattr(m, "content")
            and not isinstance(m, HumanMessage)
            and not hasattr(m, "tool_call_id")
        ]
        answer = ai_messages[-1].content if ai_messages else "I couldn't process your request."

        # Collect metadata
        tools_used: list[str] = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools_used.extend(tc["name"] for tc in msg.tool_calls)

        meta = {
            "intent": result.get("intent", "general"),
            "employee_id": result.get("employee_id"),
            "tools_used": tools_used,
        }

        st.markdown(answer)
        if tools_used:
            with st.expander(f"Tools used: {', '.join(tools_used)}", expanded=False):
                st.json(meta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": meta,
    })
