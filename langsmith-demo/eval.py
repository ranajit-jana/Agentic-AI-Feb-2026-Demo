#!/usr/bin/env python3
"""
LangSmith Evaluation & Dataset Demo

Demonstrates how to:
  1. Create a labeled test dataset in LangSmith
  2. Define custom evaluators (keyword match + LLM-as-judge)
  3. Run an experiment with langsmith.evaluation.evaluate()
  4. Print a results summary

Run:
    python eval.py
"""

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate

# Import the existing agent helpers
from agent import create_agent, run_agent

# ─── LangSmith client ────────────────────────────────────────────────────────

client = Client()

DATASET_NAME = "agent-tool-eval-v2"

# ─── 1. Test Dataset ─────────────────────────────────────────────────────────
#
# Each test case has:
#   input    – the user question sent to the agent
#   expected – a keyword / phrase that MUST appear in a correct answer
#
# Intentional failure cases are marked with  # ✗ FAIL
# to demonstrate LangSmith's ability to surface regressions.
#
TEST_CASES = [
    # Calculator tool — 2 pass / 2 fail
    {"input": "What is 15 * 23 + 7?",            "expected": "352"},          # ✓ PASS  (352)
    {"input": "What is the square root of 144?",  "expected": "999"},          # ✗ FAIL  (correct: 12)
    {"input": "What is 2 to the power of 8?",     "expected": "256"},          # ✓ PASS  (256)
    {"input": "What is 100 divided by 4?",        "expected": "999"},          # ✗ FAIL  (correct: 25)

    # Weather tool — 2 pass / 2 fail
    {"input": "What is the weather in Tokyo?",    "expected": "Rainy"},        # ✓ PASS
    {"input": "What is the weather in London?",   "expected": "Snowy"},        # ✗ FAIL  (correct: Cloudy)
    {"input": "What is the weather in Paris?",    "expected": "Clear"},        # ✓ PASS
    {"input": "What is the weather in New York?", "expected": "Foggy"},        # ✗ FAIL  (correct: Sunny)

    # Web-search tool — 1 pass / 1 fail
    {
        "input":    "Search the web for the Python programming language.",
        "expected": "its simplicity and readability",                           # ✓ PASS
    },
    {
        "input":    "Tell me about LangSmith.",
        "expected": "blockchain platform",                                      # ✗ FAIL  (correct: debugging/testing)
    },

    # Multi-step (calculator + weather) — 1 pass / 1 fail
    {
        "input":    "What is 5 * 5 and what is the weather in Paris?",
        "expected": "25",                                                       # ✓ PASS
    },
    {
        "input":    "What is 2 + 2 and what is the weather in London?",
        "expected": "99",                                                       # ✗ FAIL  (correct: 4)
    },
]


# ─── 2. Dataset helpers ───────────────────────────────────────────────────────

def create_or_get_dataset() -> str:
    """Upload the test dataset to LangSmith (skip if it already exists)."""
    existing = [ds for ds in client.list_datasets() if ds.name == DATASET_NAME]
    if existing:
        count = len(list(client.list_examples(dataset_name=DATASET_NAME)))
        print(f"✅ Using existing dataset '{DATASET_NAME}' ({count} examples)")
        return DATASET_NAME

    print(f"📦 Creating dataset '{DATASET_NAME}' in LangSmith…")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Evaluation dataset for the LangSmith agent demo. "
            "Covers the calculator, get_weather, and search_web tools, "
            "plus multi-step queries."
        ),
    )

    client.create_examples(
        inputs=[{"input": tc["input"]} for tc in TEST_CASES],
        outputs=[{"expected": tc["expected"]} for tc in TEST_CASES],
        dataset_id=dataset.id,
    )

    print(f"✅ Dataset created with {len(TEST_CASES)} examples")
    return DATASET_NAME


# ─── 3. Target function ───────────────────────────────────────────────────────
#
# evaluate() calls this once per dataset example.
# Build the agent lazily so we reuse it across calls.

_agent = None


def agent_target(inputs: dict) -> dict:
    """Invoke the agent and return its response."""
    global _agent
    if _agent is None:
        _agent = create_agent()
    response = run_agent(_agent, inputs["input"])
    return {"output": response}


# ─── 4. Evaluators ───────────────────────────────────────────────────────────
#
# Each evaluator receives (run, example) and must return a dict with:
#   key     – metric name shown in LangSmith
#   score   – numeric value (0 / 1 here)
#   comment – optional human-readable explanation


def contains_keyword(run, example) -> dict:
    """
    Keyword evaluator: passes when the expected phrase appears
    (case-insensitive) anywhere in the agent's response.
    """
    expected = (example.outputs or {}).get("expected", "").lower()
    actual   = (run.outputs   or {}).get("output",   "").lower()
    passed   = bool(expected) and (expected in actual)
    return {
        "key":     "contains_keyword",
        "score":   int(passed),
        "comment": f"Looking for '{expected}'" + (" ✓" if passed else " ✗"),
    }


def llm_correctness(run, example) -> dict:
    """
    LLM-as-judge evaluator.

    Asks gpt-4o-mini to rate whether the agent's response correctly
    answers the question (score 1) or not (score 0).
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an objective evaluator. "
            "Given a user question, an expected-answer hint, and an agent response, "
            "output ONLY valid JSON: {{\"score\": <0 or 1>, \"reason\": \"<brief reason>\"}}.\n"
            "Score 1 if the response is factually correct and addresses the question; "
            "0 otherwise.",
        ),
        (
            "human",
            "Question: {question}\n"
            "Expected hint: {expected}\n"
            "Agent response: {response}",
        ),
    ])

    chain = prompt | llm
    result = chain.invoke({
        "question": (example.inputs  or {}).get("input",    ""),
        "expected":  (example.outputs or {}).get("expected", ""),
        "response":  (run.outputs     or {}).get("output",   ""),
    })

    try:
        match  = re.search(r"\{.*\}", result.content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        score  = int(bool(parsed.get("score", 0)))
        reason = parsed.get("reason", "")
    except Exception:
        score, reason = 0, "Could not parse LLM judge output"

    return {"key": "llm_correctness", "score": score, "comment": reason}


# ─── 5. Run experiment ────────────────────────────────────────────────────────

def run_experiment(prefix: str = "agent-eval") -> None:
    """
    Upload the dataset (if needed) and run the evaluation experiment
    against the agent, then print a summary table.
    """
    print("\n🧪 LangSmith Evaluation Experiment")
    print("=" * 50)

    dataset_name = create_or_get_dataset()

    print(f"\n🔄 Evaluating agent against '{dataset_name}'…")
    print("   (Each example invokes the live agent — this may take a minute)\n")

    results = evaluate(
        agent_target,
        data=dataset_name,
        evaluators=[contains_keyword, llm_correctness],
        experiment_prefix=prefix,
        metadata={
            "model":       "gpt-4o-mini",
            "tools":       ["calculator", "get_weather", "search_web"],
            "description": "Baseline evaluation of the demo agent",
        },
        max_concurrency=2,   # run 2 examples in parallel to stay rate-limit safe
    )

    # ─── 6. Summary ──────────────────────────────────────────────────────────
    print("\n📊 Results Summary")
    print("=" * 50)

    score_buckets: dict[str, list[float]] = {}
    for result in results:
        eval_results = (result.get("evaluation_results") or {}).get("results") or []
        for er in eval_results:
            score_buckets.setdefault(er.key, []).append(er.score or 0)

    if score_buckets:
        for metric, values in score_buckets.items():
            avg    = sum(values) / len(values)
            passed = sum(v > 0 for v in values)
            print(f"  {metric:25s}: {avg:.0%}  ({passed}/{len(values)} passed)")
    else:
        print("  No scores collected — check LangSmith UI for details.")

    project = os.getenv("LANGSMITH_PROJECT", "langsmith-demo")
    print(f"\n🔗 View full results → https://smith.langchain.com  (project: {project})\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ LANGSMITH_API_KEY is not set. Add it to your .env file.")
        raise SystemExit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is not set. Add it to your .env file.")
        raise SystemExit(1)

    run_experiment()
