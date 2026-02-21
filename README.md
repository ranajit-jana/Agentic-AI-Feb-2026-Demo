# Agentic AI - LangChain Multi-Provider Setup

A LangChain project supporting multiple LLM providers: **Gemini**, **Anthropic**, **Groq**, and **OpenAI**.

---

## Prerequisites

- Python 3.10+
- `pip`

---

## 1. Clone / Open the Project

```bash
cd "Agentic AI Feb 2026"
```

---

## 2. Create & Activate Virtual Environment

```bash
python -m venv venv
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Install the provider package(s) you plan to use:

| Provider  | Install command                          |
|-----------|------------------------------------------|
| Gemini    | `pip install langchain-google-genai`     |
| Anthropic | `pip install langchain-anthropic`        |
| Groq      | `pip install langchain-groq`             |
| OpenAI    | `pip install langchain-openai`           |

---

## 4. Configure Environment Variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

Open `.env` and add the key for the provider you want to use:

```env
# Choose one provider and set its key
LLM_PROVIDER=gemini        # options: gemini | anthropic | groq | openai

GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Where to get API keys

| Provider  | URL                                         |
|-----------|---------------------------------------------|
| Gemini    | https://aistudio.google.com/app/apikey      |
| Anthropic | https://console.anthropic.com/settings/keys |
| Groq      | https://console.groq.com/keys               |
| OpenAI    | https://platform.openai.com/api-keys        |

---

## 5. Run the Project

```bash
python main.py
```

The active provider is controlled by `LLM_PROVIDER` in your `.env` file.

---

## Project Structure

```
Agentic AI Feb 2026/
├── main.py            # Main entry point
├── runnable-demo.py   # LCEL two-chain pipeline demo (see below)
├── router-chain.py    # LCEL router chain demo (see below)
├── requirements.txt   # Core dependencies
├── .env               # Your API keys (not committed)
├── .env.example       # Template for .env
└── README.md          # This file
```

---

## LCEL Runnable Demo (`runnable-demo.py`)

Demonstrates **LangChain Expression Language (LCEL)** by wiring two independent chains into a single pipeline using the `|` operator and `RunnablePassthrough`.

### How it works

```
topic ──► [Story Chain] ──► story ──► [Review Chain] ──► review
```

| Chain | Model | Temperature | Role |
|-------|-------|-------------|------|
| Story Creator | GPT-5.2 | 0.8 | Generates a 3–4 paragraph short story from a given topic |
| Content Reviewer | GPT-4o | 0.2 | Reviews the story for family-friendliness and returns a verdict, reason, and suggestions |

### Key LCEL concepts used

- **`prompt | llm | parser`** — basic chain composition
- **`RunnablePassthrough.assign()`** — adds new keys to the passing dict without dropping existing ones, allowing both `topic` and `story` to remain available downstream
- **`RunnableLambda`** — wraps a plain Python function so it can participate in the pipeline

### Running the demo

```bash
python runnable-demo.py
```

The demo runs with the built-in topic *"a young wizard who discovers a hidden library beneath the ocean"*. Change the argument passed to `run_demo()` at the bottom of the file to try other topics.

> **Note:** Requires `OPENAI_API_KEY` to be set in your `.env` file.

---

## LCEL Router Chain (`router-chain.py`)

Demonstrates **conditional routing** in LCEL by classifying a user message and dispatching it to the appropriate specialist chain.

### How it works

```
User message
  → [Classifier Chain]  — returns "billing" | "support" | "general"
  → [RunnableBranch]    — branches to the matching specialist chain
  → [Billing / Support / General Chain]
  → Response
```

| Chain | Role |
|-------|------|
| Classifier | Reads the message and outputs exactly one category label |
| Billing | Handles payment questions, invoices, subscriptions, and refunds |
| Support | Troubleshoots technical issues, bugs, and account problems |
| General | Handles greetings, company info, and everything else |

### Key LCEL concepts used

- **`RunnableBranch`** — conditional branching; evaluates predicate lambdas in order and executes the first matching chain, with a fallback default
- **`RunnablePassthrough.assign()`** — attaches the classifier result as `category` while keeping `message` in the dict
- **`RunnableLambda`** — wraps the classifier chain invocation so it fits into the `assign()` call

### Running the demo

```bash
python router-chain.py
```

The demo sends three test messages — one for each route — and prints the detected category and agent response for each.

> **Note:** Requires `OPENAI_API_KEY` to be set in your `.env` file.