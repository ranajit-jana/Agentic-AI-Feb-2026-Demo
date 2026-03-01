# Phidata Agent Demo

A basic example of building a conversational AI agent using [Phidata](https://www.phidata.com/) and OpenAI.

## Overview

This demo creates a simple agent named **Jarvis** powered by GPT-4o. It demonstrates the minimal setup required to get a Phidata agent running with streaming responses.

## Prerequisites

- Python 3.8+
- An OpenAI API key

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in this directory with your API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

Run the basic agent:

```bash
python basic.py
```

This will invoke the agent with the prompt `"what is the value of Pi?"` and stream the response to the terminal.

## Web Search Agent — Tool Interaction

Sequence diagram showing how `agent_with_websearch.py` orchestrates tools and the LLM:

```mermaid
sequenceDiagram
    actor User
    participant Agent as Jarvis (Phidata Agent)
    participant LLM as GPT-4o (OpenAI)
    participant DateTime as get_current_datetime
    participant DDG as DuckDuckGo
    participant News as Newspaper4k

    User->>Agent: Submit query (e.g. "new Hollywood movies")
    Agent->>LLM: Forward query + tool definitions + instructions
    LLM-->>Agent: Tool call: get_current_datetime()
    Agent->>DateTime: Execute get_current_datetime()
    DateTime-->>Agent: "2026-03-01 10:45:00"
    Agent->>LLM: Return current date/time
    LLM-->>Agent: Tool call: DuckDuckGo.search(query)
    Agent->>DDG: Execute web search
    DDG-->>Agent: Search results (titles, snippets, URLs)
    Agent->>LLM: Return search results
    LLM-->>Agent: Tool call: Newspaper4k.read_article(url)
    Agent->>News: Scrape full article from URL
    News-->>Agent: Full article content
    Agent->>LLM: Return article content
    LLM-->>Agent: Final synthesised response (streamed)
    Agent-->>User: Stream formatted markdown response
```

## Files

| File | Description |
|------|-------------|
| `basic.py` | Creates and runs a single conversational agent |
| `agent_with_websearch.py` | Agent with web search, scraping, and datetime tools |
| `requirements.txt` | Python dependencies |

## Dependencies

| Package | Purpose |
|---------|---------|
| `phidata` | Agent framework |
| `openai` | LLM backend (GPT-4o) |
| `python-dotenv` | Load API keys from `.env` |
| `duckduckgo-search` | Web search tool (available for extension) |
| `yfinance` | Financial data tool (available for extension) |
| `newspaper4k` | Article scraping tool (available for extension) |
| `lancedb` | Vector store for memory/RAG (available for extension) |
| `sqlalchemy` | Database support (available for extension) |
