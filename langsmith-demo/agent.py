#!/usr/bin/env python3
"""
LangChain + LangSmith + OpenAI Agent Demo

This demo shows how to create an agent with tool calling capabilities
and use LangSmith for debugging and tracing.

Uses LangGraph (the modern replacement for AgentExecutor in LangChain 1.x+)
"""

import os
import sys
from typing import Annotated
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# LangSmith setup (if configured)
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "langsmith-demo")
    print("✅ LangSmith tracing enabled")
else:
    print("ℹ️  LangSmith not configured - tracing disabled")


# Define custom tools for the agent
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "sin(3.14)")

    Returns:
        The result of the evaluation as a string
    """
    try:
        # Use eval for simple calculations (in production, use a safer method)
        import math
        # Allow math functions
        result = eval(expression, {"__builtins__": {}}, {"math": math, **vars(math)})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Get weather information for a city.

    Args:
        city: Name of the city

    Returns:
        Mock weather information (in production, call a real weather API)
    """
    # Mock weather data - in real implementation, call a weather API
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 20°C",
        "paris": "Clear, 18°C"
    }

    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def search_web(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: Search query

    Returns:
        Mock search results (in production, integrate with a search API)
    """
    # Mock search results
    mock_results = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "langsmith": "LangSmith is a platform for debugging, testing, and monitoring LLM applications."
    }

    query_lower = query.lower()
    for key, result in mock_results.items():
        if key in query_lower:
            return f"Search results for '{query}': {result}"

    return f"No specific results found for '{query}'. Try a more specific query."


def create_agent():
    """Create an agent with tool calling capabilities using LangGraph."""

    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment")
        sys.exit(1)

    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # You can use "gpt-3.5-turbo" for cost savings
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Define the tools available to the agent
    tools = [calculator, get_weather, search_web]

    # System prompt for the agent
    system_prompt = """You are a helpful AI assistant with access to various tools.

    only reply French, and only in emoji

You can use these tools to help answer questions:
- calculator: For mathematical calculations
- get_weather: For weather information
- search_web: For general web searches

When using tools, be precise and provide clear reasoning. If you need to use multiple tools, do so step by step.

Always provide helpful, accurate responses."""

    # Create the agent using LangGraph's create_react_agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    return agent


def run_agent(agent, query: str) -> str:
    """Run the agent with a query and return the response."""
    messages = [HumanMessage(content=query)]

    # Invoke the agent
    result = agent.invoke({"messages": messages})

    # Get the last message (agent's response)
    response = result["messages"][-1].content
    return response


def run_demo():
    """Run the LangSmith demo with various example queries."""

    print("🤖 LangChain + LangSmith + OpenAI Agent Demo")
    print("=" * 50)
    print("Using LangGraph (modern replacement for AgentExecutor)")
    print()

    # Create the agent
    agent = create_agent()

    # Example queries to demonstrate different capabilities
    example_queries = [
        "What is 15 * 23 + 7 and 23 * 34 + 34?",
        "What's the weather like in Tokyo?",
        "Can you search for information about LangChain?",
        "Calculate the square root of 144",
        "What's the weather in Paris and what is 2 to the power of 8?",
        "What is new york pizza famous for?"
    ]

    print("🔍 Running example queries...\n")

    for i, query in enumerate(example_queries, 1):
        print(f"Example {i}: {query}")
        print("-" * 40)

        try:
            # Run the agent with tracing
            response = run_agent(agent, query)
            print(f"Response: {response}\n")

        except Exception as e:
            print(f"❌ Error processing query: {e}\n")

        print()


def interactive_mode():
    """Run the agent in interactive mode for manual testing."""

    print("🤖 Interactive Agent Mode")
    print("Using LangGraph (modern replacement for AgentExecutor)")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 30)

    agent = create_agent()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break

            if not user_input:
                continue

            print("Agent: ", end="", flush=True)

            # Run the agent
            response = run_agent(agent, user_input)
            print(response)

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        run_demo()