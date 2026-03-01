import os
from typing import Optional, List, Dict, Any

from pathlib import Path
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.tools.sql import SQLTools

from tools import CustomSQLTools
    
from dotenv import load_dotenv
load_dotenv()

def create_sql_agent(db_path: str, use_custom_tools: bool = False) -> Agent:
    """
    Create a Phidata agent with SQL capabilities.
    
    Args:
        db_path: Path to the SQLite database
        use_custom_tools: Whether to use custom SQL tools (True) or built-in SQLTools (False)
    
    Returns:
        Configured Phidata Agent
    """

    db_url = f"sqlite:///{db_path}"
    
    # Choose tools based on preference
    if use_custom_tools:
        tools = [CustomSQLTools(db_path=db_path)]
        tool_description = "Custom SQL Tools with analytics"
    else:
        tools = [SQLTools(db_url=db_url), ]
        tool_description = "Built-in SQL Tools"
    
    agent = Agent(
        name="SQL Data Analyst",
        model=OpenAIChat(id="gpt-4o"),
        tools=tools,
        debug_mode=True,
        instructions=[
            "You are an expert SQL data analyst.",
            "You have access to a SQLite database with sales data.",
            "",
            "IMPORTANT GUIDELINES:",
            "1. Always use the get_schema tool first to understand the table structure.",
            "2. Write efficient SQL queries - use appropriate WHERE clauses and LIMIT.",
            "3. For aggregations, use GROUP BY with appropriate columns.",
            "4. Format results clearly using tables when appropriate.",
            "5. Explain your analysis and provide insights.",
            "6. If a query fails, explain why and suggest alternatives.",
            "",
            "AVAILABLE TABLES:",
            "- sales: Contains transaction data with columns for date, product, category,",
            "  quantity, unit_price, region, salesperson, customer_type, total_revenue, month",
            "",
            "When asked to analyze data:",
            "1. First understand what data is available (use get_schema)",
            "2. Write and execute appropriate SQL queries",
            "3. Present results clearly",
            "4. Provide insights and recommendations",
        ],
        show_tool_calls=True,
        markdown=True,
    )
    
    print(f"✅ SQL Agent created with {tool_description}")
    return agent

if __name__ == "__main__":
    agent = create_sql_agent("./data/sales_data.db")
    agent.print_response("What are the monthly sales trends?", stream=True)