from phi.tools import Toolkit
import sqlite3
import pandas as pd

class CustomSQLTools(Toolkit):
  
    def __init__(self, db_path: str):
        super().__init__(name="custom_sql_tools")
        self.db_path = db_path
        
        # Register functions as tools
        self.register(self.execute_query)
        self.register(self.get_schema)
        self.register(self.get_sample_data)
        self.register(self.get_column_stats)
        self.register(self.search_data)
    
    def execute_query(self, query: str) -> str:
        """
        Execute a SQL query and return results.
        
        Args:
            query: The SQL query to execute
        
        Returns:
            Query results as a formatted string
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return "Query returned no results."
            
            # Format output nicely
            result = f"Query Results ({len(df)} rows):\n\n"
            result += df.to_markdown(index=False)
            return result
            
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def get_schema(self, table_name: str = "sales") -> str:
        """
        Get the schema (column names and types) for a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            Table schema information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get sample values for each column
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = cursor.fetchone()
            
            conn.close()
            
            result = f"Schema for table '{table_name}':\n\n"
            result += "| Column | Type | Sample Value |\n"
            result += "|--------|------|-------------|\n"
            
            for i, col in enumerate(columns):
                col_name = col[1]
                col_type = col[2]
                sample_val = sample[i] if sample else "N/A"
                result += f"| {col_name} | {col_type} | {sample_val} |\n"
            
            return result
            
        except Exception as e:
            return f"Error getting schema: {str(e)}"
    
    def get_sample_data(self, table_name: str = "sales", limit: int = 5) -> str:
        """
        Get sample rows from a table.
        
        Args:
            table_name: Name of the table
            limit: Number of rows to return
        
        Returns:
            Sample data as formatted string
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
            conn.close()
            
            return f"Sample data from '{table_name}':\n\n{df.to_markdown(index=False)}"
            
        except Exception as e:
            return f"Error getting sample data: {str(e)}"
    
    def get_column_stats(self, table_name: str = "sales", column: str = "total_revenue") -> str:
        """
        Get statistics for a numeric column.
        
        Args:
            table_name: Name of the table
            column: Name of the column to analyze
        
        Returns:
            Column statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f"""
            SELECT 
                COUNT({column}) as count,
                MIN({column}) as min_value,
                MAX({column}) as max_value,
                AVG({column}) as avg_value,
                SUM({column}) as total
            FROM {table_name}
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            result = f"Statistics for '{column}' in '{table_name}':\n\n"
            result += f"- Count: {df['count'].iloc[0]:,}\n"
            result += f"- Min: {df['min_value'].iloc[0]:,.2f}\n"
            result += f"- Max: {df['max_value'].iloc[0]:,.2f}\n"
            result += f"- Average: {df['avg_value'].iloc[0]:,.2f}\n"
            result += f"- Total: {df['total'].iloc[0]:,.2f}\n"
            
            return result
            
        except Exception as e:
            return f"Error getting column stats: {str(e)}"
    
    def search_data(
        self,
        table_name: str = "sales",
        column: str = "product",
        search_term: str = "Laptop"
    ) -> str:
        """
        Search for records matching a term.
        
        Args:
            table_name: Name of the table
            column: Column to search in
            search_term: Term to search for
        
        Returns:
            Matching records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f"""
            SELECT * FROM {table_name}
            WHERE {column} LIKE '%{search_term}%'
            LIMIT 20
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return f"No records found matching '{search_term}' in column '{column}'"
            
            return f"Found {len(df)} records matching '{search_term}':\n\n{df.to_markdown(index=False)}"
            
        except Exception as e:
            return f"Error searching data: {str(e)}"