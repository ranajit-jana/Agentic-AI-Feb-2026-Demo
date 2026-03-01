import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

class CSVToSQLite:
    
    def __init__(self, db_path: str = "./data/sales_data.db"):
        """
        Initialize the CSV to SQLite converter.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.db_url = f"sqlite:///{db_path}"
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def create_sample_csv(self, csv_path: str = "./data/sales_data.csv") -> str:
        """Create sample sales data CSV for demonstration."""
        
        # Ensure directory exists
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive sample data
        data = {
            "transaction_id": list(range(1, 51)),
            "date": [
                "2024-01-05", "2024-01-08", "2024-01-12", "2024-01-15", "2024-01-18",
                "2024-01-22", "2024-01-25", "2024-01-28", "2024-02-01", "2024-02-05",
                "2024-02-08", "2024-02-12", "2024-02-15", "2024-02-18", "2024-02-22",
                "2024-02-25", "2024-02-28", "2024-03-01", "2024-03-05", "2024-03-08",
                "2024-03-12", "2024-03-15", "2024-03-18", "2024-03-22", "2024-03-25",
                "2024-03-28", "2024-04-01", "2024-04-05", "2024-04-08", "2024-04-12",
                "2024-04-15", "2024-04-18", "2024-04-22", "2024-04-25", "2024-04-28",
                "2024-05-01", "2024-05-05", "2024-05-08", "2024-05-12", "2024-05-15",
                "2024-05-18", "2024-05-22", "2024-05-25", "2024-05-28", "2024-06-01",
                "2024-06-05", "2024-06-08", "2024-06-12", "2024-06-15", "2024-06-18",
            ],
            "product": [
                "Laptop", "Smartphone", "Tablet", "Laptop", "Headphones",
                "Smartphone", "Monitor", "Keyboard", "Mouse", "Laptop",
                "Tablet", "Smartphone", "Headphones", "Webcam", "Laptop",
                "Monitor", "Keyboard", "Mouse", "Smartphone", "Tablet",
                "Laptop", "Headphones", "Monitor", "Webcam", "Keyboard",
                "Mouse", "Laptop", "Smartphone", "Tablet", "Headphones",
                "Monitor", "Webcam", "Keyboard", "Mouse", "Laptop",
                "Smartphone", "Tablet", "Headphones", "Monitor", "Webcam",
                "Keyboard", "Mouse", "Laptop", "Smartphone", "Tablet",
                "Headphones", "Monitor", "Webcam", "Keyboard", "Mouse",
            ],
            "category": [
                "Electronics", "Electronics", "Electronics", "Electronics", "Accessories",
                "Electronics", "Electronics", "Accessories", "Accessories", "Electronics",
                "Electronics", "Electronics", "Accessories", "Accessories", "Electronics",
                "Electronics", "Accessories", "Accessories", "Electronics", "Electronics",
                "Electronics", "Accessories", "Electronics", "Accessories", "Accessories",
                "Accessories", "Electronics", "Electronics", "Electronics", "Accessories",
                "Electronics", "Accessories", "Accessories", "Accessories", "Electronics",
                "Electronics", "Electronics", "Accessories", "Electronics", "Accessories",
                "Accessories", "Accessories", "Electronics", "Electronics", "Electronics",
                "Accessories", "Electronics", "Accessories", "Accessories", "Accessories",
            ],
            "quantity": [
                2, 5, 3, 1, 10, 4, 2, 8, 15, 3,
                4, 6, 12, 5, 2, 3, 10, 20, 7, 5,
                2, 8, 2, 4, 12, 18, 1, 5, 3, 15,
                2, 6, 9, 22, 2, 4, 4, 10, 3, 5,
                11, 25, 1, 6, 3, 14, 2, 7, 8, 30,
            ],
            "unit_price": [
                1200.00, 800.00, 500.00, 1200.00, 150.00,
                800.00, 350.00, 75.00, 25.00, 1200.00,
                500.00, 800.00, 150.00, 85.00, 1200.00,
                350.00, 75.00, 25.00, 800.00, 500.00,
                1200.00, 150.00, 350.00, 85.00, 75.00,
                25.00, 1200.00, 800.00, 500.00, 150.00,
                350.00, 85.00, 75.00, 25.00, 1200.00,
                800.00, 500.00, 150.00, 350.00, 85.00,
                75.00, 25.00, 1200.00, 800.00, 500.00,
                150.00, 350.00, 85.00, 75.00, 25.00,
            ],
            "region": [
                "North", "South", "East", "West", "North",
                "South", "East", "West", "North", "South",
                "East", "West", "North", "South", "East",
                "West", "North", "South", "East", "West",
                "North", "South", "East", "West", "North",
                "South", "East", "West", "North", "South",
                "East", "West", "North", "South", "East",
                "West", "North", "South", "East", "West",
                "North", "South", "East", "West", "North",
                "South", "East", "West", "North", "South",
            ],
            "salesperson": [
                "Alice", "Bob", "Charlie", "Diana", "Alice",
                "Bob", "Charlie", "Diana", "Alice", "Bob",
                "Charlie", "Diana", "Alice", "Bob", "Charlie",
                "Diana", "Alice", "Bob", "Charlie", "Diana",
                "Alice", "Bob", "Charlie", "Diana", "Alice",
                "Bob", "Charlie", "Diana", "Alice", "Bob",
                "Charlie", "Diana", "Alice", "Bob", "Charlie",
                "Diana", "Alice", "Bob", "Charlie", "Diana",
                "Alice", "Bob", "Charlie", "Diana", "Alice",
                "Bob", "Charlie", "Diana", "Alice", "Bob",
            ],
            "customer_type": [
                "Business", "Consumer", "Consumer", "Business", "Consumer",
                "Business", "Business", "Consumer", "Consumer", "Business",
                "Consumer", "Consumer", "Business", "Consumer", "Business",
                "Business", "Consumer", "Consumer", "Business", "Consumer",
                "Business", "Consumer", "Business", "Consumer", "Business",
                "Consumer", "Business", "Consumer", "Business", "Consumer",
                "Business", "Consumer", "Business", "Consumer", "Business",
                "Consumer", "Business", "Consumer", "Business", "Consumer",
                "Business", "Consumer", "Business", "Consumer", "Business",
                "Consumer", "Business", "Consumer", "Business", "Consumer",
            ],
        }
        
        df = pd.DataFrame(data)
        
        # Calculate total revenue
        df["total_revenue"] = df["quantity"] * df["unit_price"]
        
        # Add month column for easier analysis
        df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
        
        df.to_csv(csv_path, index=False)
        print(f"✅ Sample CSV created at: {csv_path}")
        return csv_path
    
    def load_csv_to_sqlite(
        self,
        csv_path: str,
        table_name: str = "sales",
        if_exists: str = "replace"
    ) -> str:
        """
        Load a CSV file into SQLite database.
        
        Args:
            csv_path: Path to the CSV file
            table_name: Name of the table to create
            if_exists: What to do if table exists ('replace', 'append', 'fail')
        
        Returns:
            Database URL for SQLAlchemy
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Connect to SQLite and load data
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        
        # Create indexes for better query performance
        cursor = conn.cursor()
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_product ON {table_name}(product)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_region ON {table_name}(region)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_salesperson ON {table_name}(salesperson)")
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ CSV loaded into SQLite database: {self.db_path}")
        print(f"   Table name: {table_name}")
        print(f"   Records: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        return self.db_url
    
    def get_table_info(self, table_name: str = "sales") -> Dict[str, Any]:
        """Get information about a table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "table_name": table_name,
            "columns": [(col[1], col[2]) for col in columns],
            "row_count": row_count
        }
    
if __name__ == "__main__":
    db_setup = CSVToSQLite(db_path="./data/sales_data.db")
    csv_path = db_setup.create_sample_csv("./data/sales_data.csv")
    db_setup.load_csv_to_sqlite(csv_path, table_name="sales")