import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import dateutil.parser

def get_weekly_sales_data():
    """
    Connects to MongoDB, retrieves sales transaction data, and aggregates it by week
    to create a pandas DataFrame with the following columns:
    - item_name
    - sku_number_of_item
    - year
    - week_number
    - total_sales_in_week
    - total_qty_sold_in_week
    """
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb+srv://aimi_admin:SC2006t3@cluster0.frqdlsi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        db = client["aimi_inventory"]
        sales_collection = db["sales"]
        
        # Retrieve all sales transactions
        print("Retrieving sales data from MongoDB...")
        sales_data = list(sales_collection.find({}, {"_id": 0}))
        print(f"Retrieved {len(sales_data)} transactions")
        
        if not sales_data:
            print("No sales data found. Using mock data for testing...")
            # Use mock data for testing if no data is found
            sales_data = [
                {
                    "transaction_id": "TRX001",
                    "sku": "LAP001",
                    "item_name": "Laptop",
                    "quantity": 2,
                    "transaction_date": "2025-03-10T10:30:00Z",
                    "total_price": 2400.0
                },
                {
                    "transaction_id": "TRX002",
                    "sku": "PHN001",
                    "item_name": "Smartphone",
                    "quantity": 1,
                    "transaction_date": "2025-03-09T14:45:00Z",
                    "total_price": 700.0
                },
                {
                    "transaction_id": "TRX003",
                    "sku": "AUD001",
                    "item_name": "Headphones",
                    "quantity": 3,
                    "transaction_date": "2025-03-08T09:15:00Z",
                    "total_price": 450.0
                },
                {
                    "transaction_id": "TRX004",
                    "sku": "LAP001",
                    "item_name": "Laptop",
                    "quantity": 1,
                    "transaction_date": "2025-03-15T11:30:00Z",
                    "total_price": 1200.0
                },
                {
                    "transaction_id": "TRX005",
                    "sku": "PHN001",
                    "item_name": "Smartphone",
                    "quantity": 2,
                    "transaction_date": "2025-03-16T13:20:00Z",
                    "total_price": 1400.0
                }
            ]
        
        # Process data
        processed_data = []
        for transaction in sales_data:
            try:
                # Parse the date (handling potential format variations)
                transaction_date = dateutil.parser.parse(transaction["transaction_date"])
                
                # Get year and week number
                iso_calendar = transaction_date.isocalendar()
                year = iso_calendar[0]
                week_number = iso_calendar[1]
                
                # Create a record with required fields
                processed_data.append({
                    "item_name": transaction["item_name"],
                    "sku_number_of_item": transaction["sku"],
                    "year": year,
                    "week_number": week_number,
                    "sales_amount": transaction["total_price"],
                    "quantity_sold": transaction["quantity"]
                })
            except (KeyError, ValueError) as e:
                print(f"Error processing transaction: {transaction.get('transaction_id', 'unknown')}, Error: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Group by item_name, sku_number_of_item, year, and week_number
        grouped_df = df.groupby(["item_name", "sku_number_of_item", "year", "week_number"]).agg({
            "sales_amount": "sum",
            "quantity_sold": "sum"
        }).reset_index()
        
        # Rename columns to match requirements
        grouped_df.rename(columns={
            "sales_amount": "total_sales_in_week",
            "quantity_sold": "total_qty_sold_in_week"
        }, inplace=True)
        
        return grouped_df
    
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Get the weekly sales data
    weekly_sales_df = get_weekly_sales_data()
    
    # Display the result
    if not weekly_sales_df.empty:
        print("\nWeekly Sales Data:")
        print(weekly_sales_df)
        
        # Optionally save to CSV
        weekly_sales_df.to_csv("weekly_sales_data.csv", index=False)
        print("\nData saved to 'weekly_sales_data.csv'")
    else:
        print("Failed to generate weekly sales data")
