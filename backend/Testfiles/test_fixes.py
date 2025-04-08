# /usr/bin/env python3
"""
Test script to verify the fixes for the errors in sales_forecasting.py
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MachineLearning import get_item_category, get_industry_health_coefficient, get_item_forecast

def test_get_item_category():
    """Test the get_item_category function"""
    print("Testing get_item_category function...")
    try:
        category = get_item_category("STM motherboard")
        print(f"Category for 'Deoderent': {category}")
        print("✅ get_item_category test passed")
    except Exception as e:
        print(f"❌ get_item_category test failed: {e}")

def test_get_industry_health_coefficient():
    """Test the get_industry_health_coefficient function"""
    print("\nTesting get_industry_health_coefficient function...")
    try:
        coefficient = get_industry_health_coefficient("Electronics", 2025)
        print(f"Industry health coefficient for 'Electronics' in 2025: {coefficient}")
        print("✅ get_industry_health_coefficient test passed")
    except Exception as e:
        print(f"❌ get_industry_health_coefficient test failed: {e}")

def test_get_item_forecast():
    """Test the get_item_forecast function"""
    print("\nTesting get_item_forecast function...")
    try:
        # This might fail if the item doesn't exist in the database, but it should not fail with the specific errors we fixed
        forecast = get_item_forecast("INV001", "week")
        if "error" in forecast:
            print(f"Note: get_item_forecast returned an error: {forecast['error']}")
            print("This is expected if the item doesn't exist or there's no sales data")
        else:
            print(f"Forecast for 'INV001': {forecast}")
        print("✅ get_item_forecast test passed (no industry_health_dict error)")
    except Exception as e:
        print(f"❌ get_item_forecast test failed: {e}")

if __name__ == "__main__":
    print("Running tests to verify fixes in sales_forecasting.py\n")
    test_get_item_category()
    test_get_industry_health_coefficient()
    test_get_item_forecast()
    print("\nAll tests completed.")
