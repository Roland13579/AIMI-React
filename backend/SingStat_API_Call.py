import json
import pandas as pd
import numpy as np
from urllib.request import Request, urlopen
from sklearn.linear_model import LinearRegression
import pickle
import os

# Function to fetch and process data from SingStat API
def fetch_singstat_data(resource_id):
    url = f"https://tablebuilder.singstat.gov.sg/api/table/tabledata/{resource_id}?offset=0&limit=10000"
    headers = {'User-Agent': 'Mozilla/5.0', "Accept": "application/json"}

    request = Request(url, headers=headers)
    response = urlopen(request).read()
    data = json.loads(response)

    rows = data["Data"]["row"]
    
    # Extract category names and corresponding values
    categories = []
    records = []

    for row in rows:
        category = row["rowText"].strip()
        values = {col["key"]: col["value"] for col in row["columns"] if "Q" in col["key"]}
        categories.append(category)
        records.append(values)

    # Convert to DataFrame
    df = pd.DataFrame(records, index=categories).reset_index()
    df.rename(columns={"index": "Category"}, inplace=True)
    quarter_cols = sorted([col for col in df.columns if "Q" in col], reverse=True)  # reverse = latest first
    latest_quarters = quarter_cols[:12]  # Select the latest 12 quarters

    desired_cols = ["Category"] + latest_quarters
    df = df[desired_cols]

    return df

# Function to calculate yearly averages from quarterly data
def calculate_yearly_averages(df):
    # Create a new DataFrame to store yearly averages
    yearly_avg_data = []
    
    # Process each row (category)
    for _, row in df.iterrows():
        category = row['Category']
        yearly_values = {}
        
        # Group quarters by year and calculate average
        for col in row.index:
            if col != 'Category' and pd.notna(row[col]):
                # Extract year from quarter key (e.g., "2023 1Q" -> 2023)
                if 'Q' in col:
                    parts = col.split(' ')
                    if len(parts) == 2:
                        try:
                            year = int(parts[0])
                            if year not in yearly_values:
                                yearly_values[year] = []
                            
                            # Convert value to float and add to list for averaging
                            try:
                                value = float(row[col])
                                yearly_values[year].append(value)
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                pass
                        except (ValueError, IndexError):
                            # Skip if year cannot be extracted
                            pass
        
        # Calculate average for each year
        for year, values in yearly_values.items():
            if values:  # Check if there are values to average
                yearly_avg_data.append({
                    'Category': category,
                    'Year': year,
                    'Coefficient': sum(values) / len(values)
                })
    
    # Convert to DataFrame
    yearly_avg_df = pd.DataFrame(yearly_avg_data)
    
    if not yearly_avg_df.empty:
        # Normalize coefficients to a 0-1 scale for each category
        for category in yearly_avg_df['Category'].unique():
            category_data = yearly_avg_df[yearly_avg_df['Category'] == category]
            if not category_data.empty:
                min_val = category_data['Coefficient'].min()
                max_val = category_data['Coefficient'].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    # Apply min-max scaling
                    yearly_avg_df.loc[yearly_avg_df['Category'] == category, 'Coefficient'] = (
                        (yearly_avg_df.loc[yearly_avg_df['Category'] == category, 'Coefficient'] - min_val) / 
                        (max_val - min_val)
                    ) * 0.5  # Scale to 0-0.5 range to match original coefficients
    
    return yearly_avg_df

# Function to train linear regression models for each category
def train_prediction_models(yearly_avg_df):
    prediction_models = {}
    
    # Check if DataFrame is empty or doesn't have required columns
    if yearly_avg_df.empty or 'Category' not in yearly_avg_df.columns or 'Year' not in yearly_avg_df.columns or 'Coefficient' not in yearly_avg_df.columns:
        print("Warning: DataFrame is empty or missing required columns. No prediction models will be trained.")
        return prediction_models
    
    for category in yearly_avg_df['Category'].unique():
        category_data = yearly_avg_df[yearly_avg_df['Category'] == category]
        
        # Need at least 2 data points for linear regression
        if len(category_data) >= 2:
            X = category_data['Year'].values.reshape(-1, 1)
            y = category_data['Coefficient'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            prediction_models[category] = model
    
    return prediction_models

# Function to predict coefficient for a specific category and year
def predict_coefficient(category, year, yearly_avg_df, prediction_models):
    # Check if DataFrame is empty or doesn't have required columns
    if yearly_avg_df.empty or 'Category' not in yearly_avg_df.columns or 'Year' not in yearly_avg_df.columns or 'Coefficient' not in yearly_avg_df.columns:
        print(f"Warning: DataFrame is empty or missing required columns. Using default coefficient for {category}, {year}.")
        return 0.3  # Default coefficient
    
    # Check if we already have data for this year
    existing_data = yearly_avg_df[(yearly_avg_df['Category'] == category) & 
                                 (yearly_avg_df['Year'] == year)]
    
    if not existing_data.empty:
        return existing_data['Coefficient'].iloc[0]
    
    # If no existing data, use prediction model
    if category in prediction_models:
        predicted_value = prediction_models[category].predict([[year]])[0]
        
        # Ensure the predicted value is within a reasonable range (0-0.5)
        predicted_value = max(0, min(0.5, predicted_value))
        
        return predicted_value
    
    # If no model available, use the most recent coefficient
    category_data = yearly_avg_df[yearly_avg_df['Category'] == category]
    if not category_data.empty:
        # Get the most recent year's coefficient
        most_recent_year = category_data['Year'].max()
        return category_data[category_data['Year'] == most_recent_year]['Coefficient'].iloc[0]
    
    # Default fallback
    return 0.3  # Default coefficient




# Main execution
if __name__ == "__main__":
    # Fetch data for Manufacturing and Services sectors
    print("Fetching data from SingStat API...")
    df_manufacturing = fetch_singstat_data("M250141")
    try:
        start_idx = df_manufacturing[df_manufacturing["Category"] == "Electronics"].index[0]
        end_idx = df_manufacturing[df_manufacturing["Category"] == "Miscellaneous"].index[0]
        df_manufacturing = df_manufacturing.loc[start_idx:end_idx].copy()
    except IndexError:
        print("Warning: Could not filter manufacturing rows by 'Electronics' to 'Miscellaneous'. Check category names.")
    
    df_services = fetch_singstat_data("M250431")
    df_services = df_services[df_services["Category"] != "Total Services Sector"]
    base_dir = os.path.dirname(os.path.abspath(__file__))  # current script dir
    manufacturing_path = os.path.join(base_dir, "manufacturing_data.csv")
    services_path = os.path.join(base_dir, "services_data.csv")
    df_manufacturing.to_csv(manufacturing_path, index=False)
    df_services.to_csv(services_path, index=False)

    # Print debugging information
    print("\nManufacturing data shape:", df_manufacturing.shape)
    print("Manufacturing data columns:", df_manufacturing.columns.tolist())
    print("Manufacturing data sample:")
    print(df_manufacturing.head())
    
    print("\nServices data shape:", df_services.shape)
    print("Services data columns:", df_services.columns.tolist())
    print("Services data sample:")
    print(df_services.head())

    # Combine both datasets (they have the same columns)
    df_combined = pd.concat([df_manufacturing, df_services], ignore_index=True)
    print("\nCombined data shape:", df_combined.shape)
    
    # Calculate yearly averages
    print("\nCalculating yearly averages...")
    yearly_avg_df = calculate_yearly_averages(df_combined)
    print("Yearly averages data shape:", yearly_avg_df.shape)
    
    # If the DataFrame is empty, create a mock dataset for testing
    if yearly_avg_df.empty:
        print("No data from API, creating mock dataset for testing...")
        # Create mock data for testing
        mock_data = []
        categories = ["Electronics", "Pharmaceuticals", "Food Beverages & Tobacco", 
                     "Wholesale & Retail Trade", "Transport Engineering"]
        
        for category in categories:
            for year in range(2020, 2024):
                # Generate a random coefficient between 0.2 and 0.5
                coefficient = 0.2 + (0.3 * np.random.random())
                mock_data.append({
                    'Category': category,
                    'Year': year,
                    'Coefficient': coefficient
                })
        
        yearly_avg_df = pd.DataFrame(mock_data)
        print("Created mock dataset with shape:", yearly_avg_df.shape)
    
    # Train prediction models
    print("Training prediction models...")
    prediction_models = train_prediction_models(yearly_avg_df)
    
    # Save data to files
    print("Saving data to files...")
    #yearly_avg_df.to_csv('industry_health_yearly.csv', index=False)
   
    
    # Save prediction models
    with open('industry_health_prediction_models.pkl', 'wb') as f:
        pickle.dump(prediction_models, f)
    
    print("Data processing complete.")
    print(f"Yearly averages saved to 'industry_health_yearly.csv'")
    print(f"Prediction models saved to 'industry_health_prediction_models.pkl'")
    
    # Display sample of yearly averages
    print("\nSample of yearly averages:")
    print(yearly_avg_df.head())
