'''
This file uses the SARIMAX model to predict the weekly sales for the next coming week 

Takes in item name, total weekly sales, total weekly qty, industry category, week number

in this file it also takes in a "holiday bonus" into account but you can leave that out for now.

also we are using example data here. so i need you to integrate it with the rest of the application.
'''


import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Example data
data = pd.DataFrame({
    'name': ['item1', 'item2', 'item3', 'item4', 'item5', 'item6'],
    'total_sales': [100, 120, 150, 130, 160, 200],  # Total sales for each week
    'qty': [10, 15, 12, 8, 10, 11],  # Quantity sold
    'category': ['electronics', 'fashion', 'health', 'tools', 'electronics', 'fashion'],
    'holiday_name': [None, None, 'christmas', None, 'black_friday', None],  # Holiday name for the week
    'week': [1, 2, 3, 4, 5, 6],
})

# Industry health dictionary
industry_health_dict = {
    "electronics": 0.4,
    "health": 0.3,
    "fashion": 0.5,
    "tools": 0.2,
}

# Holiday boost dictionary
holiday_boost_dict = {
    "christmas": 0.2,  # 20% boost for Christmas
    "black_friday": 0.15,  # 15% boost for Black Friday
}

# Add industry health column based on category
data['industry_health'] = data['category'].map(industry_health_dict)

# Define the dependent variable (sales)
y = data['total_sales']

# Define the exogenous variables (industry health)
X = data[['industry_health']]

# Fit SARIMAX model (example with (1, 0, 1) ARIMA order, no seasonal terms for simplicity)
season = 1 #4,13,52
model = SARIMAX(y, exog=X, order=(1, 0, 1), seasonal_order=(1,0,1,season))
results = model.fit()

# Make predictions for the next period
predictions = results.predict(start=0, end=5, exog=X)

# Add predictions to the dataframe
data['predictions'] = predictions

# Get prediction for item 5 (electronics) for next week
item_5_prediction = data[data['name'] == 'item5']['predictions'].iloc[0]

# Apply holiday boost for item 5 if it's during the holiday season (Black Friday)
holiday_name = data[data['name'] == 'item5']['holiday_name'].iloc[0]
holiday_boost = holiday_boost_dict.get(holiday_name, 0)  # Default to 0 if no holiday

final_prediction = item_5_prediction * (1 + holiday_boost)

# Output final prediction for item 5 for next week
print(f"Final sales prediction for item 5 for next week: {final_prediction}")

print(data[['name', 'predictions']])
