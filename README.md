# StockSight - Inventory Management & Business Forecasting System

<p align="center">
  <img src="https://github.com/Roland13579/AIMI-React/blob/main/images/logo2.png" width="300">
</p>

StockSight is a comprehensive inventory management and business forecasting system that leverages AI and real-time market data from the SingStat API to provide accurate sales predictions and business insights.

## Features

### 1. User Authentication & Management

- Secure login and registration system
- Email verification for new accounts
- Role-based access control (Manager/Employee)
- User profile management

### 2. Dashboard

- Real-time overview of business metrics
- Sales activity tracking (Pending, Packed, Shipped, Delivered)
- Low stock warnings and alerts
- Top selling items analysis
- Industry health indicators powered by SingStat data

### 3. Inventory Management

- Complete inventory tracking system
- Add, edit, and delete inventory items
- Low stock alerts and reorder point management
- Detailed item information with modal view
- Inventory status visualization
- Negative stock alerts to prevent overselling
- AI-powered reorder point suggestions based on sales trends

### 4. Sales Tracking

- Comprehensive sales data recording
- Sales history and transaction management
- Sales performance metrics and analysis
- Revenue and profit tracking

### 5. Purchase Orders

- Create and manage purchase orders
- Track order status and fulfillment
- Supplier management
- Order history and documentation

### 6. AI-Powered Forecasting

- **SingStat API Integration** for real-time industry data
- Machine learning models for sales prediction
- Industry health analysis and trend forecasting
- Product category classification
- Confidence intervals for predictions
- Time-frame adjustable forecasts (weekly, monthly, yearly)
- Focused metrics for sales and profit analysis

### SingStat API Integration

<p align="center">
  <img src="https://github.com/Roland13579/AIMI-React/blob/main/images/singstat.png" width="400">
</p>

StockSight leverages the Singapore Department of Statistics (SingStat) API to provide accurate industry insights and improve sales forecasting. The integration includes:

1. **Real-time Data Fetching**
   - Automated retrieval of latest economic indicators
   - Regular updates of industry performance metrics
   - Historical data analysis for trend identification

2. **Industry Health Analysis**
   - Manufacturing sector performance tracking
   - Services sector performance evaluation
   - Calculation of industry health coefficients
   - Trend analysis and future projections

3. **Enhanced Forecasting**
   - Industry-specific growth predictions
   - Market trend incorporation into sales forecasts
   - Confidence interval adjustments based on industry stability
   - Seasonal adjustments using historical SingStat data

4. **Category-specific Insights**
   - Product categorization aligned with SingStat sectors
   - Category-specific performance metrics
   - Comparative analysis across different industry segments
   - Growth opportunity identification

## Getting Started

### Prerequisites
- Node.js (v16+)
- Python (v3.8+)
- MongoDB

### Installation (Backend-Only)

1. Clone the repository
```bash
git clone https://github.com/yourusername/stocksight.git
cd stocksight
```

2. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables
```bash
# Create .env file in backend directory
touch .env
# Add necessary environment variables
```

4. Start the backend server
```bash
python run.py
```

5. The front end is already hosted through Vercel so go to https://stocksight-pink.vercel.app/dashboard to experience the website!

### Installation (Frond-end and Back-end)

1. Complete steps 1-4 from the above procedure to deploy the backend.

2. Install frontend dependencies
```bash
npm install
```

3. Start the frontend development server
```bash
cd ..
npm run dev
```

4. Access the application at `http://localhost:xxxx` where `xxxx` is the port you have set for hosting the frontend (See the output after running `npm run dev`).

## Technical Stack

### Frontend
- **React** (v18.2.0) - UI library
- **TypeScript** - Type-safe JavaScript
- **React Router** (v7.2.0) - Navigation and routing
- **React Bootstrap** - UI components and responsive design
- **Bootstrap Icons** - Icon library
- **Chart.js & Recharts** - Data visualization
- **Vite** - Build tool and development server

### Backend
- **Python** - Backend language
- **Flask** - Web framework
- **MongoDB** - Database for inventory and sales data
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning for classification
- **statsmodels** - Time series forecasting (SARIMAX)
- **sentence-transformers** - NLP for product categorization

### APIs and External Services
- **SingStat API** - Singapore's official statistics data
  - Manufacturing sector performance metrics
  - Services sector performance metrics
  - Economic indicators and trends
  - Industry health coefficients

## Mathematical Models & Algorithms

### 1. Forecast Confidence Calculation

The forecast confidence is calculated using an advanced time frame-specific approach based on the coefficient of variation:

```javascript
// Calculate confidence based on coefficient of variation with time frame-specific scaling
const values = data.values.slice(0, -1); // Exclude prediction
const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;

// Add time frame-specific scaling
let scalingFactor;
switch(selectedTimeFrame) {
  case "week":
    scalingFactor = 3; // Lower scaling factor for weekly data (higher variance expected)
    break;
  case "month":
    scalingFactor = 5; // Medium scaling factor for monthly data
    break;
  case "year":
    scalingFactor = 8; // Higher scaling factor for yearly data (lower variance expected)
    break;
  default:
    scalingFactor = 5;
}

// Calculate coefficient of variation (CV)
const cv = mean > 0 ? Math.sqrt(variance) / mean : 1;

// Calculate confidence with adjusted formula and data point consideration
const dataPointFactor = Math.min(1, values.length / 10); // Consider number of data points (max effect at 10+ points)
const confidenceLevel = Math.max(0, Math.min(100, 100 - (cv * scalingFactor * 100) * dataPointFactor));
```

This improved formula addresses several statistical challenges:

1. **Coefficient of Variation (CV)**: Uses the standard deviation divided by the mean, which is a more standardized measure of dispersion than the simple variance/mean ratio.

2. **Time Frame Adaptation**: Applies different scaling factors based on the time frame:
   - Weekly data (scalingFactor = 3): Accounts for naturally higher variance in weekly data
   - Monthly data (scalingFactor = 5): Moderate scaling for monthly aggregations
   - Yearly data (scalingFactor = 8): Higher scaling for yearly data which typically has lower variance

3. **Data Quantity Consideration**: Incorporates a `dataPointFactor` that adjusts confidence based on the amount of historical data available, with maximum effect at 10+ data points.

This approach produces more balanced and accurate confidence levels across different time frames, avoiding the extremes of 0% or 100% that could occur with simpler formulas.

### 2. Reorder Point Suggestion Algorithm

The reorder point suggestion uses predicted sales growth to adjust the current reorder point:

```javascript
// Calculate suggested reorder point based on predicted growth
if (predictedGrowth > 0.2) {
  // High growth: increase by 50%
  suggestedReorderPoint = Math.ceil(currentReorderPoint * 1.5);
} else if (predictedGrowth > 0.05) {
  // Moderate growth: increase by 20%
  suggestedReorderPoint = Math.ceil(currentReorderPoint * 1.2);
} else if (predictedGrowth >= 0) {
  // Stable: keep the same or slight increase
  suggestedReorderPoint = Math.ceil(currentReorderPoint * 1.05);
} else {
  // Declining: decrease by 10%
  suggestedReorderPoint = Math.max(1, Math.floor(currentReorderPoint * 0.9));
}
```

The predicted growth is calculated by comparing historical and predicted sales:

```javascript
// Calculate average predicted value
const avgPredicted = predictedValues.reduce((sum, val) => sum + val, 0) / predictedValues.length;

// Calculate average historical value
const avgHistorical = historicalValues.reduce((sum, val) => sum + val, 0) / historicalValues.length;

// Calculate growth rate
if (avgHistorical > 0) {
  predictedGrowth = (avgPredicted - avgHistorical) / avgHistorical;
}
```

### 3. SARIMAX Time Series Forecasting

The system uses SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) for time series forecasting:

```python
# Fit SARIMAX model
if seasonal_period > 1 and len(item_data) >= seasonal_period:
    model = SARIMAX(y, exog=X, order=(1, 0, 1), seasonal_order=(1, 0, 1, seasonal_period))
else:
    model = SARIMAX(y, exog=X, order=(1, 0, 1))
    
results = model.fit(disp=False)
```

The SARIMAX model parameters:
- **order=(1,0,1)**: Represents (p,d,q) where p=1 is the AR term, d=0 means no differencing, and q=1 is the MA term
- **seasonal_order=(1,0,1,seasonal_period)**: Represents (P,D,Q,s) for seasonal components
- **exog=X**: Industry health coefficients used as exogenous variables

For limited data, a simpler linear growth model is used:

```python
# Calculate historical growth rate
historical_growth = (last_value / first_value) ** (1 / (len(item_data) - 1)) - 1

# Use industry health as a factor in the growth rate
growth_rate = historical_growth * (1 + industry_health)
```

### 4. Industry Health Coefficient Calculation

Industry health coefficients are calculated from SingStat data:

```python
# Calculate cumulative average
cumulative_sum = 0
count = 0

for col in quarter_cols:
    if col <= target_col:  # Only include up to target quarter
        value = float(category_data[col].iloc[0])
        cumulative_sum += value
        count += 1

cumulative_avg = cumulative_sum / count

# Normalize to 0-0.5 range
min_val = min(values)
max_val = max(values)

normalized_avg = ((cumulative_avg - min_val) / (max_val - min_val)) * 0.5
```

This creates a normalized coefficient between 0 and 0.5 that represents the health of an industry sector, with higher values indicating stronger performance.

### 5. Confidence Interval Calculation

Confidence intervals for sales predictions use standard statistical methods:

```python
# For SARIMAX models
forecast_results = results.get_forecast(steps=periods, exog=future_exog)
confidence_intervals = forecast_results.conf_int(alpha=0.05)  # 95% confidence interval

# For simpler models
confidence_factor = 1.96 * growth_std * np.sqrt(i + 1)  # 95% confidence interval
lower_bound = max(0, next_value * (1 - confidence_factor))
upper_bound = next_value * (1 + confidence_factor)
```

The factor 1.96 corresponds to the 95% confidence level in a normal distribution, and the square root term increases the interval width for predictions further into the future.

### 6. Negative Stock Prevention

The system prevents selling more items than available in inventory with a simple comparison:

```javascript
// Check if there's enough inventory
if (quantity > selectedItem.quantity) {
  alert(`Cannot sell more than available inventory. Available: ${selectedItem.quantity}`);
  return;
}
```

## Machine Learning Components

1. **Product Categorization**
   - **NLP-based Embedding**: Uses sentence-transformers (all-MiniLM-L6-v2) to convert product names into 384-dimensional vectors
   - **Classification Model**: Pre-trained classifier that maps embeddings to industry categories
   - **Implementation**:
     ```python
     # Convert to embedding
     item_embedding = embedding_model.encode([item_name])
     # Predict category
     category = clf.predict(item_embedding)[0]
     ```

2. **Sales Prediction Models**
   - **SARIMAX Time Series Model**: Captures trend, seasonality, and external factors
   - **Linear Regression Fallback**: Used when insufficient data for SARIMAX
   - **Industry Health Integration**: Adjusts predictions based on industry performance
   - **Confidence Interval Calculation**: Widens with prediction distance
   - **Intelligent Reorder Point Suggestions**: Dynamically adjusts based on predicted growth

3. **Industry Health Prediction**
   - **Cumulative Trend Analysis**: Aggregates quarterly performance data
   - **Normalization**: Scales industry performance to 0-0.5 range
   - **Growth Projection**: Extrapolates trends for future quarters
   - **Implementation**:
     ```python
     # Apply growth to the latest cumulative average
     predicted_value = latest_cumulative_avg + (growth * quarters_ahead)
     
     # Normalize to 0-0.5 range
     normalized_value = ((predicted_value - min_val) / (max_val - min_val)) * 0.5
     ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [SingStat API](https://tablebuilder.singstat.gov.sg/api/table/tabledata/) for providing valuable economic data
- [React Bootstrap](https://react-bootstrap.github.io/) for UI components
- [Recharts](https://recharts.org/) for data visualization
- [scikit-learn](https://scikit-learn.org/) for machine learning capabilities

## Authors

<p align="center">
  <img src="https://github.com/Roland13579/AIMI-React/blob/main/images/logo.png" width="400">
</p>

- Garv Sachdev [garv001@e.ntu.edu.sg]
- Gong Yuelong [ygong005@e.ntu.edu.sg]
