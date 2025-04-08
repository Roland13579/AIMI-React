import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
} from "recharts";

interface ItemDetailsModalProps {
  item: any; // The item data passed as a prop
  onClose: () => void; // Function to close the modal
}

interface ForecastData {
  item_name: string;
  sku: string;
  category: string;
  time_frame: string;
  historical_data: any[];
  prediction_data: any[];
  has_confidence_intervals?: boolean;
  error?: string;
}

interface ReorderPointSuggestion {
  suggested_reorder_point: number;
  reason: string;
  loading: boolean;
  error: string | null;
}

const ItemDetailsModal: React.FC<ItemDetailsModalProps> = ({
  item,
  onClose,
}) => {
  const [timeFrame, setTimeFrame] = useState<"week" | "month" | "year">("week");
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState<boolean>(true);
  const [reorderPointSuggestion, setReorderPointSuggestion] = useState<ReorderPointSuggestion>({
    suggested_reorder_point: 0,
    reason: "",
    loading: true,
    error: null
  });

  // Fetch forecast data when the component mounts or when the time frame changes
  useEffect(() => {
    const fetchForecastData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Use item_id or SKU to fetch forecast data
        const itemId = item.item_id || item.SKU;
        const response = await fetch(
          `http://127.0.0.1:5000/forecast/${itemId}?time_frame=${timeFrame}`
        );

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
          setError(data.error);
          setForecastData(null);
        } else {
          setForecastData(data);
          
          // Calculate reorder point suggestion based on forecast data
          calculateReorderPointSuggestion(data);
        }
      } catch (err) {
        console.error("Error fetching forecast data:", err);
        setError("Failed to load forecast data. Please try again later.");
        setForecastData(null);
      } finally {
        setLoading(false);
      }
    };

    if (item && (item.item_id || item.SKU)) {
      fetchForecastData();
    }
  }, [item, timeFrame]);
  
  // Calculate reorder point suggestion based on forecast data
  const calculateReorderPointSuggestion = (data: ForecastData) => {
    setReorderPointSuggestion(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      if (!data || data.error) {
        throw new Error("No forecast data available");
      }
      
      // Get the current reorder point
      const currentReorderPoint = item.reorder_point || 0;
      
      // Get the predicted sales growth from the forecast data
      let predictedGrowth = 0;
      let growthReason = "";
      
      if (data.prediction_data && data.prediction_data.length > 0) {
        // Calculate average predicted value
        const predictedValues = data.prediction_data.map(p => {
          const valueKey = timeFrame === "week" 
            ? "total_sales_in_week" 
            : timeFrame === "month" 
              ? "total_sales_in_month" 
              : "total_sales_in_year";
          return p[valueKey];
        });
        
        const avgPredicted = predictedValues.reduce((sum, val) => sum + val, 0) / predictedValues.length;
        
        // Calculate average historical value
        const historicalValues = data.historical_data.map(h => {
          const valueKey = timeFrame === "week" 
            ? "total_sales_in_week" 
            : timeFrame === "month" 
              ? "total_sales_in_month" 
              : "total_sales_in_year";
          return h[valueKey];
        });
        
        const avgHistorical = historicalValues.length > 0 
          ? historicalValues.reduce((sum, val) => sum + val, 0) / historicalValues.length
          : 0;
        
        // Calculate growth rate
        if (avgHistorical > 0) {
          predictedGrowth = (avgPredicted - avgHistorical) / avgHistorical;
        }
        
        // Determine growth category
        if (predictedGrowth > 0.2) {
          growthReason = "high growth";
        } else if (predictedGrowth > 0.05) {
          growthReason = "moderate growth";
        } else if (predictedGrowth >= 0) {
          growthReason = "stable demand";
        } else {
          growthReason = "declining demand";
        }
      } else {
        growthReason = "insufficient forecast data";
      }
      
      // Calculate suggested reorder point
      let suggestedReorderPoint = currentReorderPoint;
      
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
      
      // Ensure minimum reorder point of 1
      suggestedReorderPoint = Math.max(1, suggestedReorderPoint);
      
      // Set the suggestion
      setReorderPointSuggestion({
        suggested_reorder_point: suggestedReorderPoint,
        reason: `Based on ${growthReason} (${(predictedGrowth * 100).toFixed(1)}%)`,
        loading: false,
        error: null
      });
      
    } catch (err) {
      console.error("Error calculating reorder point suggestion:", err);
      setReorderPointSuggestion({
        suggested_reorder_point: item.reorder_point || 0,
        reason: "Unable to calculate suggestion",
        loading: false,
        error: "Failed to calculate reorder point suggestion"
      });
    }
  };

  // Prepare chart data by properly separating historical and prediction data
  const prepareChartData = () => {
    if (!forecastData) return [];

    const { historical_data, prediction_data, time_frame, has_confidence_intervals } = forecastData;

    // Determine which field to use based on time frame
    let timeField: string;
    let valueField: string;
    let lowerBoundField: string | null = null;
    let upperBoundField: string | null = null;

    if (time_frame === "week") {
      timeField = "week_number";
      valueField = "total_sales_in_week";
      if (has_confidence_intervals) {
        lowerBoundField = "total_sales_in_week_lower";
        upperBoundField = "total_sales_in_week_upper";
      }
    } else if (time_frame === "month") {
      timeField = "month_number";
      valueField = "total_sales_in_month";
      if (has_confidence_intervals) {
        lowerBoundField = "total_sales_in_month_lower";
        upperBoundField = "total_sales_in_month_upper";
      }
    } else {
      timeField = "year";
      valueField = "total_sales_in_year";
      if (has_confidence_intervals) {
        lowerBoundField = "total_sales_in_year_lower";
        upperBoundField = "total_sales_in_year_upper";
      }
    }

    // Map historical data - ensure we show at least the last 2-3 data points
    const historicalChartData = historical_data.map((item) => ({
      time: `${item.year}-${item[timeField]}`,
      historical: item[valueField],
      predicted: null, // No prediction for historical data
      lower: null, // No confidence intervals for historical data
      upper: null, // No confidence intervals for historical data
    }));

    // Map prediction data
    const predictionChartData = prediction_data.map((item) => {
      const dataPoint: any = {
        time: `${item.year}-${item[timeField]}`,
        historical: null, // No historical data for future periods
        predicted: item[valueField],
      };

      // Add confidence intervals if available
      if (has_confidence_intervals && lowerBoundField && upperBoundField) {
        dataPoint.lower = item[lowerBoundField];
        dataPoint.upper = item[upperBoundField];
      } else {
        dataPoint.lower = null;
        dataPoint.upper = null;
      }

      return dataPoint;
    });

    // Combine both datasets and sort by time
    const combinedData = [...historicalChartData, ...predictionChartData];
    
    // Sort by time to ensure correct order
    combinedData.sort((a, b) => {
      const [yearA, periodA] = a.time.split('-').map(Number);
      const [yearB, periodB] = b.time.split('-').map(Number);
      
      if (yearA !== yearB) return yearA - yearB;
      return periodA - periodB;
    });
    
    return combinedData;
  };

  const chartData = prepareChartData();

  return (
    <div className="modal show" tabIndex={-1} style={{ display: "block" }}>
      <div className="modal-dialog modal-xl">
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">{item.item_name}</h5>
            <button
              type="button"
              className="btn-close"
              onClick={onClose}
            ></button>
          </div>
          <div className="modal-body">
            <div className="row">
              <div className="col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header bg-primary text-white">
                    <h6 className="mb-0">
                      <i className="bi bi-info-circle me-2"></i>
                      Item Details
                    </h6>
                  </div>
                  <div className="card-body">
                    <div className="mb-3">
                      <label className="text-muted small mb-1">SKU</label>
                      <div className="p-2 bg-light rounded">
                        <code>{item.SKU}</code>
                      </div>
                    </div>
                    
                    <div className="row mb-3">
                      <div className="col-6">
                        <label className="text-muted small mb-1">Quantity</label>
                        <div className="p-2 bg-light rounded d-flex align-items-center">
                          <i className="bi bi-boxes me-2 text-primary"></i>
                          <span className="fw-bold" style={{ fontSize: "0.9rem" }}>{item.quantity}</span>
                        </div>
                      </div>
                      <div className="col-6">
                        <label className="text-muted small mb-1">Reorder Point</label>
                        <div className="p-2 bg-light rounded d-flex align-items-center">
                          <i className="bi bi-arrow-repeat me-2 text-warning"></i>
                          <span style={{ fontSize: "0.9rem" }}>{item.reorder_point}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mb-3">
                      <label className="text-muted small mb-1">Reorder Point Suggestion</label>
                      <div className="p-2 bg-light rounded">
                        {reorderPointSuggestion.loading ? (
                          <div className="d-flex align-items-center">
                            <div className="spinner-border spinner-border-sm text-primary me-2" role="status">
                              <span className="visually-hidden">Loading...</span>
                            </div>
                            <span>Calculating suggestion...</span>
                          </div>
                        ) : reorderPointSuggestion.error ? (
                          <div className="text-danger">
                            <i className="bi bi-exclamation-triangle me-2"></i>
                            {reorderPointSuggestion.error}
                          </div>
                        ) : (
                          <div className="d-flex align-items-center">
                            <i className="bi bi-lightbulb me-2 text-primary"></i>
                            <div>
                              <span className="fw-bold" style={{ fontSize: "1.1rem" }}>{reorderPointSuggestion.suggested_reorder_point}</span>
                              <small className="d-block text-muted">{reorderPointSuggestion.reason}</small>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="row mb-3">
                      <div className="col-6">
                        <label className="text-muted small mb-1">Cost Price</label>
                        <div className="p-2 bg-light rounded d-flex align-items-center">
                          <i className="bi bi-tags me-2 text-danger"></i>
                          <span style={{ fontSize: "0.9rem" }}>${item.cost_price?.toFixed(2) || 'N/A'}</span>
                        </div>
                      </div>
                      <div className="col-6">
                        <label className="text-muted small mb-1">Selling Price</label>
                        <div className="p-2 bg-light rounded d-flex align-items-center">
                          <i className="bi bi-cash me-2 text-success"></i>
                          <span className="fw-bold" style={{ fontSize: "0.9rem" }}>${item.selling_price?.toFixed(2) || 'N/A'}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mb-3">
                      <label className="text-muted small mb-1">Description</label>
                      <div className="p-2 bg-light rounded">
                        <p className="mb-0" style={{ fontSize: "0.9rem" }}>{item.description || 'No description available'}</p>
                      </div>
                    </div>
                    
                    {item.expiration_date && (
                      <div className="mb-0">
                        <label className="text-muted small mb-1">Expiration Date</label>
                        <div className="p-2 bg-light rounded d-flex align-items-center">
                          <i className="bi bi-calendar-event me-2 text-secondary"></i>
                          <span style={{ fontSize: "0.9rem" }}>{item.expiration_date}</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header bg-success text-white">
                    <h6 className="mb-0">
                      <i className="bi bi-graph-up me-2"></i>
                      Sales Forecast
                    </h6>
                  </div>
                  <div className="card-body">
                    <div className="mb-3">
                      <label className="text-muted small mb-1">Time Frame</label>
                      <div className="btn-group w-100" role="group">
                        <button
                          type="button"
                          className={`btn ${
                            timeFrame === "week" ? "btn-primary" : "btn-outline-primary"
                          }`}
                          onClick={() => setTimeFrame("week")}
                        >
                          <i className="bi bi-calendar-week me-1"></i>
                          Weekly
                        </button>
                        <button
                          type="button"
                          className={`btn ${
                            timeFrame === "month" ? "btn-primary" : "btn-outline-primary"
                          }`}
                          onClick={() => setTimeFrame("month")}
                        >
                          <i className="bi bi-calendar-month me-1"></i>
                          Monthly
                        </button>
                        <button
                          type="button"
                          className={`btn ${
                            timeFrame === "year" ? "btn-primary" : "btn-outline-primary"
                          }`}
                          onClick={() => setTimeFrame("year")}
                        >
                          <i className="bi bi-calendar me-1"></i>
                          Yearly
                        </button>
                      </div>
                    </div>
                    {loading && (
                      <div className="text-center py-5">
                        <div className="spinner-border text-primary" role="status">
                          <span className="visually-hidden">Loading...</span>
                        </div>
                        <p className="mt-2 text-muted">Loading forecast data...</p>
                      </div>
                    )}
                    
                    {error && (
                      <div className="alert alert-danger">
                        <i className="bi bi-exclamation-triangle-fill me-2"></i>
                        {error}
                      </div>
                    )}
                    
                    {!loading && !error && forecastData && chartData.length > 0 ? (
                      <div>
                        {forecastData.has_confidence_intervals && (
                          <div className="mb-2 text-end">
                            <button
                              className="btn btn-sm btn-outline-secondary"
                              onClick={() => setShowConfidenceIntervals(!showConfidenceIntervals)}
                            >
                              <i className={`bi bi-${showConfidenceIntervals ? 'eye-slash' : 'eye'} me-1`}></i>
                              {showConfidenceIntervals ? "Hide" : "Show"} Confidence Intervals
                            </button>
                          </div>
                        )}
                        <div style={{ width: "100%", height: 320 }}>
                      <ResponsiveContainer>
                        <LineChart
                          data={chartData}
                          margin={{
                            top: 5,
                            right: 30,
                            left: 20,
                            bottom: 5,
                          }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="time" />
                          <YAxis />
                          <Tooltip 
                            formatter={(value, name) => {
                              if (value === null || value === undefined) return ['N/A', name];
                              return [Number(value).toFixed(2), name];
                            }}
                            contentStyle={{ 
                              backgroundColor: 'rgba(255, 255, 255, 0.9)',
                              border: '1px solid #ccc',
                              borderRadius: '4px',
                              padding: '5px 8px',
                              fontSize: '12px',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                            }}
                            itemStyle={{ padding: '2px 0' }}
                            labelStyle={{ marginBottom: '2px', fontWeight: 'bold' }}
                          />
                          <Legend />
                          
                          {/* Confidence interval area */}
                          {forecastData.has_confidence_intervals && showConfidenceIntervals && (
                            <Area
                              type="monotone"
                              dataKey="upper"
                              stroke="none"
                              fill="rgba(130, 202, 157, 0.2)"
                              activeDot={false}
                              connectNulls
                            />
                          )}
                          {forecastData.has_confidence_intervals && showConfidenceIntervals && (
                            <Area
                              type="monotone"
                              dataKey="lower"
                              stroke="none"
                              fill="rgba(130, 202, 157, 0.0)"
                              activeDot={false}
                              connectNulls
                            />
                          )}
                          
                          {/* Historical and predicted lines */}
                          <Line
                            type="monotone"
                            dataKey="historical"
                            name="Historical"
                            stroke="#8884d8"
                            activeDot={{ r: 8 }}
                            connectNulls
                          />
                          <Line
                            type="monotone"
                            dataKey="predicted"
                            name="Predicted"
                            stroke="#82ca9d"
                            strokeDasharray="5 5"
                            connectNulls
                            dot={{ r: 4 }}
                          />
                          
                          {/* Confidence interval bounds as dashed lines */}
                          {forecastData.has_confidence_intervals && showConfidenceIntervals && (
                            <Line
                              type="monotone"
                              dataKey="upper"
                              name="Upper Bound"
                              stroke="#82ca9d"
                              strokeDasharray="3 3"
                              strokeWidth={1}
                              dot={false}
                              activeDot={false}
                              connectNulls
                            />
                          )}
                          {forecastData.has_confidence_intervals && showConfidenceIntervals && (
                            <Line
                              type="monotone"
                              dataKey="lower"
                              name="Lower Bound"
                              stroke="#82ca9d"
                              strokeDasharray="3 3"
                              strokeWidth={1}
                              dot={false}
                              activeDot={false}
                              connectNulls
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                        <div className="text-center mt-1">
                          <small className="text-muted">
                            <i className="bi bi-robot me-1"></i>
                            powered by StockSight AI and SingStat API
                          </small>
                        </div>
                        
                        {forecastData.has_confidence_intervals && showConfidenceIntervals && (
                          <div className="alert alert-info mt-2 mb-0">
                            <small>
                              <i className="bi bi-info-circle me-1"></i>
                              Wider intervals indicate greater uncertainty in the forecast.
                            </small>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="alert alert-info">
                        <i className="bi bi-info-circle me-2"></i>
                        {!loading && !error && (
                          <span>No forecast data available for this item. Try adding more sales transactions.</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-primary"
              onClick={onClose}
            >
              <i className="bi bi-check-circle me-1"></i>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ItemDetailsModal;
