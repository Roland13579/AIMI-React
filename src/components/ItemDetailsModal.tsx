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
  error?: string;
}

const ItemDetailsModal: React.FC<ItemDetailsModalProps> = ({
  item,
  onClose,
}) => {
  const [timeFrame, setTimeFrame] = useState<"week" | "month" | "year">("week");
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

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

  // Prepare chart data by properly separating historical and prediction data
  const prepareChartData = () => {
    if (!forecastData) return [];

    const { historical_data, prediction_data, time_frame } = forecastData;

    // Determine which field to use based on time frame
    let timeField: string;
    let valueField: string;

    if (time_frame === "week") {
      timeField = "week_number";
      valueField = "total_sales_in_week";
    } else if (time_frame === "month") {
      timeField = "month_number";
      valueField = "total_sales_in_month";
    } else {
      timeField = "year";
      valueField = "total_sales_in_year";
    }

    // Map historical data - ensure we show at least the last 2-3 data points
    const historicalChartData = historical_data.map((item) => ({
      time: `${item.year}-${item[timeField]}`,
      historical: item[valueField],
      predicted: null, // No prediction for historical data
    }));

    // Map prediction data
    const predictionChartData = prediction_data.map((item) => ({
      time: `${item.year}-${item[timeField]}`,
      historical: null, // No historical data for future periods
      predicted: item[valueField],
    }));

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
      <div className="modal-dialog modal-lg">
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
                <h6>Item Details</h6>
                <p>
                  <strong>SKU:</strong> {item.SKU}
                </p>
                <p>
                  <strong>Quantity:</strong> {item.quantity}
                </p>
                <p>
                  <strong>Cost Price ($):</strong> {item.cost_price}
                </p>
                <p>
                  <strong>Selling Price ($):</strong> {item.selling_price}
                </p>
                <p>
                  <strong>Description:</strong> {item.description}
                </p>
                <p>
                  <strong>Reorder Point:</strong> {item.reorder_point}
                </p>
                <p>
                  <strong>Expiration Date:</strong> {item.expiration_date}
                </p>
              </div>
              <div className="col-md-6">
                <h6>Sales Forecast</h6>
                <div className="mb-3">
                  <div className="btn-group" role="group">
                    <button
                      type="button"
                      className={`btn ${
                        timeFrame === "week" ? "btn-primary" : "btn-outline-primary"
                      }`}
                      onClick={() => setTimeFrame("week")}
                    >
                      Weekly
                    </button>
                    <button
                      type="button"
                      className={`btn ${
                        timeFrame === "month" ? "btn-primary" : "btn-outline-primary"
                      }`}
                      onClick={() => setTimeFrame("month")}
                    >
                      Monthly
                    </button>
                    <button
                      type="button"
                      className={`btn ${
                        timeFrame === "year" ? "btn-primary" : "btn-outline-primary"
                      }`}
                      onClick={() => setTimeFrame("year")}
                    >
                      Yearly
                    </button>
                  </div>
                </div>
                {loading && <p>Loading forecast data...</p>}
                {error && <p className="text-danger">{error}</p>}
                {!loading && !error && forecastData && chartData.length > 0 ? (
                  <div style={{ width: "100%", height: 300 }}>
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
                        <Tooltip />
                        <Legend />
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
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="alert alert-info">
                    {error ? (
                      <p>{error}</p>
                    ) : loading ? (
                      <p>Loading forecast data...</p>
                    ) : (
                      <p>No forecast data available for this item. Try adding more sales transactions.</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ItemDetailsModal;
