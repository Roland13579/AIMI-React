import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

type TimeFrame = "week" | "month" | "year";

const Forecast: React.FC = () => {
  const [totalSalesData, setTotalSalesData] = useState<any>(null);
  const [totalProfitsData, setTotalProfitsData] = useState<any>(null);
  const [topProducts, setTopProducts] = useState<any[]>([]);
  const [timeFrame, setTimeFrame] = useState<TimeFrame>("week");

  // Function to fetch forecast data based on selected time frame
  const fetchForecastData = (selectedTimeFrame: TimeFrame) => {
    // Fetch total sales data
    fetch(`http://127.0.0.1:5000/forecast/total-sales?time_frame=${selectedTimeFrame}`)
      .then((res) => res.json())
      .then((data) => setTotalSalesData(data))
      .catch((err) => console.error("Error fetching total sales data:", err));

    // Fetch total profits data
    fetch(`http://127.0.0.1:5000/forecast/total-profits?time_frame=${selectedTimeFrame}`)
      .then((res) => res.json())
      .then((data) => setTotalProfitsData(data))
      .catch((err) => console.error("Error fetching total profits data:", err));

    // Fetch top 5 products
    fetch("http://127.0.0.1:5000/forecast/top-products")
      .then((res) => res.json())
      .then((data) => setTopProducts(data))
      .catch((err) => console.error("Error fetching top products:", err));
  };

  // Handle time frame change
  const handleTimeFrameChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newTimeFrame = e.target.value as TimeFrame;
    setTimeFrame(newTimeFrame);
    fetchForecastData(newTimeFrame);
  };

  useEffect(() => {
    // Initial data fetch
    fetchForecastData(timeFrame);
  }, []);

  const renderGraph = (data: any, title: string) => {
    if (!data) return <p>Loading {title}...</p>;

    const chartData = {
      labels: data.labels,
      datasets: [
        {
          label: title,
          data: data.values,
          borderColor: "rgba(75, 192, 192, 1)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          fill: true,
        },
      ],
    };

    return <Line data={chartData} />;
  };

  // Get the appropriate title suffix based on time frame
  const getTimeFrameTitle = () => {
    switch (timeFrame) {
      case "week":
        return "Weeks";
      case "month":
        return "Months";
      case "year":
        return "Years";
      default:
        return "Weeks";
    }
  };

  return (
    <div className="container mt-4">
      <h2>Forecast</h2>

      {/* Time frame selector */}
      <div className="mb-4">
        <label htmlFor="timeFrame" className="form-label">
          Time Frame:
        </label>
        <select
          id="timeFrame"
          className="form-select"
          value={timeFrame}
          onChange={handleTimeFrameChange}
          style={{ width: "200px" }}
        >
          <option value="week">Weekly</option>
          <option value="month">Monthly</option>
          <option value="year">Yearly</option>
        </select>
      </div>

      <div className="mb-5">
        <h4>Total Sales (Past 2-3 {getTimeFrameTitle()} + Prediction)</h4>
        {renderGraph(totalSalesData, "Total Sales")}
      </div>

      <div className="mb-5">
        <h4>Total Profits (Past 2-3 {getTimeFrameTitle()} + Prediction)</h4>
        {renderGraph(totalProfitsData, "Total Profits")}
      </div>

      <div>
        <h4>Top 5 Products for Next {timeFrame === "week" ? "Week" : timeFrame === "month" ? "Month" : "Year"}</h4>
        {topProducts.length > 0 ? (
          <ul>
            {topProducts.map((product, index) => (
              <li key={index}>
                <strong>{product.name}</strong> (SKU: {product.sku}) - Predicted
                Increase: {product.predicted_increase}
              </li>
            ))}
          </ul>
        ) : (
          <p>Loading top products...</p>
        )}
      </div>
    </div>
  );
};

export default Forecast;
