import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import "chart.js/auto";
import { Card, Row, Col, Badge, Button } from "react-bootstrap";
import HelpPopup from "./components/HelpPopup";

type TimeFrame = "week" | "month" | "year";

interface SummaryMetric {
  title: string;
  value: string | number;
  trend: "up" | "down" | "neutral";
  percentage?: number;
  color: string;
}

const Forecast: React.FC = () => {
  const [totalSalesData, setTotalSalesData] = useState<any>(null);
  const [totalProfitsData, setTotalProfitsData] = useState<any>(null);
  const [topProducts, setTopProducts] = useState<any[]>([]);
  const [timeFrame, setTimeFrame] = useState<TimeFrame>("week");
  const [salesMetrics, setSalesMetrics] = useState<SummaryMetric[]>([]);
  const [profitMetrics, setProfitMetrics] = useState<SummaryMetric[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [showHelpPopup, setShowHelpPopup] = useState(false);

  // Function to fetch forecast data based on selected time frame
  const fetchForecastData = (selectedTimeFrame: TimeFrame) => {
    setIsLoading(true);
    
    // Fetch total sales data
    fetch(`http://127.0.0.1:5000/forecast/total-sales?time_frame=${selectedTimeFrame}`)
      .then((res) => res.json())
      .then((data) => {
        setTotalSalesData(data);
        
        // Generate sales metrics based on the data
        if (data && data.values && data.values.length > 0) {
          const currentValue = data.values[data.values.length - 1];
          const previousValue = data.values[data.values.length - 2] || 0;
          const growthRate = previousValue ? ((currentValue - previousValue) / previousValue) * 100 : 0;
          
          // Calculate confidence based on coefficient of variation with time frame-specific scaling
          const values = data.values.slice(0, -1); // Exclude prediction
          const mean = values.reduce((sum: number, val: number) => sum + val, 0) / values.length;
          const variance = values.reduce((sum: number, val: number) => sum + Math.pow(val - mean, 2), 0) / values.length;
          
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
          
          setSalesMetrics([
            {
              title: "Predicted Sales",
              value: `$${currentValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
              trend: growthRate >= 0 ? "up" : "down",
              percentage: Math.abs(growthRate),
              color: growthRate >= 0 ? "success" : "danger"
            },
            {
              title: "Sales Growth",
              value: `${growthRate >= 0 ? "+" : ""}${growthRate.toFixed(1)}%`,
              trend: growthRate >= 0 ? "up" : "down",
              color: growthRate >= 0 ? "success" : "danger"
            },
            {
              title: "Forecast Confidence",
              value: `${confidenceLevel.toFixed(0)}%`,
              trend: confidenceLevel > 70 ? "up" : confidenceLevel > 40 ? "neutral" : "down",
              color: confidenceLevel > 70 ? "success" : confidenceLevel > 40 ? "warning" : "danger"
            }
          ]);
        }
      })
      .catch((err) => console.error("Error fetching total sales data:", err));

    // Fetch total profits data
    fetch(`http://127.0.0.1:5000/forecast/total-profits?time_frame=${selectedTimeFrame}`)
      .then((res) => res.json())
      .then((data) => {
        setTotalProfitsData(data);
        
        // Store profit data for later use when setting metrics
        if (data && data.values && data.values.length > 0) {
          const currentValue = data.values[data.values.length - 1];
          const previousValue = data.values[data.values.length - 2] || 0;
          const growthRate = previousValue ? ((currentValue - previousValue) / previousValue) * 100 : 0;
          
          // Calculate profit margin (simplified - assuming we have sales data)
          const marginTrend = growthRate > 0 ? "up" : growthRate < 0 ? "down" : "neutral";
          
          // We'll set the complete profit metrics after fetching top products
          const partialProfitMetrics: SummaryMetric[] = [
            {
              title: "Predicted Profit",
              value: `$${currentValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
              trend: growthRate >= 0 ? "up" : "down",
              percentage: Math.abs(growthRate),
              color: growthRate >= 0 ? "success" : "danger"
            },
            {
              title: "Profit Margin Trend",
              value: `${growthRate >= 0 ? "+" : ""}${growthRate.toFixed(1)}%`,
              trend: marginTrend as "up" | "down" | "neutral",
              color: marginTrend === "up" ? "success" : marginTrend === "down" ? "danger" : "warning"
            }
          ];
          
          // We'll add the Top Growth Product metric after fetching top products
          setProfitMetrics(partialProfitMetrics);
        }
      })
      .catch((err) => console.error("Error fetching total profits data:", err));

    // Fetch top 5 products
    fetch("http://127.0.0.1:5000/forecast/top-products")
      .then((res) => res.json())
      .then((data) => {
        setTopProducts(data);
        
        // Now that we have top products data, update the profit metrics to include Top Growth Product
        if (data && data.length > 0) {
          // Find the product with highest predicted increase
          const topProduct = data.reduce((prev: any, current: any) => 
            (current.predicted_increase > prev.predicted_increase) ? current : prev
          );
          
          // Check if Top Growth Product metric already exists
          setProfitMetrics(prevMetrics => {
            // Filter out any existing Top Growth Product metrics to prevent duplicates
            const filteredMetrics = prevMetrics.filter(metric => metric.title !== "Top Growth Product");
            
            // Add the Top Growth Product metric
            return [
              ...filteredMetrics,
              {
                title: "Top Growth Product",
                value: topProduct.name,
                trend: "up" as const,
                color: "success"
              }
            ];
          });
        }
        
        setIsLoading(false);
        
        setIsLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching top products:", err);
        setIsLoading(false);
      });
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

    const options = {
      plugins: {
        tooltip: {
          callbacks: {
            label: function(context: any) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              if (context.parsed.y !== null) {
                label += Number(context.parsed.y).toFixed(2);
              }
              return label;
            }
          }
        }
      }
    };
    
    return <Line data={chartData} options={options} />;
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

  // Render a metric card
  const renderMetricCard = (metric: SummaryMetric) => {
    return (
      <Card className="h-100 shadow-sm">
        <Card.Body className="d-flex flex-column">
          <Card.Title className="text-muted fs-6">{metric.title}</Card.Title>
          <div className="d-flex align-items-center mt-2">
            <h3 className="mb-0 me-2">{metric.value}</h3>
            {metric.trend !== "neutral" && (
              <span>
                <i 
                  className={`bi bi-arrow-${metric.trend === "up" ? "up" : "down"}-circle-fill text-${metric.color}`} 
                  style={{ fontSize: "1.2rem" }}
                ></i>
                {metric.percentage && (
                  <Badge bg={metric.color} className="ms-1">
                    {metric.percentage.toFixed(1)}%
                  </Badge>
                )}
              </span>
            )}
          </div>
        </Card.Body>
      </Card>
    );
  };

  return (
    <div className="container-fluid mt-4">
      <div className="mb-4">
        <div className="d-flex justify-content-between align-items-center">
          <div>
            <h2>Forecast</h2>
            <small className="text-muted">powered by StockSight AI and SingStat API</small>
          </div>
          
          <div className="d-flex align-items-center">
            {/* Help Bulb Button */}
            <Button 
              variant="link" 
              className="help-bulb-button me-3" 
              onClick={() => setShowHelpPopup(true)}
            >
              <i className="bi bi-lightbulb-fill"></i>
            </Button>
            
            {/* Time frame selector */}
            <div className="d-flex align-items-center">
              <label htmlFor="timeFrame" className="form-label me-2 mb-0">
                Time Frame:
              </label>
              <select
                id="timeFrame"
                className="form-select"
                value={timeFrame}
                onChange={handleTimeFrameChange}
                style={{ width: "150px" }}
              >
                <option value="week">Weekly</option>
                <option value="month">Monthly</option>
                <option value="year">Yearly</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      
      {/* Help Popup */}
      <HelpPopup 
        show={showHelpPopup} 
        onHide={() => setShowHelpPopup(false)} 
        tabName="Forecast"
        content={{
          features: (
            <div>
              <h5>Forecast Overview</h5>
              <p>
                The Forecast tab provides AI-powered predictions for your business's future performance. 
                It helps you anticipate sales trends, optimize inventory, and make data-driven decisions.
              </p>
              
              <h6 className="mt-4">Key Features</h6>
              <ul>
                <li>
                  <strong>Sales Metrics:</strong> View predicted sales, growth rates, and forecast confidence levels.
                </li>
                <li>
                  <strong>Profit Metrics:</strong> Analyze predicted profits, profit margin trends, and identify top growth products.
                </li>
                <li>
                  <strong>Sales Forecast Chart:</strong> Visualize past sales data alongside predictions for future periods.
                </li>
                <li>
                  <strong>Profit Forecast Chart:</strong> Track and predict profit trends over time.
                </li>
                <li>
                  <strong>Top Products Prediction:</strong> Identify which products are expected to perform best in the upcoming period.
                </li>
                <li>
                  <strong>Time Frame Selection:</strong> Switch between weekly, monthly, and yearly views to analyze different time horizons.
                </li>
              </ul>
            </div>
          ),
          calculations: (
            <div>
              <h5>Calculations & Definitions</h5>
              
              <div className="mb-4">
                <h6>Predicted Sales</h6>
                <p>
                  StockSightAI uses time series forecasting models to predict future sales based on historical data. The prediction algorithm:
                </p>
                <ul>
                  <li>Analyzes past sales patterns to identify trends, seasonality, and cyclical patterns</li>
                  <li>Incorporates external factors like industry health indicators from SingStat API</li>
                  <li>Uses machine learning to generate predictions for future time periods</li>
                </ul>
                <p>
                  <strong>Sales Growth:</strong> Calculated as the percentage change between the current predicted value and the previous period's value:
                  <br />
                  <code>Growth Rate = ((Current Value - Previous Value) / Previous Value) × 100%</code>
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Forecast Confidence</h6>
                <p>
                  StockSightAI calculates confidence levels based on:
                </p>
                <ul>
                  <li><strong>Coefficient of Variation (CV):</strong> A measure of relative variability calculated as the ratio of the standard deviation to the mean of historical data</li>
                  <li><strong>Time Frame Scaling:</strong> Different scaling factors are applied based on the selected time frame (weekly, monthly, yearly)</li>
                  <li><strong>Data Point Consideration:</strong> The number of available historical data points affects confidence (more data generally leads to higher confidence)</li>
                </ul>
                <p>
                  The confidence formula is:
                  <br />
                  <code>Confidence = 100 - (CV × Scaling Factor × 100 × Data Point Factor)</code>
                </p>
                <p>
                  Higher confidence levels (70%+) indicate more reliable forecasts, while lower levels suggest greater uncertainty.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Profit Predictions</h6>
                <p>
                  StockSightAI calculates predicted profits by:
                </p>
                <ul>
                  <li>Analyzing historical profit margins for each product</li>
                  <li>Applying these margins to predicted sales volumes</li>
                  <li>Adjusting for known cost changes and market trends</li>
                </ul>
                <p>
                  <strong>Profit Margin Trend:</strong> Shows the directional movement of profit margins over time, helping identify if profitability is improving or declining.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Top Growth Products</h6>
                <p>
                  StockSightAI identifies products with the highest predicted percentage increase in sales. The algorithm:
                </p>
                <ul>
                  <li>Calculates the predicted percentage increase for each product</li>
                  <li>Ranks products based on this metric</li>
                  <li>Presents the top 5 products with the highest growth potential</li>
                </ul>
                <p>
                  This helps you focus inventory and marketing efforts on products with the highest growth potential.
                </p>
              </div>
            </div>
          )
        }}
      />

      {/* Summary Metrics Section */}
      <Row className="mb-4 g-3">
        <Col md={6}>
          <Card className="shadow-sm">
            <Card.Header className="bg-white">
              <h5 className="mb-0">Sales Metrics</h5>
            </Card.Header>
            <Card.Body>
              <Row className="g-2">
                {isLoading ? (
                  <div className="text-center py-3">
                    <div className="spinner-border text-primary" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                  </div>
                ) : (
                  salesMetrics.map((metric, index) => (
                    <Col key={index} xs={12}>
                      {renderMetricCard(metric)}
                    </Col>
                  ))
                )}
              </Row>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={6}>
          <Card className="shadow-sm">
            <Card.Header className="bg-white">
              <h5 className="mb-0">Profit Metrics</h5>
            </Card.Header>
            <Card.Body>
              <Row className="g-2">
                {isLoading ? (
                  <div className="text-center py-3">
                    <div className="spinner-border text-primary" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                  </div>
                ) : (
                  profitMetrics.map((metric, index) => (
                    <Col key={index} xs={12}>
                      {renderMetricCard(metric)}
                    </Col>
                  ))
                )}
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Charts Section */}
      <Row className="mb-4">
        <Col md={12}>
          <Card className="shadow-sm">
            <Card.Header className="bg-white">
              <h5 className="mb-0">Total Sales (Past 2-3 {getTimeFrameTitle()} + Prediction)</h5>
            </Card.Header>
            <Card.Body>
              {renderGraph(totalSalesData, "Total Sales")}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={12}>
          <Card className="shadow-sm">
            <Card.Header className="bg-white">
              <h5 className="mb-0">Total Profits (Past 2-3 {getTimeFrameTitle()} + Prediction)</h5>
            </Card.Header>
            <Card.Body>
              {renderGraph(totalProfitsData, "Total Profits")}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={12}>
          <Card className="shadow-sm">
            <Card.Header className="bg-white">
              <h5 className="mb-0">Top 5 Products for Next {timeFrame === "week" ? "Week" : timeFrame === "month" ? "Month" : "Year"}</h5>
            </Card.Header>
            <Card.Body>
              {topProducts.length > 0 ? (
                <ul className="list-group">
                  {topProducts.map((product, index) => (
                    <li key={index} className="list-group-item d-flex justify-content-between align-items-center">
                      <div>
                        <strong>{product.name}</strong> (SKU: {product.sku})
                      </div>
                      <span className="badge bg-success rounded-pill">
                        +{product.predicted_increase}%
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-center py-3">
                  {isLoading ? (
                    <div className="spinner-border text-primary" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                  ) : (
                    <p>No top products data available</p>
                  )}
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Forecast;
