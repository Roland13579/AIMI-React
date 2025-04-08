import React, { useEffect, useState } from "react";
import WelcomePopup from "./components/WelcomePopup";
import HelpPopup from "./components/HelpPopup";
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

import {
  Dropdown,
  Modal,
  Button,
  Card,
  Row,
  Col,
  Table,
  Badge,
} from "react-bootstrap";

const Dashboard = () => {
  const [inventory, setInventory] = useState<any[]>([]);
  const [sales, setSales] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [salesData, setSalesData] = useState<any[]>([]);
  const [totalSales, setTotalSales] = useState(0);
  const [timeFilter, setTimeFilter] = useState("monthly");
  const [servicesTop, setServicesTop] = useState<any[]>([]);
  const [manufacturingTop, setManufacturingTop] = useState<any[]>([]);

  // State for modals
  const [showLowStockWarning, setShowLowStockWarning] = useState(false);
  const [showWelcomePopup, setShowWelcomePopup] = useState(false);
  const [showHelpPopup, setShowHelpPopup] = useState(false);

  // Show welcome popup only after login/registration
  useEffect(() => {
    // Check if the welcome popup flag exists in localStorage
    const shouldShowWelcomePopup =
      localStorage.getItem("show_welcome_popup") === "true";

    if (shouldShowWelcomePopup) {
      // Show the welcome popup
      setShowWelcomePopup(true);
      // Remove the flag so it doesn't show again until next login
      localStorage.removeItem("show_welcome_popup");
    }
  }, []);

  // Fetch inventory and sales data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const inventoryResponse = await fetch(
          "http://127.0.0.1:5000/inventory"
        );
        if (!inventoryResponse.ok) throw new Error("Failed to fetch inventory");
        const inventoryData = await inventoryResponse.json();

        const salesResponse = await fetch("http://127.0.0.1:5000/sales");
        if (!salesResponse.ok) throw new Error("Failed to fetch sales");
        const salesData = await salesResponse.json();

        setInventory(inventoryData);
        setSales(salesData);

        // Check for low stock items after fetching inventory
        const lowStockItems = inventoryData.filter(
          (item: any) => item.quantity <= item.reorder_point
        );

        // We'll show the low stock warning after the welcome popup is closed
        // We don't show it immediately since the welcome popup should appear first
        // The low stock warning will be shown after the welcome popup is closed
      } catch (error) {
        console.error("Error fetching data:", error);
        setError("Failed to load data.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    const fetchSalesData = async () => {
      try {
        const response = await fetch(
          `http://127.0.0.1:5000/sales/summary?filter=${timeFilter}`
        );
        const data = await response.json();
        setSalesData(data.data);
        setTotalSales(data.total_sales);
      } catch (error) {
        console.error("Error fetching sales summary:", error);
      }
    };

    fetchSalesData();
  }, [timeFilter]);

  // Calculate low stock items (200% of reorder point)
  const lowStockItems = inventory.filter(
    (item: any) => item.quantity <= item.reorder_point * 2
  );

  // Calculate top 5 selling items
  const topSellingItems = sales
    .reduce((acc: any[], transaction: any) => {
      const item = acc.find((i: any) => i.sku === transaction.sku);
      if (item) {
        item.quantity += transaction.quantity;
      } else {
        acc.push({ ...transaction, quantity: transaction.quantity });
      }
      return acc;
    }, [])
    .sort((a: any, b: any) => b.quantity - a.quantity)
    .slice(0, 5);

  // Calculate sales activity (status counts)
  const salesActivity = sales.reduce(
    (acc: Record<string, number>, transaction: any) => {
      acc[transaction.status] += 1;
      return acc;
    },
    { pending: 0, packed: 0, shipped: 0, delivered: 0 }
  );

  useEffect(() => {
    const fetchTopSectors = async () => {
      try {
        const res = await fetch(
          "http://127.0.0.1:5000/industry-health/top-sectors"
        );
        const data = await res.json();

        const servicesData = data.services.map((sector: any) => ({
          category: sector.category,
          data: sector.trend.map((point: any) => ({
            quarter: point.quarter_label,
            value: point.value,
          })),
        }));

        const manufacturingData = data.manufacturing.map((sector: any) => ({
          category: sector.category,
          data: sector.trend.map((point: any) => ({
            quarter: point.quarter_label,
            value: point.value,
          })),
        }));

        setServicesTop(servicesData);
        setManufacturingTop(manufacturingData);
      } catch (error) {
        console.error("Error fetching industry health data:", error);
      }
    };

    fetchTopSectors();
  }, []);

  // Get badge color for status
  const getStatusBadgeColor = (status: string) => {
    switch (status) {
      case "pending":
        return "warning";
      case "packed":
        return "info";
      case "shipped":
        return "primary";
      case "delivered":
        return "success";
      default:
        return "secondary";
    }
  };

  if (loading) return <p>Loading dashboard...</p>;
  if (error) return <p>{error}</p>;
  const timeFilterLabels: Record<string, string> = {
    daily: "Last 30 days",
    monthly: "Last 12 months",
    yearly: "Last 5 years",
  };

  // Handle closing the welcome popup
  const handleCloseWelcomePopup = () => {
    setShowWelcomePopup(false);

    // Check if we need to show low stock warning after welcome popup is closed
    const lowStockItems = inventory.filter(
      (item: any) => item.quantity <= item.reorder_point
    );
    if (lowStockItems.length > 0) {
      setShowLowStockWarning(true);
    }
  };

  return (
    <div className="container-fluid mt-4">
      <div className="d-flex justify-content-between align-items-center">
        <div>
          <h2>Dashboard</h2>
          <p className="text-muted mb-4">
            Here you can find general information about your business performance,
            inventory status, and industry trends.
          </p>
        </div>
        <Button 
          variant="link" 
          className="help-bulb-button" 
          onClick={() => setShowHelpPopup(true)}
        >
          <i className="bi bi-lightbulb-fill"></i>
        </Button>
      </div>

      {/* Welcome Popup */}
      <WelcomePopup show={showWelcomePopup} onHide={handleCloseWelcomePopup} />
      
      {/* Help Popup */}
      <HelpPopup 
        show={showHelpPopup} 
        onHide={() => setShowHelpPopup(false)} 
        tabName="Dashboard"
        content={{
          features: (
            <div>
              <h5>Dashboard Overview</h5>
              <p>
                The Dashboard provides a comprehensive overview of your business performance, inventory status, and industry trends. It's designed to give you quick insights into your business operations at a glance.
              </p>
              
              <h6 className="mt-4">Key Features</h6>
              <ul>
                <li>
                  <strong>Sales Activity:</strong> Monitor the status of your sales transactions across different stages (pending, packed, shipped, delivered).
                </li>
                <li>
                  <strong>Sales Overview:</strong> Visualize your sales performance over time with customizable time frames (daily, monthly, yearly).
                </li>
                <li>
                  <strong>Low Stock Items:</strong> Quickly identify items that are running low and need to be reordered.
                </li>
                <li>
                  <strong>Top Selling Items:</strong> See which products are performing best in your inventory.
                </li>
                <li>
                  <strong>Industry Health Graphs:</strong> Track the performance of top sectors in services and manufacturing to inform your business strategy.
                </li>
              </ul>
            </div>
          ),
          calculations: (
            <div>
              <h5>Calculations & Definitions</h5>
              
              <div className="mb-4">
                <h6>Sales Activity</h6>
                <p>
                  StockSightAI tracks the status of each transaction and categorizes them into four stages:
                </p>
                <ul>
                  <li><strong>Pending:</strong> Orders that have been placed but not yet processed</li>
                  <li><strong>Packed:</strong> Orders that have been processed and packed for shipping</li>
                  <li><strong>Shipped:</strong> Orders that have been dispatched to customers</li>
                  <li><strong>Delivered:</strong> Orders that have been successfully delivered</li>
                </ul>
              </div>
              
              <div className="mb-4">
                <h6>Sales Overview</h6>
                <p>
                  The sales chart displays your total sales over time. StockSightAI aggregates transaction data to show:
                </p>
                <ul>
                  <li><strong>Daily View:</strong> Shows sales for the last 30 days</li>
                  <li><strong>Monthly View:</strong> Shows sales for the last 12 months</li>
                  <li><strong>Yearly View:</strong> Shows sales for the last 5 years</li>
                </ul>
                <p>
                  <strong>Total Sales:</strong> The sum of all transaction amounts within the selected time period.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Low Stock Items</h6>
                <p>
                  StockSightAI identifies items as "low stock" when their current quantity falls below 200% of their reorder point. This provides an early warning system before items reach critical levels.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Industry Health Indicators</h6>
                <p>
                  StockSightAI uses data from the SingStat API to calculate and display:
                </p>
                <ul>
                  <li>
                    <strong>Net Weighted Balance:</strong> A measure of business sentiment calculated as the percentage of positive responses minus the percentage of negative responses, weighted by the degree of positivity or negativity.
                  </li>
                  <li>
                    <strong>Cumulative Average:</strong> StockSightAI calculates a running average of the net weighted balance over time to smooth out short-term fluctuations and identify long-term trends.
                  </li>
                </ul>
              </div>
            </div>
          )
        }}
      />

      {/* Low Stock Warning Modal */}
      <Modal
        show={showLowStockWarning}
        onHide={() => setShowLowStockWarning(false)}
        backdrop="static" // Prevent closing by clicking outside
        keyboard={false} // Prevent closing by pressing ESC
      >
        <Modal.Header closeButton>
          <Modal.Title>⚠️ Low Stock / Out Of Stock Warning</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>
            The following items are low in stock or out of stock. Please reorder
            them as soon as possible:
          </p>
          <Table striped bordered hover>
            <thead>
              <tr>
                <th>Item Name</th>
                <th>SKU</th>
                <th>Quantity</th>
                <th>Reorder Point</th>
              </tr>
            </thead>
            <tbody>
              {lowStockItems.map((item: any, index: number) => (
                <tr key={index}>
                  <td>{item.item_name}</td>
                  <td>{item.SKU}</td>
                  <td>{item.quantity}</td>
                  <td>{item.reorder_point}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="primary"
            onClick={() => setShowLowStockWarning(false)}
          >
            Acknowledge
          </Button>
        </Modal.Footer>
      </Modal>

      {/* Sales Activity Section */}
      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Header>
              <h5>Sales Activity</h5>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col>
                  <Card className="text-center">
                    <Card.Body>
                      <h3>{salesActivity.pending}</h3>
                      <Badge bg={getStatusBadgeColor("pending")}>Pending</Badge>
                    </Card.Body>
                  </Card>
                </Col>
                <Col>
                  <Card className="text-center">
                    <Card.Body>
                      <h3>{salesActivity.packed}</h3>
                      <Badge bg={getStatusBadgeColor("packed")}>Packed</Badge>
                    </Card.Body>
                  </Card>
                </Col>
                <Col>
                  <Card className="text-center">
                    <Card.Body>
                      <h3>{salesActivity.shipped}</h3>
                      <Badge bg={getStatusBadgeColor("shipped")}>Shipped</Badge>
                    </Card.Body>
                  </Card>
                </Col>
                <Col>
                  <Card className="text-center">
                    <Card.Body>
                      <h3>{salesActivity.delivered}</h3>
                      <Badge bg={getStatusBadgeColor("delivered")}>
                        Delivered
                      </Badge>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      {/* Sales Overview Section */}
      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Header className="d-flex justify-content-between align-items-center">
              <h5>Sales Overview</h5>
              <div className="d-flex align-items-center">
                <span className="me-3">
                  Total Sales: ${totalSales.toFixed(2)}
                </span>
                <Dropdown>
                  <Dropdown.Toggle variant="primary" id="time-filter-dropdown">
                    {timeFilterLabels[timeFilter]}
                  </Dropdown.Toggle>
                  <Dropdown.Menu>
                    <Dropdown.Item onClick={() => setTimeFilter("daily")}>
                      Last 30 days
                    </Dropdown.Item>
                    <Dropdown.Item onClick={() => setTimeFilter("monthly")}>
                      Last 12 months
                    </Dropdown.Item>
                    <Dropdown.Item onClick={() => setTimeFilter("yearly")}>
                      Last 5 years
                    </Dropdown.Item>
                  </Dropdown.Menu>
                </Dropdown>
              </div>
            </Card.Header>
            <Card.Body>
              <div style={{ height: "400px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={salesData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(value) => {
                        if (timeFilter === "daily")
                          return new Date(value).toLocaleDateString();
                        if (timeFilter === "monthly")
                          return new Date(value).toLocaleString("default", {
                            month: "short",
                          });
                        return value;
                      }}
                    />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="total"
                      stroke="#8884d8"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Low Stock Items Section */}
      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Header>
              <h5>Low Stock Items</h5>
            </Card.Header>
            <Card.Body>
              <Table striped bordered hover>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Item Name</th>
                    <th>SKU</th>
                    <th>Quantity</th>
                    <th>Reorder Point</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {lowStockItems.length > 0 ? (
                    lowStockItems.map((item: any, index: number) => (
                      <tr key={index}>
                        <td>{index + 1}</td>
                        <td>{item.item_name}</td>
                        <td>{item.SKU}</td>
                        <td>{item.quantity}</td>
                        <td>{item.reorder_point}</td>
                        <td>
                          <Badge bg="danger">Low Stock</Badge>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="text-center">
                        No low stock items.
                      </td>
                    </tr>
                  )}
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Top Selling Items Section */}
      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Header>
              <h5>Top 5 Selling Items</h5>
            </Card.Header>
            <Card.Body>
              <Table striped bordered hover>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Item Name</th>
                    <th>SKU</th>
                    <th>Quantity Sold</th>
                  </tr>
                </thead>
                <tbody>
                  {topSellingItems.length > 0 ? (
                    topSellingItems.map((item: any, index: number) => (
                      <tr key={index}>
                        <td>{index + 1}</td>
                        <td>{item.item_name}</td>
                        <td>{item.sku}</td>
                        <td>{item.quantity}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={4} className="text-center">
                        No sales data available.
                      </td>
                    </tr>
                  )}
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      {/* Industry Health Graphs */}
      <Row className="mb-4">
        <Col md={6}>
          <Card>
            <Card.Header>
              <h5 className="mb-0">Top 5 Services Sectors</h5>
            </Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart
                  data={
                    servicesTop.length > 0
                      ? servicesTop[0].data.map((point: any, index: number) => {
                          const quarter = point.quarter;
                          const entry: any = { quarter };

                          servicesTop.forEach((sector: any) => {
                            if (index < sector.data.length) {
                              entry[sector.category] = sector.data[index].value;
                            }
                          });
                          return entry;
                        })
                      : []
                  }
                  margin={{ top: 5, right: 30, left: 20, bottom: 30 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="quarter"
                    tickFormatter={(value) => {
                      if (!value) return "";
                      const [year, quarter] = value.split(" ");
                      return `${quarter} '${year.slice(2)}`;
                    }}
                    height={50}
                    tick={{ fontSize: 12 }}
                    interval={0}
                  />
                  <YAxis />
                  <Tooltip
                    formatter={(value: any) => [
                      `${Number(value).toFixed(2)}`,
                      "",
                    ]}
                    labelFormatter={(label) => `Quarter: ${label}`}
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={36}
                    wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }}
                  />
                  {servicesTop.map((sector: any, index: number) => (
                    <Line
                      key={sector.category}
                      type="monotone"
                      dataKey={sector.category}
                      name={sector.category}
                      stroke={
                        ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#413ea0"][
                          index % 5
                        ]
                      }
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                      isAnimationActive={true}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div style={{ marginTop: "2rem" }}>
                {/* Heading */}
                <h3
                  style={{
                    fontSize: "1.6rem",
                    fontWeight: "bold",
                    color: "#333",
                    marginBottom: "0.75rem",
                  }}
                >
                  Ranked List of Top 5 Service Sectors in Singapore
                </h3>

                {/* Description Paragraph */}
                <p
                  style={{
                    fontSize: "1rem",
                    lineHeight: "1.6",
                    color: "#555",
                    marginBottom: "1rem",
                    maxWidth: "700px",
                  }}
                >
                  These top 5 sectors have been filtered out from the complete
                  list based on their cumulative average net weighted balance
                  over time.
                </p>

                {/* SingStat Link */}
                <p style={{ marginBottom: "0.75rem" }}>
                  <strong>Powered by </strong>
                  <a
                    href="https://tablebuilder.singstat.gov.sg/table/TS/M250431"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "#007bff",
                      textDecoration: "none",
                      fontSize: "1rem",
                    }}
                  >
                    SingStat API (Services)
                  </a>
                </p>

                {/* Footnote */}
                <p
                  style={{
                    fontSize: "0.95rem",
                    color: "#777",
                    maxWidth: "700px",
                  }}
                >
                  <em>
                    * We use cumulative averages to smooth out short-term
                    fluctuations, ensuring that enduring trends are accurately
                    captured.
                  </em>
                </p>
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          <Card>
            <Card.Header>
              <h5 className="mb-0">Top 5 Manufacturing Sectors</h5>
            </Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart
                  data={
                    manufacturingTop.length > 0
                      ? manufacturingTop[0].data.map(
                          (point: any, index: number) => {
                            const quarter = point.quarter;
                            const entry: any = { quarter };

                            manufacturingTop.forEach((sector: any) => {
                              if (index < sector.data.length) {
                                entry[sector.category] =
                                  sector.data[index].value;
                              }
                            });
                            return entry;
                          }
                        )
                      : []
                  }
                  margin={{ top: 5, right: 30, left: 20, bottom: 30 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="quarter"
                    tickFormatter={(value) => {
                      if (!value) return "";
                      const [year, quarter] = value.split(" ");
                      return `${quarter} '${year.slice(2)}`;
                    }}
                    height={50}
                    tick={{ fontSize: 12 }}
                    interval={0}
                  />
                  <YAxis />
                  <Tooltip
                    formatter={(value: any) => [
                      `${Number(value).toFixed(2)}`,
                      "",
                    ]}
                    labelFormatter={(label) => `Quarter: ${label}`}
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={36}
                    wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }}
                  />
                  {manufacturingTop.map((sector: any, index: number) => (
                    <Line
                      key={sector.category}
                      type="monotone"
                      dataKey={sector.category}
                      name={sector.category}
                      stroke={
                        ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#413ea0"][
                          index % 5
                        ]
                      }
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                      isAnimationActive={true}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div style={{ marginTop: "2rem" }}>
                {/* Heading */}
                <h3
                  style={{
                    fontSize: "1.6rem",
                    fontWeight: "bold",
                    color: "#333",
                    marginBottom: "0.75rem",
                  }}
                >
                  Ranked List of Top 5 Manufacturing Sectors in Singapore
                </h3>

                {/* Description Paragraph */}
                <p
                  style={{
                    fontSize: "1rem",
                    lineHeight: "1.6",
                    color: "#555",
                    marginBottom: "1rem",
                    maxWidth: "700px",
                  }}
                >
                  These top 5 sectors have been filtered out from the complete
                  list based on their cumulative average net weighted balance
                  over time — a robust indicator of long-term performance.
                </p>

                {/* SingStat Link */}
                <p style={{ marginBottom: "0.75rem" }}>
                  <strong>Powered by </strong>
                  <a
                    href="https://tablebuilder.singstat.gov.sg/table/TS/M250141"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "#007bff",
                      textDecoration: "none",
                      fontSize: "1rem",
                    }}
                  >
                    SingStat API (Manufacturing)
                  </a>
                </p>

                {/* Footnote */}
                <p
                  style={{
                    fontSize: "0.95rem",
                    color: "#777",
                    maxWidth: "700px",
                  }}
                >
                  <em>
                    * We use cumulative averages to smooth out short-term
                    fluctuations, ensuring that enduring trends are accurately
                    captured.
                  </em>
                </p>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
