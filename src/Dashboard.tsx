import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Dropdown, Modal, Button } from "react-bootstrap"; // Import Modal and Button
import { Card, Row, Col, Table, Badge } from "react-bootstrap";

const Dashboard = () => {
  const [inventory, setInventory] = useState<any[]>([]);
  const [sales, setSales] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [salesData, setSalesData] = useState<any[]>([]);
  const [totalSales, setTotalSales] = useState(0);
  const [timeFilter, setTimeFilter] = useState("monthly");

  // State for low stock warning modal
  const [showLowStockWarning, setShowLowStockWarning] = useState(false);

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
          (item) => item.quantity <= item.reorder_point * 2
        );
        if (lowStockItems.length > 0) {
          setShowLowStockWarning(true); // Show warning modal if there are low stock items
        }
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
    (item) => item.quantity <= item.reorder_point * 2
  );

  // Calculate top 5 selling items
  const topSellingItems = sales
    .reduce((acc, transaction) => {
      const item = acc.find((i) => i.sku === transaction.sku);
      if (item) {
        item.quantity += transaction.quantity;
      } else {
        acc.push({ ...transaction, quantity: transaction.quantity });
      }
      return acc;
    }, [])
    .sort((a, b) => b.quantity - a.quantity)
    .slice(0, 5);

  // Calculate sales activity (status counts)
  const salesActivity = sales.reduce(
    (acc, transaction) => {
      acc[transaction.status] += 1;
      return acc;
    },
    { pending: 0, packed: 0, shipped: 0, delivered: 0 }
  );

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
  const timeFilterLabels = {
    daily: "Last 30 days",
    monthly: "Last 12 months",
    yearly: "Last 5 years",
  };

  return (
    <div className="container-fluid mt-4">
      <h2>Dashboard</h2>

      {/* Low Stock Warning Modal */}
      <Modal
        show={showLowStockWarning}
        onHide={() => setShowLowStockWarning(false)}
        backdrop="static" // Prevent closing by clicking outside
        keyboard={false} // Prevent closing by pressing ESC
      >
        <Modal.Header closeButton>
          <Modal.Title>⚠️ Low Stock Warning</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>
            The following items are low in stock. Please reorder them as soon as
            possible:
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
              {lowStockItems.map((item, index) => (
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
                    lowStockItems.map((item, index) => (
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
                    topSellingItems.map((item, index) => (
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
    </div>
  );
};

export default Dashboard;
