import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Table, Badge, Modal, Form, InputGroup } from "react-bootstrap";
import HelpPopup from "../components/HelpPopup";

interface PurchaseOrder {
  reference_number: string;
  name: string;
  SKU: string;
  vendor: string;
  quantity: number;
  status: "pending" | "approved";
  created_at: string;
}

const PurchaseOrder = () => {
  const [orders, setOrders] = useState<PurchaseOrder[]>([]);
  const [filteredOrders, setFilteredOrders] = useState<PurchaseOrder[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedOrder, setSelectedOrder] = useState<PurchaseOrder | null>(
    null
  );
  const [showApproveModal, setShowApproveModal] = useState(false);
  const [accessLevel, setAccessLevel] = useState("Employee");
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [showHelpPopup, setShowHelpPopup] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetch("http://127.0.0.1:5000/purchase-orders")
      .then((response) => {
        if (!response.ok) throw new Error("Failed to fetch orders");
        return response.json();
      })
      .then((data) => {
        setOrders(data);
        setFilteredOrders(data);
        setLoading(false);
      })
      .catch((error) => {
        setError(error.message);
        setLoading(false);
      });

    // Fetch user access level
    const username = localStorage.getItem("username");
    if (username) {
      fetch("http://127.0.0.1:5000/profile", {
        headers: { Username: username },
      })
        .then((res) => res.json())
        .then((data) => setAccessLevel(data.access_level));
    }
  }, []);

  // Handle search and filter
  useEffect(() => {
    let result = orders.filter((order) => {
      const matchesSearch =
        order.reference_number
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        order.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        order.SKU.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesStatus =
        statusFilter === "all" || order.status === statusFilter;

      return matchesSearch && matchesStatus;
    });

    setFilteredOrders(result);
  }, [searchTerm, statusFilter, orders]);

  const handleApprove = async () => {
    if (!selectedOrder) return;

    try {
      const response = await fetch(
        `http://127.0.0.1:5000/purchase-orders/${selectedOrder.reference_number}/approve`,
        { method: "PUT" }
      );

      if (!response.ok) throw new Error("Approval failed");

      setOrders((prevOrders) =>
        prevOrders.map((order) =>
          order.reference_number === selectedOrder.reference_number
            ? { ...order, status: "approved" }
            : order
        )
      );
      setShowApproveModal(false);
    } catch (error) {
      console.error("Approval error:", error);
    }
  };

  const handleDeleteOrder = (referenceNumber: string) => {
    if (!window.confirm("Are you sure you want to delete this purchase order?"))
      return;

    fetch(`http://127.0.0.1:5000/purchase-orders/${referenceNumber}`, {
      method: "DELETE",
    })
      .then((res) => {
        if (!res.ok) throw new Error("Failed to delete");
        return res.json();
      })
      .then(() => {
        setOrders((prev) =>
          prev.filter((order) => order.reference_number !== referenceNumber)
        );
      })
      .catch((err) => {
        console.error("Delete error:", err);
        alert("Failed to delete purchase order.");
      });
  };

  if (loading) return <p>Loading purchase orders...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div className="purchase-order-container">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div className="d-flex align-items-center">
          <h2 className="mb-0 me-3">Purchase Orders</h2>
          <Button 
            variant="link" 
            className="help-bulb-button" 
            onClick={() => setShowHelpPopup(true)}
          >
            <i className="bi bi-lightbulb-fill"></i>
          </Button>
        </div>
        <Button
          variant="primary"
          onClick={() => navigate("/create-purchase-order")}
        >
          + Create Purchase Order
        </Button>
      </div>
      
      {/* Help Popup */}
      <HelpPopup 
        show={showHelpPopup} 
        onHide={() => setShowHelpPopup(false)} 
        tabName="Purchase Orders"
        content={{
          features: (
            <div>
              <h5>Purchase Orders Overview</h5>
              <p>
                The Purchase Orders tab allows you to create and manage orders for restocking your inventory from vendors.
              </p>
              
              <h6 className="mt-4">Key Features</h6>
              <ul>
                <li>
                  <strong>Order Management:</strong> Create, view, and delete purchase orders for inventory items.
                </li>
                <li>
                  <strong>Status Tracking:</strong> Monitor the status of each purchase order (pending or approved).
                </li>
                <li>
                  <strong>Search & Filter:</strong> Easily find purchase orders by reference number, item name, SKU, or status.
                </li>
                <li>
                  <strong>Approval Workflow:</strong> Managers can approve purchase orders, which automatically updates inventory when received.
                </li>
                <li>
                  <strong>Vendor Management:</strong> Track which vendors supply different items in your inventory.
                </li>
              </ul>
            </div>
          ),
          calculations: (
            <div>
              <h5>Calculations & Definitions</h5>
              
              <div className="mb-4">
                <h6>Purchase Order Status</h6>
                <p>
                  StockSightAI tracks purchase orders through two main statuses:
                </p>
                <ul>
                  <li><strong>Pending:</strong> Initial state when a purchase order is created but not yet approved</li>
                  <li><strong>Approved:</strong> The purchase order has been approved by a manager and is ready for processing</li>
                </ul>
              </div>
              
              <div className="mb-4">
                <h6>Inventory Integration</h6>
                <p>
                  When a purchase order is approved, StockSightAI:
                </p>
                <ul>
                  <li>Records the expected delivery of new inventory</li>
                  <li>Updates inventory projections in the forecast system</li>
                  <li>Adjusts reorder recommendations based on pending orders</li>
                </ul>
                <p>
                  This integration ensures that your inventory forecasts account for incoming stock.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Access Level Controls</h6>
                <p>
                  StockSightAI implements role-based access controls for purchase orders:
                </p>
                <ul>
                  <li><strong>Employees:</strong> Can create purchase orders and view their status</li>
                  <li><strong>Managers:</strong> Can approve purchase orders and delete them if necessary</li>
                </ul>
                <p>
                  This separation of duties helps maintain proper inventory control and prevents unauthorized purchases.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Reference Number Generation</h6>
                <p>
                  StockSightAI automatically generates unique reference numbers for each purchase order using:
                </p>
                <ul>
                  <li>A timestamp component to ensure uniqueness</li>
                  <li>A vendor identifier to easily track orders by supplier</li>
                  <li>A sequential counter to maintain order within the system</li>
                </ul>
                <p>
                  This system ensures that each purchase order can be uniquely identified and tracked throughout the procurement process.
                </p>
              </div>
            </div>
          )
        }}
      />

      {/* Search and Filter Section */}
      <div className="row mb-3">
        <div className="col-md-6">
          <InputGroup>
            <Form.Control
              type="text"
              placeholder="Search by SKU, Reference Number, or Name..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Button
              variant="outline-secondary"
              onClick={() => setSearchTerm("")}
            >
              Clear
            </Button>
          </InputGroup>
        </div>
        <div className="col-md-3">
          <Form.Select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All Statuses</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
          </Form.Select>
        </div>
      </div>

      {/* Orders Table */}
      <div className="table-responsive">
        <Table striped bordered hover>
        <thead className="table-dark">
          <tr>
            <th>Reference #</th>
            <th>Item Name</th>
            <th>SKU</th>
            <th>Vendor</th>
            <th>Quantity</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {filteredOrders.map((order) => (
            <tr key={order.reference_number}>
              <td>{order.reference_number}</td>
              <td>{order.name}</td>
              <td>{order.SKU}</td>
              <td>{order.vendor}</td>
              <td>{order.quantity}</td>
              <td>
                <Badge bg={order.status === "pending" ? "warning" : "success"}>
                  {order.status}
                </Badge>
              </td>
              <td>
                {accessLevel === "Manager" && order.status === "pending" && (
                  <Button
                    variant="success"
                    size="sm"
                    onClick={() => {
                      setSelectedOrder(order);
                      setShowApproveModal(true);
                    }}
                  >
                    Approve
                  </Button>
                )}
                {accessLevel === "Manager" && (
                  <Button
                    variant="danger"
                    size="sm"
                    className="ms-2"
                    onClick={() => handleDeleteOrder(order.reference_number)}
                  >
                    Delete
                  </Button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
        </Table>
      </div>

      {/* Approve Modal */}
      <Modal show={showApproveModal} onHide={() => setShowApproveModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Approve Purchase Order</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Are you sure you want to approve {selectedOrder?.reference_number}?
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="secondary"
            onClick={() => setShowApproveModal(false)}
          >
            Cancel
          </Button>
          <Button variant="primary" onClick={handleApprove}>
            Confirm Approval
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};

export default PurchaseOrder;
