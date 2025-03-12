import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Table, Badge, Modal, Form, InputGroup } from "react-bootstrap";

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

  if (loading) return <p>Loading purchase orders...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div className="container mt-4">
      <div className="d-flex justify-content-between mb-3">
        <h2>Purchase Orders</h2>
        <Button
          variant="primary"
          onClick={() => navigate("/create-purchase-order")}
        >
          + Create Purchase Order
        </Button>
      </div>

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
              </td>
            </tr>
          ))}
        </tbody>
      </Table>

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
