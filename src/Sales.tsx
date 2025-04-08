import React, { useState, useEffect } from "react";
import {
  Modal,
  Button,
  Form,
  Badge,
  Toast,
  ToastContainer,
} from "react-bootstrap";
import HelpPopup from "./components/HelpPopup";
import { v4 as uuidv4 } from "uuid";

interface SalesTransaction {
  transaction_id: string;
  sku: string;
  item_name: string;
  quantity: number;
  customer_name: string;
  payment_method: string;
  transaction_date: string; // Date is now part of the transaction
  status: "pending" | "packed" | "shipped" | "delivered";
  total_price: number;
}

interface InventoryItem {
  item_id: string;
  item_name: string;
  SKU: string;
  quantity: number;
  selling_price: number;
  description: string;
  reorder_point: number;
  cost_price: number;
  expiration_date: string | null;
}

const Sales: React.FC = () => {
  const [transactions, setTransactions] = useState<SalesTransaction[]>([]);
  const [filteredTransactions, setFilteredTransactions] = useState<
    SalesTransaction[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [dateFilter, setDateFilter] = useState<string>("");

  // Modal states
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [showStatusModal, setShowStatusModal] = useState(false);
  const [selectedTransaction, setSelectedTransaction] =
    useState<SalesTransaction | null>(null);
  const [refreshingForecast, setRefreshingForecast] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const [toastVariant, setToastVariant] = useState<"success" | "danger">(
    "success"
  );
  const [showHelpPopup, setShowHelpPopup] = useState(false);

  // Form states for adding new transaction
  const [inventoryItems, setInventoryItems] = useState<InventoryItem[]>([]);
  const [selectedSku, setSelectedSku] = useState("");
  const [selectedItemName, setSelectedItemName] = useState("");
  const [quantity, setQuantity] = useState(1);
  const [customerName, setCustomerName] = useState("");
  const [paymentMethod, setPaymentMethod] = useState("cash");
  const [totalPrice, setTotalPrice] = useState(0);
  const [transactionDate, setTransactionDate] = useState(
    new Date().toISOString().split("T")[0]
  ); // Date input state
  const [newStatus, setNewStatus] = useState<
    "pending" | "packed" | "shipped" | "delivered"
  >("pending");

  // Fetch sales transactions
  useEffect(() => {
    fetchTransactions();
    fetchInventoryItems();
  }, []);

  // Function to refresh forecast data
  const refreshForecastData = () => {
    setRefreshingForecast(true);

    fetch("http://127.0.0.1:5000/refresh-forecast-data", {
      method: "POST",
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (data.status === "success") {
          setToastVariant("success");
          setToastMessage(
            `Forecast data refreshed successfully. ${data.items_processed} items processed.`
          );
        } else {
          setToastVariant("danger");
          setToastMessage(`Error: ${data.message}`);
        }
        setShowToast(true);
      })
      .catch((error) => {
        console.error("Error refreshing forecast data:", error);
        setToastVariant("danger");
        setToastMessage("Failed to refresh forecast data. Please try again.");
        setShowToast(true);
      })
      .finally(() => {
        setRefreshingForecast(false);
      });
  };

  // Apply filters when search term, status filter, or date filter changes
  useEffect(() => {
    applyFilters();
  }, [searchTerm, statusFilter, dateFilter, transactions]);

  const fetchTransactions = () => {
    setLoading(true);
    fetch("http://127.0.0.1:5000/sales")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setTransactions(data);
        setFilteredTransactions(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching sales transactions:", error);
        setError("Failed to load sales transactions.");
        setLoading(false);
      });
  };

  const fetchInventoryItems = () => {
    fetch("http://127.0.0.1:5000/inventory")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setInventoryItems(data);
      })
      .catch((error) => {
        console.error("Error fetching inventory items:", error);
      });
  };

  const applyFilters = () => {
    let filtered = [...transactions];

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(
        (transaction) =>
          transaction.transaction_id
            .toLowerCase()
            .includes(searchTerm.toLowerCase()) ||
          transaction.item_name
            .toLowerCase()
            .includes(searchTerm.toLowerCase()) ||
          transaction.customer_name
            .toLowerCase()
            .includes(searchTerm.toLowerCase())
      );
    }

    // Apply status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter(
        (transaction) => transaction.status === statusFilter
      );
    }

    // Apply date filter
    if (dateFilter) {
      filtered = filtered.filter((transaction) => {
        const transactionDate = new Date(transaction.transaction_date)
          .toISOString()
          .split("T")[0];
        return transactionDate === dateFilter;
      });
    }

    setFilteredTransactions(filtered);
  };

  const handleAddTransaction = () => {
    // Find the selected item to get its selling price
    const selectedItem = inventoryItems.find(
      (item) => item.SKU === selectedSku
    );
    if (!selectedItem) {
      alert("Please select a valid item");
      return;
    }

    // Check if there's enough inventory
    if (quantity > selectedItem.quantity) {
      alert(
        `Cannot sell more than available inventory. Available: ${selectedItem.quantity}`
      );
      return;
    }

    if (!customerName.trim()) {
      alert("Customer Name must not be empty.");
      return;
    }

    // Calculate total price
    const calculatedTotalPrice = selectedItem.selling_price * quantity;

    // Create new transaction object
    const newTransaction: SalesTransaction = {
      transaction_id: uuidv4(),
      sku: selectedSku,
      item_name: selectedItemName,
      quantity: quantity,
      customer_name: customerName,
      payment_method: paymentMethod,
      transaction_date: transactionDate, // Use the user-provided date
      status: "pending",
      total_price: calculatedTotalPrice,
    };

    // Send POST request to add the transaction
    fetch("http://127.0.0.1:5000/sales", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(newTransaction),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(() => {
        // Refresh transactions and inventory
        fetchTransactions();
        fetchInventoryItems();
        // Reset form fields
        resetFormFields();
        // Close modal
        setShowAddModal(false);
      })
      .catch((error) => {
        console.error("Error adding sales transaction:", error);
        alert("Failed to add sales transaction. Please try again.");
      });
  };

  const handleUpdateStatus = () => {
    if (!selectedTransaction || !newStatus) return;

    // Send PUT request to update the transaction status
    fetch(`http://127.0.0.1:5000/sales/${selectedTransaction.transaction_id}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ status: newStatus }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(() => {
        // Refresh transactions
        fetchTransactions();
        // Close modal
        setShowStatusModal(false);
      })
      .catch((error) => {
        console.error("Error updating transaction status:", error);
        alert("Failed to update transaction status. Please try again.");
      });
  };

  const handleDeleteTransaction = (transactionId: string) => {
    if (!confirm("Are you sure you want to delete this transaction?")) return;

    // Send DELETE request
    fetch(`http://127.0.0.1:5000/sales/${transactionId}`, {
      method: "DELETE",
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(() => {
        // Refresh transactions and inventory
        fetchTransactions();
        fetchInventoryItems();
      })
      .catch((error) => {
        console.error("Error deleting sales transaction:", error);
        alert("Failed to delete sales transaction. Please try again.");
      });
  };

  const resetFormFields = () => {
    setSelectedSku("");
    setSelectedItemName("");
    setQuantity(1);
    setCustomerName("");
    setPaymentMethod("cash");
    setTotalPrice(0);
    setTransactionDate(new Date().toISOString().split("T")[0]); // Reset date to today
  };

  const handleItemSelect = (sku: string) => {
    setSelectedSku(sku);
    const item = inventoryItems.find((item) => item.SKU === sku);
    if (item) {
      setSelectedItemName(item.item_name);
      setTotalPrice(item.selling_price * quantity);
    }
  };

  const handleQuantityChange = (newQuantity: number) => {
    setQuantity(newQuantity);
    const item = inventoryItems.find((item) => item.SKU === selectedSku);
    if (item) {
      setTotalPrice(item.selling_price * newQuantity);
    }
  };

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

  if (loading) return <p>Loading sales transactions...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div className="sales-container">
      <div className="row mb-4">
        <div className="col-md-6 d-flex align-items-center">
          <h2 className="mb-0 me-3">Sales Transactions</h2>
          <Button 
            variant="link" 
            className="help-bulb-button" 
            onClick={() => setShowHelpPopup(true)}
          >
            <i className="bi bi-lightbulb-fill"></i>
          </Button>
        </div>
        <div className="col-md-6 text-end">
          <Button
            variant="success"
            className="me-2"
            onClick={refreshForecastData}
            disabled={refreshingForecast}
          >
            {refreshingForecast ? "Refreshing..." : "Refresh Forecast Data"}
          </Button>
          <Button variant="primary" onClick={() => setShowAddModal(true)}>
            + Add Transaction
          </Button>
        </div>
      </div>

      {/* Help Popup */}
      <HelpPopup 
        show={showHelpPopup} 
        onHide={() => setShowHelpPopup(false)} 
        tabName="Sales"
        content={{
          features: (
            <div>
              <h5>Sales Transactions Overview</h5>
              <p>
                The Sales tab allows you to manage all your sales transactions, track customer purchases, and monitor sales performance.
              </p>
              
              <h6 className="mt-4">Key Features</h6>
              <ul>
                <li>
                  <strong>Transaction Management:</strong> Add, view, update, and delete sales transactions.
                </li>
                <li>
                  <strong>Status Tracking:</strong> Monitor the status of each transaction (pending, packed, shipped, delivered).
                </li>
                <li>
                  <strong>Search & Filter:</strong> Easily find transactions by ID, item name, customer, status, or date.
                </li>
                <li>
                  <strong>Forecast Data Refresh:</strong> Update the forecast data based on recent sales to improve prediction accuracy.
                </li>
                <li>
                  <strong>Transaction Details:</strong> View comprehensive information about each sale, including payment method and total price.
                </li>
              </ul>
            </div>
          ),
          calculations: (
            <div>
              <h5>Calculations & Definitions</h5>
              
              <div className="mb-4">
                <h6>Transaction Status</h6>
                <p>
                  StockSightAI tracks each transaction through four stages:
                </p>
                <ul>
                  <li><strong>Pending:</strong> Initial state when a transaction is created but not yet processed</li>
                  <li><strong>Packed:</strong> The items have been collected and packaged for shipping</li>
                  <li><strong>Shipped:</strong> The package has been dispatched to the customer</li>
                  <li><strong>Delivered:</strong> The customer has received the items</li>
                </ul>
              </div>
              
              <div className="mb-4">
                <h6>Total Price Calculation</h6>
                <p>
                  StockSightAI automatically calculates the total price of a transaction using the formula:
                </p>
                <p>
                  <code>Total Price = Item Selling Price × Quantity</code>
                </p>
                <p>
                  This calculation is performed when selecting an item and quantity during transaction creation.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Inventory Impact</h6>
                <p>
                  When a sales transaction is created, StockSightAI automatically:
                </p>
                <ul>
                  <li>Reduces the available quantity of the item in inventory</li>
                  <li>Checks if the new quantity falls below the reorder point</li>
                  <li>Triggers low stock warnings when necessary</li>
                </ul>
              </div>
              
              <div className="mb-4">
                <h6>Forecast Data Refresh</h6>
                <p>
                  The "Refresh Forecast Data" button triggers StockSightAI to:
                </p>
                <ul>
                  <li>Reanalyze all recent sales transactions</li>
                  <li>Update time series models with the latest data</li>
                  <li>Recalculate sales and profit predictions</li>
                  <li>Adjust confidence intervals based on new information</li>
                </ul>
                <p>
                  This process ensures that your forecasts remain accurate as new sales data becomes available.
                </p>
              </div>
            </div>
          )
        }}
      />

      {/* Toast notification */}
      <ToastContainer position="top-end" className="p-3">
        <Toast
          show={showToast}
          onClose={() => setShowToast(false)}
          delay={5000}
          autohide
          bg={toastVariant}
        >
          <Toast.Header>
            <strong className="me-auto">Forecast Update</strong>
          </Toast.Header>
          <Toast.Body className={toastVariant === "danger" ? "" : "text-white"}>
            {toastMessage}
          </Toast.Body>
        </Toast>
      </ToastContainer>

      <div className="row mb-4">
        <div className="col-md-6">
          <input
            type="text"
            className="form-control"
            placeholder="Search by ID, item name, or customer..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="col-md-3">
          <select
            className="form-select"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All Statuses</option>
            <option value="pending">Pending</option>
            <option value="packed">Packed</option>
            <option value="shipped">Shipped</option>
            <option value="delivered">Delivered</option>
          </select>
        </div>
        <div className="col-md-3">
          <input
            type="date"
            className="form-control"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
          />
        </div>
      </div>

      <div className="table-responsive">
        <table className="table table-bordered table-hover table-striped">
          <thead className="table-dark">
            <tr>
              <th>Transaction ID</th>
              <th>Item</th>
              <th>Quantity</th>
              <th>Customer</th>
              <th>Date</th> {/* New column for date */}
              <th>Status</th>
              <th>Total Price</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredTransactions.length > 0 ? (
              filteredTransactions.map((transaction) => (
                <tr key={transaction.transaction_id}>
                  <td>{transaction.transaction_id}</td>
                  <td>{transaction.item_name}</td>
                  <td>{transaction.quantity}</td>
                  <td>{transaction.customer_name}</td>
                  <td>{transaction.transaction_date}</td>{" "}
                  {/* Display the date */}
                  <td>
                    <Badge bg={getStatusBadgeColor(transaction.status)}>
                      {transaction.status.charAt(0).toUpperCase() +
                        transaction.status.slice(1)}
                    </Badge>
                  </td>
                  <td>${transaction.total_price.toFixed(2)}</td>
                  <td>
                    <Button
                      variant="info"
                      size="sm"
                      className="me-2"
                      onClick={() => {
                        setSelectedTransaction(transaction);
                        setShowDetailsModal(true);
                      }}
                    >
                      <i className="bi bi-eye"></i>
                    </Button>
                    <Button
                      variant="warning"
                      size="sm"
                      className="me-2"
                      onClick={() => {
                        setSelectedTransaction(transaction);
                        setNewStatus(transaction.status);
                        setShowStatusModal(true);
                      }}
                    >
                      <i className="bi bi-pencil"></i>
                    </Button>
                    <Button
                      variant="danger"
                      size="sm"
                      onClick={() =>
                        handleDeleteTransaction(transaction.transaction_id)
                      }
                    >
                      <i className="bi bi-trash"></i>
                    </Button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={8} className="text-center">
                  No sales transactions found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Add Transaction Modal */}
      <Modal
        show={showAddModal}
        onHide={() => setShowAddModal(false)}
        size="lg"
      >
        <Modal.Header closeButton>
          <Modal.Title>Add Sales Transaction</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Form.Group className="mb-3">
              <Form.Label>Item</Form.Label>
              <Form.Select
                value={selectedSku}
                onChange={(e) => handleItemSelect(e.target.value)}
                required
              >
                <option value="">Select an item</option>
                {inventoryItems.map((item) => (
                  <option key={item.SKU} value={item.SKU}>
                    {item.item_name} (SKU: {item.SKU}) - $
                    {item.selling_price.toFixed(2)} - Available: {item.quantity}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Quantity</Form.Label>
              <Form.Control
                type="number"
                min="1"
                value={quantity}
                onChange={(e) => handleQuantityChange(parseInt(e.target.value))}
                required
              />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Customer Name</Form.Label>
              <Form.Control
                type="text"
                value={customerName}
                onChange={(e) => setCustomerName(e.target.value)}
                required
              />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Payment Method</Form.Label>
              <Form.Select
                value={paymentMethod}
                onChange={(e) => setPaymentMethod(e.target.value)}
                required
              >
                <option value="cash">Cash</option>
                <option value="credit">Credit Card</option>
                <option value="debit">Debit Card</option>
                <option value="bank_transfer">Bank Transfer</option>
              </Form.Select>
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Transaction Date</Form.Label>
              <Form.Control
                type="date"
                value={transactionDate}
                onChange={(e) => setTransactionDate(e.target.value)}
                required
              />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Total Price</Form.Label>
              <Form.Control
                type="text"
                value={`$${totalPrice.toFixed(2)}`}
                readOnly
              />
            </Form.Group>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowAddModal(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleAddTransaction}>
            Add Transaction
          </Button>
        </Modal.Footer>
      </Modal>

      {/* Transaction Details Modal */}
      <Modal show={showDetailsModal} onHide={() => setShowDetailsModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Transaction Details</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedTransaction && (
            <div>
              <p>
                <strong>Transaction ID:</strong>{" "}
                {selectedTransaction.transaction_id}
              </p>
              <p>
                <strong>SKU:</strong> {selectedTransaction.sku}
              </p>
              <p>
                <strong>Item Name:</strong> {selectedTransaction.item_name}
              </p>
              <p>
                <strong>Quantity:</strong> {selectedTransaction.quantity}
              </p>
              <p>
                <strong>Customer Name:</strong>{" "}
                {selectedTransaction.customer_name}
              </p>
              <p>
                <strong>Payment Method:</strong>{" "}
                {selectedTransaction.payment_method}
              </p>
              <p>
                <strong>Transaction Date:</strong>{" "}
                {selectedTransaction.transaction_date}
              </p>
              <p>
                <strong>Status:</strong>{" "}
                <Badge bg={getStatusBadgeColor(selectedTransaction.status)}>
                  {selectedTransaction.status.charAt(0).toUpperCase() +
                    selectedTransaction.status.slice(1)}
                </Badge>
              </p>
              <p>
                <strong>Total Price:</strong> $
                {selectedTransaction.total_price.toFixed(2)}
              </p>
            </div>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="secondary"
            onClick={() => setShowDetailsModal(false)}
          >
            Close
          </Button>
        </Modal.Footer>
      </Modal>

      {/* Update Status Modal */}
      <Modal show={showStatusModal} onHide={() => setShowStatusModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Update Transaction Status</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedTransaction && (
            <Form>
              <Form.Group className="mb-3">
                <Form.Label>Current Status</Form.Label>
                <div>
                  <Badge bg={getStatusBadgeColor(selectedTransaction.status)}>
                    {selectedTransaction.status.charAt(0).toUpperCase() +
                      selectedTransaction.status.slice(1)}
                  </Badge>
                </div>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>New Status</Form.Label>
                <Form.Select
                  value={newStatus}
                  onChange={(e) =>
                    setNewStatus(
                      e.target.value as
                        | "pending"
                        | "packed"
                        | "shipped"
                        | "delivered"
                    )
                  }
                >
                  <option value="pending">Pending</option>
                  <option value="packed">Packed</option>
                  <option value="shipped">Shipped</option>
                  <option value="delivered">Delivered</option>
                </Form.Select>
              </Form.Group>
            </Form>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowStatusModal(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleUpdateStatus}>
            Update Status
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};

export default Sales;
