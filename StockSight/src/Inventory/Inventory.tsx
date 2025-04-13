import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom"; // ✅ Import for navigation
import ItemDetailsModal from "../components/ItemDetailsModal"; // Import the modal component
import HelpPopup from "../components/HelpPopup"; // Import the help popup component
import { Button } from "react-bootstrap";

// Define interface for inventory item
interface InventoryItem {
  item_id: string;
  item_name: string;
  SKU: string;
  quantity: number;
  selling_price: number;
  description?: string;
  cost_price?: number;
  reorder_point?: number;
  expiration_date?: string;
}

const Inventory = () => {
  const [items, setItems] = useState<InventoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedItem, setSelectedItem] = useState<InventoryItem | null>(null); // State for selected item
  const [showHelpPopup, setShowHelpPopup] = useState(false); // State for help popup
  const navigate = useNavigate(); // ✅ For redirection

  // 📌 Fetch Inventory Items from Flask Backend
  useEffect(() => {
    fetchInventoryItems();
  }, []);

  const fetchInventoryItems = () => {
    fetch("http://127.0.0.1:5000/inventory")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setItems(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching inventory:", error);
        setError("Failed to load inventory.");
        setLoading(false);
      });
  };

  const handleItemClick = (item: InventoryItem) => {
    setSelectedItem(item); // Set the selected item to display in the modal
  };

  const handleCloseModal = () => {
    setSelectedItem(null); // Close the modal by setting selected item to null
  };

  const handleDeleteItem = (sku: string) => {
    if (!window.confirm("Are you sure you want to delete this item?")) return;

    fetch(`http://127.0.0.1:5000/inventory/${sku}`, {
      method: "DELETE",
    })
      .then((res) => {
        if (!res.ok) throw new Error("Failed to delete item");
        return res.json();
      })
      .then(() => {
        setItems((prevItems) => prevItems.filter((item) => item.SKU !== sku));
      })
      .catch((error) => {
        console.error("Error deleting item:", error);
        alert("Failed to delete item.");
      });
  };

  if (loading) return <p>Loading inventory...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div className="inventory-container">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div className="d-flex align-items-center">
          <h2 className="mb-0 me-3">Inventory</h2>
          <Button 
            variant="link" 
            className="help-bulb-button" 
            onClick={() => setShowHelpPopup(true)}
          >
            <i className="bi bi-lightbulb-fill"></i>
          </Button>
          <p className="text-muted small mb-0 ms-3">
            <i className="bi bi-info-circle me-1"></i>
            Click on a product name for more details
          </p>
        </div>

        {/* Always show the "Add Item" button */}
        <button
          className="btn btn-primary"
          onClick={() => navigate("/add-item")}
        >
          + Add Item
        </button>
      </div>
      
      {/* Help Popup */}
      <HelpPopup 
        show={showHelpPopup} 
        onHide={() => setShowHelpPopup(false)} 
        tabName="Inventory"
        content={{
          features: (
            <div>
              <h5>Inventory Management Overview</h5>
              <p>
                The Inventory tab provides a comprehensive view of all your stock items, allowing you to efficiently manage your inventory.
              </p>
              
              <h6 className="mt-4">Key Features</h6>
              <ul>
                <li>
                  <strong>Item Listing:</strong> View all inventory items with essential information like name, SKU, quantity, and selling price.
                </li>
                <li>
                  <strong>Item Details:</strong> Click on any item name to view detailed information, including cost price, description, and reorder point.
                </li>
                <li>
                  <strong>Sales Forecast:</strong> Access AI-powered sales predictions for individual items to help with inventory planning.
                </li>
                <li>
                  <strong>Reorder Point Suggestions:</strong> Receive intelligent suggestions for optimal reorder points based on sales patterns.
                </li>
                <li>
                  <strong>Add New Items:</strong> Easily add new products to your inventory with the "Add Item" button.
                </li>
                <li>
                  <strong>Delete Items:</strong> Remove products that are no longer part of your inventory.
                </li>
              </ul>
            </div>
          ),
          calculations: (
            <div>
              <h5>Calculations & Definitions</h5>
              
              <div className="mb-4">
                <h6>Inventory Metrics</h6>
                <p>
                  StockSightAI tracks several key metrics for each inventory item:
                </p>
                <ul>
                  <li><strong>Quantity:</strong> The current number of units available in stock</li>
                  <li><strong>Reorder Point:</strong> The threshold quantity at which new stock should be ordered</li>
                  <li><strong>Cost Price:</strong> The price paid to acquire or produce each unit</li>
                  <li><strong>Selling Price:</strong> The price at which each unit is sold to customers</li>
                </ul>
              </div>
              
              <div className="mb-4">
                <h6>Reorder Point Calculation</h6>
                <p>
                  StockSightAI calculates suggested reorder points based on:
                </p>
                <ul>
                  <li><strong>Historical Sales Data:</strong> Analysis of past sales patterns and velocity</li>
                  <li><strong>Lead Time:</strong> The time it takes to receive new stock after placing an order</li>
                  <li><strong>Sales Forecast:</strong> Predicted future demand for the item</li>
                </ul>
                <p>
                  The basic formula used is:
                  <br />
                  <code>Reorder Point = (Average Daily Sales × Lead Time) + Safety Stock</code>
                </p>
                <p>
                  Where Safety Stock is calculated based on sales variability and desired service level.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Profit Margin</h6>
                <p>
                  StockSightAI calculates profit margins for each item using:
                </p>
                <p>
                  <code>Profit Margin = ((Selling Price - Cost Price) / Selling Price) × 100%</code>
                </p>
                <p>
                  This helps identify which products are most profitable and which may need price adjustments.
                </p>
              </div>
              
              <div className="mb-4">
                <h6>Low Stock Detection</h6>
                <p>
                  StockSightAI automatically identifies items with low stock when:
                </p>
                <p>
                  <code>Current Quantity ≤ Reorder Point</code>
                </p>
                <p>
                  When this condition is met, the system generates alerts to help prevent stockouts.
                </p>
              </div>
            </div>
          )
        }}
      />

      <div className="table-responsive">
        <table className="table table-bordered table-hover table-striped">
        <thead className="table-dark">
          <tr>
            <th>#</th>
            <th>Name</th>
            <th>SKU</th>
            <th>Quantity</th>
            <th>Selling Price ($)</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {items.length > 0 ? (
            items.map((item, index) => (
              <tr key={index}>
                <td>{index + 1}</td>
                <td>
                  <button
                    className="btn btn-link"
                    onClick={() => handleItemClick(item)}
                  >
                    {item.item_name}
                  </button>
                </td>
                <td>{item.SKU}</td>
                <td>{item.quantity}</td>
                <td>
                  $
                  {typeof item.selling_price === "number"
                    ? item.selling_price.toFixed(2)
                    : "N/A"}
                </td>
                <td>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDeleteItem(item.SKU)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={6} className="text-center">
                No inventory items available.
              </td>
            </tr>
          )}
        </tbody>
        </table>
      </div>

      {selectedItem && (
        <ItemDetailsModal
          item={selectedItem}
          onClose={handleCloseModal} // Close the modal when the user clicks "Close"
        />
      )}
    </div>
  );
};

export default Inventory;
