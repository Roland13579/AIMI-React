import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom"; // ✅ Import for navigation
import ItemDetailsModal from "../components/ItemDetailsModal"; // Import the modal component

const Inventory = () => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedItem, setSelectedItem] = useState<any>(null); // State for selected item
  const navigate = useNavigate(); // ✅ For redirection

  // 📌 Fetch Inventory Items from Flask Backend
  useEffect(() => {
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
  }, []);

  const handleItemClick = (item: any) => {
    setSelectedItem(item); // Set the selected item to display in the modal
  };

  const handleCloseModal = () => {
    setSelectedItem(null); // Close the modal by setting selected item to null
  };

  if (loading) return <p>Loading inventory...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div className="container mt-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h2>Inventory</h2>

        {/* Always show the "Add Item" button */}
        <button
          className="btn btn-primary"
          onClick={() => navigate("/add-item")}
        >
          + Add Item
        </button>
      </div>

      <table className="table table-bordered table-hover">
        <thead className="table-dark">
          <tr>
            <th>#</th>
            <th>Name</th>
            <th>SKU</th>
            <th>Quantity</th>
            <th>Selling Price ($)</th>
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
                    onClick={() => {
                      // Open the modal with selected item details
                      handleItemClick(item);
                    }}
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
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan="5" className="text-center">
                No inventory items available.
              </td>
            </tr>
          )}
        </tbody>
      </table>

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
