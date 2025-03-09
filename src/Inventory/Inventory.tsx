import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom"; // ✅ Import for navigation

const Inventory = () => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
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
        if (data.length === 0) {
          setError("No inventory items found.");
        }
        setItems(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching inventory:", error);
        setError("Failed to load inventory.");
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading inventory...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div className="container mt-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h2>Inventory</h2>

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
                <td>{item.item_name}</td>
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
    </div>
  );
};

export default Inventory;
