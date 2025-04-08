import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Form, Alert } from "react-bootstrap";

interface InventoryItem {
  SKU: string;
  item_name: string;
  quantity: number;
}

const CreatePurchaseOrder = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: "",
    sku: "",
    vendor: "",
    quantity: 1,
  });
  const [inventoryItems, setInventoryItems] = useState<InventoryItem[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Fetch inventory items on component mount
  useEffect(() => {
    fetch("http://127.0.0.1:5000/inventory")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch inventory items");
        }
        return response.json();
      })
      .then((data) => setInventoryItems(data))
      .catch((error) => {
        console.error("Error fetching inventory:", error);
        setError("Failed to load inventory items. Please try again later.");
      });
  }, []);

  // Handle SKU changes
  useEffect(() => {
    if (formData.sku) {
      const foundItem = inventoryItems.find(
        (item) => item.SKU.toUpperCase() === formData.sku.toUpperCase().trim()
      );
      if (foundItem) {
        setFormData((prev) => ({ ...prev, name: foundItem.item_name }));
      }
    }
  }, [formData.sku, inventoryItems]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:5000/purchase-orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formData.name,
          SKU: formData.sku.toUpperCase(), // Changed to uppercase SKU
          vendor: formData.vendor,
          quantity: formData.quantity,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Failed to create purchase order");
      }

      // Redirect to purchase orders page on success
      navigate("/purchase-order");
    } catch (error) {
      console.error("Error creating purchase order:", error);
      setError(error.message || "An error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-4">
      <h2>Create New Purchase Order</h2>
      {error && <Alert variant="danger">{error}</Alert>}

      <Form onSubmit={handleSubmit}>
        <Form.Group className="mb-3">
          <Form.Label>Item Name</Form.Label>
          <Form.Control
            type="text"
            placeholder="Will auto-fill from SKU"
            value={formData.name}
            readOnly // Make this field read-only
            required
          />
        </Form.Group>

        <Form.Group className="mb-3">
          <Form.Label>SKU</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter SKU"
            name="sku"
            value={formData.sku}
            onChange={(e) =>
              setFormData({ ...formData, sku: e.target.value.toUpperCase() })
            }
            list="existingSkus"
            required
          />
          <datalist id="existingSkus">
            {inventoryItems.map((item) => (
              <option key={item.SKU} value={item.SKU}>
                {item.item_name} (SKU: {item.SKU})
              </option>
            ))}
          </datalist>
          <Form.Text className="text-muted">
            Type or select a valid SKU from inventory
          </Form.Text>
        </Form.Group>

        <Form.Group className="mb-3">
          <Form.Label>Vendor</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter vendor name"
            value={formData.vendor}
            onChange={(e) =>
              setFormData({ ...formData, vendor: e.target.value })
            }
            required
          />
        </Form.Group>

        <Form.Group className="mb-3">
          <Form.Label>Quantity</Form.Label>
          <Form.Control
            type="number"
            min="1"
            placeholder="Enter quantity"
            value={formData.quantity}
            onChange={(e) =>
              setFormData({ ...formData, quantity: parseInt(e.target.value) })
            }
            required
          />
        </Form.Group>

        <Button variant="primary" type="submit" disabled={loading}>
          {loading ? "Creating..." : "Create Purchase Order"}
        </Button>
        <Button
          variant="secondary"
          className="ms-2"
          onClick={() => navigate("/purchase-order")}
        >
          Cancel
        </Button>
      </Form>
    </div>
  );
};

export default CreatePurchaseOrder;
