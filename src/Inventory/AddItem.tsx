import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const AddItem = () => {
  const navigate = useNavigate();
  const [isDirty, setIsDirty] = useState(false); // Tracks if form is modified

  const [formData, setFormData] = useState({
    item_name: "",
    SKU: "",
    quantity: "",
    cost_price: "",
    selling_price: "",
    reorder_point: "",
    description: "",
    expiration_date: "",
  });

  // 📌 Handle form input change
  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setIsDirty(true);
  };

  // 📌 Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsDirty(false); // Reset dirty state after submission

    const response = await fetch("http://127.0.0.1:5000/inventory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });

    if (response.ok) {
      alert("Item added successfully!");
      navigate("/inventory"); // ✅ Redirect to inventory page
    } else {
      alert("Failed to add item.");
    }
  };

  // 📌 Prevent Browser Refresh / Tab Close
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue =
          "You have unsaved changes. Are you sure you want to leave?";
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [isDirty]);

  // 📌 Prevent Internal Navigation (React Router)
  useEffect(() => {
    const handleNavigation = (e: PopStateEvent) => {
      if (
        isDirty &&
        !window.confirm(
          "You have unsaved changes. Are you sure you want to leave?"
        )
      ) {
        e.preventDefault();
        navigate("/add-item"); // Keep user on the page
      }
    };

    window.addEventListener("popstate", handleNavigation);
    return () => window.removeEventListener("popstate", handleNavigation);
  }, [isDirty, navigate]);

  return (
    <div className="container mt-4">
      <h2>Add New Item</h2>
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label className="form-label">Item Name</label>
          <input
            type="text"
            className="form-control"
            name="item_name"
            onChange={handleChange}
            required
          />
        </div>

        <div className="mb-3">
          <label className="form-label">SKU</label>
          <input
            type="text"
            className="form-control"
            name="SKU"
            onChange={handleChange}
            required
          />
        </div>

        <div className="mb-3">
          <label className="form-label">Quantity</label>
          <input
            type="number"
            className="form-control"
            name="quantity"
            onChange={handleChange}
            required
          />
        </div>

        <div className="mb-3">
          <label className="form-label">Reorder Point</label>
          <input
            type="number"
            className="form-control"
            name="reorder_point"
            onChange={handleChange}
            required
          />
        </div>

        <div className="mb-3">
          <label className="form-label">Cost Price ($)</label>
          <input
            type="number"
            step="0.01"
            className="form-control"
            name="cost_price"
            onChange={handleChange}
            required
          />
        </div>

        <div className="mb-3">
          <label className="form-label">Selling Price ($)</label>
          <input
            type="number"
            step="0.01"
            className="form-control"
            name="selling_price"
            onChange={handleChange}
            required
          />
        </div>

        <div className="mb-3">
          <label className="form-label">Expiration Date</label>
          <input
            type="date"
            className="form-control"
            name="expiration_date"
            onChange={handleChange}
          />
        </div>

        <div className="mb-3">
          <label className="form-label">Description</label>
          <textarea
            className="form-control"
            name="description"
            rows={3}
            onChange={handleChange}
            required
          ></textarea>
        </div>

        <button type="submit" className="btn btn-success">
          Add Item
        </button>
        <button
          type="button"
          className="btn btn-secondary ms-2"
          onClick={() => {
            if (
              !isDirty ||
              window.confirm(
                "You have unsaved changes. Do you want to discard them?"
              )
            ) {
              navigate("/inventory");
            }
          }}
        >
          Cancel
        </button>
      </form>
    </div>
  );
};

export default AddItem;
