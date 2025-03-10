import React from "react";

interface ItemDetailsModalProps {
  item: any; // The item data passed as a prop
  onClose: () => void; // Function to close the modal
}

const ItemDetailsModal: React.FC<ItemDetailsModalProps> = ({
  item,
  onClose,
}) => {
  return (
    <div className="modal show" tabIndex={-1} style={{ display: "block" }}>
      <div className="modal-dialog">
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">{item.item_name}</h5>
            <button
              type="button"
              className="btn-close"
              onClick={onClose}
            ></button>
          </div>
          <div className="modal-body">
            <p>
              <strong>SKU:</strong> {item.SKU}
            </p>
            <p>
              <strong>Quantity:</strong> {item.quantity}
            </p>
            <p>
              <strong>Cost Price ($):</strong> {item.cost_price}
            </p>
            <p>
              <strong>Selling Price ($):</strong> {item.selling_price}
            </p>
            <p>
              <strong>Description:</strong> {item.description}
            </p>
            <p>
              <strong>Reorder Point:</strong> {item.reorder_point}
            </p>
            <p>
              <strong>Expiration Date:</strong> {item.expiration_date}
            </p>
          </div>
          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ItemDetailsModal;
