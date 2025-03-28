import React from "react";
import { Link, useLocation } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css"; // Bootstrap
import "bootstrap-icons/font/bootstrap-icons.css"; // Bootstrap Icons

const Sidebar: React.FC = () => {
  const location = useLocation(); // Get current page path

  return (
    <div
      className="d-flex flex-column p-3 bg-dark text-white vh-100"
      style={{ width: "250px", position: "fixed" }}
    >
      <h2 className="text-center">StockSight</h2>
      <ul className="nav nav-pills flex-column">
        <li className="nav-item">
          <Link
            to="/dashboard"
            className={`nav-link text-white ${
              location.pathname === "/dashboard" ? "active bg-primary" : ""
            }`}
          >
            <i className="bi bi-house-door me-2"></i> Dashboard
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to="/forecast"
            className={`nav-link text-white ${
              location.pathname === "/forecast" ? "active bg-primary" : ""
            }`}
          >
            <i className="bi bi-graph-up me-2"></i> Forecast
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to="/inventory"
            className={`nav-link text-white ${
              location.pathname === "/inventory" ? "active bg-primary" : ""
            }`}
          >
            <i className="bi bi-box-seam me-2"></i> Inventory
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to="/sales"
            className={`nav-link text-white ${
              location.pathname === "/sales" ? "active bg-primary" : ""
            }`}
          >
            <i className="bi bi-cart me-2"></i> Sales
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to="/purchase-order"
            className={`nav-link text-white ${
              location.pathname === "/purchase-order" ? "active bg-primary" : ""
            }`}
          >
            <i className="bi bi-clipboard-check me-2"></i> Purchase Orders
          </Link>
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;
