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
      {/* Cube Logo at the top (with extra top margin to avoid being hidden by the header) */}
      <div className="text-center mb-4 mt-5">
        <div 
          className="bg-white text-primary rounded-circle mx-auto d-flex align-items-center justify-content-center" 
          style={{ width: "60px", height: "60px" }}
        >
          <i className="bi bi-box-seam" style={{ fontSize: "2rem" }}></i>
        </div>
      </div>
      
      {/* Centered Navigation */}
      <ul className="nav nav-pills flex-column flex-grow-1 justify-content-center">
        <li className="nav-item">
          <Link
            to="/dashboard"
            className={`nav-link text-white ${
              location.pathname === "/dashboard" ? "active bg-primary" : ""
            }`}
            style={{ fontSize: "1.1rem" }}
          >
            <i className="bi bi-house-door me-2" style={{ fontSize: "1.3rem" }}></i> Dashboard
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to="/forecast"
            className={`nav-link text-white ${
              location.pathname === "/forecast" ? "active bg-primary" : ""
            }`}
            style={{ fontSize: "1.1rem" }}
          >
            <i className="bi bi-graph-up me-2" style={{ fontSize: "1.3rem" }}></i> Forecast
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to="/inventory"
            className={`nav-link text-white ${
              location.pathname === "/inventory" ? "active bg-primary" : ""
            }`}
            style={{ fontSize: "1.1rem" }}
          >
            <i className="bi bi-box-seam me-2" style={{ fontSize: "1.3rem" }}></i> Inventory
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to="/sales"
            className={`nav-link text-white ${
              location.pathname === "/sales" ? "active bg-primary" : ""
            }`}
            style={{ fontSize: "1.1rem" }}
          >
            <i className="bi bi-cart me-2" style={{ fontSize: "1.3rem" }}></i> Sales
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to="/purchase-order"
            className={`nav-link text-white ${
              location.pathname === "/purchase-order" ? "active bg-primary" : ""
            }`}
            style={{ fontSize: "1.1rem" }}
          >
            <i className="bi bi-clipboard-check me-2" style={{ fontSize: "1.3rem" }}></i> Purchase Orders
          </Link>
        </li>
      </ul>
      
      {/* StockSight text at the bottom */}
      <h2 className="text-center mt-4" style={{ fontSize: "1.8rem" }}>StockSight</h2>
    </div>
  );
};

export default Sidebar;
