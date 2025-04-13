import React from "react";
import { Link, useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";

const Header = () => {
  const navigate = useNavigate();

  // ✅ Logout Function (Clears Token & Redirects)
  const handleLogout = () => {
    localStorage.removeItem("token"); // Clear auth token
    window.dispatchEvent(new Event("storage")); // Manually trigger storage event
    navigate("/login"); // Redirect to login page
  };

  return (
    <nav className="navbar navbar-dark bg-dark px-3 fixed-top">
      {/* Logo or App Name */}
      <span className="navbar-brand d-flex align-items-center">
        <div className="bg-white text-primary rounded-circle me-2 d-flex align-items-center justify-content-center" 
             style={{ width: "30px", height: "30px" }}>
          <i className="bi bi-box-seam" style={{ fontSize: "1.1rem" }}></i>
        </div>
        <span style={{ fontSize: "1.3rem" }}>StockSight</span>
      </span>

      {/* Profile & Logout Buttons */}
      <div className="d-flex align-items-center">
        {/* Profile Icon */}
        <Link to="/profile" className="btn btn-outline-light me-2">
          <i className="bi bi-person-circle"></i> Profile
        </Link>

        {/* Logout Button */}
        <button className="btn btn-danger" onClick={handleLogout}>
          <i className="bi bi-box-arrow-right"></i> Logout
        </button>
      </div>
    </nav>
  );
};

export default Header;
