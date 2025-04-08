import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "bootstrap-icons/font/bootstrap-icons.css"; // Bootstrap Icons

interface LoginProps {
  setIsAuthenticated: (auth: boolean) => void; // Define the prop
}

const Login: React.FC<LoginProps> = ({ setIsAuthenticated }) => {
  // Accept the prop
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const navigate = useNavigate(); // React Router navigation function

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const response = await fetch("http://127.0.0.1:5000/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();
    alert(data.message); // Show success/error message

    if (response.ok) {
      localStorage.setItem("token", data.token); // Store token
      setIsAuthenticated(true); // ✅ Update authentication state
      localStorage.setItem("username", data.username); // ✅ Store username (for profile.tsx)
      
      // Set flag to show welcome popup on dashboard
      localStorage.setItem("show_welcome_popup", "true");
      
      console.log("Username saved:", data.username);
      navigate("/dashboard"); // ✅ Redirect user to dashboard
    }
  };

  return (
    <div className="d-flex justify-content-center align-items-center" style={{ minHeight: "100vh" }}>
      <div 
        style={{ 
          width: "900px", 
          maxWidth: "90%", 
          display: "flex", 
          boxShadow: "0 0.5rem 1rem rgba(0, 0, 0, 0.15)", 
          borderRadius: "10px",
          overflow: "hidden"
        }}
      >
        {/* Left side - Logo and description */}
        <div 
          style={{ 
            flex: "0 0 40%", 
            background: "rgba(3, 38, 75, 0.8)", 
            padding: "2rem", 
            display: "flex", 
            flexDirection: "column", 
            justifyContent: "center",
            color: "white"
          }}
        >
          <div className="text-center mb-4">
            <div 
              className="bg-white text-primary mx-auto mb-3" 
              style={{ 
                width: "70px", 
                height: "70px", 
                borderRadius: "50%", 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "center" 
              }}
            >
              <i className="bi bi-box-seam" style={{ fontSize: "2rem" }}></i>
            </div>
            <h1 className="fw-bold mb-2" style={{ fontSize: "2rem" }}>StockSight</h1>
            <p className="mb-1" style={{ fontSize: "0.9rem" }}>Inventory Management & Business Forecasting</p>
            <p className="mb-4" style={{ fontSize: "0.75rem", opacity: "0.8" }}>Powered by StockSight AI and SingStat API</p>
          </div>
          
          <div className="mt-2">
            <div className="d-flex align-items-center mb-3">
              <i className="bi bi-graph-up-arrow me-3" style={{ fontSize: "1.2rem" }}></i>
              <div>
                <h6 className="mb-0" style={{ fontSize: "0.9rem" }}>AI-Powered Forecasting</h6>
                <p className="mb-0" style={{ fontSize: "0.75rem", opacity: "0.8" }}>Predict sales trends with precision</p>
              </div>
            </div>
            <div className="d-flex align-items-center mb-3">
              <i className="bi bi-box-seam me-3" style={{ fontSize: "1.2rem" }}></i>
              <div>
                <h6 className="mb-0" style={{ fontSize: "0.9rem" }}>Smart Inventory</h6>
                <p className="mb-0" style={{ fontSize: "0.75rem", opacity: "0.8" }}>Automated stock management</p>
              </div>
            </div>
            <div className="d-flex align-items-center">
              <i className="bi bi-clipboard-data me-3" style={{ fontSize: "1.2rem" }}></i>
              <div>
                <h6 className="mb-0" style={{ fontSize: "0.9rem" }}>Industry Insights</h6>
                <p className="mb-0" style={{ fontSize: "0.75rem", opacity: "0.8" }}>Real-time market analytics</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Right side - Login form */}
        <div 
          style={{ 
            flex: "0 0 60%", 
            background: "rgba(234, 230, 230, 0.95)", 
            padding: "2rem"
          }}
        >
          <div className="text-center mb-4">
            <h2 className="fw-bold mb-1" style={{ fontSize: "1.5rem", color: "rgba(3, 38, 75, 0.8)" }}>Welcome Back</h2>
            <p className="text-muted mb-3" style={{ fontSize: "0.9rem" }}>Sign in to continue to StockSight</p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <div className="input-group">
                <span className="input-group-text bg-light border-end-0">
                  <i className="bi bi-person text-primary"></i>
                </span>
                <input
                  type="text"
                  className="form-control border-start-0 ps-0"
                  placeholder="Username"
                  style={{ fontSize: "1rem" }}
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </div>
            </div>
            <div className="mb-4">
              <div className="input-group">
                <span className="input-group-text bg-light border-end-0">
                  <i className="bi bi-lock text-primary"></i>
                </span>
                <input
                  type="password"
                  className="form-control border-start-0 ps-0"
                  placeholder="Password"
                  style={{ fontSize: "1rem" }}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>
            <button type="submit" className="btn btn-primary w-100 py-2 mb-3 fw-bold" style={{ fontSize: "1rem" }}>
              <i className="bi bi-box-arrow-in-right me-2"></i>
              Sign In
            </button>
          </form>

          <div className="text-center mt-2">
            <p className="mb-0" style={{ fontSize: "0.85rem", color: "gray"}}>
              Don't have an account?{" "}
              <span
                className="text-primary"
                style={{ cursor: "pointer", fontSize: "0.85rem" }}
                onClick={() => navigate("/signup")}
              >
                Sign Up
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
