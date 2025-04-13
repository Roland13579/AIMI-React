import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import VerificationPopup from "../components/VerificationPopup";
import "../styles.css"; // Ensure styles are correctly imported
import "bootstrap-icons/font/bootstrap-icons.css"; // Bootstrap Icons

const SignUp: React.FC = () => {
  const [fullName, setFullName] = useState("");
  const [handphone, setHandphone] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [accessLevel, setAccessLevel] = useState("Employee"); // Default to Employee
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [showVerificationPopup, setShowVerificationPopup] = useState(false);

  const navigate = useNavigate();

  const validatePassword = (password: string): string | null => {
    const regex =
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
    if (!regex.test(password)) {
      return "Password must contain at least 1 special character, mixed case letters, and be at least 8 characters long.";
    }
    return null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    // Validate password
    const passwordError = validatePassword(password);
    if (passwordError) {
      setError(passwordError);
      setIsLoading(false);
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fullName,
          handphone,
          email,
          username,
          password,
          accessLevel,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        setSuccessMessage("Verification code sent to your email.");
        setShowVerificationPopup(true); // Show the verification popup
      } else {
        // Handle specific error messages from the backend
        if (data.message.includes("Phone number taken")) {
          setError("User account already exists: Phone number taken.");
        } else if (data.message.includes("Email taken")) {
          setError("User account already exists: Email taken.");
        } else if (data.message.includes("Username already taken")) {
          setError(
            "Username already taken. Please choose a different username."
          );
        } else {
          setError(data.message || "Signup failed. Please try again.");
        }
      }
    } catch (err) {
      setError("An error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerify = async (code: string) => {
    const response = await fetch("http://127.0.0.1:5000/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, code }),
    });

    const data = await response.json();
    if (response.ok) {
      setSuccessMessage("Email verified successfully! Redirecting to login...");
      
      // Set flag to show welcome popup on dashboard after login
      localStorage.setItem("show_welcome_popup", "true");
      
      setTimeout(() => {
        navigate("/login"); // Redirect to login page after successful verification
      }, 2000); // Redirect after 2 seconds
    } else {
      setError(data.message || "Invalid verification code.");
    }
  };

  return (
    <div className="d-flex justify-content-center align-items-center" style={{ minHeight: "100vh" }}>
      <div 
        style={{ 
          width: "1000px", 
          maxWidth: "95%", 
          display: "flex", 
          boxShadow: "0 0.5rem 1rem rgba(0, 0, 0, 0.15)", 
          borderRadius: "10px",
          overflow: "hidden"
        }}
      >
        {/* Left side - Logo and description */}
        <div 
          style={{ 
            flex: "0 0 35%", 
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
            <h1 className="fw-bold mb-2" style={{ fontSize: "1.8rem" }}>StockSight</h1>
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
        
        {/* Right side - Signup form */}
        <div 
          style={{ 
            flex: "0 0 65%", 
            background: "rgba(234, 230, 230, 0.95)", 
            padding: "2rem"
          }}
        >
          <div className="text-center mb-3">
            <div 
              className="bg-primary text-white mx-auto mb-2" 
              style={{ 
                width: "50px", 
                height: "50px", 
                borderRadius: "50%", 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "center" 
              }}
            >
              <i className="bi bi-person-plus-fill" style={{ fontSize: "1.5rem" }}></i>
            </div>
            <h2 className="fw-bold mb-1" style={{ fontSize: "1.5rem", color: "rgba(3, 38, 75, 0.8)" }}>Create Account</h2>
            <p className="text-muted mb-3" style={{ fontSize: "0.9rem" }}>Join StockSight to manage your inventory efficiently</p>
          </div>
          
          {successMessage && (
            <div className="alert alert-success">{successMessage}</div>
          )}
          {error && (
            <div className="alert alert-danger">
              {error}
              {error.includes("Username already taken") && (
                <span
                  className="text-primary"
                  style={{ cursor: "pointer", marginLeft: "5px" }}
                  onClick={() => setUsername("")}
                >
                  Try a different username
                </span>
              )}
            </div>
          )}
          
          <form onSubmit={handleSubmit}>
            <div className="row">
              <div className="col-md-6 mb-3">
                <div className="input-group">
                  <span className="input-group-text bg-light border-end-0">
                    <i className="bi bi-person text-primary"></i>
                  </span>
                  <input
                    type="text"
                    className="form-control border-start-0 ps-0"
                    placeholder="Full Name"
                    style={{ fontSize: "0.95rem" }}
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    required
                  />
                </div>
              </div>
              <div className="col-md-6 mb-3">
                <div className="input-group">
                  <span className="input-group-text bg-light border-end-0">
                    <i className="bi bi-phone text-primary"></i>
                  </span>
                  <input
                    type="tel"
                    className="form-control border-start-0 ps-0"
                    placeholder="Handphone Number"
                    style={{ fontSize: "0.95rem" }}
                    value={handphone}
                    onChange={(e) => setHandphone(e.target.value)}
                    required
                  />
                </div>
              </div>
            </div>

            <div className="row">
              <div className="col-md-6 mb-3">
                <div className="input-group">
                  <span className="input-group-text bg-light border-end-0">
                    <i className="bi bi-envelope text-primary"></i>
                  </span>
                  <input
                    type="email"
                    className="form-control border-start-0 ps-0"
                    placeholder="Email Address"
                    style={{ fontSize: "0.95rem" }}
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
              </div>
              <div className="col-md-6 mb-3">
                <div className="input-group">
                  <span className="input-group-text bg-light border-end-0">
                    <i className="bi bi-person-badge text-primary"></i>
                  </span>
                  <input
                    type="text"
                    className="form-control border-start-0 ps-0"
                    placeholder="Username"
                    style={{ fontSize: "0.95rem" }}
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                  />
                </div>
              </div>
            </div>

            <div className="row">
              <div className="col-md-6 mb-3">
                <div className="input-group">
                  <span className="input-group-text bg-light border-end-0">
                    <i className="bi bi-lock text-primary"></i>
                  </span>
                  <input
                    type="password"
                    className="form-control border-start-0 ps-0"
                    placeholder="Password"
                    style={{ fontSize: "0.95rem" }}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
                <small className="text-muted d-block mt-1" style={{ fontSize: "0.75rem" }}>
                  <i className="bi bi-info-circle me-1"></i>
                  Password must contain at least 1 special character, mixed case
                  letters, and be at least 8 characters long.
                </small>
              </div>
              <div className="col-md-6 mb-3">
                <div className="input-group">
                  <span className="input-group-text bg-light border-end-0">
                    <i className="bi bi-shield-lock text-primary"></i>
                  </span>
                  <select
                    className="form-select border-start-0 ps-0"
                    style={{ fontSize: "0.95rem" }}
                    value={accessLevel}
                    onChange={(e) => setAccessLevel(e.target.value)}
                  >
                    <option value="" disabled>Select Access Level</option>
                    <option value="Manager">Manager</option>
                    <option value="Employee">Employee</option>
                  </select>
                </div>
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary w-100 py-2 mb-3 fw-bold"
              style={{ fontSize: "1rem" }}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                  Creating Account...
                </>
              ) : (
                <>
                  <i className="bi bi-check-circle me-2"></i>
                  Create Account
                </>
              )}
            </button>
          </form>

          <div className="text-center mt-2">
            <p className="mb-0" style={{ fontSize: "0.85rem", color: "gray" }}>
              Already have an account?{" "}
              <span
                className="text-primary"
                style={{ cursor: "pointer", fontSize: "0.85rem" }}
                onClick={() => navigate("/login")}
              >
                Sign In
              </span>
            </p>
          </div>
        </div>
      </div>

      {/* Verification Popup */}
      {showVerificationPopup && (
        <VerificationPopup
          email={email}
          onVerify={handleVerify}
          onClose={() => setShowVerificationPopup(false)}
        />
      )}
    </div>
  );
};

export default SignUp;
