import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import VerificationPopup from "../components/VerificationPopup";
import "../styles.css"; // Ensure styles are correctly imported

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
      setTimeout(() => {
        navigate("/login"); // Redirect to login page after successful verification
      }, 2000); // Redirect after 2 seconds
    } else {
      setError(data.message || "Invalid verification code.");
    }
  };

  return (
    <div className="d-flex justify-content-center align-items-center vh-100 bg-light">
      {/* Website Header */}
      <div className="position-absolute top-0 start-50 translate-middle-x mt-4">
        <h1 className="fw-bold">StockSight</h1>
      </div>

      {/* Sign Up Card */}
      <div className="card p-4 shadow-lg" style={{ width: "400px" }}>
        <h2 className="text-center mb-4">Sign Up</h2>
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
                onClick={() => setUsername("")} // Clear the username field
              >
                Try a different username
              </span>
            )}
          </div>
        )}
        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label">Full Name</label>
            <input
              type="text"
              className="form-control"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required
            />
          </div>

          <div className="mb-3">
            <label className="form-label">Handphone Number</label>
            <input
              type="tel"
              className="form-control"
              value={handphone}
              onChange={(e) => setHandphone(e.target.value)}
              required
            />
          </div>

          <div className="mb-3">
            <label className="form-label">Email Address</label>
            <input
              type="email"
              className="form-control"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="mb-3">
            <label className="form-label">Username</label>
            <input
              type="text"
              className="form-control"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div className="mb-3">
            <label className="form-label">Password</label>
            <input
              type="password"
              className="form-control"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <small className="text-muted">
              Password must contain at least 1 special character, mixed case
              letters, and be at least 8 characters long.
            </small>
          </div>

          <div className="mb-3">
            <label className="form-label">Access Level</label>
            <select
              className="form-select"
              value={accessLevel}
              onChange={(e) => setAccessLevel(e.target.value)}
            >
              <option value="Manager">Manager</option>
              <option value="Employee">Employee</option>
            </select>
          </div>

          <button
            type="submit"
            className="btn btn-success w-100"
            disabled={isLoading}
          >
            {isLoading ? "Signing up..." : "Sign Up"}
          </button>
        </form>

        {/* Redirect to Login */}
        <p className="text-center mt-3">
          Already have an account?{" "}
          <span
            className="text-primary"
            style={{ cursor: "pointer" }}
            onClick={() => navigate("/login")}
          >
            Login here
          </span>
        </p>
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
