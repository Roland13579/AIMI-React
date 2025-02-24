import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles.css"; // Ensure styles are correctly imported

const SignUp: React.FC = () => {
  const [fullName, setFullName] = useState("");
  const [handphone, setHandphone] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [accessLevel, setAccessLevel] = useState("Employee"); // Default to Employee

  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert(
      `Full Name: ${fullName}\nHandphone: ${handphone}\nEmail: ${email}\nUsername: ${username}\nAccess Level: ${accessLevel}`
    );
    navigate("/login"); // Redirect to login after signup
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

          <button type="submit" className="btn btn-success w-100">
            Sign Up
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
    </div>
  );
};

export default SignUp;
