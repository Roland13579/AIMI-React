import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

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
      console.log("Username saved:", data.username);
      navigate("/dashboard"); // ✅ Redirect user to dashboard
    }
  };

  return (
    <div className="d-flex justify-content-center align-items-center vh-100 bg-light">
      <div className="card p-4 shadow-lg" style={{ width: "350px" }}>
        <h2 className="text-center mb-4">Login</h2>
        <form onSubmit={handleSubmit}>
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
          <button type="submit" className="btn btn-primary w-100">
            Login
          </button>
        </form>

        <p className="text-center mt-3">
          Don't have an account?{" "}
          <span
            className="text-primary"
            style={{ cursor: "pointer" }}
            onClick={() => navigate("/signup")}
          >
            Sign Up
          </span>
        </p>
      </div>
    </div>
  );
};

export default Login;
