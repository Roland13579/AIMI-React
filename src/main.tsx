import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import Profile from "./Login_SignUp/Profile";
import Login from "./Login_SignUp/Login";
import SignUp from "./Login_SignUp/Signup";
import Dashboard from "./Dashboard";
import Forecast from "./Forecast";
import Sales from "./Sales";
import Inventory from "./Inventory/Inventory";
import AddItem from "./Inventory/AddItem";
import PurchaseOrder from "./PurchaseOrder";
import "bootstrap/dist/css/bootstrap.min.css";

const App = () => {
  // 🔹 Reset auth on every reload (for testing only)
  //useEffect(() => {
  //localStorage.removeItem("token"); // Clear stored token
  //}, []);

  //const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(
    !!localStorage.getItem("token")
  );
  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem("token");
      setIsAuthenticated(!!token);
    };

    window.addEventListener("storage", checkAuth);
    return () => window.removeEventListener("storage", checkAuth);
  }, []);

  return (
    <Router>
      <div className="d-flex flex-column vh-100">
        {/* ✅ Show Header only if user is authenticated */}
        {isAuthenticated && <Header />}

        <div className="d-flex flex-grow-1">
          {/* ✅ Show Sidebar only if user is authenticated */}
          {isAuthenticated && <Sidebar />}

          <div
            className="flex-grow-1 p-3"
            style={{
              marginLeft: isAuthenticated ? "250px" : "0",
              marginTop: isAuthenticated ? "60px" : "0", // Adjust margin for header
            }}
          >
            <Routes>
              {/* ✅ Show Login & Signup only when NOT authenticated */}
              {!isAuthenticated ? (
                <>
                  <Route
                    path="/"
                    element={<Login setIsAuthenticated={setIsAuthenticated} />}
                  />
                  <Route
                    path="/login"
                    element={<Login setIsAuthenticated={setIsAuthenticated} />}
                  />
                  <Route path="/signup" element={<SignUp />} />
                </>
              ) : (
                <>
                  <Route path="/" element={<Navigate to="/dashboard" />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/forecast" element={<Forecast />} />
                  <Route path="/sales" element={<Sales />} />
                  <Route path="/inventory" element={<Inventory />} />
                  <Route path="/add-item" element={<AddItem />} />
                  <Route path="/purchase-order" element={<PurchaseOrder />} />
                  <Route path="/profile" element={<Profile />} />
                </>
              )}
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
