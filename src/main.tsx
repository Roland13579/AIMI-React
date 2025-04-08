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
import PurchaseOrder from "./Purchase_Order/PurchaseOrder";
import CreatePurchaseOrder from "./Purchase_Order/CreatePurchaseOrder";
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
              paddingTop: isAuthenticated ? "10px" : "0", // Add padding for better spacing
              overflowY: "auto", // Enable scrolling for content
              height: isAuthenticated ? "calc(100vh - 60px)" : "100vh", // Adjust height based on authentication
              background: "linear-gradient(135deg, #e8ecf2 0%, #d8e1eb 100%)" // More bluish-gray gradient
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
                  <Route
                    path="/create-purchase-order"
                    element={<CreatePurchaseOrder />}
                  />
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
