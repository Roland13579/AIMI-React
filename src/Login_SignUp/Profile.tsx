import React, { useEffect, useState } from "react";

const Profile = () => {
  const [user, setUser] = useState({ full_name: "", email: "", username: "" });
  const [error, setError] = useState("");

  useEffect(() => {
    const username = localStorage.getItem("username");

    if (!username || username.trim() === "") {
      console.error("No username found in localStorage");
      setError("User not logged in.");
      return;
    }

    console.log("Fetching profile for username:", username); // ✅ Debugging log

    fetch("http://127.0.0.1:5000/profile", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Username: username, // ✅ Send username to Flask
      },
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch user data");
        }
        return response.json();
      })
      .then((data) => {
        console.log("Profile data received:", data); // ✅ Debugging log
        setUser(data);
      })
      .catch((error) => setError(error.message));
  }, []);

  if (error) return <p className="text-danger">{error}</p>;

  return (
    <div className="container mt-4">
      <h2>Profile</h2>
      <div className="card p-3">
        <p>
          <strong>Full Name:</strong> {user.full_name}
        </p>
        <p>
          <strong>Email:</strong> {user.email}
        </p>
        <p>
          <strong>Username:</strong> {user.username}
        </p>
        <p>
          <strong>Access level:</strong> {user.access_level}
        </p>
      </div>
    </div>
  );
};

export default Profile;
