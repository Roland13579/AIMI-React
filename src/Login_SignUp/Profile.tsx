import React, { useEffect, useState } from "react";

const Profile = () => {
  const [user, setUser] = useState({
    full_name: "",
    email: "",
    username: "",
    access_level: "",
  });
  const [error, setError] = useState("");
  const [isEditing, setIsEditing] = useState({ username: false, email: false });
  const [editedUser, setEditedUser] = useState({
    full_name: "",
    email: "",
    username: "",
  });

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
        setEditedUser(data); // Initialize editedUser with current user data
      })
      .catch((error) => setError(error.message));
  }, []);

  const handleEdit = (field: string) => {
    setIsEditing({ ...isEditing, [field]: true });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditedUser({ ...editedUser, [e.target.name]: e.target.value });
  };

  const handleSave = () => {
    // Validation: if editing email or username, ensure they're not empty
    if (isEditing.email) {
      if (!editedUser.email.trim()) {
        alert("Email cannot be empty");
        return;
      }
      if (!editedUser.email.includes("@")) {
        alert("Please enter a valid email address containing '@'");
        return;
      }
    }
    if (isEditing.username) {
      if (!editedUser.username.trim()) {
        alert("Username cannot be empty");
        return;
      }
    }

    const updatedUser = {
      username: editedUser.username,
      email: editedUser.email,
      full_name: editedUser.full_name,
      access_level: user.access_level,
    };

    fetch("http://127.0.0.1:5000/update-profile", {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updatedUser),
    })
      .then((response) => response.json())
      .then((data) => {
        alert("Profile updated successfully");
        setIsEditing({ username: false, email: false });
        setUser(updatedUser); // Update the user state with the new values
      })
      .catch((error) => {
        console.error("Error updating profile:", error);
        alert("Failed to update profile");
      });
  };

  if (error) return <p className="text-danger">{error}</p>;

  return (
    <div className="container mt-4">
      <h2>Profile</h2>
      <div className="card p-3">
        <div>
          <strong>Full Name: </strong> {user.full_name}
        </div>
        <div>
          <strong>Access level: </strong> {user.access_level}
        </div>
        <div>
          <strong>Email: </strong>
          {isEditing.email ? (
            <input
              type="email"
              name="email"
              value={editedUser.email}
              onChange={handleChange}
              className="form-control"
            />
          ) : (
            <>
              {user.email}{" "}
              <button
                onClick={() => handleEdit("email")}
                className="btn btn-link"
              >
                Edit
              </button>
            </>
          )}
        </div>
        <div>
          <strong>Username: </strong>
          {isEditing.username ? (
            <input
              type="text"
              name="username"
              value={editedUser.username}
              onChange={handleChange}
              className="form-control"
            />
          ) : (
            <>
              {user.username}{" "}
              <button
                onClick={() => handleEdit("username")}
                className="btn btn-link"
              >
                Edit
              </button>
            </>
          )}
        </div>
        {(isEditing.username || isEditing.email) && (
          <div className="mt-3">
            <button onClick={handleSave} className="btn btn-success">
              Save
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Profile;
