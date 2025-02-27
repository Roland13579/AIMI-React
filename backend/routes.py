from flask import request, jsonify
from app import app, db
from models import User

# Route for user sign-up
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()  # Get JSON data from the request

    # Create a new User object with the data from the request
    user = User(
        full_name=data['fullName'],
        handphone=data['handphone'],
        email=data['email'],
        username=data['username'],
        password=data['password'],  # In a real application, you should hash the password
        access_level=data['accessLevel']
    )

    db.session.add(user)  # Add the new user to the database session
    db.session.commit()  # Commit the session to save the user to the database

    return jsonify({'message': 'User created successfully'}), 201  # Return a success message

# Route for user login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()  # Get JSON data from the request

    # Query the database for a user with the provided username
    user = User.query.filter_by(username=data['username']).first()

    if user and user.password == data['password']:  # Check if the user exists and the password matches
        return jsonify({'message': 'Login successful', 'token': 'dummy_token'})  # Return a success message and a dummy token
    else:
        return jsonify({'message': 'Invalid username or password'}), 401  # Return an error message