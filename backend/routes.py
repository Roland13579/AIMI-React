from flask import request, jsonify
from app import app, db, mail
from models import User
import random
from flask_mail import Message

# Route for user sign-up
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()  # Get JSON data from the request

    #Check if user exist by handphone, email, and username
    if (User.query.filter_by(handphone=data['handphone']).first()):
        return jsonify({'message': 'User account already exists: Phone number taken.'}), 400
    if(User.query.filter_by(email=data['email']).first()):
        return jsonify({'message': 'User account already exists: Email taken.'}), 400
    if(User.query.filter_by(username=data['username']).first()):
        return jsonify({'message': 'User account already exists: Username taken.'}), 400
    
    verification_code = generate_verification_code()
    verification_codes[data['email']] = verification_code  # Store the code temporarily

     # Send the verification code via email
    msg = Message(
        subject="Verify Your Email",
        recipients=[data['email']],
        body=f"Your verification code is: {verification_code}"
    )
    mail.send(msg)

    # Create a new User object with the data from the request (Dont commit yet)
    user = User(
        full_name=data['fullName'],
        handphone=data['handphone'],
        email=data['email'],
        username=data['username'],
        password=data['password'],  # Hash the password before saving
        access_level=data['accessLevel'],
        is_verified=False  # Set verification as false
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'Verification code sent to your email.'}), 200


@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()

    #Check if code match
    if data['email'] in verification_codes and verification_codes[data['email']] == data['code']:

        user = User.query.filter_by(email=data['email']).first()
        user.is_verified = True
        db.session.commit()

        #Remove the code from temporary storage
        del verification_codes[data['email']]
        return jsonify({'message': 'Email verified successfully'}), 200
    else:
        return jsonify({'message': 'Invalid verification code'}), 400

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
    

# Temporary storage for verification codes (in production, use a database or Redis)
verification_codes = {}

def generate_verification_code():
    return str(random.randint(100000, 999999))  # Generate a 6-digit code