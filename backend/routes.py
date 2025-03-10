from flask import request, jsonify
from app import app, db, mail, inventory_collection
from models import User
import random
from flask_mail import Message
from werkzeug.security import generate_password_hash, check_password_hash

# Temporary storage for verification codes (in production, use a database or Redis)
verification_codes = {}

def generate_verification_code():
    return str(random.randint(100000, 999999))  # Generate a 6-digit code

# Route for user sign-up
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    # Check if user already exists
    if User.query.filter_by(handphone=data['handphone']).first():
        return jsonify({'message': 'User account already exists: Phone number taken.'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'User account already exists: Email taken.'}), 400
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'User account already exists: Username taken.'}), 400

    verification_code = generate_verification_code()
    verification_codes[data['email']] = verification_code  

    # Send email verification code
    msg = Message(
        subject="Verify Your Email",
        recipients=[data['email']],
        body=f"Your verification code is: {verification_code}"
    )
    mail.send(msg)

    # Create a new User with hashed password
    user = User(
        full_name=data['fullName'],
        handphone=data['handphone'],
        email=data['email'],
        username=data['username'],
        password=generate_password_hash(data['password']),  # Hash password
        access_level=data['accessLevel'],
        is_verified=False
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'Verification code sent to your email.'}), 200

# Route for verifying email
@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()

    if data['email'] in verification_codes and verification_codes[data['email']] == data['code']:
        user = User.query.filter_by(email=data['email']).first()
        user.is_verified = True
        db.session.commit()

        del verification_codes[data['email']]
        return jsonify({'message': 'Email verified successfully'}), 200
    else:
        return jsonify({'message': 'Invalid verification code'}), 400

# Route for user login (No Secret Key, Using Dummy Token)
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()

    if user and check_password_hash(user.password, data['password']):
        if not user.is_verified:
            return jsonify({'message': 'Please verify your email before logging in.'}), 403
        
        return jsonify({
            'message': 'Login successful',
            'token': 'dummy_token',  # Replace this with a real JWT token later
            'username': user.username  # ✅ Send username to frontend
        })
    else:
        return jsonify({'message': 'Invalid username or password'}), 401


@app.route('/profile', methods=['GET'])
def get_profile():
    username = request.headers.get("Username")  # ✅ Get username from request headers

    if not username or username.strip() == "":
        return jsonify({"error": "Unauthorized - No Username Provided"}), 401

    user = User.query.filter_by(username=username).first()  # ✅ Fetch correct user

    if not user:
        return jsonify({"error": "User not found", "username": username}), 404

    return jsonify({
        "full_name": user.full_name,
        "email": user.email,
        "username": user.username,
        "access_level": user.access_level
    })

@app.route('/update-profile', methods=['PUT'])
def update_profile():
    data = request.get_json()
    print("Received data:", data)  # Debugging log

    if "username" not in data or "email" not in data:
        return jsonify({"error": "Missing username or email"}), 400
    
    # Check if the new username already exists
    new_username = data.get("username")
    if User.query.filter_by(username=new_username).first():
        return jsonify({"error": "Username already exists"}), 400

    # Fetch the user by full_name (which we assume is unique)
    user = User.query.filter_by(full_name=data["full_name"]).first()

    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Update the user's data if present in the request
    if "username" in data:
        user.username = data["username"]
    if "email" in data:
        user.email = data["email"]
    if "full_name" in data:
        user.full_name = data["full_name"]

    try:
        db.session.commit()  # Commit the changes to the database
        return jsonify({"message": "Profile updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500





#----------------------Inventory----------------------#
#Inventory

#Get all items
@app.route('/inventory', methods=['GET'])
def get_inventory():
    inventory_items = list(inventory_collection.find({}, {"_id": 0}))  # Fetch all items
    return jsonify(inventory_items)

#Add an item
@app.route('/inventory', methods=['POST'])
def add_inventory():
    data = request.get_json()

    # ✅ Ensure all required fields exist
    new_item = {
        "item_id": data.get("item_id", ""),
        "item_name": data.get("item_name", ""),
        "description": data.get("description", ""),
        "SKU": data.get("SKU", ""),
        "quantity": data.get("quantity", 0),
        "reorder_point": data.get("reorder_point", 0),
        "cost_price": data.get("cost_price", 0.0),
        "selling_price": float(data.get("selling_price", 0.0)),
        "expiration_date": data.get("expiration_date", None)
    }

    inventory_collection.insert_one(new_item)  # ✅ Insert into MongoDB
    return jsonify({'message': 'Item added successfully'}), 201


#Update an item
@app.route('/inventory/<string:item_id>', methods=['PUT']) #PUT method to update an item
def update_inventory(item_id):
    data = request.get_json()
    inventory_collection.update_one({"item_id": item_id}, {"$set": data}) #Update the item with the specified item_id
    return jsonify({'message': 'Item updated successfully'})

#Delete an item
@app.route('/inventory/<string:item_id>', methods=['DELETE'])
def delete_inventory(item_id):
    inventory_collection.delete_one({"item_id": item_id})
    return jsonify({"message": "Item deleted successfully"})

#----------------------Sales----------------------#