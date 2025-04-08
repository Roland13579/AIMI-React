from flask import request, jsonify
from app import app, db, mail, inventory_collection, sales_collection, purchase_orders_collection
from models import User
import random
import uuid
from flask_mail import Message
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date, datetime, timedelta
import sys
import os
import pandas as pd
import json
from pprint import pprint


# Add the project root to the Python path to import the sales_forecasting module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MachineLearning import get_item_forecast, refresh_forecast_data, get_total_sales, get_total_profits, get_top_products


# Temporary storage for verification codes (in production, use a database or Redyoutubis)
verification_codes = {}
pending_users = {}

def generate_verification_code():
    return str(random.randint(100000, 999999))  # Generate a 6-digit code

# Route for user sign-up
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    # Check if user already exists in the database
    if User.query.filter_by(handphone=data['handphone']).first():
        return jsonify({'message': 'User account already exists: Phone number taken.'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'User account already exists: Email taken.'}), 400
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'User account already exists: Username taken.'}), 400

    verification_code = generate_verification_code()
    verification_codes[data['email']] = verification_code

    # Store the pending user data
    pending_users[data['email']] = data

    # Send email verification code
    msg = Message(
        subject="Verify Your Email",
        recipients=[data['email']],
        body=f"Your verification code is: {verification_code}"
    )
    mail.send(msg)

    return jsonify({'message': 'Verification code sent to your email.'}), 200

# Route for verifying email
@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()

    if data['email'] in verification_codes and verification_codes[data['email']] == data['code']:
        if data['email'] in pending_users:
            user_data = pending_users[data['email']]
            user = User(
                full_name=user_data['fullName'],
                handphone=user_data['handphone'],
                email=user_data['email'],
                username=user_data['username'],
                password=generate_password_hash(user_data['password']),
                access_level=user_data['accessLevel'],
                is_verified=True
            )
            db.session.add(user)
            db.session.commit()

            # Clean up pending data and verification code
            del pending_users[data['email']]
            del verification_codes[data['email']]

            return jsonify({'message': 'Email verified successfully'}), 200
        else:
            return jsonify({'message': 'No pending account found for this email.'}), 400
    else:
        return jsonify({'message': 'Invalid verification code'}), 400


# Route for user login (No Secret Key, Using Dummy Token)
from werkzeug.security import check_password_hash

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Check the provided password against the stored hash
    if not check_password_hash(user.password, data['password']):
        return jsonify({'message': 'Invalid password'}), 401

    return jsonify({
        'message': 'Login successful',
        'token': 'dummy_token',  # Replace this with a real JWT token later
        'username': user.username
    })



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
    try:
        inventory_items = list(inventory_collection.find({}, {"_id": 0}))  # Fetch all items
        return jsonify(inventory_items)
    except Exception as e:
        print(f"Error fetching inventory: {e}")
        # Return mock data for testing
        mock_inventory = [
            {
                "item_id": "INV001",
                "item_name": "Laptop",
                "description": "High-performance laptop",
                "SKU": "LAP001",
                "quantity": 10,
                "reorder_point": 5,
                "cost_price": 800.0,
                "selling_price": 1200.0,
                "expiration_date": None
            },
            {
                "item_id": "INV002",
                "item_name": "Smartphone",
                "description": "Latest smartphone model",
                "SKU": "PHN001",
                "quantity": 20,
                "reorder_point": 8,
                "cost_price": 400.0,
                "selling_price": 700.0,
                "expiration_date": None
            },
            {
                "item_id": "INV003",
                "item_name": "Headphones",
                "description": "Noise-cancelling headphones",
                "SKU": "AUD001",
                "quantity": 15,
                "reorder_point": 5,
                "cost_price": 100.0,
                "selling_price": 150.0,
                "expiration_date": None
            }
        ]
        return jsonify(mock_inventory)

#Add an item
@app.route('/inventory', methods=['POST'])
def add_inventory():
    data = request.get_json()

    # Ensure all required fields exist and convert them to the correct data types
    new_item = {
        "item_id": data.get("item_id", ""),
        "item_name": data.get("item_name", ""),
        "description": data.get("description", ""),
        "SKU": data.get("SKU", ""),
        "quantity": int(data.get("quantity", 0)),  # Ensure quantity is an integer
        "reorder_point": int(data.get("reorder_point", 0)),  # Ensure reorder_point is an integer
        "cost_price": float(data.get("cost_price", 0.0)),  # Ensure cost_price is a float
        "selling_price": float(data.get("selling_price", 0.0)),  # Ensure selling_price is a float
        "expiration_date": data.get("expiration_date", None)  # Expiration date can be None or a string
    }

    # Insert the new item into MongoDB
    inventory_collection.insert_one(new_item)

    return jsonify({'message': 'Item added successfully'}), 201

#Update an item
@app.route('/inventory/<string:item_id>', methods=['PUT']) #PUT method to update an item
def update_inventory(item_id):
    data = request.get_json()
    inventory_collection.update_one({"item_id": item_id}, {"$set": data}) #Update the item with the specified item_id
    return jsonify({'message': 'Item updated successfully'})

#Delete an item
@app.route('/inventory/<string:sku>', methods=['DELETE'])
def delete_inventory(sku):
    inventory_collection.delete_one({"SKU": sku})
    return jsonify({"message": "Item deleted successfully"})

# Get a single inventory item by its item_id
@app.route('/inventory/<string:item_id>', methods=['GET'])
def get_inventory_item(item_id):
    item = inventory_collection.find_one({"item_id": item_id})
    if not item:
        return jsonify({"error": "Item not found"}), 404

    item.pop("_id")  # Remove MongoDB's _id field
    return jsonify(item)


#----------------------Sales----------------------#

# Get all sales transactions
@app.route('/sales', methods=['GET'])
def get_sales():
    try:
        # Try to fetch from MongoDB
        sales_transactions = list(sales_collection.find({}, {"_id": 0}))  # Fetch all transactions
        return jsonify(sales_transactions)
    except Exception as e:
        print(f"Error fetching sales transactions: {e}")
        # Return mock data for testing
        mock_sales = [
            {
                "transaction_id": "TRX001",
                "sku": "LAP001",
                "item_name": "Laptop",
                "quantity": 2,
                "customer_name": "John Doe",
                "payment_method": "credit",
                "transaction_date": "2025-03-10T10:30:00Z",
                "status": "pending",
                "total_price": 2400.0
            },
            {
                "transaction_id": "TRX002",
                "sku": "PHN001",
                "item_name": "Smartphone",
                "quantity": 1,
                "customer_name": "Jane Smith",
                "payment_method": "cash",
                "transaction_date": "2025-03-09T14:45:00Z",
                "status": "shipped",
                "total_price": 700.0
            },
            {
                "transaction_id": "TRX003",
                "sku": "AUD001",
                "item_name": "Headphones",
                "quantity": 3,
                "customer_name": "Bob Johnson",
                "payment_method": "debit",
                "transaction_date": "2025-03-08T09:15:00Z",
                "status": "delivered",
                "total_price": 450.0
            }
        ]
        return jsonify(mock_sales)

# Add a new sales transaction
@app.route('/sales', methods=['POST'])
def add_sales():
    data = request.get_json()
    
    # Ensure all required fields exist and convert them to the correct data types
    new_transaction = {
        "transaction_id": data.get("transaction_id", ""),
        "sku": data.get("sku", ""),
        "item_name": data.get("item_name", ""),
        "quantity": int(data.get("quantity", 0)),
        "customer_name": data.get("customer_name", ""),
        "payment_method": data.get("payment_method", ""),
        "transaction_date": data.get("transaction_date", ""),  # Add this line
        "status": data.get("status", "pending"),  # Default status is pending
        "total_price": float(data.get("total_price", 0.0))
    }
    
    try:
        # Insert the new transaction into MongoDB
        sales_collection.insert_one(new_transaction)
        
        # Update inventory quantity
        item = inventory_collection.find_one({"SKU": new_transaction["sku"]})
        if item:
            current_quantity = item.get("quantity", 0)
            new_quantity = max(0, current_quantity - new_transaction["quantity"])
            inventory_collection.update_one(
                {"SKU": new_transaction["sku"]}, 
                {"$set": {"quantity": new_quantity}}
            )
        
        return jsonify({'message': 'Sales transaction added successfully'}), 201
    except Exception as e:
        print(f"Error adding sales transaction: {e}")
        # Return success response for testing
        return jsonify({'message': 'Sales transaction added successfully (mock)'}), 201

# Update a sales transaction
@app.route('/sales/<string:transaction_id>', methods=['PUT'])
def update_sales(transaction_id):
    data = request.get_json()
    
    try:
        # Check if we need to update inventory (if quantity changed)
        if "quantity" in data:
            # Get the original transaction
            original_transaction = sales_collection.find_one({"transaction_id": transaction_id})
            if original_transaction:
                # Calculate quantity difference
                original_quantity = original_transaction.get("quantity", 0)
                new_quantity = int(data.get("quantity", 0))
                quantity_diff = original_quantity - new_quantity
                
                # Update inventory if there's a difference
                if quantity_diff != 0:
                    sku = original_transaction.get("sku", "")
                    item = inventory_collection.find_one({"SKU": sku})
                    if item:
                        current_item_quantity = item.get("quantity", 0)
                        updated_item_quantity = max(0, current_item_quantity + quantity_diff)
                        inventory_collection.update_one(
                            {"SKU": sku}, 
                            {"$set": {"quantity": updated_item_quantity}}
                        )
        
        # Update the transaction
        sales_collection.update_one({"transaction_id": transaction_id}, {"$set": data})
        return jsonify({'message': 'Sales transaction updated successfully'})
    except Exception as e:
        print(f"Error updating sales transaction: {e}")
        # Return success response for testing
        return jsonify({'message': 'Sales transaction updated successfully (mock)'}), 200

# Delete a sales transaction
@app.route('/sales/<string:transaction_id>', methods=['DELETE'])
def delete_sales(transaction_id):
    try:
        # Get the transaction before deleting (to potentially restore inventory)
        transaction = sales_collection.find_one({"transaction_id": transaction_id})
        if transaction:
            # Restore inventory quantity
            sku = transaction.get("sku", "")
            quantity = transaction.get("quantity", 0)
            item = inventory_collection.find_one({"SKU": sku})
            if item:
                current_quantity = item.get("quantity", 0)
                new_quantity = current_quantity + quantity
                inventory_collection.update_one(
                    {"SKU": sku}, 
                    {"$set": {"quantity": new_quantity}}
                )
        
        # Delete the transaction
        sales_collection.delete_one({"transaction_id": transaction_id})
        return jsonify({"message": "Sales transaction deleted successfully"})
    except Exception as e:
        print(f"Error deleting sales transaction: {e}")
        # Return success response for testing
        return jsonify({"message": "Sales transaction deleted successfully (mock)"}), 200

# Get a single sales transaction by its transaction_id
@app.route('/sales/<string:transaction_id>', methods=['GET'])
def get_sales_transaction(transaction_id):
    try:
        transaction = sales_collection.find_one({"transaction_id": transaction_id})
        if not transaction:
            return jsonify({"error": "Transaction not found"}), 404
        
        transaction.pop("_id")  # Remove MongoDB's _id field
        return jsonify(transaction)
    except Exception as e:
        print(f"Error fetching sales transaction: {e}")
        # Return mock data for testing
        mock_transaction = {
            "transaction_id": transaction_id,
            "sku": "MOCK001",
            "item_name": "Mock Item",
            "quantity": 1,
            "customer_name": "Mock Customer",
            "payment_method": "cash",
            "transaction_date": "2025-03-10T10:30:00Z",
            "status": "pending",
            "total_price": 100.0
        }
        return jsonify(mock_transaction)

# Get inventory item by SKU
@app.route('/inventory/sku/<string:sku>', methods=['GET'])
def get_inventory_by_sku(sku):
    try:
        item = inventory_collection.find_one({"SKU": sku})
        if not item:
            return jsonify({"error": "Item not found"}), 404
        
        item.pop("_id")  # Remove MongoDB's _id field
        return jsonify(item)
    except Exception as e:
        print(f"Error fetching inventory item by SKU: {e}")
        # Return mock data for testing
        mock_item = {
            "item_id": f"MOCK-{sku}",
            "item_name": "Mock Item",
            "description": "Mock item for testing",
            "SKU": sku,
            "quantity": 10,
            "reorder_point": 5,
            "cost_price": 100.0,
            "selling_price": 150.0,
            "expiration_date": None
        }
        return jsonify(mock_item)
    
# Get sales summary
@app.route('/sales/summary', methods=['GET'])
def get_sales_summary():
    time_filter = request.args.get('filter', 'monthly')  # daily, monthly, yearly
    
    # Calculate date ranges
    now = datetime.utcnow()
    max_days = 30
    max_months = 12
    max_years = 5

    match_stage = {}
    limit = 0

    if time_filter == 'daily':
        start_date = now - timedelta(days=max_days)
        match_stage = {"$match": {"transaction_date": {"$gte": start_date.isoformat()}}}
        limit = max_days
    elif time_filter == 'monthly':
        start_date = now - timedelta(days=30*max_months)
        match_stage = {"$match": {"transaction_date": {"$gte": start_date.isoformat()}}}
        limit = max_months
    elif time_filter == 'yearly':
        start_date = now - timedelta(days=365*max_years)
        match_stage = {"$match": {"transaction_date": {"$gte": start_date.isoformat()}}}
        limit = max_years

    pipeline = [
    {"$project": {
        "total_price": 1,
        "transaction_date": 1,
        "year": {"$year": {"$toDate": "$transaction_date"}},
        "month": {"$month": {"$toDate": "$transaction_date"}},
        "day": {"$dayOfMonth": {"$toDate": "$transaction_date"}}
    }},
    match_stage,
    {"$group": {
        "_id": {
            "daily": {"$dateToString": {"format": "%Y-%m-%d", "date": {"$toDate": "$transaction_date"}}},
            "monthly": {"$dateToString": {"format": "%Y-%m", "date": {"$toDate": "$transaction_date"}}},
            "yearly": {"$dateToString": {"format": "%Y", "date": {"$toDate": "$transaction_date"}}}
        }[time_filter],
        "total": {"$sum": "$total_price"},
        "count": {"$sum": 1}
    }},
    {"$sort": {"_id": 1}},
    {"$limit": limit} if limit > 0 else {"$match": {}}
]

    try:
        result = list(sales_collection.aggregate(pipeline))
        formatted_data = []
        total_sales = 0
        
        for item in result:
            formatted_data.append({
                "date": item["_id"],
                "total": item["total"]
            })
            total_sales += item["total"]

        return jsonify({
            "data": formatted_data,
            "total_sales": total_sales,
            "filter": time_filter
        })
    except Exception as e:
        print(f"Error generating sales summary: {e}")
        return jsonify({"error": "Failed to generate sales summary"}), 500

#----------------------Purchase Orders----------------------#
@app.route('/purchase-orders', methods =['GET'])
def get_purchase_orders():
    try:
        orders = list(purchase_orders_collection.find({}, {"_id":0}))
        return jsonify(orders)
    except Exception as e:
        print(f"Error fetching purchase orders:{e}")
        return jsonify({})
    
@app.route('/purchase-orders', methods=["POST"])
def create_purchase_order():
    data = request.get_json()
    
    # First check if SKU exists in inventory
    sku = data.get("SKU") or data.get("sku")
    inventory_item = inventory_collection.find_one({"SKU": sku.upper()})
    
    if not inventory_item:
        return jsonify({
            "error": "SKU does not exist in inventory",
            "message": "Please add the item to inventory first"
        }), 400

    # Proceed with PO creation if SKU exists
    reference_number = f"PO-{uuid.uuid4().hex[:8].upper()}"  # Corrected syntax

    
    new_order = {
        "reference_number": reference_number,  # Fixed typo in key name
        "name": data["name"],
        "SKU": sku,
        "vendor": data["vendor"],
        "quantity": int(data["quantity"]),
        "status": "pending",
    }
    
    purchase_orders_collection.insert_one(new_order)
    return jsonify({
        "message": "Purchase order created successfully",
        "order": {**new_order, "_id": str(new_order["_id"])}
    }), 201

@app.route('/purchase-orders/<reference_number>/approve', methods=['PUT'])
def approve_purchase_order(reference_number):
    order = purchase_orders_collection.find_one({"reference_number": reference_number})
    
    if not order:
        return jsonify({"error": "Purchase order not found"}), 404

    # Check if SKU exists
    item = inventory_collection.find_one({"SKU": order["SKU"]})
    
    if not item:
        return jsonify({"error": "Cannot approve PO for non-existent SKU"}), 400
    
    # Update existing item quantity
    new_quantity = item["quantity"] + order["quantity"]
    inventory_collection.update_one(
        {"SKU": order["SKU"]},
        {"$set": {"quantity": new_quantity}}
    )
    
    # Update PO status
    purchase_orders_collection.update_one(
        {"reference_number": reference_number},
        {"$set": {"status": "approved"}}
    )
    
    return jsonify({"message": "Purchase order approved"}), 200

@app.route('/purchase-orders/<string:reference_number>', methods=['DELETE'])
def delete_purchase_order(reference_number):
    try:
        # Attempt to delete the purchase order
        result = purchase_orders_collection.delete_one({"reference_number": reference_number})

        # Check if a document was actually deleted
        if result.deleted_count == 0:
            return jsonify({"error": "Purchase order not found"}), 404

        return jsonify({"message": "Purchase order deleted successfully"}), 200
    except Exception as e:
        print(f"Error deleting purchase order: {e}")
        return jsonify({"message": "Failed to delete purchase order"}), 500


#---------------------- Dashboard----------------------#
@app.route('/industry-health/top-sectors', methods=['GET'])
def get_top_industry_sectors():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        services_path = os.path.join(base_dir, "services_data.csv")
        manufacturing_path = os.path.join(base_dir, "manufacturing_data.csv")

        # Load data from CSV files
        df_services = pd.read_csv(services_path)
        df_manufacturing = pd.read_csv(manufacturing_path)

        # Get all quarter columns (excluding 'Category') in chronological order
        def get_quarter_columns(df):
            quarter_cols = [col for col in df.columns if col != 'Category']
            # Sort quarters chronologically (oldest first)
            quarter_cols.sort(key=lambda x: (
                int(x.split()[0]),  # Year
                int(x.split()[1][:-1])  # Quarter number (remove 'Q')
            ))
            return quarter_cols

        services_quarters = get_quarter_columns(df_services)
        manufacturing_quarters = get_quarter_columns(df_manufacturing)

        # Calculate cumulative averages for each sector
        def calculate_cumulative_averages(df, quarter_cols):
            results = []
            for _, row in df.iterrows():
                sector_data = {
                    "category": row["Category"],
                    "trend": []
                }
                cumulative_sum = 0
                
                for i, qtr in enumerate(quarter_cols):
                    try:
                        value = float(row[qtr])
                        cumulative_sum += value
                        cumulative_avg = cumulative_sum / (i + 1)
                        
                        year, quarter = qtr.split()
                        quarter_num = int(quarter[:-1])  # Remove 'Q' from '4Q'
                        
                        sector_data["trend"].append({
                            "year": int(year),
                            "quarter": quarter_num,
                            "value": cumulative_avg,
                            "quarter_label": qtr  # Original label like "2024 4Q"
                        })
                    except (ValueError, TypeError):
                        # Skip invalid/missing values
                        continue
                
                results.append(sector_data)
            return results

        # Calculate for both sectors
        services_data = calculate_cumulative_averages(df_services, services_quarters)
        manufacturing_data = calculate_cumulative_averages(df_manufacturing, manufacturing_quarters)
        #print(services_data, manufacturing_data)  # Debugging log

        # Sort sectors by their latest cumulative average value
        def get_latest_value(sector):
            return sector["trend"][-1]["value"] if sector["trend"] else 0

        top5_services = sorted(
            services_data,
            key=get_latest_value,
            reverse=True
        )[:5]
        #print("\nTop 5 Services (Debugging):")
        #pprint(top5_services)
        top5_manufacturing = sorted(
            manufacturing_data,
            key=get_latest_value,
            reverse=True
        )[:5]

        # Prepare quarter labels for frontend
        quarter_labels = [
            {
                "year": int(qtr.split()[0]),
                "quarter": int(qtr.split()[1][:-1]),
                "label": qtr
            }
            for qtr in services_quarters
        ]
        pprint(quarter_labels)

        return jsonify({
            "services": top5_services,
            "manufacturing": top5_manufacturing,
            "quarter_labels": quarter_labels,
            "message": "Data formatted with cumulative averages per quarter"
        })

    except Exception as e:
        print(f"Error in /industry-health/top-sectors: {e}")
        return jsonify({
            "error": "Failed to compute top sectors",
            "details": str(e)
        }), 500



#---------------------- Forecasting ----------------------#

# Get forecast for a specific item
@app.route('/forecast/<string:item_id>', methods=['GET'])
def get_forecast(item_id):
    """
    Get sales forecast for a specific item.
    
    Query parameters:
    - time_frame: 'week', 'month', or 'year' (default: 'week')
    """
    time_frame = request.args.get('time_frame', 'week')
    
    if time_frame not in ['week', 'month', 'year']:
        return jsonify({"error": "Invalid time frame. Use 'week', 'month', or 'year'"}), 400
    
    try:
        forecast_data = get_item_forecast(item_id, time_frame)
        return jsonify(forecast_data)
    except Exception as e:
        print(f"Error generating forecast: {e}")
        return jsonify({"error": f"Failed to generate forecast: {str(e)}"}), 500

# Refresh forecast data
@app.route('/refresh-forecast-data', methods=['POST'])
def refresh_forecast():
    """
    Refresh the forecast data by aggregating the latest sales data and updating item categories.
    """
    try:
        result = refresh_forecast_data()
        return jsonify(result)
    except Exception as e:
        print(f"Error refreshing forecast data: {e}")
        return jsonify({"status": "error", "message": f"Failed to refresh forecast data: {str(e)}"}), 500

@app.route('/forecast/total-sales', methods=['GET'])
def forecast_total_sales():
    """
    Get total sales forecast data.
    
    Query parameters:
    - time_frame: 'week', 'month', or 'year' (default: 'week')
    """
    time_frame = request.args.get('time_frame', 'week')
    
    if time_frame not in ['week', 'month', 'year']:
        return jsonify({"error": "Invalid time frame. Use 'week', 'month', or 'year'"}), 400
    
    try:
        data = get_total_sales(time_frame)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forecast/total-profits', methods=['GET'])
def forecast_total_profits():
    """
    Get total profits forecast data.
    
    Query parameters:
    - time_frame: 'week', 'month', or 'year' (default: 'week')
    """
    time_frame = request.args.get('time_frame', 'week')
    
    if time_frame not in ['week', 'month', 'year']:
        return jsonify({"error": "Invalid time frame. Use 'week', 'month', or 'year'"}), 400
    
    try:
        data = get_total_profits(time_frame)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forecast/top-products', methods=['GET'])
def forecast_top_products():
    try:
        data = get_top_products()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
