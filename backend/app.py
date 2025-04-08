from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS  # Communication between frontend and backend
from pymongo import MongoClient
from flask_mail import Mail

import os  # ✅ Import os for environment variables

# Initialize the Flask application
app = Flask(__name__)

# ✅ Secure Email Credentials using Environment Variables (IMPORTANT)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'sc2006t3@gmail.com')  
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'liye innd pwux jntl')  
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']

mail = Mail(app)

# ✅ Enable CORS (Frontend & Backend Communication)
CORS(app)

# ✅ MongoDB Connection
MONGO_URI = "mongodb+srv://aimi_admin:SC2006t3@cluster0.frqdlsi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # MongoDB Atlas connection
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["aimi_inventory"]  # Database name
print(mongo_db.list_collection_names())  # List all collections in the database
inventory_collection = mongo_db["inventory"]  # Collection for inventory items
sales_collection = mongo_db["sales"]  # Collection for sales transactions
purchase_orders_collection = mongo_db["purchase_orders"]  # Collection for purchase orders

# ✅ Configure the SQLite database (For User Authentication)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  

# ✅ Initialize the SQLAlchemy Database (Renamed to `db`)
db = SQLAlchemy(app)

# ✅ Initialize Flask-Migrate for SQL Migrations
migrate = Migrate(app, db)

# ✅ Import routes after initializing the app to avoid circular imports
from routes import *
