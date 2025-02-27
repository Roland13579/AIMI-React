from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS #Communication between frontend and backend

from flask_mail import Mail, Message

# Initialize the Flask application
app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Replace with your email provider's SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sc2006t3@gmail.com'  
app.config['MAIL_PASSWORD'] = 'nozm ubpa egrd xujv'  
app.config['MAIL_DEFAULT_SENDER'] = 'sc2006t3@gmail.com'  

mail = Mail(app)
CORS(app)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Users.db'  # SQLite database file will be created in the project directory
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking to avoid overhead

# Initialize the database
db = SQLAlchemy(app)

# Initialize Flask-Migrate for database migrations
migrate = Migrate(app, db)

# Import routes after initializing the app and db to avoid circular imports
from routes import *