from app import db

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # Primary key
    full_name = db.Column(db.String(100), nullable=False)  # Full name of the user
    handphone = db.Column(db.String(15), nullable=False)  # Handphone number
    email = db.Column(db.String(120), unique=True, nullable=False)  # Email address, must be unique
    username = db.Column(db.String(80), unique=True, nullable=False)  # Username, must be unique
    password = db.Column(db.String(60), nullable=False)  # Password
    access_level = db.Column(db.String(20), nullable=False)  # Access level (Manager or Employee)
    is_verified = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"  # String representation of the User object