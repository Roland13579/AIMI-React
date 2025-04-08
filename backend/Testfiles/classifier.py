'''
Item classifier

Takes in the name + description of an item and assign it a category (industry)

so an item like "STMr34 Chip : A chip designed by STM" would be assigned into the semiconductors industry category

Right now im using a test products.csv file for testing. It includes two columns, item name and its correct category for training.

The Categories are : 

Use this info to gauge what this component does. We need to get the categories for each item and have as a new column in the original sales by week dataframe we made in weekly_sales_aggregator.py
'''

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Load dataset (expects a CSV with 'Item Name' and 'Category' columns)
df = pd.read_csv("~/Desktop/SoftEngg/AIMI-React-draft1_sales/products.csv")

# Extract features and labels
X_train = df["Item Name"].tolist()  # Item names
y_train = df["Category"].tolist()  # Corresponding categories

# Load a pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert item names into embeddings
X_train_embeddings = model.encode(X_train)

# Split data (80% train, 20% test)
X_train_emb, X_test_emb, y_train, y_test = train_test_split(
    X_train_embeddings, y_train, test_size=0.2, random_state=42
)

# Train an SVM classifier
clf = SVC(kernel="linear")
clf.fit(X_train_emb, y_train)

# Evaluate accuracy
accuracy = clf.score(X_test_emb, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(clf, "category_classifier.pkl")
joblib.dump(model, "embedding_model.pkl")

# Function to predict categories for new items1]
def predict_category(item_name):
    item_embedding = model.encode([item_name])  # Convert to embedding
    category = clf.predict(item_embedding)[0]  # Predict category
    return category

# Test example
new_item = "Deoderant"
print(f"{new_item} → {predict_category(new_item)}")