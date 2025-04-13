#!/usr/bin/env python3
"""
Category Classifier Training Script

This script trains a classifier to categorize items based on their names.
It uses a sentence transformer to convert item names into embeddings,
then trains an SVM classifier on those embeddings.

Usage:
    python train_category_classifier.py [--data_file FILENAME] [--test_size FLOAT] [--random_state INT]

Arguments:
    --data_file: Path to the CSV file containing item names and categories (default: products.csv)
    --test_size: Proportion of data to use for testing (default: 0.2)
    --random_state: Random seed for reproducibility (default: 42)
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a category classifier for items.')
    parser.add_argument('--data_file', type=str, default='products.csv',
                        help='Path to the CSV file containing item names and categories')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def load_data(data_file):
    """Load and preprocess the data."""
    logger.info(f"Loading data from {data_file}")
    try:
        df = pd.read_csv(data_file)
        # Check if required columns exist
        if 'Item Name' not in df.columns or 'Category' not in df.columns:
            logger.error(f"Required columns 'Item Name' and 'Category' not found in {data_file}")
            raise ValueError(f"Required columns 'Item Name' and 'Category' not found in {data_file}")
        
        # Extract features and labels
        X = df["Item Name"].tolist()  # Item names
        y = df["Category"].tolist()  # Corresponding categories
        
        logger.info(f"Loaded {len(X)} items with {len(set(y))} unique categories")
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_embeddings(X, model_name="all-MiniLM-L6-v2"):
    """Convert item names into embeddings using a sentence transformer."""
    logger.info(f"Creating embeddings using {model_name}")
    try:
        model = SentenceTransformer(model_name)
        X_embeddings = model.encode(X)
        logger.info(f"Created embeddings with shape {X_embeddings.shape}")
        return X_embeddings, model
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise

def train_classifier(X_train, y_train, X_test, y_test):
    """Train an SVM classifier and evaluate its performance."""
    logger.info("Training SVM classifier")
    try:
        # Train the classifier
        clf = SVC(kernel="linear", probability=True)
        clf.fit(X_train, y_train)
        
        # Evaluate on test set
        accuracy = clf.score(X_test, y_test)
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f}")
        
        # Detailed classification report
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification report:\n{report}")
        
        return clf, y_pred
    except Exception as e:
        logger.error(f"Error training classifier: {e}")
        raise

def plot_confusion_matrix(y_test, y_pred, categories):
    """Plot a confusion matrix to visualize classification results."""
    logger.info("Plotting confusion matrix")
    try:
        cm = confusion_matrix(y_test, y_pred, labels=categories)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
        
        plt.figure(figsize=(12, 10))
        disp.plot(xticks_rotation=45)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('confusion_matrix.png')
        logger.info("Confusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        logger.warning("Continuing without confusion matrix plot")

def save_models(clf, embedding_model):
    """Save the trained classifier and embedding model."""
    logger.info("Saving models")
    try:
        joblib.dump(clf, "category_classifier.pkl")
        joblib.dump(embedding_model, "embedding_model.pkl")
        logger.info("Models saved as 'category_classifier.pkl' and 'embedding_model.pkl'")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise

def test_classifier(clf, embedding_model, test_items):
    """Test the classifier on some example items."""
    logger.info("Testing classifier on example items")
    try:
        for item in test_items:
            item_embedding = embedding_model.encode([item])
            category = clf.predict(item_embedding)[0]
            probabilities = clf.predict_proba(item_embedding)[0]
            max_prob = max(probabilities)
            logger.info(f"Item: {item} → Category: {category} (confidence: {max_prob:.4f})")
    except Exception as e:
        logger.error(f"Error testing classifier: {e}")
        logger.warning("Continuing without test examples")

def main():
    """Main function to train and evaluate the classifier."""
    args = parse_arguments()
    
    # Load data
    X, y = load_data(args.data_file)
    
    # Get unique categories
    categories = sorted(list(set(y)))
    
    # Create embeddings
    X_embeddings, embedding_model = create_embeddings(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    logger.info(f"Split data into {len(X_train)} training and {len(X_test)} testing samples")
    
    # Train and evaluate classifier
    clf, y_pred = train_classifier(X_train, y_train, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, categories)
    
    # Save models
    save_models(clf, embedding_model)
    
    # Test on some examples
    test_items = [
        "Gaming Laptop",
        "Wireless Earbuds",
        "Winter Coat",
        "Multivitamin Supplement",
        "Electric Drill",
        "Organic Apples",
        "Facial Cleanser",
        "Coffee Table",
        "Running Shoes",
        "Remote Control Helicopter",
        "Engine Oil",
        "Ballpoint Pens",
        "Cat Food",
        "Travel Mug"
    ]
    test_classifier(clf, embedding_model, test_items)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
