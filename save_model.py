import joblib
from train_model import train_model  # Import the train_model function from your training script
import pandas as pd
import numpy as np
import os

# List of gesture labels
gesture_labels = ['siapa', 'bila', 'mana', 'apa', 'kenapa', 'bagaimana']

# Load all gesture data CSVs into one DataFrame
def load_data(gesture_labels):
    data = []
    for label in gesture_labels:
        csv_file = f'{label}_gesture_data.csv'
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            data.append(df)
        else:
            print(f"File not found: {csv_file}")
    
    # Combine all gesture data into a single DataFrame
    full_data = pd.concat(data, ignore_index=True)
    return full_data

# Preprocess the data and split it into features and labels
def preprocess_data(df):
    # Separate features and labels
    X = df.drop(columns=['label'])  # Drop the 'label' column to get features
    y = df['label']  # Get the 'label' column as the target
    
    # Convert to numpy arrays for training
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Save the trained model
def save_model(model, filename='gesture_classifier.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data(gesture_labels)
    X, y = preprocess_data(df)
    
    # Train and evaluate the model
    trained_model = train_model(X, y)
    
    # Save the trained model
    save_model(trained_model)
