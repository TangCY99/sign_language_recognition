import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

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

# Train and evaluate the model
def train_model(X, y):
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return clf

if __name__ == "__main__":
    # List of gesture labels
    gesture_labels = ['siapa', 'bila', 'mana', 'apa', 'kenapa', 'bagaimana']
    
    # Load and preprocess the data
    df = load_data(gesture_labels)
    X, y = preprocess_data(df)
    
    # Train and evaluate the model
    trained_model = train_model(X, y)
