import joblib
import numpy as np
from sklearn.svm import SVC  # Import the SVM classifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Replace with your gesture dataset

# Load or generate your dataset
# Here we use Iris dataset as an example. Replace with your gesture dataset.
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = SVC()  # Replace with your model
model.fit(X_train, y_train)

# Save the trained model
def save_model(model, filename='gesture_classifier.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

save_model(model)
