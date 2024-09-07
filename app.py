import streamlit as st
import numpy as np
import cv2
import joblib

# Load the trained model
model = joblib.load('gesture_classifier.pkl')

def preprocess_frame(frame):
    """Preprocess the frame for the model."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Adjust size as needed
    normalized = resized / 255.0  # Normalize if required
    return normalized

def main():
    st.title("Sign Language Recognition")

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        
        # Ensure preprocessed_frame is in the correct shape for the model
        reshaped_frame = np.reshape(preprocessed_frame, (1, -1))
        
        # Predict using the model
        try:
            prediction = model.predict(reshaped_frame)
            st.write(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.write(f"Error during prediction: {e}")

        # Display the frame
        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
