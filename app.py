import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('gesture_classifier.pkl')
    return model

model = load_model()

# Set up the Streamlit app layout
st.title('Sign Language Recognition')

# Initialize a video capture object
cap = cv2.VideoCapture(0)

st.write("Press 'Space' to capture a frame.")

# Create a unique key for the video capture start button
if st.button('Start Capture', key='start_capture'):
    st.write("Capture started. Press 'Space' to capture a frame.")

    # Display a frame to show video feed
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break
        
        # Display the video feed
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Convert frame to grayscale and preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray, (64, 64))
        reshaped_frame = resized_frame.reshape(1, -1)  # Flatten the image

        # Use a unique key for the prediction button
        if st.button('Predict', key='predict_button'):
            try:
                prediction = model.predict(reshaped_frame)
                st.write(f"Prediction: {prediction[0]}")
            except Exception as e:
                st.write(f"Error: {e}")

        # Stop capturing when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
