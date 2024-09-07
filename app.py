import streamlit as st
import cv2
import numpy as np
import pickle

# Load the trained model
model_path = "gesture_classifier.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define labels for gestures
labels = {
    0: "siapa",
    1: "bila",
    2: "mana",
    3: "apa",
    4: "kenapa",
    5: "bagaimana"
}

# Webcam input
def main():
    st.title("Sign Language Recognition")

    # Open the webcam
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Cannot access the camera.")
                break
            
            # Preprocess the frame for the model
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (64, 64)).flatten()
            reshaped_frame = np.reshape(resized_frame, (1, -1))

            # Predict gesture
            prediction = model.predict(reshaped_frame)
            predicted_gesture = labels[int(prediction[0])]

            # Display the frame and prediction
            FRAME_WINDOW.image(frame)
            st.write(f"Predicted Gesture: {predicted_gesture}")

        cap.release()

if __name__ == "__main__":
    main()
