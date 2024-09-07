import streamlit as st
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the pre-trained model
model = joblib.load('gesture_classifier.pkl')

def recognize_gesture(frame):
    # Pre-process the frame and predict the gesture
    # This is a placeholder; adjust based on your model's requirements
    # For example, convert the frame to grayscale, resize, etc.
    # Example dummy implementation:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Extract features from the frame
    features = extract_features(gray)
    # Predict using the model
    prediction = model.predict([features])
    return prediction[0]

def extract_features(image):
    # Implement feature extraction based on your model
    return np.random.rand(100)  # Placeholder: Replace with actual feature extraction

def main():
    st.title("Sign Language Recognition")

    st.write("Upload an image or video to recognize gestures.")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        # If the file is an image
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            image = np.array(bytearray(uploaded_file.read()))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            # Convert image to OpenCV format
            nparr = np.fromstring(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gesture = recognize_gesture(img)
            st.write(f"Recognized Gesture: {gesture}")

        # If the file is a video
        elif uploaded_file.type == "video/mp4":
            st.video(uploaded_file)

            # Process video
            cap = cv2.VideoCapture(uploaded_file)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                gesture = recognize_gesture(frame)
                st.write(f"Recognized Gesture: {gesture}")

            cap.release()

    # Buttons with unique keys
    if st.button("Recognize Siapa", key="siapa_button"):
        st.write("Button Siapa clicked")

    if st.button("Recognize Bila", key="bila_button"):
        st.write("Button Bila clicked")

    if st.button("Recognize Mana", key="mana_button"):
        st.write("Button Mana clicked")

    if st.button("Recognize Apa", key="apa_button"):
        st.write("Button Apa clicked")

    if st.button("Recognize Kenapa", key="kenapa_button"):
        st.write("Button Kenapa clicked")

    if st.button("Recognize Bagaimana", key="bagaimana_button"):
        st.write("Button Bagaimana clicked")

if __name__ == "__main__":
    main()
