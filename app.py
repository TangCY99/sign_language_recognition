import streamlit as st
import cv2
import joblib
import numpy as np
import mediapipe as mp

# Load the trained model
model = joblib.load('gesture_classifier.pkl')

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def recognize_gesture(frame):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe hands
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.extend([landmark.x, landmark.y, landmark.z])
            if len(hand_data) == 63:
                hand_data = np.array(hand_data).reshape(1, -1)
                prediction = model.predict(hand_data)
                return prediction[0]
    return "Unknown"

st.title('Sign Language Recognition')
st.write("Use the webcam to recognize gestures.")

# Add a placeholder for webcam input
run = st.button('Start Webcam')
if run:
    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gesture = recognize_gesture(frame)
        st.image(frame, channels="BGR", use_column_width=True)
        st.write(f'Gesture: {gesture}')
        if st.button('Stop Webcam'):
            break
    cap.release()
