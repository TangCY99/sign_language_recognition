import streamlit as st
import numpy as np
import cv2
import joblib  # Used for loading the model

def main():
    st.title("Sign Language Recognition")

    # Load your trained model
    model = joblib.load('gesture_classifier.pkl')

    # Create a video capture object
    cap = cv2.VideoCapture(0)

    st.write("Click the button to start prediction")

    if st.button('Start Video', key='start_video'):
        st.write("Video is running...")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image")
                break

            # Preprocess the frame
            resized_frame = cv2.resize(frame, (64, 64))
            reshaped_frame = resized_frame.reshape(1, 64, 64, 3)

            # Make prediction
            prediction = model.predict(reshaped_frame)
            predicted_class = np.argmax(prediction, axis=1)

            # Display the prediction
            st.write(f"Predicted class: {predicted_class}")

            # Display the video frame
            st.image(frame, channels="BGR")

    if st.button('Stop Video', key='stop_video'):
        st.write("Stopping video...")
        cap.release()

if __name__ == "__main__":
    main()
