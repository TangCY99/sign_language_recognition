import joblib
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model = joblib.load('gesture_classifier.pkl')

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam feed for real-time gesture recognition
def real_time_gesture_recognition():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally
        frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame with MediaPipe hands
        results = hands.process(image_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract hand landmarks
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])

                if len(hand_data) == 63:  # Ensure 63 data points for 21 landmarks
                    hand_data = np.array(hand_data).reshape(1, -1)
                    prediction = model.predict(hand_data)

                    # Display the predicted gesture label
                    cv2.putText(frame, f'Gesture: {prediction[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Real-Time Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_gesture_recognition()
