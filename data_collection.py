import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define labels for gestures
labels = ['siapa', 'bila', 'mana', 'apa', 'kenapa', 'bagaimana']

# Function to capture data
def capture_data(label):
    cap = cv2.VideoCapture(0)
    data = []

    print(f"Recording data for {label}. Press 'q' to stop.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a later selfie-view display
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

                # Ensure the data has the correct length (21 landmarks * 3 coordinates)
                if len(hand_data) == 63:  # 21 landmarks for each hand, 3 coordinates (x, y, z) per landmark
                    hand_data.append(label)  # Add the label at the end
                    data.append(hand_data)

        cv2.imshow(f"Recording {label}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert data to DataFrame with proper column names
    columns = []
    for i in range(21):  # For each of the 21 landmarks
        columns.extend([f'x{i}', f'y{i}', f'z{i}'])

    # Add 'label' column at the end
    columns.append('label')

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'{label}_gesture_data.csv', index=False)
    print(f"Data saved for {label}.")

if __name__ == '__main__':
    for label in labels:
        capture_data(label)
