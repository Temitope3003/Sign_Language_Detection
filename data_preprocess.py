import os
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_path = r"C:\Users\PC\Desktop\RAIN\Semester 3\Image_classification\alphabets_2\alphabets\A_Z\data"

data = []
labels = []

try:
    for dir_ in os.listdir(data_path):
        for img_path in os.listdir(os.path.join(data_path, dir_)):
            data_aux = []
            img = cv2.imread(os.path.join(data_path, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        data_aux.append(x)
                        data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_)

    hands.close()

    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
        print("Pickle file saved successfully.")

except Exception as e:
    print(f"Error: {e}")


  