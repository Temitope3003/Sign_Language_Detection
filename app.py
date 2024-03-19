from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)

model_dict = pickle.load(open('model_and_scaler.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

# Global variable to store predicted text
predicted_text = ""

@app.route('/')
def index():
    return render_template('index.html', predicted_text=predicted_text)

def generate_frames():
    global predicted_text  # Access the global variable

    cap = cv2.VideoCapture(0)
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            max_length = 84
            data_padded = data_aux + [0] * (max_length - len(data_aux))
            data_scaled = scaler.transform([data_padded])

            try:
                prediction = model.predict(data_scaled)[0]
                predicted_text = str(prediction)
            except Exception as e:
                print(f"Error in prediction: {e}")
                predicted_text = ""

            # Display the predicted result as text
            cv2.putText(frame, predicted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            # Clear predicted text when no hands are detected
            predicted_text = ""

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_predicted_text')
def get_predicted_text():
    global predicted_text
    print("predicted_text = " + predicted_text)
    return jsonify({'predicted_text': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)

