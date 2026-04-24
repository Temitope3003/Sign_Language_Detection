# Sign Language Detection

A real-time sign language to text translation system built to bridge communication gaps for the hearing-impaired community.

## Problem
Hearing-impaired individuals face significant communication barriers in everyday interactions. This system translates sign language gestures into readable text in real time, making communication more accessible.

## Approach
- Captured and processed hand gesture data using MediaPipe for landmark detection
- Used YOLO for object detection and gesture localisation
- Trained LSTM networks to recognise gesture sequences over time
- Deployed as a real-time translation prototype using OpenCV

## Results
- Successfully developed a working prototype capable of real-time sign language to text translation
- Won Best Performing Project at Robotics and AI Nigeria (RAIN)

## Tech Stack
- Python
- TensorFlow, Keras
- CNN, LSTM
- OpenCV
- MediaPipe
- YOLO
- Scikit-Learn
- JavaScript

## How to Run
```bash
git clone https://github.com/Temitope3003/Sign_Language_Detection
cd Sign_Language_Detection
pip install -r requirements.txt
python main.py
```

## Use Case
Practical accessibility technology for hearing-impaired individuals in everyday communication settings.
