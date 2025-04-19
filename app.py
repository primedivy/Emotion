from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

# Load emotion detection model
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Emotion labels and their YouTube links
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_responses = {
    'angry': 'https://www.youtube.com/results?search_query=calming+music',
    'disgust': 'https://www.youtube.com/results?search_query=uplifting+music',
    'fear': 'https://www.youtube.com/results?search_query=soothing+music',
    'happy': 'https://www.youtube.com/results?search_query=happy+songs',
    'neutral': 'https://www.youtube.com/results?search_query=instrumental+music',
    'sad': 'https://www.youtube.com/results?search_query=cheer+up+music',
    'surprise': 'https://www.youtube.com/results?search_query=surprise+songs'
}

# Global variable to hold current emotion
current_emotion = "neutral"

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocessing function
def extract_features(image):
    image = cv2.resize(image, (48, 48))
    image = np.reshape(image, (1, 48, 48, 1))
    image = image / 255.0
    return image

# Generator function for video frames
def generate_frames():
    global current_emotion
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            features = extract_features(face_img)
            pred = model.predict(features)
            prediction_label = labels[pred.argmax()]
            current_emotion = prediction_label  # update global emotion

            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x, y-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    return jsonify({'emotion': current_emotion, 'url': emotion_responses.get(current_emotion, '#')})

# Run the app
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use the port Render provides
    app.run(host='0.0.0.0', port=port)
