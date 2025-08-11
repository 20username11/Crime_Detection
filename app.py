from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the trained model
model = load_model('crime_detection_model.h5')

# Define labels
labels = ['Robbery','Abuse','Shoplifting','NormalVedios','Robbery', 'Burglary', 'Arson', 'Assault','Stealing', 'Arrest', 'Explosion',
         'Shooting', 'Vandalism','Fighting']

# Hardcoded path to the video file
VIDEO_PATH = os.path.join('static', 'AdobeStock_984294293_Video_HD_Preview.mp4') # Update to your actual video path

def preprocess_video(video_path, resize=(64, 64)):
    print("Attempting to read video from:", video_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return None
    frame = cv2.resize(frame, resize)
    cap.release()
    return np.expand_dims(frame / 255.0, axis=0)

# Prediction function
def predict_crime(video_path):
    preprocessed_video = preprocess_video(video_path)
    if preprocessed_video is None:
        return None
    predictions = model.predict(preprocessed_video)
    predicted_class = np.argmax(predictions, axis=1)[0]
    if predicted_class < len(labels):
        return labels[predicted_class]
    else:
        return None

@app.route('/dashboard')
def dashboard():
    detected_class = predict_crime(VIDEO_PATH)
    if detected_class and detected_class in labels:
        message = f"{detected_class} detected!"
        return render_template('admin_dashboard.html', video=os.path.basename(VIDEO_PATH), message=message)
    else:
        return render_template('admin_dashboard.html', message="No suspicious activity detected.")

@app.route('/send_to_police', methods=['POST'])
def send_to_police():
    return jsonify({"message": "Video sent to authorities successfully!"})

@app.route('/test_prediction')
def test_prediction():
    result = predict_crime(VIDEO_PATH)
    return jsonify({"Prediction": result}) 

if __name__ == "__main__":
    app.run(debug=True,port=5001)
