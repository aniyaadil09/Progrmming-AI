from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded."})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return jsonify({"error": "No face detected!"})

    profile_text = ""
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 20)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my+h//2), (mx+mw, my+mh+h//2), (0, 0, 255), 2)
            break  # only first mouth

        # Simple personality / emotion logic
        if len(eyes) >= 2:
            eye_distance = np.linalg.norm(np.array(eyes[0][:2]) - np.array(eyes[1][:2]))
        else:
            eye_distance = 0

        personality = "Outgoing" if eye_distance > 40 else "Calm"
        emotion = "Happy" if len(mouth) > 0 else "Neutral"

        profile_text = f"Personality: {personality} | Emotion: {emotion}"

    result_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)
    cv2.imwrite(result_path, img)

    return jsonify({"image": result_path, "profile": profile_text})

if __name__ == "__main__":
    app.run(debug=True)