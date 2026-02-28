from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load OpenCV Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    profile_text = ""
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Read image
            img = cv2.imread(filepath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                profile_text = "No face detected!"
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Detect eyes inside face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

                # Detect mouth inside face
                mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 20)
                for (mx, my, mw, mh) in mouth:
                    cv2.rectangle(roi_color, (mx, my+h//2), (mx+mw, my+mh+h//2), (0, 0, 255), 2)
                    break  # Only first mouth

                # Simple measurements
                if len(eyes) >= 2:
                    eye_distance = np.linalg.norm(np.array(eyes[0][:2]) - np.array(eyes[1][:2]))
                else:
                    eye_distance = 0
                profile_text = f"Eye Distance: {int(eye_distance)}. "
                if eye_distance > 40:
                    profile_text += "Personality: Outgoing"
                else:
                    profile_text += "Personality: Calm"

            # Save result
            result_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)
            cv2.imwrite(result_path, img)
            image_path = result_path

    return render_template("index.html", image_path=image_path, profile_text=profile_text)

if __name__ == "__main__":
    app.run(debug=True)