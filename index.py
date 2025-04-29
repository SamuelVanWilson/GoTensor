from flask import Flask, render_template, Response, request, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import io

app = Flask(__name__, static_folder="asset")
# ——— Load Models ————————————————————————————
# Emotion Detector model
emotion_model = load_model('emotion_detector_model.keras', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Style Transfer model
style_model = load_model('style_transfer_model.keras', compile=False)

# ——— Helper: Style image loader ——————————————————
def load_and_prep(img_file, size=(256,256)):
    img = Image.open(img_file).convert('RGB')
    img = img.resize(size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0).astype(np.float32)

@app.route('/')
def main():
    return render_template('home.html')

# ——— Project 1: Emotion Detector ——————————————————
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_model.predict(roi)[0]
            label = emotion_labels[np.argmax(preds)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/emotion')
def emotion():
    return render_template('emotion.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ——— Project 2: Style Transfer ——————————————————
@app.route('/style_transfer', methods=['GET','POST'])
def style_transfer():
    result_filename = None

    if request.method == 'POST':
        content_file = request.files['content_image']
        style_file   = request.files['style_image']

        # preprocess
        content_arr = load_and_prep(content_file)
        style_arr   = load_and_prep(style_file)

        # inference
        output = style_model.predict([content_arr, style_arr])[0]
        output = (output * 255).astype(np.uint8)

        # save result
        os.makedirs(app.static_folder, exist_ok=True)
        result_filename = f"stylized_{content_file.filename}"
        output_path = os.path.join(app.static_folder, result_filename)
        Image.fromarray(output).save(output_path)

    return render_template('style_transfer.html',
                           result=result_filename)


if __name__ == "__main__":
    app.run(debug=True)