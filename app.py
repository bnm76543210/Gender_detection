from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Загрузка модели
model = YOLO('best.pt')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    results = model.predict(source=file_path, conf=0.5, save=False)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        gender = 'Male' if int(box.cls[0]) == 1 else 'Female'
        detections.append({
            "gender": gender,
            "confidence": conf,
            "box": [x1, y1, x2, y2]
        })

    os.remove(file_path)
    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
