from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Загрузка модели
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Make sure it is included in your repository.")

model = YOLO(model_path)

@app.route("/")
def home():
    return jsonify({"message": "Face Detection API is live! Use /process to send POST requests with images."}), 200

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    # Предсказания YOLO
    try:
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

        return jsonify({"detections": detections}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
