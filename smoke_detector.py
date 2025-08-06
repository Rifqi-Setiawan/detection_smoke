from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import time
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model YOLOv8
model = YOLO("runs/detect/train6/weights/best.pt")

# Buka kamera
cap = cv2.VideoCapture(0)

# Variabel global
detect_count = 0
last_sent = 0
frame_result = None
last_confidence = 0.0
last_detected_time = 0  # ⬅️ waktu terakhir terdeteksi smoking


def gen_frames():
    global detect_count, last_sent, frame_result, last_confidence, last_detected_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Tetap gunakan resolusi asli
        results = model(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes

        frame_result = frame.copy()
        confidence = 0.0

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results[0].names[cls_id]

            if conf < 0.25:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if label == 'smoking':
                confidence = conf
                last_detected_time = time.time()  # ⬅️ simpan waktu deteksi terakhir
                if time.time() - last_sent > 5:
                    detect_count += 1
                    last_sent = time.time()
                    print(f"🚬 Smoking detected with confidence {conf:.2f}")

            # Gambar bounding box
            cv2.rectangle(frame_result, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 0, 255), 2)
            label_pos = (xyxy[0], max(0, xyxy[1] - 10))
            cv2.putText(frame_result, f"{label} {conf:.2f}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        last_confidence = confidence

        ret, buffer = cv2.imencode('.jpg', frame_result)
        frame = buffer.tobytes()

        time.sleep(0.01)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    global last_detected_time
    current_time = time.time()
    is_recently_detected = (current_time - last_detected_time) < 3  # ⬅️ tetap "detected" selama 3 detik

    return jsonify({
        'detections_today': detect_count,
        'detected': is_recently_detected,
        'confidence': round(last_confidence, 2)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
