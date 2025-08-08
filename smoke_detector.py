from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import time
import numpy as np

# Initialize Flask application and enable CORS for cross-origin requests
app = Flask(__name__)
CORS(app)

# Load pre-trained YOLOv8 model for smoke detection
model = YOLO("runs/detect/train6/weights/best.pt")

# Initialize video capture from default camera (index 0)
cap = cv2.VideoCapture(0)

# Global variables for tracking detection state
detect_count = 0           # Total number of smoking detections recorded
last_sent = 0             # Timestamp of last detection notification sent
frame_result = None       # Latest processed frame with annotations
last_confidence = 0.0     # Confidence score of most recent detection
last_detected_time = 0    # Timestamp when smoking was last detected


def gen_frames():
    """
    Generate video frames with real-time smoke detection annotations.
    Continuously captures frames, processes through YOLO model, and yields encoded frames.
    """
    global detect_count, last_sent, frame_result, last_confidence, last_detected_time

    while True:
        # Capture frame from camera
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference with confidence threshold of 0.25
        results = model(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes

        # Create copy of frame for drawing annotations
        frame_result = frame.copy()
        confidence = 0.0

        # Process each detected object
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results[0].names[cls_id]

            # Skip detections below confidence threshold
            if conf < 0.25:
                continue

            # Extract bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Handle smoking detection with cooldown mechanism
            if label == 'smoking':
                confidence = conf
                last_detected_time = time.time()  # Update last detection timestamp
                
                # Prevent spam by enforcing 5-second cooldown between notifications
                if time.time() - last_sent > 5:
                    detect_count += 1
                    last_sent = time.time()
                    print(f"🚬 Smoking detected with confidence {conf:.2f}")

            # Draw bounding box around detected object
            cv2.rectangle(frame_result, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 0, 255), 2)
            
            # Add label with confidence score above bounding box
            label_pos = (xyxy[0], max(0, xyxy[1] - 10))
            cv2.putText(frame_result, f"{label} {conf:.2f}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Update global confidence with latest detection
        last_confidence = confidence

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame_result)
        frame = buffer.tobytes()

        # Small delay to prevent excessive CPU usage
        time.sleep(0.01)

        # Yield frame in MJPEG format for HTTP streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    """Video streaming endpoint that returns MJPEG stream with detection overlay."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """
    API endpoint to get current detection status and statistics.
    Returns JSON with detection count, current status, and confidence score.
    """
    global last_detected_time
    current_time = time.time()
    
    # Check if smoking was detected within the last 3 seconds
    is_recently_detected = (current_time - last_detected_time) < 3

    return jsonify({
        'detections_today': detect_count,
        'detected': is_recently_detected,
        'confidence': round(last_confidence, 2)
    })


if __name__ == '__main__':
    # Start Flask development server on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000)
