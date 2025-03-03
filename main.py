from flask import Flask, request, jsonify, send_file, render_template_string, Response
import cv2
import numpy as np
from pyngrok import ngrok
from io import BytesIO
from ultralytics import YOLO
import os
import threading
from queue import Queue
import json
from flask_sock import Sock
from sklearn.cluster import AgglomerativeClustering
import base64

app = Flask(__name__)

sock = Sock(app)

# # Setup ngrok tunnel
# ngrok.set_auth_token("2sesgJLFXrYNkfute01Q5xT38uk_6QUSnwWtT5tfsuRSUA8zn")
# tunnel = ngrok.connect(5000)
# print(f"Public URL: {tunnel.public_url}")

# Load YOLO model
model = YOLO("trained_models/rug_knot_classifier_v4.pt")

# Create temp directory if it doesn't exist
os.makedirs('/tmp', exist_ok=True)

# Global variables for streaming
stream_active = False
frame_queue = Queue(maxsize=1)
processed_frame_queue = Queue(maxsize=1)

# Index page template
index_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rug Knot Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }

        .container {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        h2 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .upload-form {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .file-input-container {
            position: relative;
            width: 100%;
            height: 200px;
            border: 2px dashed #ccc;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: border-color 0.3s;
        }

        .file-input-container:hover {
            border-color: #2c3e50;
        }

        .file-input-container input[type="file"] {
            position: absolute;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }

        .upload-icon {
            font-size: 48px;
            color: #ccc;
            margin-bottom: 10px;
        }

        .upload-text {
            text-align: center;
            color: #666;
        }

        button {
            background-color: #2c3e50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #34495e;
        }

        .result-container {
            margin-top: 2rem;
        }

        .result-image {
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        #selected-file-name {
            margin-top: 10px;
            color: #666;
        }

        .loading {
            display: none;
            text-align: center;
            margin-top: 1rem;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #2c3e50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Rug Knot Analysis</h2>
        <form class="upload-form" action="/predict" method="POST" enctype="multipart/form-data" id="upload-form">
            <div class="file-input-container">
                <input type="file" name="file" accept="image/*" required id="file-input">
                <div class="upload-text">
                    <div class="upload-icon">📁</div>
                    <p>Drag and drop your image here<br>or click to browse</p>
                </div>
            </div>
            <div id="selected-file-name"></div>
            <div style="margin: 10px 0;">
                <label>
                    <input type="checkbox" name="show_boxes" value="true">
                    Show detection boxes
                </label>
            </div>
            <button type="submit">Analyze Image</button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing image...</p>
        </div>

        {% if image_url %}
        <div class="result-container">
            <h3>Analysis Result</h3>
            <img src="{{ image_url }}" alt="Processed Image" class="result-image">
        </div>
        {% endif %}
    </div>

    <script>
        const fileInput = document.getElementById('file-input');
        const fileNameDisplay = document.getElementById('selected-file-name');
        const uploadForm = document.getElementById('upload-form');
        const loading = document.getElementById('loading');

        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                fileNameDisplay.textContent = `Selected file: ${this.files[0].name}`;
            }
        });

        uploadForm.addEventListener('submit', function() {
            loading.style.display = 'block';
        });

        // Drag and drop functionality
        const dropZone = document.querySelector('.file-input-container');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            dropZone.style.borderColor = '#2c3e50';
        }

        function unhighlight(e) {
            dropZone.style.borderColor = '#ccc';
        }

        dropZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            if (files && files[0]) {
                fileNameDisplay.textContent = `Selected file: ${files[0].name}`;
            }
        }
    </script>
</body>
</html>
"""

stream_page_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Rug Knot Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .video-container {
            width: 100%;
            margin: 20px 0;
            display: flex;
            justify-content: center;
        }
        #videoFeed {
            width: 100%;
            max-width: 320px;
            height: auto;
            border-radius: 8px;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
        }
        button {
            background-color: #2c3e50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #34495e;
        }
        button:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        #measurements {
            margin-top: 20px;
            text-align: center;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Live Rug Knot Detection</h2>
        <div class="video-container">
            <canvas id="videoFeed" width="320" height="320"></canvas>
            <canvas id="hiddenCanvas" width="320" height="320" style="display:none;"></canvas>
            <video id="clientVideo" style="display:none;" autoplay playsinline></video>
        </div>
        <div class="controls">
            <button id="startBtn" onclick="startStream()">Start Stream</button>
            <button id="stopBtn" onclick="stopStream()" disabled>Stop Stream</button>
            <button id="captureBtn" onclick="captureFrame()" disabled>Measure</button>
        </div>
        <div id="measurements"></div>
    </div>

    <script>
        const clientVideo = document.getElementById('clientVideo');
        const videoFeed = document.getElementById('videoFeed');
        const hiddenCanvas = document.getElementById('hiddenCanvas');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const captureBtn = document.getElementById('captureBtn');
        const measurementsDiv = document.getElementById('measurements');
        let ws;
        let streamInterval;

        async function startStream() {
            try {
                // Get camera stream with back camera constraint
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: {
                        facingMode: { exact: "environment" }, // Use back camera
                        width: { ideal: 320 },
                        height: { ideal: 320 }
                    }
                });
                clientVideo.srcObject = stream;

                // Setup WebSocket connection
                ws = new WebSocket('wss://deed-2-50-137-17.ngrok-free.app/ws');
                
                ws.onopen = () => {
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    captureBtn.disabled = false;
                    streamInterval = setInterval(sendFrame, 2000); // Send frame every 2s
                };

                ws.onmessage = (event) => {
                    const data = event.data;
                    if (typeof data === 'string' && data.startsWith('{')) {
                        // Handle measurement data
                        const measurements = JSON.parse(data);
                        measurementsDiv.innerHTML = `
                            Rows: ${measurements.rows}<br>
                            Columns: ${measurements.cols}<br>
                            Total Knots: ${measurements.total}
                        `;
                    } else {
                        // Handle image data
                        const img = new Image();
                        img.onload = () => {
                            const ctx = videoFeed.getContext('2d');
                            ctx.drawImage(img, 0, 0, 320, 320);
                        };
                        img.src = URL.createObjectURL(event.data);
                    }
                };

                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    stopStream();
                };

                ws.onclose = () => {
                    console.log('WebSocket connection closed');
                    stopStream();
                };

            } catch (error) {
                console.error('Error starting stream:', error);
                // If back camera fails, try without the exact constraint
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: "environment", // Try back camera but allow fallback
                            width: { ideal: 320 },
                            height: { ideal: 320 }
                        }
                    });
                    clientVideo.srcObject = stream;
                } catch (fallbackError) {
                    console.error('Fallback camera error:', fallbackError);
                    stopStream();
                }
            }
        }

        function sendFrame() {
            if (ws && ws.readyState === WebSocket.OPEN && clientVideo.videoWidth > 0) {
                const ctx = hiddenCanvas.getContext('2d');
                // Draw and resize the video frame to 320x320
                ctx.drawImage(clientVideo, 0, 0, 320, 320);
                
                hiddenCanvas.toBlob((blob) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(blob);
                    }
                }, 'image/jpeg', 0.8);
            }
        }

        function captureFrame() {
            if (ws && ws.readyState === WebSocket.OPEN && clientVideo.videoWidth > 0) {
                const ctx = hiddenCanvas.getContext('2d');
                ctx.drawImage(clientVideo, 0, 0, 320, 320);
                
                hiddenCanvas.toBlob((blob) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send('MEASURE');  // Signal server to measure this frame
                        ws.send(blob);
                    }
                }, 'image/jpeg', 0.8);
            }
        }

        async function stopStream() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('STOP');  // Send stop signal to server
                ws.close();
                ws = null;
            }
            if (streamInterval) {
                clearInterval(streamInterval);
                streamInterval = null;
            }
            if (clientVideo.srcObject) {
                clientVideo.srcObject.getTracks().forEach(track => track.stop());
                clientVideo.srcObject = null;
            }
            startBtn.disabled = false;
            stopBtn.disabled = true;
            captureBtn.disabled = true;
            measurementsDiv.innerHTML = '';
            const ctx = videoFeed.getContext('2d');
            ctx.clearRect(0, 0, 320, 320);
        }

        // Clean up when page is closed/refreshed
        window.addEventListener('beforeunload', stopStream);
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(index_page)

@app.route("/stream")
def stream_page():
    return render_template_string(stream_page_template)

@sock.route('/ws')
def websocket(ws):
    measure_next_frame = False
    try:
        while True:
            data = ws.receive()
            if data is None:
                break
              
            if data == 'STOP':
                break

            if data == 'MEASURE':
                measure_next_frame = True
                continue
                
            # Convert received data to image
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Run YOLO detection
            results = model.predict(frame, imgsz=640, conf=0.55, iou=0.1)[0]
            
            # Draw boxes and calculate measurements if requested
            boxes = []
            for r in results:
                for box in r.boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Calculate and draw center point
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2
                        cv2.circle(frame, (x_center, y_center), 3, (255, 0, 0), -1)
                        
                        if measure_next_frame:
                            boxes.append([x1, y1, x2, y2])
                    except Exception as e:
                        print(f"Error drawing box: {e}")
                        continue

            if measure_next_frame:
                # Calculate rows and columns
                if boxes:
                    boxes = np.array(boxes)
                    y_centers = [(box[1] + box[3]) // 2 for box in boxes]
                    x_centers = [(box[0] + box[2]) // 2 for box in boxes]
                    
                    # Group centers into rows and columns
                    y_centers = np.array(y_centers)
                    x_centers = np.array(x_centers)
                    
                    # Cluster y-coordinates for rows
                    y_clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=30,
                        linkage='complete'
                    ).fit(y_centers.reshape(-1, 1))
                    
                    # Cluster x-coordinates for columns 
                    x_clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=30,
                        linkage='complete'
                    ).fit(x_centers.reshape(-1, 1))
                    
                    measurements = {
                        'rows': len(set(y_clustering.labels_)),
                        'cols': len(set(x_clustering.labels_)),
                        'total': len(set(y_clustering.labels_)) * len(set(x_clustering.labels_))
                    }
                    
                    ws.send(json.dumps(measurements))
                measure_next_frame = False
            
            # Convert processed frame back to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            
            if ws.connected:
                ws.send(buffer.tobytes())
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket connection closed")


@app.route("/predict", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the file data into memory
        print("[DEBUG] Reading file data...")
        img_data = file.read()
        
        # Convert image data to numpy array
        print("[DEBUG] Converting to numpy array...")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        
        # Run inference
        print("[DEBUG] Running YOLO inference...")
        results = model.predict(img, imgsz=320, conf=0.3, iou=0.1)[0]

        # Get show_boxes parameter
        show_boxes = request.form.get('show_boxes') == 'true'
        print(f"[DEBUG] Show boxes: {show_boxes}")

        # Lists to store center points of knots
        centers_x = []
        centers_y = []

        # Process each result and extract boxes
        boxes = []  # Store all boxes and their centers
        height, width = img.shape[:2]
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # get box coordinates
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Calculate box area and visible area
                box_width = x2 - x1
                box_height = y2 - y1
                total_area = box_width * box_height
                
                # Calculate visible area
                visible_x1 = max(0, x1)
                visible_y1 = max(0, y1)
                visible_x2 = min(width, x2)
                visible_y2 = min(height, y2)
                
                visible_width = visible_x2 - visible_x1
                visible_height = visible_y2 - visible_y1
                visible_area = visible_width * visible_height
                
                # Only include box if at least 50% is visible
                if visible_area >= 0.5 * total_area:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    boxes.append({
                        'coords': (x1, y1, x2, y2),
                        'center': (center_x, center_y)
                    })
                    centers_x.append(center_x)
                    centers_y.append(center_y)

        print(f"[DEBUG] Found {len(boxes)} potential knots")

        # Sort centers
        centers_y.sort()
        centers_x.sort()

        # Set tolerances based on average knot size
        if len(boxes) > 0:
            avg_width = sum((b['coords'][2] - b['coords'][0]) for b in boxes) / len(boxes)
            avg_height = sum((b['coords'][3] - b['coords'][1]) for b in boxes) / len(boxes)
            print(f"[DEBUG] Found {avg_height, avg_width} avg heigth/width")
            x_tolerance = int(avg_width * 0.5)  # 50% of average knot width
            y_tolerance = int(avg_height * 0.5)  # 60% of average knot height
        else:
            x_tolerance = y_tolerance = 5  # Default fallback values
            
        # x_tolerance = 30
        # y_tolerance = 30
        # Find representative points for rows and columns using clustering
        rows = []
        cols = []
        
        # Cluster y-coordinates into rows
        if centers_y:
            current_row = [centers_y[0]]
            for y in centers_y[1:]:
                if y - current_row[-1] <= y_tolerance:
                    current_row.append(y)
                else:
                    # Add average y-coordinate of current row
                    rows.append(sum(current_row) // len(current_row))
                    current_row = [y]
            # Don't forget the last row
            if current_row:
                rows.append(sum(current_row) // len(current_row))

        # Cluster x-coordinates into columns
        if centers_x:
            current_col = [centers_x[0]]
            for x in centers_x[1:]:
                if x - current_col[-1] <= x_tolerance:
                    current_col.append(x)
                else:
                    # Add average x-coordinate of current column
                    cols.append(sum(current_col) // len(current_col))
                    current_col = [x]
            # Don't forget the last column
            if current_col:
                cols.append(sum(current_col) // len(current_col))

        print(f"[DEBUG] Found {len(rows)} rows and {len(cols)} columns")

        # Add padding for measurements
        padding = 100
        height, width = img.shape[:2]
        padded_img = np.full((height + 2*padding, width + 2*padding, 3), 255, dtype=np.uint8)
        padded_img[padding:padding+height, padding:padding+width] = img

        # Draw boxes if requested
        if show_boxes:
            print("[DEBUG] Drawing detection boxes...")
            for box in boxes:
                x1, y1, x2, y2 = box['coords']
                cv2.rectangle(padded_img, 
                            (x1 + padding, y1 + padding),
                            (x2 + padding, y2 + padding),
                            (0, 255, 0), 2)

        # Calculate knot count
        knot_count = len(rows) * len(cols)
        print(f"[DEBUG] Final knot count: {knot_count}")

        # Draw horizontal measurement lines and labels
        cv2.line(padded_img, (padding, padding//2), (width+padding, padding//2), (0, 0, 0), 2)
        cv2.putText(padded_img, f"{len(cols)} knots",
                    (padding + width//2 - 100, padding//2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        # Draw vertical measurement lines and labels
        cv2.line(padded_img, (width+padding+padding//2, padding), (width+padding+padding//2, height+padding), (0, 0, 0), 2)
        cv2.putText(padded_img, f"{len(rows)} knots",
                    (width+padding+padding//2 + 10, padding + height//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        # Add total knot count in bottom padding
        cv2.putText(padded_img, f"{int(knot_count)} Total Knots",
                    (padding + width//2 - 100, height + padding + padding//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        # Convert padded image to JPEG format
        print("[DEBUG] Encoding final image...")
        _, img_encoded = cv2.imencode('.jpg', padded_img)
        img_bytes = img_encoded.tobytes()

        # Store the padded image temporarily
        img_file = '/tmp/processed_image.jpg'
        with open(img_file, 'wb') as f:
            f.write(img_bytes)

        return render_template_string(index_page, image_url=f"/download_image")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_api", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the file data into memory
        img_data = file.read()
        
        # Convert image data to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get confidence threshold from request, default to 0.5
        conf = float(request.form.get('conf', 0.5))

        # Run inference
        results = model.predict(img, imgsz=640, conf=conf, iou=0.1)[0]

        # Get show_boxes parameter
        show_boxes = request.form.get('show_boxes') == 'true'

        # Lists to store center points of knots
        centers_x = []
        centers_y = []

        # Process each result and extract boxes
        boxes = []  # Store all boxes and their centers
        height, width = img.shape[:2]
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Calculate box area and visible area
                box_width = x2 - x1
                box_height = y2 - y1
                total_area = box_width * box_height
                
                # Calculate visible area
                visible_x1 = max(0, x1)
                visible_y1 = max(0, y1)
                visible_x2 = min(width, x2)
                visible_y2 = min(height, y2)
                
                visible_width = visible_x2 - visible_x1
                visible_height = visible_y2 - visible_y1
                visible_area = visible_width * visible_height
                
                if visible_area >= 0.5 * total_area:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    boxes.append({
                        'coords': (x1, y1, x2, y2),
                        'center': (center_x, center_y)
                    })
                    centers_x.append(center_x)
                    centers_y.append(center_y)

        # Sort centers
        centers_y.sort()
        centers_x.sort()

        # Set tolerances based on average knot size
        if len(boxes) > 0:
            avg_width = sum((b['coords'][2] - b['coords'][0]) for b in boxes) / len(boxes)
            avg_height = sum((b['coords'][3] - b['coords'][1]) for b in boxes) / len(boxes)
            x_tolerance_input = int(request.form.get('x_tolerance', 0.5))
            y_tolerance_input = int(request.form.get('y_tolerance', 0.5))
            x_tolerance = int(avg_width * x_tolerance_input)
            y_tolerance = int(avg_height * y_tolerance_input)
        else:
            x_tolerance = y_tolerance = 5

        # Find representative points for rows and columns
        rows = []
        cols = []
        
        # Cluster y-coordinates into rows
        if centers_y:
            current_row = [centers_y[0]]
            for y in centers_y[1:]:
                if y - current_row[-1] <= y_tolerance:
                    current_row.append(y)
                else:
                    rows.append(sum(current_row) // len(current_row))
                    current_row = [y]
            if current_row:
                rows.append(sum(current_row) // len(current_row))

        # Cluster x-coordinates into columns
        if centers_x:
            current_col = [centers_x[0]]
            for x in centers_x[1:]:
                if x - current_col[-1] <= x_tolerance:
                    current_col.append(x)
                else:
                    cols.append(sum(current_col) // len(current_col))
                    current_col = [x]
            if current_col:
                cols.append(sum(current_col) // len(current_col))

        # Add padding and draw boxes if requested
        padding = 1
        height, width = img.shape[:2]
        padded_img = np.full((height + 2*padding, width + 2*padding, 3), 255, dtype=np.uint8)
        padded_img[padding:padding+height, padding:padding+width] = img

        if show_boxes:
            for box in boxes:
                x1, y1, x2, y2 = box['coords']
                cv2.rectangle(padded_img, 
                            (x1 + padding, y1 + padding),
                            (x2 + padding, y2 + padding),
                            (0, 255, 0), 2)

        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', padded_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image": f"data:image/jpeg;base64,{img_base64}",
            "rows": len(rows),
            "columns": len(cols),
            "total_knots": len(rows) * len(cols)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_image")
def download_image():
    img_file = "/tmp/processed_image.jpg"
    return send_file(img_file, mimetype='image/jpeg')

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
