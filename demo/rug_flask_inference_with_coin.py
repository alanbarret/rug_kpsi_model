from flask import Flask, request, jsonify, send_file, render_template_string
import cv2
import numpy as np
from pyngrok import ngrok
from io import BytesIO
from ultralytics import YOLO
import os

app = Flask(__name__)

# Setup ngrok tunnel
# ngrok.set_auth_token("2sesgJLFXrYNkfute01Q5xT38uk_6QUSnwWtT5tfsuRSUA8zn")
# tunnel = ngrok.connect(5000)
# print(f"Public URL: {tunnel.public_url}")

# Load YOLO model
model = YOLO("trained_models/rug_knot_classifier_v3.pt")

# Create temp directory if it doesn't exist
os.makedirs('/tmp', exist_ok=True)

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

@app.route("/", methods=["GET"])
def index():
    return render_template_string(index_page)

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
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Use imdecode instead of imread
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define HSV range for silvery/gray colors
        lower_gray = np.array([0, 0, 100])   # Lower bound for gray/silver
        upper_gray = np.array([138, 131, 123]) # Upper bound for gray/silver

        # Create mask for gray regions
        mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Identify the largest circular contour (assumed to be the coin)
        if len(contours) > 0:
            coin_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(coin_contour)
            x, y, radius = int(x), int(y), int(radius)

            # Refine the coin edge detection
            # Apply Canny edge detection around the detected circle region
            roi = img[max(0,y-radius-10):min(img.shape[0],y+radius+10), 
                   max(0,x-radius-10):min(img.shape[1],x+radius+10)]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours in the edge image
            roi_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(roi_contours) > 0:
                # Find the largest contour in ROI which is likely the coin edge
                coin_edge = max(roi_contours, key=cv2.contourArea)
                # Draw the refined edge on the image
                cv2.drawContours(img[max(0,y-radius-10):min(img.shape[0],y+radius+10), 
                                   max(0,x-radius-10):min(img.shape[1],x+radius+10)], 
                               [coin_edge], -1, (0, 255, 0), 2)

            # Compute scale factor using known coin diameter (24mm)
            coin_diameter_px = 2 * radius
            mm_per_px = 24 / coin_diameter_px

            # Calculate the square size in pixels (1 cm²)
            square_side_px = int(10 / mm_per_px)

            # Define the position of the square next to the coin
            square_x = x + radius + 10  # Slightly to the right of the coin
            square_y = y - square_side_px // 2  # Center it vertically


            # Crop the 1cm² area for analysis
            cropped_img = img[square_y:square_y + square_side_px,
                            square_x:square_x + square_side_px]
            # Run inference on cropped area
            print("[DEBUG] Running YOLO inference...")
            results = model.predict(cropped_img, imgsz=640, conf=0.15, iou=0.45)[0]
            # Get show_boxes parameter
            show_boxes = request.form.get('show_boxes') == 'true'
            print(f"[DEBUG] Show boxes: {show_boxes}")

            # Lists to store center points of knots
            centers_x = []
            centers_y = []

            # Process each result and extract boxes
            boxes = []  # Store all boxes and their centers
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # get box coordinates
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
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
                x_tolerance = int(avg_width * 0.5)  # 50% of average knot width
                y_tolerance = int(avg_height * 0.5)  # 50% of average knot height
            else:
                x_tolerance = y_tolerance = 5  # Default fallback values

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

            # Draw boxes if requested
            if show_boxes:
                print("[DEBUG] Drawing detection boxes...")
                for box in boxes:
                    x1, y1, x2, y2 = box['coords']
                    cv2.rectangle(cropped_img, 
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0), 2)

            # Calculate knot count
            knot_count = len(rows) * len(cols)
            print(f"[DEBUG] Final knot count: {knot_count}")

            # Add padding to cropped image for measurements
            padding = 50
            height, width = cropped_img.shape[:2]
            padded_img = np.full((height + 2*padding, width + 2*padding, 3), 255, dtype=np.uint8)
            padded_img[padding:padding+height, padding:padding+width] = cropped_img

            # Draw horizontal measurement lines and labels
            cv2.line(padded_img, (padding, padding//2), (width+padding, padding//2), (0, 0, 0), 2)
            cv2.putText(padded_img, f"{len(cols)} knots",
                        (padding + width//2 - 100, padding//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Draw vertical measurement lines and labels
            cv2.line(padded_img, (width+padding+padding//2, padding), (width+padding+padding//2, height+padding), (0, 0, 0), 2)
            cv2.putText(padded_img, f"{len(rows)} knots",
                        (width+padding+padding//2 + 10, padding + height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Add knots per square cm in bottom padding
            cv2.putText(padded_img, f"{int(knot_count)} KPCM",
                        (padding + width//2 - 100, height + padding + padding//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Convert padded image to JPEG format
            print("[DEBUG] Encoding final image...")
            _, img_encoded = cv2.imencode('.jpg', padded_img)
            img_bytes = img_encoded.tobytes()

            # Store the padded image temporarily
            img_file = '/tmp/processed_image.jpg'
            with open(img_file, 'wb') as f:
                f.write(img_bytes)

            return render_template_string(index_page, image_url=f"/download_image")
            
        else:
            return jsonify({"error": "No coin detected in image"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download_image")
def download_image():
    img_file = "/tmp/processed_image.jpg"
    return send_file(img_file, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
