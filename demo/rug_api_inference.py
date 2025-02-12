from flask import Flask, request, jsonify, send_file, render_template_string
import requests
import cv2
import numpy as np
from pyngrok import ngrok
from io import BytesIO

app = Flask(__name__)

# Setup ngrok tunnel
ngrok.set_auth_token("2sesgJLFXrYNkfute01Q5xT38uk_6QUSnwWtT5tfsuRSUA8zn")
tunnel = ngrok.connect(5000)
print(f"Public URL: {tunnel.public_url}")

API_URL = "https://predict.ultralytics.com"
HEADERS = {"x-api-key": "d259bc72855a3168ff91758044c533536e7441cc17"}

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
        <form class="upload-form" action="/predict/" method="POST" enctype="multipart/form-data" id="upload-form">
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

@app.route("/predict/", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the file data into memory
        img_data = file.read()
        
        # Send the image data to the prediction API
        data = {"model": "https://hub.ultralytics.com/models/Dh0OzsK2RPI64ejpS4SO", "imgsz": 640, "conf": 0.04, "iou": 0.45}
        files = {"file": (file.filename, img_data, file.mimetype)}
        response = requests.post(API_URL, headers=HEADERS, data=data, files=files)
        response.raise_for_status()  # Will raise an exception for invalid responses
        results = response.json()
        
        if "images" not in results or not results["images"]:
            return jsonify({"error": "Invalid response format"}), 500

        # Get image shape from response
        img_shape = results["images"][0]["shape"]
        height, width = img_shape[0], img_shape[1]

        # Decode the image data (buffer) back into a usable OpenCV image
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        # Create padded image with white background
        padding = 100  # Padding size in pixels
        padded_img = np.full((height + 2*padding, width + 3*padding, 3), 255, dtype=np.uint8)
        padded_img[padding:padding+height, padding:padding+width] = img
        
        # Lists to store center points of knots
        centers_x = []
        centers_y = []
        # Get show_boxes parameter
        show_boxes = request.form.get('show_boxes') == 'true'

        # Process each result and extract boxes
        boxes = []  # Store all boxes and their centers
        for result in results["images"][0]["results"]:
            box = result["box"]
            x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            boxes.append({
                'coords': (x1, y1, x2, y2),
                'center': (center_x, center_y)
            })
            centers_x.append(center_x)
            centers_y.append(center_y)

        # Sort centers and find unique rows/columns with tolerance
        centers_y.sort()
        centers_x.sort()
        
        def calculate_tolerance(sorted_coords):
            if len(sorted_coords) < 2:
                return 20
            diffs = [sorted_coords[i+1] - sorted_coords[i] for i in range(len(sorted_coords)-1)]
            median_diff = sorted(diffs)[len(diffs)//2]
            return max(10, int(median_diff * 0.5))

        x_tolerance = calculate_tolerance(centers_x)
        y_tolerance = calculate_tolerance(centers_y)
        
        # Find representative points for rows and columns
        rows = []
        cols = []
        row_points = set()  # Store y-coordinates of accepted points
        col_points = set()  # Store x-coordinates of accepted points
        
        prev_y = -y_tolerance
        for y in centers_y:
            if y - prev_y > y_tolerance:
                rows.append(y)
                row_points.add(y)
                prev_y = y
                
        prev_x = -x_tolerance        
        for x in centers_x:
            if x - prev_x > x_tolerance:
                cols.append(x)
                col_points.add(x)
                prev_x = x

        # Draw boxes only for points that match our row/column criteria
        if show_boxes:
            for box in boxes:
                center_x, center_y = box['center']
                # Check if this box's center is close to any accepted row/column intersection
                for row_y in rows:
                    for col_x in cols:
                        if (abs(center_x - col_x) <= x_tolerance and 
                            abs(center_y - row_y) <= y_tolerance):
                            x1, y1, x2, y2 = box['coords']
                            # Draw box with padding adjustment
                            cv2.rectangle(padded_img, 
                                        (x1 + padding, y1 + padding),
                                        (x2 + padding, y2 + padding),
                                        (0, 255, 0), 2)
                            break
                    else:
                        continue
                    break

        # Calculate knot count from rows and columns
        knot_count = len(rows) * len(cols)

        # Draw horizontal measurement lines and labels in top padding
        cv2.line(padded_img, (padding, padding//2), (width+padding, padding//2), (0, 0, 0), 2)
        cv2.putText(padded_img, f"{len(cols)} knots", 
                    (padding + width//2 - 100, padding//2 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Draw vertical measurement lines and labels in right padding
        cv2.line(padded_img, (width+padding+padding//2, padding), (width+padding+padding//2, height+padding), (0, 0, 0), 2)
        cv2.putText(padded_img, f"{len(rows)} knots",
                    (width+padding+padding//2 + 10, padding + height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # Add knots per square inch and total knots in bottom padding
        cv2.putText(padded_img, f"{int(knot_count)} KPSI", 
                    (padding + width//2 - 100, height + padding + padding//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Convert image to JPEG format
        _, img_encoded = cv2.imencode('.jpg', padded_img)
        img_bytes = img_encoded.tobytes()

        # Convert to a URL that will be accessible in the browser
        image_url = tunnel.public_url + "/download_image"
        
        # Store the image temporarily and return the URL
        img_file = '/tmp/processed_image.jpg'
        with open(img_file, 'wb') as f:
            f.write(img_bytes)
        
        return render_template_string(index_page, image_url=f"/download_image")

    except requests.exceptions.RequestException as e:
        # Handle errors with the API request
        return jsonify({"error": f"Error connecting to prediction API: {str(e)}"}), 500
    except Exception as e:
        # Handle general errors
        return jsonify({"error": str(e)}), 500

@app.route("/download_image")
def download_image():
    img_file = "/tmp/processed_image.jpg"
    return send_file(img_file, mimetype='image/jpeg', as_attachment=True, download_name='processed_image.jpg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
