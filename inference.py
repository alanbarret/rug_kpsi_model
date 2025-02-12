from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
# model = YOLO("yolo11m.pt")  # load an official model
model = YOLO("trained_models/rug_knot_classifier_v5.pt")  # load a custom model

# Predict with the model
results = model("images/1x1CM_rug_2.jpeg",
    imgsz=320,  # image size
    conf=0.5,  # confidence threshold
    iou=0.5    # NMS IOU threshold
    )  # predict on an image
# print(results)

# Plot the results with green boxes and count knots
for r in results:
    # Get the original image
    im_array = r.orig_img
    height, width = im_array.shape[:2]
    
    # Get all box coordinates
    boxes = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    # Convert to numpy array for easier manipulation
    boxes = np.array(boxes)
    
    # Count knots in rows (height)
    if len(boxes) > 0:
        # Sort boxes by y-coordinate (height)
        sorted_by_y = boxes[boxes[:, 1].argsort()]
        row_count = 1
        current_y = sorted_by_y[0][1]
        
        for box in sorted_by_y[1:]:
            if box[1] - current_y > height * 0.05:  # 5% threshold for new row
                row_count += 1
                current_y = box[1]
        
        # Count knots in columns (width)
        sorted_by_x = boxes[boxes[:, 0].argsort()]
        col_count = 1
        current_x = sorted_by_x[0][0]
        
        for box in sorted_by_x[1:]:
            if box[0] - current_x > width * 0.05:  # 5% threshold for new column
                col_count += 1
                current_x = box[0]
        
        total_knots = len(boxes)
        
        # Draw counts on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im_array, f'Rows: {row_count}', (10, 30), font, 1, (255, 0, 0), 2)
        cv2.putText(im_array, f'Columns: {col_count}', (10, 70), font, 1, (255, 0, 0), 2)
        cv2.putText(im_array, f'Total Knots: {total_knots}', (10, 110), font, 1, (255, 0, 0), 2)
    
    # Plot each detection box
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(im_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image
cv2.imshow("Detected Knots", im_array)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the image
# cv2.imwrite("detected_knots_green.jpg", im_array)