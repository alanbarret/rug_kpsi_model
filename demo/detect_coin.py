import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_coins(image_path):
    # Load the image and convert to HSV color space
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for silvery/gray colors
    lower_gray = np.array([0, 0, 100])   # Lower bound for gray/silver
    upper_gray = np.array([138,131,123]) # Upper bound for gray/silver

    # Create mask for gray regions
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours on a copy of the original image for visualization
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Identify the largest circular contour (assumed to be the coin)
    if len(contours) > 0:
        coin_contour = max(contours, key=cv2.contourArea)
        
        # Find the best fitting circle
        (x, y), radius = cv2.minEnclosingCircle(coin_contour)
        x, y, radius = int(x), int(y), int(radius)

        # Refine the coin edge detection
        # Apply Canny edge detection around the detected circle region
        roi = image[max(0,y-radius-10):min(image.shape[0],y+radius+10), 
                   max(0,x-radius-10):min(image.shape[1],x+radius+10)]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours in the edge image
        roi_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(roi_contours) > 0:
            # Find the largest contour in ROI which is likely the coin edge
            coin_edge = max(roi_contours, key=cv2.contourArea)
            # Draw the refined edge on the image
            cv2.drawContours(image[max(0,y-radius-10):min(image.shape[0],y+radius+10), 
                                 max(0,x-radius-10):min(image.shape[1],x+radius+10)], 
                           [coin_edge], -1, (0, 255, 0), 2)

        # Compute scale factor using known coin diameter (24mm)
        coin_diameter_px = 2 * radius
        mm_per_px = 24 / coin_diameter_px

        # Calculate the square size in pixels (1 cm²)
        square_side_px = int(10 / mm_per_px)

        # Define the position of the square next to the coin
        square_x = x + radius + 10  # Slightly to the right of the coin
        square_y = y - square_side_px // 2  # Center it vertically

        # Draw the detected coin circle as a reference
        cv2.circle(image, (x, y), radius, (255, 0, 0), 2)  # Draw blue circle around coin
        
        cv2.rectangle(image, 
                     (square_x, square_y),
                     (square_x + square_side_px, square_y + square_side_px),
                     (255, 0, 0), 2)  # Draw blue square
        
        # Add text showing actual size
        cv2.putText(image, "1 cm²", 
                    (square_x, square_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display results
        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Coin with 1cm² Reference')
        plt.axis('off')
        plt.show()
        
        return [(x,y), radius]
    else:
        print("No coins detected!")
        return None

# Example usage
detect_coins("coin.jpeg")