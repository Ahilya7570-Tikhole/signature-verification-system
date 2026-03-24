import cv2
import numpy as np
import os
from glob import glob

def preprocess_image(input_path, output_path):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        return False
        
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise Removal (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Thresholding (Inv) to find the text easily
    # We use inverse binary so the signature is white (255) and background is black (0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Crop Signature Region
    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter negligible noise
            if w > 10 and h > 10:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        
        # If valid bounding box found
        if x_min < np.inf and (x_max - x_min) > 0 and (y_max - y_min) > 0:
            pad = 10
            # crop from ORIGINAL grayscale image, not the thresholded one
            start_y = max(0, int(y_min) - pad)
            end_y = min(gray.shape[0], int(y_max) + pad)
            start_x = max(0, int(x_min) - pad)
            end_x = min(gray.shape[1], int(x_max) + pad)
            
            cropped = gray[start_y:end_y, start_x:end_x]
        else:
            cropped = gray
    else:
        cropped = gray
        
    # 5. Resize (224 x 224)
    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
    
    # 6. Normalize
    # We min-max normalize standardizing the pixel range strictly to 0-255 
    normalized = cv2.normalize(resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Save the final image
    cv2.imwrite(output_path, normalized)
    return True

def process_dataset(local_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    count = 0
    
    for r, d, f in os.walk(local_dir):
        for file in f:
            if file.lower().endswith(('.png', '.tif', '.jpg', '.jpeg')):
                in_path = os.path.join(r, file)
                rel_path = os.path.relpath(r, local_dir)
                out_folder = os.path.join(processed_dir, rel_path)
                
                os.makedirs(out_folder, exist_ok=True)
                out_path = os.path.join(out_folder, file)
                
                if preprocess_image(in_path, out_path):
                    count += 1
    print(f"Processed {count} images and saved to {processed_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_dataset_path = os.path.join(base_dir, "Local_dataset")
    processed_dataset_path = os.path.join(base_dir, "Processed_dataset")
    print(f"Running processing in {base_dir}...")
    process_dataset(local_dataset_path, processed_dataset_path)
