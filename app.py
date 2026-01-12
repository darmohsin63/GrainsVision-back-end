import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS to allow your React app (on a different port) to communicate with this API
CORS(app)

def base64_to_image(base64_string):
    """Decodes a base64 string into an OpenCV image."""
    try:
        # Remove metadata header if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        decoded_data = base64.b64decode(base64_string)
        np_data = np.frombuffer(decoded_data, np.uint8)
        image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

def image_to_base64(image):
    """Encodes an OpenCV image back to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        # --- 1. Decode Image ---
        original_img = base64_to_image(data['image'])
        if original_img is None:
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400

        # --- 2. Preprocessing ---
        # Convert to Grayscale
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Adaptive Thresholding (handles inconsistent lighting better than simple thresholding)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to separate touching grains
        kernel = np.ones((3, 3), np.uint8)
        # Opening removes small noise
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # Dilate slightly to make sure valid grains are solid
        sure_bg = cv2.dilate(opening, kernel, iterations=1)

        # --- 3. Find Contours ---
        contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 4. Analysis Logic ---
        total_grains = 0
        broken_grains = 0
        chalky_grains = 0
        damaged_grains = 0
        
        output_img = original_img.copy()
        
        # Filter noise based on area
        min_area = 50 
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Calculate average area to establish a baseline for "Whole" grains
        if valid_contours:
            avg_area = np.mean([cv2.contourArea(cnt) for cnt in valid_contours])
        else:
            avg_area = 0

        for cnt in valid_contours:
            total_grains += 1
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Aspect Ratio Calculation (Longest side / Shortest side)
            dim1, dim2 = (w, h) if w > h else (h, w)
            aspect_ratio = dim1 / dim2 if dim2 > 0 else 0
            
            is_defect = False
            color = (0, 255, 0) # Green (Good)

            # Check 1: Broken Grains
            # Logic: If area is significantly smaller than average OR aspect ratio is too round (for rice)
            if area < (avg_area * 0.5) or aspect_ratio < 1.5:
                broken_grains += 1
                is_defect = True
                color = (0, 0, 255) # Red

            # Check 2: Chalky Grains (Whiteness/Opacity)
            # Logic: Look at the average intensity of pixels inside the grain
            if not is_defect:
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]
                
                # If grain is very bright/white (chalky), mark it
                # Threshold depends on lighting, 160-200 is a safe range for white rice on dark bg
                if mean_intensity > 190: 
                    chalky_grains += 1
                    is_defect = True
                    color = (0, 255, 255) # Yellow

            # Draw bounding box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)

        # --- 5. Statistics Calculation ---
        if total_grains == 0: 
            return jsonify({
                'success': True,
                'processedImage': data['image'],
                'stats': {
                    'grainCount': 0, 'purityPercentage': 0, 'defectPercentage': 0,
                    'brokenPercentage': 0, 'chalkyPercentage': 0, 'damagedPercentage': 0,
                    'impuritiesPercentage': 0, 'summary': "No grains detected.",
                    'grade': "N/A", 'recommendation': "Check lighting and focus."
                }
            })

        broken_pct = (broken_grains / total_grains) * 100
        chalky_pct = (chalky_grains / total_grains) * 100
        # Simulating impurities based on remaining noise or specific color detection (omitted for speed)
        impurities_pct = 0.5 
        damaged_pct = 0.5
        
        total_defect_pct = broken_pct + chalky_pct + damaged_pct + impurities_pct
        purity_pct = max(0, 100 - total_defect_pct)
        
        # Grading Logic
        grade = "Standard"
        if purity_pct >= 92: grade = "Premium Grade A"
        elif purity_pct >= 80: grade = "Grade B"
        else: grade = "Feed Grade"

        # Summary Generation
        summary = f"Analysis complete. Detected {total_grains} grains. " \
                  f"Purity is {purity_pct:.1f}%. " \
                  f"Found {broken_grains} broken and {chalky_grains} chalky kernels."

        recommendation = "Approved for packaging."
        if broken_pct > 15: recommendation = "High broken percentage. Sifting recommended."
        if purity_pct < 75: recommendation = "Quality below standard. Reject batch."

        return jsonify({
            'success': True,
            'processedImage': image_to_base64(output_img),
            'stats': {
                'grainCount': total_grains,
                'purityPercentage': round(purity_pct, 1),
                'defectPercentage': round(total_defect_pct, 1),
                'brokenPercentage': round(broken_pct, 1),
                'chalkyPercentage': round(chalky_pct, 1),
                'damagedPercentage': round(damaged_pct, 1),
                'impuritiesPercentage': round(impurities_pct, 1),
                'summary': summary,
                'grade': grade,
                'recommendation': recommendation
            }
        })

    except Exception as e:
        print(f"Analysis Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Run server on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
