from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = YOLO("./runs/detect/train7/weights/best.pt").to(device)

# Dictionary mapping parking spot IDs to names and coordinates
spot_names = {37: {'name': '624', 'coordinates': [235, 270, 318, 415]}, 19: {'name': '623', 'coordinates': [324, 260, 410, 409]}, 35: {'name': '622', 'coordinates': [406, 261, 492, 409]}, 30: {'name': '621', 'coordinates': [485, 262, 569, 414]}, 47: {'name': '620', 'coordinates': [574, 261, 636, 393]}, 45: {'name': '619', 'coordinates': [655, 249, 719, 393]}, 25: {'name': '618', 'coordinates': [736, 267, 794, 403]}, 8: {'name': '617', 'coordinates': [810, 271, 884, 421]}, 16: {'name': '616', 'coordinates': [888, 271, 964, 420]}, 13: {'name': '615', 'coordinates': [968, 272, 1019, 419]}, 31: {'name': '614', 'coordinates': [1049, 266, 1111, 392]}, 53: {'name': '613', 'coordinates': [1134, 274, 1194, 391]}, 41: {'name': '612', 'coordinates': [1208, 267, 1287, 420]}, 55: {'name': '611', 'coordinates': [1289, 257, 1373, 412]}, 24: {'name': '610', 'coordinates': [1369, 266, 1450, 422]}, 46: {'name': '609', 'coordinates': [1460, 277, 1525, 417]}, 11: {'name': '608', 'coordinates': [1528, 274, 1605, 428]}, 23: {'name': '607', 'coordinates': [1615, 274, 1687, 417]}, 29: {'name': '606', 'coordinates': [1686, 273, 1768, 428]}, 56: {'name': '643', 'coordinates': [432, 611, 522, 780]}, 58: {'name': '642', 'coordinates': [524, 615, 590, 774]}, 52: {'name': '641', 'coordinates': [614, 629, 682, 778]}, 1: {'name': '640', 'coordinates': [694, 617, 772, 780]}, 3: {'name': '639', 'coordinates': [776, 615, 860, 780]}, 48: {'name': '638', 'coordinates': [879, 613, 943, 764]}, 20: {'name': '637', 'coordinates': [950, 614, 1028, 779]}, 50: {'name': '636', 'coordinates': [1047, 633, 1112, 771]}, 18: {'name': '635', 'coordinates': [1119, 616, 1206, 783]}, 9: {'name': '634', 'coordinates': [1208, 618, 1294, 783]}, 39: {'name': '633', 'coordinates': [1297, 621, 1380, 785]}, 4: {'name': '632', 'coordinates': [1379, 618, 1465, 787]}, 32: {'name': '631', 'coordinates': [1480, 624, 1549, 778]}, 5: {'name': '630', 'coordinates': [1551, 618, 1634, 784]}, 33: {'name': '629', 'coordinates': [1643, 618, 1724, 781]}, 51: {'name': '664', 'coordinates': [334, 786, 405, 934]}, 60: {'name': '663', 'coordinates': [421, 786, 492, 945]}, 27: {'name': '662', 'coordinates': [510, 786, 599, 963]}, 28: {'name': '661', 'coordinates': [597, 814, 668, 980]}, 0: {'name': '660', 'coordinates': [689, 788, 771, 965]}, 7: {'name': '659', 'coordinates': [775, 787, 860, 965]}, 2: {'name': '658', 'coordinates': [865, 786, 946, 965]}, 6: {'name': '657', 'coordinates': [952, 786, 1035, 934]}, 49: {'name': '656', 'coordinates': [1048, 800, 1117, 943]}, 17: {'name': '655', 'coordinates': [1127, 789, 1212, 965]}, 40: {'name': '654', 'coordinates': [1216, 788, 1304, 967]}, 14: {'name': '653', 'coordinates': [1307, 793, 1386, 965]}, 15: {'name': '652', 'coordinates': [1391, 790, 1485, 967]}, 36: {'name': '651', 'coordinates': [1485, 797, 1570, 966]}, 12: {'name': '650', 'coordinates': [1569, 791, 1661, 970]}, 38: {'name': '649', 'coordinates': [1650, 790, 1761, 976]}}

# Create a lookup dictionary for spot names to coordinates
new_spot_names = {}
for id, spot in spot_names.items():
    new_spot_names[spot['name']] = spot['coordinates']

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process the image and detect parking spots"""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to load image"}, []
    
    # Run inference
    results = model(img, device=0 if torch.cuda.is_available() else "cpu")
    
    # Extract detected coordinates
    coordinates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bbox coordinates
            coordinates.append([x1, y1, x2, y2])
            cls = int(box.cls[0])  # Class ID
            conf = box.conf[0].item()  # Confidence score
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls} {x1, y1, x2, y2}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Save the processed image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "detected_" + os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    
    # Determine empty spots and occupied spots
    empty_spots = []
    for name, spot_coord in new_spot_names.items():
        if spot_coord not in coordinates:
            empty_spots.append(name)
    
    # Determine unknown coordinates (not matching any known parking spot)
    unknown_coords = []
    for coordinate in coordinates:
        found = False
        for name, spot in new_spot_names.items():
            if coordinate == spot:
                found = True
                break
        if not found:
            unknown_coords.append(coordinate)
    
    # Calculate occupied spots
    occupied_spots = [name for name in new_spot_names.keys() if name not in empty_spots]
    
    # Group empty spots by section
    grouped_empty_spots = []
    for spot in empty_spots:
        # Using first digit as section identifier
        section = spot[0] if len(spot) > 0 else "?"
        grouped_empty_spots.append({
            "id": spot,
            "section": section,
            "status": "empty"
        })
    
    return {
        "processed_image": output_path,
        "empty_spots": empty_spots,
        "occupied_spots": occupied_spots,
        "grouped_empty_spots": grouped_empty_spots,
        "unknown_detections": unknown_coords,
        "total_spots": len(new_spot_names),
        "total_empty": len(empty_spots),
        "total_occupied": len(occupied_spots)
    }, coordinates

@app.route('/api/detect', methods=['POST'])
def detect_parking():
    """API endpoint to process parking lot images"""
    # Check if request has file
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image
        result, _ = process_image(file_path)
        
        # Return the API response
        return jsonify(result)
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/spots', methods=['GET'])
def get_parking_spots():
    """API endpoint to get all parking spot information"""
    return jsonify({
        "spots": new_spot_names,
        "total_spots": len(new_spot_names)
    })

@app.route('/uploads/<filename>')
def serve_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    # Use port 5001 instead of 5000 to avoid conflict with AirPlay on macOS
    app.run(debug=True, host='0.0.0.0', port=5001)