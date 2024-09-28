from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the fixed frame when the app starts
frame_path = os.path.join(os.path.dirname(__file__), 'frame.jpg')
frame = cv2.imread(frame_path)

# Frame position details (as defined earlier)
frame_positions = [
    ((34, 110), (295, 450)),  # Top-left square
    ((323, 110), (583, 450)), # Top-right square
    ((33, 472), (293, 813)),  # Bottom-left square
    ((322, 472), (583, 813))  # Bottom-right square
]

def resize_and_crop(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    # Compute aspect ratios
    img_aspect = w / h
    target_aspect = target_w / target_h

    # Resize while maintaining aspect ratio
    if img_aspect > target_aspect:
        # Image is wider, fit by height
        scale_factor = target_h / h
    else:
        # Image is taller, fit by width
        scale_factor = target_w / w
    
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    resized_img = cv2.resize(img, (new_w, new_h))

    # Crop the extra part if necessary
    if new_w > target_w:
        # Crop width
        start_x = (new_w - target_w) // 2
        cropped_img = resized_img[:, start_x:start_x + target_w]
    elif new_h > target_h:
        # Crop height
        start_y = (new_h - target_h) // 2
        cropped_img = resized_img[start_y:start_y + target_h, :]
    else:
        cropped_img = resized_img

    return cropped_img

def insert_images_in_frame(image_list):
    # Copy the original frame to avoid modifying the loaded frame
    final_frame = frame.copy()

    # Loop over each of the 4 positions and insert the respective images
    for i, (img, pos) in enumerate(zip(image_list, frame_positions)):
        # Get the target size based on the frame position
        (x1, y1), (x2, y2) = pos
        target_w = x2 - x1
        target_h = y2 - y1

        # Resize and crop the image to fit the frame
        resized_cropped_img = resize_and_crop(img, (target_w, target_h))

        # Place the resized and cropped image into the frame
        final_frame[y1:y2, x1:x2] = resized_cropped_img

    return final_frame

@app.route('/upload', methods=['POST'])
def upload_files():
    images = []
    
    for i in range(1, 5):
        if f'image{i}' not in request.files:
            return jsonify({"error": f"No image{i} uploaded"}), 400
        image_file = request.files[f'image{i}']
        # Convert image to OpenCV format
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        images.append(img)

    # Insert the images into the frame
    final_image = insert_images_in_frame(images)

    # Convert final_image back to a file-like object for sending
    _, img_encoded = cv2.imencode('.jpg', final_image)
    return send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='4cutresult.jpg'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
