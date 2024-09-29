from flask import Flask, request, jsonify, url_for, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
import qrcode
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Set folder to save result images
RESULT_FOLDER = 'static/results/'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the fixed frames when the app starts
frame_black_path = os.path.join(os.path.dirname(__file__), 'frame.jpg')
frame_hadam4_path = os.path.join(os.path.dirname(__file__), 'hadam4.png')
frame_hadam2_path = os.path.join(os.path.dirname(__file__), 'hadam2.png')
frame_hadam1_path = os.path.join(os.path.dirname(__file__), 'hadam1.png')
frame_hadam_deer_path = os.path.join(os.path.dirname(__file__), 'hadam_deer.png')

frame_black = cv2.imread(frame_black_path)
frame_hadam4 = cv2.imread(frame_hadam4_path)
frame_hadam2 = cv2.imread(frame_hadam2_path)
frame_hadam1 = cv2.imread(frame_hadam1_path)
frame_hadam_deer = cv2.imread(frame_hadam_deer_path)

# Check if frames are loaded properly
if frame_black is None or frame_hadam4 is None or frame_hadam2 is None:
    raise FileNotFoundError("One or more frame images could not be loaded.")

# Frame position details for each frame
frame_positions_black = [
    ((34, 110), (295, 450)),  # Top-left square
    ((323, 110), (583, 450)), # Top-right square
    ((33, 472), (293, 813)),  # Bottom-left square
    ((322, 472), (583, 813))  # Bottom-right square
]

frame_positions_hadam4 = [
    ((17, 17), (322, 240)),  # Top-left square
    ((336, 17), (635, 240)), # Top-right square
    ((17, 255), (322, 473)),  # Bottom-left square
    ((336, 255), (635, 473))  # Bottom-right square
]

frame_positions_hadam2 = [
    ((17, 17), (323, 388)),  # Top-left square
    ((336, 17), (634, 388))  # Top-right square
]

frame_positions_hadam1 = [
    ((86, 229), (1800, 1945))
]

frame_positions_hadam_deer = [
    ((86, 229), (1800, 1945))
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

    # Ensure the final size is exactly target_w x target_h (this handles any off-by-one errors)
    cropped_img = cv2.resize(cropped_img, (target_w, target_h))

    return cropped_img

def insert_images_in_frame(frame, image_list, positions):
    # Copy the original frame to avoid modifying the loaded frame
    final_frame = frame.copy()

    # Loop over each of the positions and insert the respective images
    for i, (img, pos) in enumerate(zip(image_list, positions)):
        # Get the target size based on the frame position
        (x1, y1), (x2, y2) = pos
        target_w = x2 - x1
        target_h = y2 - y1

        # Resize and crop the image to fit the frame
        resized_cropped_img = resize_and_crop(img, (target_w, target_h))

        # Place the resized and cropped image into the frame
        final_frame[y1:y2, x1:x2] = resized_cropped_img

    return final_frame

def generate_qr_code(url):
    # Generate QR code from URL
    qr = qrcode.QRCode(
        version=1,  # Controls the size of the QR Code
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    qr_img = qr.make_image(fill='black', back_color='white')
    
    # Convert to OpenCV format
    qr_img = np.array(qr_img.convert('RGB'))
    qr_img = cv2.cvtColor(qr_img, cv2.COLOR_RGB2BGR)
    
    print(f"Generated QR code size: {qr_img.shape}")
    
    return qr_img

def overlay_qr_code(image, qr_code, scale_factor):
    # Get the dimensions of the final image
    img_h, img_w = image.shape[:2]

    # Calculate the desired size for the square QR code based on the scale factor
    target_size = img_w // scale_factor  # Scale the width by the given factor

    # Resize the QR code to fit the desired size (keep it square)
    qr_code_resized = cv2.resize(qr_code, (target_size, target_size))

    # Get the dimensions of the resized QR code
    qr_h, qr_w = qr_code_resized.shape[:2]

    # Calculate the position for the QR code (bottom-right corner)
    x_offset = img_w - qr_w - 10  # 10 px padding from the edge
    y_offset = img_h - qr_h - 10  # 10 px padding from the edge

    print(f"Overlaying QR code at position: {x_offset}, {y_offset}")

    # Overlay the resized QR code on the final image
    image[y_offset:y_offset + qr_h, x_offset:x_offset + qr_w] = qr_code_resized

    return image

def save_image_to_server(image):
    # Generate a unique filename using current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'4cutresult_{timestamp}.jpg'
    filepath = os.path.join(RESULT_FOLDER, filename)

    # Save the image to the server
    cv2.imwrite(filepath, image)
    print(f"Image saved to {filepath}")

    return filename

@app.route('/black', methods=['POST'])
def upload_files_black():
    return upload_files_for_frame(frame_black, frame_positions_black, 4, qr_scale_factor=10)

@app.route('/hadam4', methods=['POST'])
def upload_files_hadam4():
    return upload_files_for_frame(frame_hadam4, frame_positions_hadam4, 4, qr_scale_factor=10)

@app.route('/hadam2', methods=['POST'])
def upload_files_hadam2():
    return upload_files_for_frame(frame_hadam2, frame_positions_hadam2, 2, qr_scale_factor=10)

@app.route('/hadam1', methods=['POST'])
def upload_files_hadam1():
    return upload_files_for_frame(frame_hadam1, frame_positions_hadam1, 1, qr_scale_factor=5)

@app.route('/hadam_deer', methods=['POST'])
def upload_files_hadam_deer():
    return upload_files_for_frame(frame_hadam_deer, frame_positions_hadam_deer, 1, qr_scale_factor=5)

def upload_files_for_frame(frame, positions, num_images, qr_scale_factor):
    images = []

    # Check for the required number of images
    for i in range(1, num_images + 1):
        if f'image{i}' not in request.files:
            return jsonify({"error": f"No image{i} uploaded"}), 400
        image_file = request.files[f'image{i}']
        print(f"Received image{i}: {image_file.filename}")

        # Convert image to OpenCV format
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Check if the image was properly decoded
        if img is None:
            return jsonify({"error": f"Failed to decode image{i}"}), 400

        images.append(img)

    # Insert the images into the frame
    final_image = insert_images_in_frame(frame, images, positions)

    # Save the image to the server
    saved_filename = save_image_to_server(final_image)
    image_url = url_for('static', filename=f'results/{saved_filename}', _external=True)

    # Generate a QR code for the URL
    qr_code_img = generate_qr_code(image_url)

    # Overlay the QR code on the bottom-right of the final image with the scale factor
    final_image_with_qr = overlay_qr_code(final_image, qr_code_img, qr_scale_factor)

    # Save the final image with QR to the server
    saved_filename_with_qr = save_image_to_server(final_image_with_qr)
    final_image_url_with_qr = url_for('static', filename=f'results/{saved_filename_with_qr}', _external=True)

    # Return the image file directly to the client
    _, img_encoded = cv2.imencode('.jpg', final_image_with_qr)

    if img_encoded is None:
        return jsonify({"error": "Failed to encode the final image"}), 500

    return send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='4cutresult_with_qr.jpg'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
