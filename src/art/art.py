import os
import threading
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from io import BytesIO
from PIL import Image

# Constants
RESIZE_HEIGHT = 607
NUM_ITER = 1500
CONTENT_WEIGHT = 8e-4  # Content loss weight
STYLE_WEIGHT = 8e-1  # Style loss weight
CONTENT_LAYER_NAME = "block5_conv2"  # The layer to use for content loss
STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

# Set the local directory paths
base_path = './images/'  # You can modify this to your desired folder path
result_path = os.path.join(base_path, 'result/')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Flask setup
app = Flask(__name__)

def get_result_image_size(image_path, result_height):
    image_width, image_height = keras.preprocessing.image.load_img(image_path).size
    result_width = int(image_width * result_height / image_height)
    return result_height, result_width

def preprocess_image(image_path, target_height, target_width):
    img = keras.preprocessing.image.load_img(image_path, target_size=(target_height, target_width))
    arr = keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = vgg19.preprocess_input(arr)
    return tf.convert_to_tensor(arr)

def get_model():
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights='imagenet', include_top=False)
    outputs_dict = {layer.name: layer.output for layer in model.layers}
    return keras.Model(inputs=model.inputs, outputs=outputs_dict)

def get_optimizer():
    return keras.optimizers.Adam(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=8.0, decay_steps=445, decay_rate=0.98
        )
    )

def compute_loss(feature_extractor, combination_image, content_features, style_features):
    combination_features = feature_extractor(combination_image)
    loss_content = compute_content_loss(content_features, combination_features)
    loss_style = compute_style_loss(style_features, combination_features, combination_image.shape[1] * combination_image.shape[2])
    return CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style

def compute_content_loss(content_features, combination_features):
    original_image = content_features[CONTENT_LAYER_NAME]
    generated_image = combination_features[CONTENT_LAYER_NAME]
    return tf.reduce_sum(tf.square(generated_image - original_image)) / 2

def compute_style_loss(style_features, combination_features, combination_size):
    loss_style = 0
    for layer_name in STYLE_LAYER_NAMES:
        style_feature = style_features[layer_name][0]
        combination_feature = combination_features[layer_name][0]
        loss_style += style_loss(style_feature, combination_feature, combination_size) / len(STYLE_LAYER_NAMES)
    return loss_style

def style_loss(style_features, combination_features, combination_size):
    S = gram_matrix(style_features)
    C = gram_matrix(combination_features)
    channels = style_features.shape[2]
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (combination_size ** 2))

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def save_result(generated_image, result_height, result_width, name):
    img = deprocess_image(generated_image, result_height, result_width)
    keras.preprocessing.image.save_img(name, img)

def deprocess_image(tensor, result_height, result_width):
    tensor = tensor.numpy()
    tensor = tensor.reshape((result_height, result_width, 3))
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.680
    tensor = tensor[:, :, ::-1]  # 'BGR'->'RGB'
    return np.clip(tensor, 0, 255).astype("uint8")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def process_images(content_path, style_path):
    result_height, result_width = get_result_image_size(content_path, RESIZE_HEIGHT)
    print("result resolution: (%d, %d)" % (result_height, result_width))

    # Image preprocessing
    content_tensor = preprocess_image(content_path, result_height, result_width)
    style_tensor = preprocess_image(style_path, result_height, result_width)
    generated_image = tf.Variable(tf.random.uniform(style_tensor.shape, dtype=tf.dtypes.float32))

    # Model setup
    model = get_model()
    optimizer = get_optimizer()
    print(model.summary())

    # Extract content and style features
    content_features = model(content_tensor)
    style_features = model(style_tensor)

    # Optimize result image
    for iter in range(NUM_ITER):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, generated_image, content_features, style_features)

        grads = tape.gradient(loss, generated_image)
        print(f"iter: {iter:4d}, loss: {loss:8.2f}")
        optimizer.apply_gradients([(grads, generated_image)])

        if (iter + 1) % 100 == 0:
            name = os.path.join(result_path, f"result.png")
            save_result(generated_image, result_height, result_width, name)

    name = os.path.join(result_path, f"result.png")
    save_result(generated_image, result_height, result_width, name)
    return name

# Define a route to upload and process images
@app.route('/upload', methods=['POST'])
def upload_file():
    content_file = request.files.get('content')
    style_file = request.files.get('style')
    
    if content_file and style_file and allowed_file(content_file.filename) and allowed_file(style_file.filename):
        # Secure filenames
        content_filename = secure_filename(content_file.filename)
        style_filename = secure_filename(style_file.filename)
        
        # Paths to save files
        content_path = os.path.join(base_path, content_filename)
        style_path = os.path.join(base_path, style_filename)
        
        # Save files locally
        content_file.save(content_path)
        style_file.save(style_path)
        
        # Process the images
        processed_image_path = process_images(content_path, style_path)
        
        # Open the processed image and prepare it for sending
        with Image.open(processed_image_path) as img:
            img_io = BytesIO()  # Create a BytesIO object to hold the image data
            img.save(img_io, 'PNG')  # Save the image in PNG format to the BytesIO object
            img_io.seek(0)  # Seek back to the beginning of the BytesIO object
            
        # Return the image data as a file
        return send_file(img_io, mimetype='image/png')
    else:
        return 'Invalid files or no files uploaded', 400

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
