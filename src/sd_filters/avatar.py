import os
import requests
from PIL import Image
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import base64

app = Flask(__name__)

# URL to the Stable Diffusion WebUI API
WEBUI_API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

# Model name or hash for duchaitenPonyXLNo_v52.safetensors
MODEL_CHECKPOINT = "duchaitenPonyXLNo_v52.safetensors"

# Positive and negative prompts
POSITIVE_PROMPT = "realistic style photo, outstanding style, tall, cute, in teens, soft lighting to cast gentle shadows on the subject, score_9, score_8_up, score_7_up"
NEGATIVE_PROMPT = "score_6, score_5, score_4, source_pony, (worst quality:1.2), (low quality:1.2), (normal quality:1.2), lowres, bad anatomy, bad hands, signature, watermarks, ugly, imperfect eyes, skewed eyes, unnatural face, unnatural body, error, extra limb, missing limbs, painting by bad-artist"

# Define a temporary directory for storing images
TEMP_DIR = "/tmp/stable_diffusion_images"
os.makedirs(TEMP_DIR, exist_ok=True)

def generate_image(input_image_path):
    """
    Function to send a POST request to the Stable Diffusion WebUI API for image generation.
    """
    # Open the input image and encode it as base64
    with open(input_image_path, "rb") as image_file:
        input_image_bytes = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Prepare the payload for the WebUI API
    payload = {
        "init_images": [input_image_bytes],
        "prompt": POSITIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "denoising_strength": 0.75,  # Adjust based on your preference
        "sampler_name": "Euler a",  # You can change the sampling method
        "cfg_scale": 7.0,  # Classifier-free guidance scale
        "steps": 50,  # Number of steps
        "override_settings": {
            "sd_model_checkpoint": MODEL_CHECKPOINT  # Explicitly setting the model checkpoint
        }
    }

    # Make the POST request to the WebUI API
    response = requests.post(WEBUI_API_URL, json=payload)
    
    if response.status_code != 200:
        raise RuntimeError(f"Stable Diffusion API request failed: {response.text}")
    
    # Get the generated image from the API response
    result = response.json()
    image_base64 = result['images'][0]

    # Convert the base64-encoded image to binary
    image_bytes = BytesIO(base64.b64decode(image_base64))

    return image_bytes

@app.route("/generate", methods=["POST"])
def generate():
    # Check if an image file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    # Open and save the image to a temporary file
    image = Image.open(BytesIO(file.read()))
    temp_input_path = os.path.join(TEMP_DIR, "input_image.png")
    image.save(temp_input_path)

    try:
        # Call the function to generate an image using the WebUI API
        output_image_bytes = generate_image(temp_input_path)

        # Send the generated image back to the client
        return send_file(output_image_bytes, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
