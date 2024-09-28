import os
import requests
import openai
from PIL import Image
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import base64
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Initialize BLIP model for image captioning (optional, in case you use GPT for refinement)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# URL to the Stable Diffusion WebUI API
WEBUI_API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

# Model name or hash for pixelArtDiffusionXL_spriteShaper.safetensors
MODEL_CHECKPOINT = "pixelArtDiffusionXL_spriteShaper.safetensors"

# Positive and negative prompts
BASE_POSITIVE_PROMPT = "Pixel Art, a person, best quality, Golden ratio, (masterpiece, best quality:1.2), HDR, chromatic aberration, depth of field, shaded, good looking, charm, anime, high resolution"
NEGATIVE_PROMPT = "bad quality, bad anatomy, worst quality, low quality, low resolution, extra fingers, blur, blurry, ugly, wrong proportions, watermark, image artifacts, lowres, jpeg artifacts, deformed, noisy image, deformation, corrupt image"

# Securely load OpenAI GPT-4 API key from an environment variable (optional, if using GPT)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a temporary directory for storing images
TEMP_DIR = "/tmp/stable_diffusion_images"
os.makedirs(TEMP_DIR, exist_ok=True)

def generate_image(input_image_path):
    """
    Function to send a POST request to the Stable Diffusion WebUI API for image generation.
    Uses img2img to modify the input image while keeping it similar.
    """
    # Open the input image and encode it as base64
    with open(input_image_path, "rb") as image_file:
        input_image_bytes = base64.b64encode(image_file.read()).decode("utf-8")

    # Prepare the payload for the WebUI API
    payload = {
        "init_images": [input_image_bytes],  # input image to guide generation
        "prompt": BASE_POSITIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "denoising_strength": 0.75,  # Controls how much variation there is from the original image
        "sampler_name": "Euler a",  # You can change the sampling method
        "cfg_scale": 7.0,  # Classifier-free guidance scale
        "steps": 28,  # Number of steps
        "width": 512,  # Image width
        "height": 512,  # Image height
        "enable_hr": True,  # Enable high resolution
        "hr_scale": 1.75,  # High-resolution scale
        "hr_upscaler": "Latent (nearest-exact)",  # Upscaler to use
        "hr_second_pass_steps": 28,  # Second pass steps for high-res
        "ensd": -1,  # Random seed for reproducibility
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
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    # Open and save the image to a temporary file
    image = Image.open(BytesIO(file.read()))
    temp_input_path = os.path.join(TEMP_DIR, "input_image.png")
    image.save(temp_input_path)

    try:
        # Generate the image using the WebUI API
        output_image_bytes = generate_image(temp_input_path)

        # Send the generated image back to the client
        return send_file(output_image_bytes, mimetype='image/png')

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)
