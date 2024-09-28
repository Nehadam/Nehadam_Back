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

# Initialize BLIP model for image captioning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# URL to the Stable Diffusion WebUI API
WEBUI_API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

# Model name or hash for asyncsMIXPONY_ponyV20.safetensors
MODEL_CHECKPOINT = "asyncsMIXPONY_ponyV20.safetensors"

# Base positive and negative prompts
BASE_POSITIVE_PROMPT = "score_9, score_8_up, score_7_up, source_anime, masterpiece, best quality, perfect anatomy, very aesthetic, official art, close-up of face, looking at viewer, high quality lighting, with all details"
NEGATIVE_PROMPT = ("nipples, pussy, nude, nsfw, cleavage, score_4, score_5, score_6, simple background, censored, "
                   "lowres, (bad anatomy, wrinkle, dutch angle, wedgie:1.2), watermark, negativeXL_D, mole on breast, submerged, "
                   "hyperrealistic, source_furry, source_pony, source_cartoon, mosaic censoring, bar censor, worst quality, low quality, "
                   "extra digits, fewer digits, missing fingers, bad anatomy, bad hands, extra fingers, extra feet, extra legs, extra hands, "
                   "extra breasts, twisted, jpeg artifacts, greyscale, futanari, kemonomimi, low quality, bad proportions, extra legs, "
                   "deformed anatomy, messy color, deformed fingers, bad, distracted, 3d")

# Securely load OpenAI GPT-4 API key from an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure the API key is loaded
if not openai.api_key:
    raise Exception("OpenAI API key not found. Set OPENAI_API_KEY as an environment variable.")

# Define a temporary directory for storing images
TEMP_DIR = "/tmp/stable_diffusion_images"
os.makedirs(TEMP_DIR, exist_ok=True)

def generate_image_description_blip(image_path):
    """
    Function to describe an image using BLIP (pre-trained model).
    The generated description will be based on the visual content of the image.
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Process the image and generate a description using BLIP
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return description

def refine_description_with_gpt(image_description):
    """
    Refine the basic image description using GPT-4.
    """
    # A system message to instruct GPT-4 to refine the description
    system_message = {
        "role": "system",
        "content": "You are an assistant that improves and enhances image descriptions."
    }

    # A user message that provides the basic description of the image
    user_message = {
        "role": "user",
        "content": f"Here is a basic description of an image: '{image_description}'. Please enhance it to be more detailed, positive, and artistic."
    }

    # Call the OpenAI GPT-4 API using the chat-based endpoint
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[system_message, user_message],
        max_tokens=100,
        temperature=0.7
    )

    # Get the refined description from GPT-4
    refined_description = response['choices'][0]['message']['content'].strip()
    return refined_description

def generate_image(input_image_path, gpt_description):
    """
    Function to send a POST request to the Stable Diffusion WebUI API for image generation.
    Uses img2img to modify the input image while keeping it similar.
    Adds the GPT-4 description to the positive prompt.
    """
    # Open the input image and encode it as base64
    with open(input_image_path, "rb") as image_file:
        input_image_bytes = base64.b64encode(image_file.read()).decode("utf-8")

    # Combine the base positive prompt with the GPT-4 generated description
    POSITIVE_PROMPT = f"{BASE_POSITIVE_PROMPT}, {gpt_description}"

    # Print the generated prompt to the console for debugging
    print(f"Generated Positive Prompt: {POSITIVE_PROMPT}")

    # Prepare the payload for the WebUI API
    payload = {
        "init_images": [input_image_bytes],  # input image to guide generation
        "prompt": POSITIVE_PROMPT,
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
        # Use BLIP to generate a basic description of the image
        basic_description = generate_image_description_blip(temp_input_path)
        print(f"BLIP generated description: {basic_description}")

        # Use GPT-4 to refine the description
        gpt_description = refine_description_with_gpt(basic_description)
        print(f"GPT-4 refined description: {gpt_description}")

        # Call the function to generate an image using the WebUI API
        output_image_bytes = generate_image(temp_input_path, gpt_description)

        # Send the generated image back to the client
        return send_file(output_image_bytes, mimetype='image/png')

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)

