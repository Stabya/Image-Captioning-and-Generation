from flask import Flask, request, jsonify, session, send_file
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
import os, uuid, time, requests, io, json, base64
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)
app.secret_key = os.urandom(24)

SAVE_DIR = 'saved_images'
SUMMARY_FILE = 'summaries.json'
os.makedirs(SAVE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB limit for uploads

model_id = "Salesforce/blip-vqa-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

api_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
api_keys = [
    "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",
    "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",
    "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"
]
current_key_index = 0

headers = {"Authorization": f"Bearer {api_keys[current_key_index]}"}

# Load models
try:
    vqa_model = BlipForQuestionAnswering.from_pretrained(model_id).to(device)
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    processor = BlipProcessor.from_pretrained(model_id)
except Exception as e:
    print(f"Error loading models or processor: {e}")
    exit()

if os.path.exists(SUMMARY_FILE):
    with open(SUMMARY_FILE, "r") as f:
        image_summaries = json.load(f)
else:
    image_summaries = {}

def allowed_file(filename):
    """Check if the file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_summaries():
    """Helper function to save image summaries to a JSON file."""
    with open(SUMMARY_FILE, "w") as f:
        json.dump(image_summaries, f)

def rotate_api_key():
    """Rotate the Hugging Face API key upon expiration, and handle exhausted key list."""
    global current_key_index, headers
    if current_key_index + 1 >= len(api_keys):
        print("All API keys have been exhausted. Please update the API keys.")
        raise Exception("All API keys exhausted")
    current_key_index = (current_key_index + 1) % len(api_keys)
    headers = {"Authorization": f"Bearer {api_keys[current_key_index]}"}
    print(f"Rotated to API key index: {current_key_index}")

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image, generate a caption, and save the image and caption."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'}), 400

    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_length=50)
    summary = processor.decode(out[0], skip_special_tokens=True)

    image_id = uuid.uuid4().hex
    saved_image_path = os.path.join(SAVE_DIR, f"{image_id}.png")
    image.save(saved_image_path)

    image_summaries[image_id] = {'summary': summary, 'questions': []}
    save_summaries()

    # Base64 encode the saved image
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'summary': summary, 'image_id': image_id, 'image_base64': base64_image})

@app.route('/question', methods=['POST'])
def ask_question():
    """Ask a question about an uploaded image and receive an answer."""
    if 'question' not in request.form:
        return jsonify({'error': 'No question provided'}), 400
    if 'image_id' not in request.form:
        return jsonify({'error': 'No image_id provided'}), 400

    question = request.form['question']
    image_id = request.form['image_id']

    image_data = image_summaries.get(image_id)
    if not image_data:
        return jsonify({'error': 'Image summary not found'}), 404

    summary = image_data['summary']
    image_path = os.path.join(SAVE_DIR, f"{image_id}.png")
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404

    image = Image.open(image_path).convert('RGB')

    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vqa_model.generate(**inputs, max_length=200, num_beams=5, no_repeat_ngram_size=2)
    answer = processor.decode(outputs[0], skip_special_tokens=True)

    detailed_answer = f"Based on the image, {answer}."
    image_summaries[image_id]['questions'].append({
        'question': question,
        'answer': detailed_answer
    })
    save_summaries()

    return jsonify({'answer': detailed_answer, 'question': question, 'summary': summary})

@app.route('/history', methods=['GET'])
def get_history():
    """Get the entire history of uploaded images, summaries, and questions."""
    return jsonify(image_summaries)

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate an image based on a description and return it as a Base64-encoded string."""
    description = request.json.get('description')
    if not description:
        return jsonify({'error': 'No description provided'}), 400

    # Allow optional parameters for customization
    height = request.json.get('height', 512)  # Default height is 512
    width = request.json.get('width', 512)    # Default width is 512
    num_inference_steps = request.json.get('num_inference_steps', 50)  # Default is 50 steps

    data = {
        "inputs": description,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps
    }

    max_retries = 10
    delay = 60

    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}: Sending request to the API...")
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            try:
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                print(f"Error: Unable to process image response: {str(e)}")
                return jsonify({'error': 'Unable to process image from API response'}), 500
            image_file_path = os.path.join(SAVE_DIR, f"generated_{uuid.uuid4().hex}.png")
            image.save(image_file_path)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({
                'message': 'Image generated successfully',
                'image_base64': base64_image, 
                'image_path': image_file_path
            }), 200
        elif response.status_code == 503:
            print(f"Model is currently loading. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        elif response.status_code == 401:
            print("API key expired. Rotating API key...")
            rotate_api_key()  
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return jsonify({'error': response.text}), response.status_code

    return jsonify({'error': 'Max retries exceeded'}), 500

if __name__ == '__main__':
    app.run(debug=True)
