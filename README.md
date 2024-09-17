# Flask Image Processing System

This Flask-based image processing system facilitates image captioning, question-answering about images, and image generation using pre-trained models and the Hugging Face API. The system integrates the BLIP (Bootstrapped Language Image Pre-training) model for visual question answering (VQA) and image captioning, and uses the Stable Diffusion model for image generation. Users can upload an image, ask questions about it, generate captions, and create new images from text descriptions. The system handles large image uploads, tracks session history, and rotates API keys for sustained usage.

## Prerequisites

- **Flask** - [instructions](https://flask.palletsprojects.com/en/2.1.x/installation/)
- **Torch** - [instructions](https://pytorch.org/get-started/locally/)
- **Transformers** - [instructions](https://huggingface.co/docs/transformers/installation)
- **Pillow (PIL)** - [instructions](https://pillow.readthedocs.io/en/stable/installation.html)
- **Requests** - [instructions](https://docs.python-requests.org/en/master/user/install/)
- **Werkzeug** - [instructions](https://werkzeug.palletsprojects.com/en/2.0.x/installation/)
- **Base64** - Comes with Python

## Usage

### Upload an Image
Upload an image in PNG, JPG, or JPEG format, generate a caption, and save the image along with its caption.

- **Endpoint:** `/upload`
- **Method:** `POST`
- **Parameters:**
  - `image`: The image file to be uploaded.
- **Response:**
  - A generated caption for the image.
  - The Base64-encoded image data.
  - A unique `image_id` for further interactions.

### Ask a Question About an Image
Submit a question related to a previously uploaded image and receive an answer.

- **Endpoint:** `/question`
- **Method:** `POST`
- **Parameters:**
  - `question`: A textual question about the image.
  - `image_id`: The ID of the image (obtained from the upload response).
- **Response:**
  - The generated answer to the question.
  - The image caption and history.

### Generate an Image from a Description
Generate an image based on a text description using Stable Diffusion.

- **Endpoint:** `/generate`
- **Method:** `POST`
- **Parameters:**
  - `description`: Text description of the image.
  - Optional: `height`, `width`, `num_inference_steps` for customizing the generation.
- **Response:**
  - The Base64-encoded generated image.
  - The image file path.

### View Image Summaries and Question History
View the history of uploaded images, their captions, and the questions asked about them.

- **Endpoint:** `/history`
- **Method:** `GET`
- **Response:**
  - A JSON object containing image summaries and questions.

## Model Setup and Key Components

### Pre-trained Models
There is no explicit model training in this system, but pre-trained models are loaded on startup:
- **Visual Question Answering (VQA) Model:** `Salesforce/blip-vqa-base`
- **Image Captioning Model:** `Salesforce/blip-image-captioning-base`

These models are initialized using PyTorch and utilize the GPU if available. Hugging Face API keys are used for image generation.

### Key Components
- **File Upload Validation:** Only PNG, JPG, and JPEG formats are accepted, with a maximum file size of 16 MB.
- **API Key Rotation:** Hugging Face API keys are rotated if they expire or become rate-limited, ensuring continuous operation.
- **Session Handling:** Flask sessions store secret keys for security, and uploaded images and generated results are saved persistently.
- **Image Processing:** Image files are handled using Pillow, converted to Base64 format for API responses.
- **Error Handling:** Robust error handling for invalid file types, missing parameters, model loading issues, and API failures.

## Results
The system efficiently handles:
- Generating captions and answering questions about uploaded images.
- Creating new images from textual descriptions.
- Managing image and question histories for tracking user interactions.

Generated images, captions, and answers to questions are saved and can be retrieved using the `image_id`.

## References
- **BLIP:** [Bootstrapped Language Image Pre-training](https://arxiv.org/abs/2201.12086)
- **Hugging Face:** [Stable Diffusion API](https://huggingface.co/)
- **Flask Documentation:** [Flask](https://flask.palletsprojects.com/)


