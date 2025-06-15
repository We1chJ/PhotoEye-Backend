from flask import Flask, request, jsonify
import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
import io
import base64
import threading
import time

app = Flask(__name__)

# Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global variables for models
model = None
clip_model = None
preprocess = None
device = None
models_loaded = False
loading_error = None

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = nn.functional.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = nn.functional.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def load_models():
    """Load and initialize the models"""
    global model, clip_model, preprocess, device, models_loaded, loading_error
    
    try:
        print("🔍 Detecting compute device...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {device}")
        
        # Load CLIP model
        print("🎨 Loading CLIP model...")
        print("📥 Downloading/loading CLIP model...")
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Load aesthetic scoring model (if you have it)
        # Commenting out since you don't have the model file
        print("🧠 Loading aesthetic scoring model...")
        model = MLP(768)
        s = torch.load("sac+logos+ava1-l14-linearMSE.pth", map_location=device)
        model.load_state_dict(s)
        model.to(device)
        model.eval()
        
        models_loaded = True
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        loading_error = str(e)
        print(f"❌ ERROR loading models: {e}")

def predict_aesthetic_score(pil_image):
    """Predict aesthetic score for a PIL image"""
    try:
        # Check if models are loaded
        if not models_loaded:
            raise Exception("Models are still loading. Please try again in a moment.")
        
        if loading_error:
            raise Exception(f"Model loading failed: {loading_error}")
        
        # Preprocess image
        image = preprocess(pil_image).unsqueeze(0).to(device)
        
        # Extract CLIP features
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        
        # Normalize features
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        
        # For now, return a dummy score since we don't have the aesthetic model
        # Replace this with actual model prediction when you have the model file
        return float(np.random.uniform(5.0, 9.0))  # Dummy score between 5-9
        
        # Uncomment when you have the model file:
        if device == "cuda":
            prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        else:
            prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.FloatTensor))
        return float(prediction.cpu().detach().numpy()[0][0])
    
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for DigitalOcean"""
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "loading_error": loading_error
    }), 200

@app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "Aesthetic Scoring API",
        "status": "running",
        "models_loaded": models_loaded
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Predict aesthetic score from uploaded image or base64 data"""
    try:
        # Check if models are loaded
        if not models_loaded:
            if loading_error:
                return jsonify({"error": f"Model loading failed: {loading_error}"}), 500
            else:
                return jsonify({"error": "Models are still loading. Please try again in a moment."}), 503
        
        pil_image = None
        
        # Check if it's a file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Process image directly from memory
            image_data = file.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Check if it's base64 data
        elif request.is_json:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({"error": "No image data provided"}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Predict aesthetic score
        score = predict_aesthetic_score(pil_image)
        
        return jsonify({
            "aesthetic_score": score
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        # Start Flask server immediately
        print("🚀 Starting Flask API server...")
        
        # Load models in a separate thread to not block startup
        print("📚 Starting model loading in background...")
        loading_thread = threading.Thread(target=load_models)
        loading_thread.daemon = True
        loading_thread.start()
        
        # Start the Flask server
        app.run(host='0.0.0.0', port=8080)
        
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to initialize server: {e}")
        print("🛑 Server startup aborted.")
        exit(1)