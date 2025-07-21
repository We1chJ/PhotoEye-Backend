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

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def project_1024_to_768(embeddings):
    """
    Simple projection methods to convert 1024-dim to 768-dim
    """
    # Method 1: Truncate (take first 768 dimensions)
    # return embeddings[:, :768]
    
    # Method 2: PCA-like projection (learned linear transformation)
    # This would need to be trained, but here's a random projection
    if not hasattr(project_1024_to_768, 'projection_matrix'):
        # Create a fixed random projection matrix (you'd train this properly)
        torch.manual_seed(42)  # For reproducibility
        project_1024_to_768.projection_matrix = torch.randn(1024, 768) * 0.1
    
    projection_matrix = project_1024_to_768.projection_matrix.to(embeddings.device)
    return torch.mm(embeddings, projection_matrix)
    
    # Method 3: Average pooling groups of dimensions
    # embeddings_reshaped = embeddings.view(-1, 4, 256)  # Group every 4 dimensions
    # return embeddings_reshaped.mean(dim=1)[:, :768]

def load_models():
    global model, clip_model, preprocess, device, models_loaded, loading_error
    try:
        print("🔍 Detecting compute device...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {device}")
        
        print("🎨 Loading CLIP RN50 model...")
        clip_model, preprocess = clip.load("RN50", device=device)
        
        print("🧠 Loading aesthetic scoring model...")
        model = MLP(768)  # Still expects 768-dim input
        s = torch.load("sac+logos+ava1-l14-linearMSE.pth", map_location=device)
        model.load_state_dict(s)
        model.to(device)
        model.eval()
        
        models_loaded = True
        print("✅ Models loaded successfully!")
        print("📝 Using simple projection from 1024D → 768D")
    except Exception as e:
        loading_error = str(e)
        print(f"❌ ERROR loading models: {e}")

# Start loading models in background
print("🚀 Starting Flask API server...")
print("📚 Starting model loading in background...")
loading_thread = threading.Thread(target=load_models)
loading_thread.daemon = True
loading_thread.start()

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Aesthetic Scoring API with RN50 + Projection",
        "status": "running",
        "models_loaded": models_loaded
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not models_loaded:
            if loading_error:
                return jsonify({"error": f"Model loading failed: {loading_error}"}), 500
            else:
                return jsonify({"error": "Models are still loading. Please try again in a moment."}), 503

        pil_image = None

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            image_data = file.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        elif request.is_json:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({"error": "No image data provided"}), 400
            image_data = base64.b64decode(data['image'])
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        else:
            return jsonify({"error": "No image provided"}), 400

        score = predict_aesthetic_score(pil_image)
        return jsonify({"aesthetic_score": score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def predict_aesthetic_score(pil_image):
    try:
        if not models_loaded:
            raise Exception("Models are still loading. Please try again in a moment.")
        if loading_error:
            raise Exception(f"Model loading failed: {loading_error}")
        
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            # Get RN50 embeddings (1024-dim)
            image_features = clip_model.encode_image(image)
            
            # Project to 768-dim
            projected_features = project_1024_to_768(image_features)
            
        im_emb_arr = normalized(projected_features.cpu().detach().numpy())
        
        if device == "cuda":
            prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        else:
            prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.FloatTensor))
        
        return float(prediction.cpu().detach().numpy()[0][0])
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)