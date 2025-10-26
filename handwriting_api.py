# handwriting_api.py - MINIMAL VERSION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# TensorFlow memory optimizations
import tensorflow as tf

# Configure TensorFlow to use less memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.set_soft_device_placement(True)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import base64
import cv2
import tempfile
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Your weights URL
WEIGHTS_URL = "https://www.logifera.com/model.weights.h5"
WEIGHTS_CACHE_PATH = "/tmp/model.weights.h5"

# Simplified model architecture to reduce memory usage
def create_model(input_shape=(64, 64, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Simplified CNN
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model

# Global model variable
model = None
weights_downloaded = False

def download_weights():
    """Download weights from your Hostinger domain"""
    global weights_downloaded
    
    if os.path.exists(WEIGHTS_CACHE_PATH):
        file_size = os.path.getsize(WEIGHTS_CACHE_PATH)
        logger.info(f"Weights found in cache: {file_size / (1024*1024):.2f} MB")
        weights_downloaded = True
        return WEIGHTS_CACHE_PATH
    
    logger.info("Downloading model weights...")
    
    try:
        response = requests.get(WEIGHTS_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        with open(WEIGHTS_CACHE_PATH, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        actual_size = os.path.getsize(WEIGHTS_CACHE_PATH)
        logger.info(f"Download completed: {actual_size / (1024*1024):.2f} MB")
        
        weights_downloaded = True
        return WEIGHTS_CACHE_PATH
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        if os.path.exists(WEIGHTS_CACHE_PATH):
            logger.warning("Using cached weights despite download error")
            return WEIGHTS_CACHE_PATH
        raise Exception(f"Failed to download weights: {str(e)}")

def load_model_once():
    """Load model only once when the API starts"""
    global model
    
    if model is not None:
        return
    
    logger.info("Initializing model...")
    model = create_model(input_shape=(64, 64, 3))
    
    try:
        weights_path = download_weights()
        logger.info("Loading weights into model...")
        model.load_weights(weights_path)
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        logger.info("Model loaded and compiled successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model = None
        raise

# Simple saliency map (without LIME)
def generate_saliency_map(model, img_array, class_idx):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        class_score = predictions[:, class_idx]
    gradients = tape.gradient(class_score, img_tensor)
    gradients = tf.abs(gradients)
    saliency = tf.reduce_max(gradients, axis=-1)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))
    saliency = tf.cast(saliency * 255, tf.uint8)
    return saliency.numpy()

def overlay_saliency_on_image(saliency_map, img_path, alpha=0.4):
    saliency_map = np.squeeze(saliency_map)
    if saliency_map.dtype != np.uint8:
        saliency_map = np.uint8(255 * saliency_map / saliency_map.max())
    img = cv2.imread(img_path)
    saliency_map_resized = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))
    saliency_map_colored = cv2.applyColorMap(saliency_map_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, alpha, saliency_map_colored, 1 - alpha, 0)
    return overlay

@app.route('/analyze-handwriting', methods=['POST'])
def analyze_handwriting():
    try:
        load_model_once()
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            file.save(tmp_file.name)
            img_path = tmp_file.name

        img = load_img(img_path, target_size=(64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        
        result = "Dyslexic" if class_index == 1 else "Non-Dyslexic"

        # Generate saliency map only (skip LIME for now)
        saliency_map = generate_saliency_map(model, img_array, class_index)
        overlay_image = overlay_saliency_on_image(saliency_map, img_path)

        # Encode image
        _, saliency_buffer = cv2.imencode('.png', overlay_image)
        saliency_base64 = base64.b64encode(saliency_buffer).decode('utf-8')

        os.unlink(img_path)

        return jsonify({
            "filename": file.filename,
            "result": result,
            "confidence": confidence,
            "saliency_base64": saliency_base64,
            "lime_base64": ""  # Empty for now
        })

    except Exception as e:
        logger.error(f"Error in analyze_handwriting: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        load_model_once()
        return jsonify({
            "status": "healthy", 
            "model_loaded": model is not None,
            "weights_downloaded": weights_downloaded,
            "ready": True
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "weights_downloaded": weights_downloaded
        }), 500

@app.route('/')
def home():
    return jsonify({
        "message": "Handwriting Analysis API", 
        "status": "running",
        "weights_url": WEIGHTS_URL
    })

if __name__ == '__main__':
    logger.info("Starting Handwriting Analysis API...")
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Server starting on port {port}")
    
    # Try to pre-load but don't fail startup
    try:
        load_model_once()
    except Exception as e:
        logger.warning(f"Initial model load failed: {str(e)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
