# handwriting_api.py - UPDATED FOR LIGHTWEIGHT MODEL
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
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

# Your weights URL - NOW POINTING TO SMALL MODEL
WEIGHTS_URL = "https://www.logifera.com/small_model.weights.h5"
WEIGHTS_CACHE_PATH = "/tmp/model.weights.h5"

# LIGHTWEIGHT architecture that matches your small_model.weights.h5
def create_model(input_shape=(64, 64, 3)):
    inputs = Input(shape=input_shape)
    
    # Lightweight CNN - EXACTLY as in CNN_model.py create_lightweight_model
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='small_conv1')(inputs)
    x = MaxPooling2D((2, 2), name='small_pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='small_conv2')(x)
    x = MaxPooling2D((2, 2), name='small_pool2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='small_conv3')(x)
    x = MaxPooling2D((2, 2), name='small_pool3')(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='small_dense1')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', name='small_dense2')(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='softmax', name='small_output')(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model

# Global model variable
model = None

def download_weights():
    """Download weights from your Hostinger domain"""
    if os.path.exists(WEIGHTS_CACHE_PATH):
        file_size = os.path.getsize(WEIGHTS_CACHE_PATH)
        logger.info(f"Weights found in cache: {file_size / (1024*1024):.2f} MB")
        return WEIGHTS_CACHE_PATH
    
    logger.info("Downloading model weights...")
    
    try:
        response = requests.get(WEIGHTS_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        with open(WEIGHTS_CACHE_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        actual_size = os.path.getsize(WEIGHTS_CACHE_PATH)
        logger.info(f"Download completed: {actual_size / (1024*1024):.2f} MB")
        return WEIGHTS_CACHE_PATH
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        if os.path.exists(WEIGHTS_CACHE_PATH):
            return WEIGHTS_CACHE_PATH
        raise

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

def generate_saliency_map(model, img_array, class_idx):
    """Generate saliency map for visualization"""
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
    """Overlay saliency map on original image"""
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

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            file.save(tmp_file.name)
            img_path = tmp_file.name

        # Process image
        img = load_img(img_path, target_size=(64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        
        result = "Dyslexic" if class_index == 1 else "Non-Dyslexic"

        # Generate saliency map
        saliency_map = generate_saliency_map(model, img_array, class_index)
        overlay_image = overlay_saliency_on_image(saliency_map, img_path)

        # Encode image
        _, saliency_buffer = cv2.imencode('.png', overlay_image)
        saliency_base64 = base64.b64encode(saliency_buffer).decode('utf-8')

        # Clean up
        os.unlink(img_path)

        return jsonify({
            "filename": file.filename,
            "result": result,
            "confidence": confidence,
            "saliency_base64": saliency_base64,
            "lime_base64": ""  # Empty since we removed LIME
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
            "ready": True
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/')
def home():
    return jsonify({"message": "Handwriting Analysis API", "status": "running"})

if __name__ == '__main__':
    logger.info("Starting Handwriting Analysis API...")
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Server starting on port {port}")
    
    # Pre-load model
    try:
        load_model_once()
    except Exception as e:
        logger.warning(f"Initial model load failed: {str(e)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
