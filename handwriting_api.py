# handwriting_api.py - OPTIMIZED FOR RAILWAY
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import base64
import cv2
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Your model architecture
def create_custom_cnn(x):
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='custom_conv1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='custom_pool1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='custom_conv2')(x)
    return x

def create_model(input_shape=(64, 64, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = create_custom_cnn(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu', name='dense1')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, name='dense2')(x)
    x = tf.keras.layers.Reshape((8, 8))(x)
    gru = tf.keras.layers.GRU(64, return_sequences=True, name='gru')(x)
    attention = tf.keras.layers.Attention(name='attention')([gru, gru])
    combined = tf.keras.layers.concatenate([gru, attention])
    flattened = tf.keras.layers.Flatten()(combined)
    output = tf.keras.layers.Dense(2, activation='softmax', name='output')(flattened)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model

# Global model variable
model = None

def load_model_once():
    """Load model only once when the API starts"""
    global model
    if model is None:
        logger.info("Loading model for the first time...")
        model = create_model(input_shape=(64, 64, 3))
        
        # Try to load weights from different possible locations
        weights_paths = [
            'model.weights.h5',
            '/app/model.weights.h5',
            './model.weights.h5'
        ]
        
        weights_loaded = False
        for weights_path in weights_paths:
            if os.path.exists(weights_path):
                logger.info(f"Loading weights from: {weights_path}")
                model.load_weights(weights_path)
                weights_loaded = True
                break
        
        if not weights_loaded:
            raise Exception("Could not find model.weights.h5 in any location")
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        logger.info("Model loaded successfully!")

# Your visualization functions
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

def generate_lime_explanation(model, img_array, class_index):
    explainer = lime_image.LimeImageExplainer()
    def batch_predict(images):
        return model.predict(images)
    explanation = explainer.explain_instance(
        img_array[0].astype('double'),
        batch_predict,
        top_labels=3,
        hide_color=0,
        num_samples=50  # Reduced for faster processing
    )
    temp, mask = explanation.get_image_and_mask(
        class_index,
        positive_only=True,
        num_features=3,
        hide_rest=False
    )
    lime_img = mark_boundaries(temp, mask)
    lime_img = (lime_img * 255).astype(np.uint8)
    lime_img_bgr = cv2.cvtColor(lime_img, cv2.COLOR_RGB2BGR)
    return lime_img_bgr

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

        # Generate visualizations
        saliency_map = generate_saliency_map(model, img_array, class_index)
        overlay_image = overlay_saliency_on_image(saliency_map, img_path)
        lime_img = generate_lime_explanation(model, img_array, class_index)

        # Encode images
        _, saliency_buffer = cv2.imencode('.png', overlay_image)
        _, lime_buffer = cv2.imencode('.png', lime_img)
        
        saliency_base64 = base64.b64encode(saliency_buffer).decode('utf-8')
        lime_base64 = base64.b64encode(lime_buffer).decode('utf-8')

        # Clean up
        os.unlink(img_path)

        return jsonify({
            "filename": file.filename,
            "result": result,
            "confidence": confidence,
            "saliency_base64": saliency_base64,
            "lime_base64": lime_base64
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
    # Pre-load model when starting
    logger.info("Starting Handwriting Analysis API...")
    try:
        load_model_once()
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Server starting on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
