# app.py

from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_audio

app = Flask(__name__)
MODEL_PATH = 'model/cnn_model11111.h5'

# Load the model once
model = load_model(MODEL_PATH)
class_names = ['bonafide', 'spoof']  # Ensure the same order used during training

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(file_path)

    # Preprocess and predict
    try:
        X_input = preprocess_audio(file_path)
        prediction = model.predict(X_input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = class_names[predicted_class]
        confidence = float(np.max(prediction))

        return jsonify({
            'prediction': label,
            'confidence': f"{confidence:.2f}"
        })
    finally:
        os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
