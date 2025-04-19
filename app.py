from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_audio

app = Flask(__name__)

# Use absolute path for temp directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'cnn_model11111.h5')

# Load the model once
model = load_model(MODEL_PATH)
class_names = ['bonafide', 'spoof']  # Ensure the same order used during training

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        app.logger.error("No audio file provided in request")
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        app.logger.error("No file selected")
        return jsonify({'error': 'No selected file'}), 400

    # Validate file extension
    valid_extensions = {'.wav', '.mp3', '.flac'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in valid_extensions:
        app.logger.error(f"Invalid file extension: {file_ext}")
        return jsonify({'error': f"Invalid file format. Only {', '.join(valid_extensions)} are allowed"}), 400

    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Save file with secure filename
    file_path = os.path.join(TEMP_DIR, file.filename)
    app.logger.info(f"Saving file to: {file_path}")
    try:
        file.save(file_path)
        if not os.path.exists(file_path):
            app.logger.error(f"File was not saved: {file_path}")
            return jsonify({'error': 'Failed to save file'}), 500
    except Exception as e:
        app.logger.error(f"Error saving file: {str(e)}")
        return jsonify({'error': 'Error saving file'}), 500

    # Preprocess and predict
    try:
        app.logger.info(f"Preprocessing file: {file_path}")
        X_input = preprocess_audio(file_path)
        prediction = model.predict(X_input)
        print("prediction:  "+str(prediction))
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = class_names[predicted_class]
        confidence = float(np.max(prediction))

        return jsonify({
            'prediction': label,
            'confidence': f"{confidence:.2f}"
        })
    except Exception as e:
        app.logger.error(f"Error during preprocessing/prediction: {str(e)}")
        return jsonify({'error': 'Error processing audio file'}), 500
    finally:
        # Comment out file deletion for debugging
        # if os.path.exists(file_path):
        #     try:
        #         os.remove(file_path)
        #         app.logger.info(f"Deleted file: {file_path}")
        #     except Exception as e:
        #         app.logger.error(f"Error deleting file: {str(e)}")
        pass

if __name__ == '__main__':
    app.run(debug=True)