# from flask import Flask, request, render_template, jsonify
# import os
# import torchaudio
# from transformers import pipeline

# app = Flask(__name__)

# # Load Hugging Face audio classification pipeline
# classifier = pipeline("audio-classification", model="mo-thecreator/Deepfake-audio-detection")

# # Set up temporary directory for audio uploads
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TEMP_DIR = os.path.join(BASE_DIR, 'temp')
# os.makedirs(TEMP_DIR, exist_ok=True)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400

#     file = request.files['audio']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Validate file extension
#     valid_extensions = {'.wav', '.mp3', '.flac'}
#     file_ext = os.path.splitext(file.filename)[1].lower()
#     if file_ext not in valid_extensions:
#         return jsonify({'error': f"Invalid file format. Only {', '.join(valid_extensions)} allowed"}), 400

#     # Save file
#     file_path = os.path.join(TEMP_DIR, file.filename)
#     file.save(file_path)

#     try:
#         # Convert to WAV at 16kHz
#         waveform, sr = torchaudio.load(file_path)
#         waveform = torchaudio.functional.resample(waveform, sr, 16000)
#         converted_path = os.path.join(TEMP_DIR, 'converted.wav')
#         torchaudio.save(converted_path, waveform, 16000)

#         # Get predictions from pipeline
#         results = classifier(converted_path)
#         if not results:
#             return jsonify({'error': 'Model returned no results'}), 500

#         # Return top prediction
#         top_result = results[0]
#         return jsonify({
#             'prediction': top_result['label'],
#             'confidence': f"{top_result['score']:.2f}"
#         })

#     except Exception as e:
#         app.logger.error(f"Prediction error: {str(e)}")
#         return jsonify({'error': 'Failed to process audio file'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, render_template, jsonify
import os
import torchaudio
from transformers import pipeline
from utils.preprocess import preprocess_audio

app = Flask(__name__)

# Use absolute path for temp directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# Load the Hugging Face pipeline once
classifier = pipeline("audio-classification", model="mo-thecreator/Deepfake-audio-detection")

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
        converted_path = preprocess_audio(file_path)
        app.logger.info(f"Predicting with file: {converted_path}")
        results = classifier(converted_path)

        # Extract the top prediction
        top_result = max(results, key=lambda x: x['score'])
        label = top_result['label']
        confidence = top_result['score']

        return jsonify({
            'prediction': label,
            'confidence': f"{confidence:.2f}"
        })
    except Exception as e:
        app.logger.error(f"Error during preprocessing/prediction: {str(e)}")
        return jsonify({'error': 'Error processing audio file'}), 500
    finally:
        # Clean up files (uncomment for production)
        # for path in [file_path, converted_path]:
        #     if os.path.exists(path):
        #         try:
        #             os.remove(path)
        #             app.logger.info(f"Deleted file: {path}")
        #         except Exception as e:
        #             app.logger.error(f"Error deleting file: {str(e)}")
        pass

if __name__ == '__main__':
    app.run(debug=True)