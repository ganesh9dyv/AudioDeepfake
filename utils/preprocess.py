# utils/preprocess.py

import numpy as np
import librosa

def preprocess_audio(file_path, sample_rate=16000, duration=3, n_mels=128, max_time_steps=109):
    audio, _ = librosa.load(file_path, sr=sample_rate, duration=duration)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure consistent shape
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    mel_spectrogram = mel_spectrogram.reshape(1, 128, 109, 1)  # Add batch and channel dimension
    return mel_spectrogram
