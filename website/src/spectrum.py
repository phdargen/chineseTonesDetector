from flask import Flask, request, jsonify
from flask_cors import CORS  
import librosa.display
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')  # (non-interactive)
import matplotlib.pyplot as plt
from scipy import signal
from pydub import AudioSegment
from pydub.playback import play
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/get_spectrum": {"origins": "http://localhost:3000"}}) 
#CORS(app)

@app.route("/get_spectrum", methods=["POST"])
def get_spectrum():
    
    try:
        audio_data = request.files["audio"]
        
        # Convert audio from webm to wav format
        audio = AudioSegment.from_file(audio_data, format="webm")
        wav_audio = audio.set_frame_rate(44100)  
        wav_audio.export("temp_audio.wav", format="wav")
        
        # Process audio
        audio, sr = librosa.load("temp_audio.wav")
        
        # Compute the short-time Fourier transform (STFT)
        D = librosa.stft(audio)

        # Compute the magnitude spectrum
        magnitude = np.abs(D)

        # Convert to dB scale
        log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Plot the spectrum
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(log_magnitude, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        
        # Create a placeholder spectrum plot for testing
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(np.random.rand(10, 100), sr=44100)
        plt.axis('off')

    ## Save plot
    plt.savefig('spectrum.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Encode the spectrum plot as base64
    with open('spectrum.png', 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return jsonify({"spectrum": img_base64})

if __name__ == "__main__":
    app.run(debug=True)
