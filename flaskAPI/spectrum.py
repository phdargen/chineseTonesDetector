from flask import Flask, request, jsonify
from flask_cors import CORS  
import librosa.display
import numpy as np
import base64
import matplotlib
matplotlib.use('Agg')  # (non-interactive)
import matplotlib.pyplot as plt
from pydub import AudioSegment
import tensorflow as tf

app = Flask(__name__)
CORS(app, resources={
    r"/get_spectrum": {"origins": ["http://localhost:3000/*", "https://chinese-tones-detector.vercel.app/*"]},
    r"/about": {"origins": ["http://localhost:3000/*", "https://chinese-tones-detector.vercel.app/*"]}
})

# load tf model   
model = tf.keras.models.load_model('../prepareData/tfModelTones')

def load_and_preprocess_image(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
    img = tf.keras.preprocessing.image.img_to_array(img)
    #img = datagen.random_transform(img)
    img = img / 255.0
    return img

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

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
        audio, index = librosa.effects.trim(audio, top_db=30, ref=np.max, frame_length=2048, hop_length=512)

        # Compute the short-time Fourier transform (STFT)
        D = librosa.stft(audio)

        # Compute the magnitude spectrum
        magnitude = np.abs(D)

        # Convert to dB scale
        log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Plot the spectrum
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(log_magnitude, sr=sr, x_axis='time', y_axis='log')
        plt.axis('off')
        # plt.ylim([32, 8192])
        plt.savefig('input.png', bbox_inches='tight', pad_inches=0, transparent=True)

        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.axis('on')

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        # Create a placeholder spectrum 
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(np.random.rand(10, 100), sr=44100)
        plt.axis('off')

    ## Save plot
    plt.savefig('spectrum.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Encode the spectrum plot as base64
    with open('spectrum.png', 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    ## Predict tone
    # Predict for example file
    input_image = load_and_preprocess_image('input.png')
    predictions = model.predict(np.expand_dims(input_image, axis=0))
    highest_index = np.argmax(predictions)
    print(f"Predictions: {predictions}")
    print(f"The highest prediction is for tone: {highest_index+1}")

    response_data = {
        "spectrum": img_base64,
        "prediction": int(highest_index)+1  
    }

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
