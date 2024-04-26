from flask import Flask, request, jsonify, abort
from flask_cors import CORS  
import librosa.display
import numpy as np
import base64
import matplotlib
matplotlib.use('Agg')  # (non-interactive)
import matplotlib.pyplot as plt
from pydub import AudioSegment
import tensorflow as tf

import boto3
import random
import re

app = Flask(__name__)
CORS(app, resources={
    r"/api/get_spectrum": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
    r"/api/about": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
    r"/api/random-sound": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
    r"/api/get_file_access": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
})

# AWS S3 setup
s3_client = boto3.client('s3')
bucket_name = 'chinesetonesdata'

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
    return 'Server is running and serving files from AWS S3'

@app.route('/api/about')
def about():
    return 'About'

@app.route("/api/get_spectrum", methods=["POST"])
def get_spectrum():
    try:
        audio_data = request.files["audio"]
        audio_format = audio_data.content_type.split('/')[-1]  

        print(f"audio_data: {str(audio_data)}")
        print(f"audio_format: {str(audio_format)}")

        # Convert audio from webm/mp3 to wav format
        audio = AudioSegment.from_file(audio_data, format="mp3" if audio_format == "mpeg" or audio_format == "mp3" else "webm")
        wav_audio = audio.set_frame_rate(44100)  
        wav_audio.export("temp_audio.wav", format="wav")
        
        # Process audio
        #audio, sr = librosa.load(audio_data)
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

@app.route('/api/random-sound')
def random_sound():
    try:
        # Retrieve list of files in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='raw_data/')
        files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.mp3')]

        if not files:
            abort(404, 'No valid sound files found')

        # Select a random file
        selected_file_key = random.choice(files)
        file_name = selected_file_key.split('/')[-1]

        # Extract sound, tone, speaker from file name
        match = re.match(r'(\w+)(\d+)_(\w+)_MP3\.mp3', file_name)
        if match:
            sound, tone, speaker = match.groups()
            # Generate a pre-signed URL for temporary access
            url = s3_client.generate_presigned_url('get_object',
                                                   Params={'Bucket': bucket_name, 'Key': selected_file_key},
                                                   ExpiresIn=3600)  # URL expires in 1 hour
            
            print(url)

            return jsonify({
                "sound": sound,
                "tone": tone,
                "speaker": speaker,
                "url": url
            })
        else:
            abort(404, 'Failed to parse the selected sound file')

    except Exception as e:
        print(f"Error accessing AWS S3: {str(e)}")
        abort(500, str(e))

@app.route('/api/get_file_access')
def get_file_access():
    # Retrieve the filename from query parameters
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename parameter is required'}), 400

    try:
        file_key = f'raw_data/{filename}'

        # Generate a presigned URL for the requested file
        presigned_url = s3_client.generate_presigned_url('get_object',
                                                         Params={'Bucket': bucket_name, 'Key': file_key},
                                                         ExpiresIn=3600)  # URL expires in 1 hour

        return jsonify({'file_name': filename, 'url': presigned_url})

    except Exception as e:
        print(f"Error generating presigned URL: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

