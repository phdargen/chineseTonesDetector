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

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from trainML.processAudio import make_spectrum
samplingRate = 22050
#modelName = '../trainML/tfModelTones_v3'
#modelName = '../trainML/test'
modelName = '../trainML/tfModelTones'

app = Flask(__name__)
CORS(app, resources={
    r"/api/get_spectrum": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
    r"/api/about": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
    r"/api/random-sound": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
    r"/api/get_file_access": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
    r"/api/upload": {"origins": ["*", "https://chinese-tones-detector.vercel.app/*"]},
})

# AWS S3 setup
s3_client = boto3.client('s3')
bucket_name = 'chinesetonesdata'
#bucket_folder = 'noise_data'
bucket_folder = 'user_data'

# load tf model   
model = tf.keras.models.load_model(modelName)

def load_and_preprocess_image(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
    img = tf.keras.preprocessing.image.img_to_array(img)
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
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.data)
    app.logger.debug('Files: %s', request.files)
    
    try:
        audio_data = request.files["audio"]
        audio_format = audio_data.content_type.split('/')[-1]  

        print(f"audio_data: {str(audio_data)}")
        print(f"audio_format: {str(audio_format)}")

        # Convert audio from webm/mp3 to wav format
        audio = AudioSegment.from_file(audio_data, format="mp3" if audio_format == "mpeg" or audio_format == "mp3" else "mp4" if audio_format == "mp4" else "webm")
        wav_audio = audio.set_frame_rate(samplingRate)  
        wav_audio.export("temp_audio.wav", format="wav")
        
        # Process audio
        audio, sr = librosa.load("temp_audio.wav",sr=samplingRate)
        m_db = make_spectrum(audio, sr, max_lenght=1, normalize=False)
        img = librosa.display.specshow(m_db, x_axis='time',y_axis='mel')
        plt.axis('off')
        plt.savefig('input.png', bbox_inches='tight', pad_inches=0, transparent=True)

        cbar = plt.colorbar(img, format='%+2.0f')
        cbar.set_label('Amplitude (dB)')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.ylim([0, 4096])        
        plt.axis('on')

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        # Create a placeholder spectrum 
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(np.random.rand(10, 100), sr=samplingRate)
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

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or 'filename' not in request.form:
        return jsonify({'error': 'No file or filename part in the request'}), 400
    
    file = request.files['file']
    filename = request.form['filename']
    content_type = request.form['contentType']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    try:
        file_name = f"{bucket_folder}/{filename}"
        s3_client.upload_fileobj(file, bucket_name, file_name, ExtraArgs={'ContentType': content_type})
        file_url = f"https://{bucket_name}.s3.us-east-2.amazonaws.com/{file_name}"
        return jsonify({'url': file_url}), 200
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000)

