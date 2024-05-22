import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

import os
import csv
import re

from processAudio import get_mp3_info, get_spectrum
samplingRate = 22050

def prepareExamples():

    # Example audio files
    audio_files = ["../raw_data/ma1_FV1_MP3.mp3","../raw_data/ma2_FV1_MP3.mp3","../raw_data/ma3_FV1_MP3.mp3","../raw_data/ma4_FV1_MP3.mp3"]

    for audio_file in audio_files:
        # Load audio data 
        audio, sr = librosa.load(audio_file, sr=samplingRate)
        print('Duration ', librosa.get_duration(y=audio,sr=sr))
        print('Sampling rate', sr)

        base_name = audio_file.split('/')[-1]
        image_fileName = f"examples/{base_name.replace('_MP3.mp3', '.png')}"

        get_spectrum(audio=audio,sr=sr,max_lenght=1,output_file=image_fileName)


def prepareAll(outDir="spectrum_data", csv_file = 'output.csv'):
    # Create a list to store the extracted data
    data = []

    # Loop through the files in the directory
    directory = '../raw_data/'
    maxFiles = -1

    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):

            # Extract info from filename
            match = re.match(r'(\w+)(\d+)_(\w+)_MP3\.mp3', filename)
            if match:
                sound, tone, speaker = match.groups()

                audio, sr = librosa.load(directory+filename, sr=samplingRate)
                info = get_mp3_info(audio,sr)

                img_filename = f"{outDir}/{sound}{tone}_{speaker}.png"
                get_spectrum(audio=audio,sr=sr, max_lenght=1, normalize=False, output_file=img_filename, plot_axis=False)
                
                data.append([filename, img_filename, sound, tone, speaker, info["Duration (seconds)"], info["Frames"], info["Sampling Rate (Hz)"] ])
            else:
                print(f"Error: No match found for filename '{filename}'")

            if maxFiles != -1 and len(data) >= maxFiles:
                break

    # Write data to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'img_filename', 'sound', 'tone', 'speaker', 'Duration (seconds)','Frames','Sampling Rate (Hz)'])
        writer.writerows(data)

    print(f'CSV file "{csv_file}" has been created')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="Prepare example spectrograms")
    parser.add_argument('--csv_file', type=str, default="output.csv", help="CSV file name")
    parser.add_argument('--outDir', type=str, default="spectrum_data", help="Path to img dir")
    args = parser.parse_args()

    if args.test:
        prepareExamples()
    else: prepareAll(args.outDir, args.csv_file)

if __name__ == "__main__":
    main()