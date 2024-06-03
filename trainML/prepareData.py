import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

import os
import csv
import re
import glob

from processAudio import get_mp3_info, get_spectrum, augment_audio, augment_mel_spectrogram
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


def prepareAll(outDir="spectrum_data", csv_file = 'output.csv', num_augmentations=0):

    os.makedirs(outDir, exist_ok=True)

    # Create a list to store the extracted data
    data = []
    data_augmented = []

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

                # Augment the audio data
                for j in range(num_augmentations):
                    augmented_audio = augment_audio(audio, sr)
                    augmented_filename = f"{outDir}/{sound}{tone}_{speaker}_aug{j+1}"

                    augmented_img_filename = f"{outDir}/{sound}{tone}_{speaker}_aug{j+1}.png"
                    get_spectrum(audio=augmented_audio, sr=sr, max_lenght=1, normalize=False, output_file=augmented_img_filename, plot_axis=False)

                    augmented_info = get_mp3_info(augmented_audio, sr)
                    data_augmented.append([augmented_filename, augmented_img_filename, sound, tone, speaker, augmented_info["Duration (seconds)"], augmented_info["Frames"], augmented_info["Sampling Rate (Hz)"]])

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

    if(num_augmentations>0):
        augmented_csv_file = csv_file.replace('.csv', '_augmented.csv')
        with open(augmented_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'img_filename', 'sound', 'tone', 'speaker', 'Duration (seconds)','Frames','Sampling Rate (Hz)'])
            writer.writerows(data_augmented)
        print(f'CSV file "{augmented_csv_file}" has been created')

def prepareNoise(inDir="noise_data", outDir="spectrum_noise", csv_file='noise_output.csv', num_augmentations=0):

    # Ensure output directory exists
    os.makedirs(outDir, exist_ok=True)

    # List to store extracted data
    data = []
    data_augmented = []

    # Find all .webm files in directory
    webm_files = glob.glob(os.path.join(inDir, '*'))

    for i, webm_file in enumerate(webm_files):

        # Load audio data 
        audio, sr = librosa.load(webm_file, sr=samplingRate)

        # Generate and save the spectrogram
        img_filename = f"{outDir}/noise_sample_{i+1}.png"
        get_spectrum(audio=audio, sr=samplingRate, max_lenght=1, normalize=False, output_file=img_filename, plot_axis=False)

        # Collect noise sample info
        info = get_mp3_info(audio, sr)
        data.append([webm_file, img_filename, 'noise', 5, 'none', info["Duration (seconds)"], info["Frames"], info["Sampling Rate (Hz)"]])

        # Augment the audio data
        for j in range(num_augmentations):
            augmented_audio = augment_audio(audio, sr)
            augmented_filename = f"augmented_noise_sample_{i+1}_{j+1}"

            augmented_img_filename = f"{outDir}/augmented_noise_sample_{i+1}_{j+1}.png"
            get_spectrum(audio=augmented_audio, sr=sr, max_lenght=1, normalize=False, output_file=augmented_img_filename, plot_axis=False)

            augmented_info = get_mp3_info(augmented_audio, sr)
            data_augmented.append([augmented_filename, augmented_img_filename, 'augmented_noise', 5, 'none', augmented_info["Duration (seconds)"], augmented_info["Frames"], augmented_info["Sampling Rate (Hz)"]])

    # Create random combinations of noise samples
    num_combinations=len(webm_files)*10
    for i in range(num_combinations):
        # Randomly select noise samples to combine
        file1, file2, file3, file4 = np.random.choice(webm_files, 4, replace=False)
        audio1, sr1 = librosa.load(file1, sr=samplingRate)
        audio2, sr2 = librosa.load(file2, sr=samplingRate)
        audio3, sr3 = librosa.load(file3, sr=samplingRate)
        audio4, sr4 = librosa.load(file4, sr=samplingRate)

        # Ensure audios have same length
        min_len = min(len(audio1), len(audio2), len(audio3), len(audio4))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        audio3 = audio3[:min_len]
        audio4 = audio4[:min_len]

        # Combine audio files
        random_number = np.random.randint(0, 10)
        combined_audio = audio1 + audio2
        if (random_number>5): combined_audio += audio3
        if (random_number>8): combined_audio += audio4

        # Generate and save the spectrogram
        combined_img_filename = f"{outDir}/combined_noise_sample_{i+1}.png"
        get_spectrum(audio=combined_audio, sr=samplingRate, max_lenght=1, normalize=False, output_file=combined_img_filename, plot_axis=False)

        # Collect combined noise sample info
        combined_info = get_mp3_info(combined_audio, samplingRate)
        data.append([f"combined_noise_sample_{i+1}", combined_img_filename, 'combined_noise', 5, 'none', combined_info["Duration (seconds)"], combined_info["Frames"], combined_info["Sampling Rate (Hz)"]])

        # Augment the audio data
        for j in range(num_augmentations):
            augmented_audio = augment_audio(combined_audio, sr)
            augmented_filename = f"augmented_combined_noise_sample_{i+1}_{j+1}"

            augmented_img_filename = f"{outDir}/augmented_combined_noise_sample_{i+1}_{j+1}.png"
            get_spectrum(audio=augmented_audio, sr=sr, max_lenght=1, normalize=False, output_file=augmented_img_filename, plot_axis=False)

            augmented_info = get_mp3_info(augmented_audio, sr)
            data_augmented.append([augmented_filename, augmented_img_filename, 'augmented_combined_noise', 5, 'none', augmented_info["Duration (seconds)"], augmented_info["Frames"], augmented_info["Sampling Rate (Hz)"]])

    # Write data to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'img_filename', 'sound', 'tone', 'speaker', 'Duration (seconds)', 'Frames', 'Sampling Rate (Hz)'])
        writer.writerows(data)
    print(f'CSV file "{csv_file}" has been created with {len(webm_files)} noise samples')

    if(num_augmentations>0):
        augmented_csv_file = csv_file.replace('.csv', '_augmented.csv')
        with open(augmented_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'img_filename', 'sound', 'tone', 'speaker', 'Duration (seconds)','Frames','Sampling Rate (Hz)'])
            writer.writerows(data_augmented)
        print(f'CSV file "{augmented_csv_file}" has been created')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="Prepare example spectrograms")

    parser.add_argument('--noise', action='store_true', help="Prepare noise spectrograms")
    parser.add_argument('--noise_csv_file', type=str, default="noise.csv", help="Noise CSV file name")
    parser.add_argument('--noise_inDir', type=str, default="noise_data", help="Noise data dir")

    parser.add_argument('--csv_file', type=str, default="output.csv", help="CSV file name")
    parser.add_argument('--outDir', type=str, default="spectrum_data", help="Path to img dir")

    parser.add_argument('--num_augmentations', type=int, default=0, help="Number of data augmentations")

    args = parser.parse_args()

    if args.test:
        prepareExamples()
    elif args.noise:
        prepareNoise(args.noise_inDir, args.outDir, args.noise_csv_file, args.num_augmentations)
    else: prepareAll(args.outDir, args.csv_file, args.num_augmentations)

if __name__ == "__main__":
    main()