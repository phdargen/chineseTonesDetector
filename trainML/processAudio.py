import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import butter, sosfilt

import os
import csv
import re

def time_stretch(audio, rate=1.25):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio
    
def time_shift(audio, shift_max=0.2, shift_direction='both', roll=False):
    shift = int(len(audio) * shift_max * np.random.rand())
    
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        shift = -shift if np.random.rand() > 0.5 else shift
    
    if roll:
        return np.roll(audio, shift)
    else:
        shifted_audio = np.zeros_like(audio)
        if shift > 0:
            shifted_audio[shift:] = audio[:-shift]
        elif shift < 0:
            shift = -shift
            shifted_audio[:-shift] = audio[shift:]
        else:
            shifted_audio = audio  
        return shifted_audio
    
def change_volume(audio, gain=1.5):
    return audio * gain

def spec_augment(mel_spectrogram, freq_mask_param=10, time_mask_param=10):
    mel_spectrogram = mel_spectrogram.copy()
    num_mel_channels = mel_spectrogram.shape[0]

    # Frequency masking
    f = np.random.uniform(low=0.0, high=freq_mask_param)
    f0 = int(np.random.uniform(low=0.0, high=num_mel_channels - f))
    mel_spectrogram[f0:f0 + int(f), :] = 0

    # Time masking
    t = np.random.uniform(low=0.0, high=time_mask_param)
    t0 = int(np.random.uniform(low=0.0, high=mel_spectrogram.shape[1] - t))
    mel_spectrogram[:, t0:t0 + int(t)] = 0

    return mel_spectrogram

def augment_audio(audio, sr):
    new_audio = audio
    new_audio = time_stretch(audio=new_audio, rate=np.random.uniform(0.8, 1.2))
    new_audio = time_shift(audio=new_audio, shift_max=0.2, shift_direction='both', roll=False)
    new_audio = pitch_shift(audio=new_audio, sr=sr, n_steps=np.random.randint(-5, 5))
    new_audio = add_noise(audio=new_audio, noise_factor=np.random.uniform(0.0, 0.02))
    return new_audio

def augment_mel_spectrogram(mel_spectrogram, freq_mask_param=10, time_mask_param=10):
    mel_spectrogram = spec_augment(mel_spectrogram, freq_mask_param, time_mask_param)
    return mel_spectrogram


def apply_filter(audio, sr, low_freq=32, high_freq=4096):
    # Design a bandpass filter
    sos = butter(10, [low_freq, high_freq], btype='bandpass', fs=sr, output='sos')
    # Apply the filter
    filtered_audio = sosfilt(sos, audio)
    return filtered_audio

def dynamic_range_compression(audio):
    return librosa.effects.preemphasis(y=audio)

def pad_or_truncate(audio, target_length):
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), 'constant')
    else:
        return audio
    

def get_mp3_info(y,sr):

    # Get sampling rate
    sampling_rate = sr

    # Get frames
    total_samples = len(y)

    # Get duration
    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "Sampling Rate (Hz)": sampling_rate,
        "Frames": total_samples,
        "Duration (seconds)": duration
    }

def make_spectrum(audio, sr, max_lenght=1, normalize=False):

    # Trim silence
    audio, index = librosa.effects.trim(audio, top_db=30, ref=np.max, frame_length=256, hop_length=64)
    #print(librosa.get_duration(y=audio,sr=sr))

    # Filter frequencies
    audio = apply_filter(audio, sr)

    # Use consistent time lenght for audio
    audio = pad_or_truncate(audio, sr*max_lenght)

    # Create spectrogram
    m = librosa.feature.melspectrogram(y=audio,sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    m_db = librosa.amplitude_to_db(m, ref=np.max)
    #m_db = librosa.power_to_db(m, ref=np.max)

    # Normalize spectrogram
    if(normalize): m_db = (m_db - np.mean(m_db)) / np.std(m_db)

    return m_db

def get_spectrum(audio, sr, max_lenght=1, normalize=False, output_file=None, plot_axis=True):

    #audio = time_stretch(audio=audio, rate=2)
    #audio = pitch_shift(audio, sr, 2)
    #audio = add_noise(audio, noise_factor=0.01)
    #audio = time_shift(audio, shift_max=0.2, shift_direction='both')
    #audio = change_volume(audio,100)
    #audio = dynamic_range_compression(audio)
        
    m_db = make_spectrum(audio, sr, max_lenght, normalize)
    img = librosa.display.specshow(m_db, x_axis='time',y_axis='mel')

    plt.ylim([0, 4096])
    if(plot_axis):
        cbar = plt.colorbar(img, format='%+2.0f')
        cbar.set_label('Amplitude (dB)')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
    else:plt.axis('off')
    
    plt.tight_layout()
    if output_file: plt.savefig(output_file, bbox_inches='tight', pad_inches=0, transparent=True)
    else: plt.show()    
    plt.close()