import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from processAudio import get_mp3_info, get_spectrum
samplingRate = 22050

def trainModel(csv_file='output.csv', outDir="spectrum_data", modelName='tfModelTones'):

    # Extract file paths and tone labels
    data = pd.read_csv(csv_file)
    file_paths = data['img_filename']
    tones = data['tone']

    num_entries = data.shape[0]
    print(f'Number of entries in output csv: {num_entries}')

    # Encode tone labels
    label_encoder = LabelEncoder()
    tones_encoded = label_encoder.fit_transform(tones)
    num_classes = len(label_encoder.classes_)

    print("File paths:")
    print(file_paths)

    print("\nTones encoded:")
    print(tones_encoded)

    print("\nNumber of classes:")
    print(num_classes)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(file_paths, tones_encoded, test_size=0.1, random_state=42)
    print(f'Number of training events: {X_train.shape[0]}')
    print(f'Number of test events: {X_test.shape[0]}')

    def load_and_preprocess_image(file_path):
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        return img

    # Test with sample image
    #sample_image = load_and_preprocess_image(file_paths[0])
    #plt.figure(figsize=(6, 6))
    #plt.imshow(sample_image)
    #plt.axis('off')
    #plt.show()

    # Load training images
    X_train = np.array([load_and_preprocess_image(fp) for fp in X_train])
    X_test = np.array([load_and_preprocess_image(fp) for fp in X_test])

    # Define model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Train model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    result = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save(modelName)

    # Print performance
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    plt.title('Accuracy')
    plt.plot(result.history['accuracy'], 'r')
    plt.plot(result.history['val_accuracy'], 'b')
    plt.savefig('accuracy.png')

    # Predict for example file
    example_image_file = f"{outDir}/ao4_MV3.png"  
    example_image = load_and_preprocess_image(example_image_file)
    predictions = model.predict(np.expand_dims(example_image, axis=0))
    print(f"Predictions: {predictions}")

    # Get probabilities
    label_probabilities = predictions[0]
    class_names = [ '1', '2', '3', '4']  
    label_probabilities_dict = {class_names[i]: label_probabilities[i] for i in range(len(class_names))}
    predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])[0]
    print(f"Predicted Class: {predicted_class}")

    # Print probabilities
    for class_name, probability in label_probabilities_dict.items():
        print(f"Class: {class_name}, Probability: {probability:.4f}")

    # Get predicted probabilities for test set
    predicted_probabilities = model.predict(X_test)
    true_labels = y_test

    # Compute ROC curve and AUC for each class 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    num_classes = len(class_names)  
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels == i, predicted_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('roc.png')

    audio_file = "../website/temp_audio.wav"
    audio, sr = librosa.load(audio_file, sr=samplingRate)
    info = get_mp3_info(audio,sr)
    print("MP3 File Information:")
    for key, value in info.items():
        print(f"{key}: {value}")

    plt.figure(figsize=(10, 6))
    get_spectrum(audio=audio,output_file="test.png",sr=sr,max_lenght=1,normalize=False,plot_axis=False)
    example_image = load_and_preprocess_image("test.png")
    predictions = model.predict(np.expand_dims(example_image, axis=0))
    highest_index = np.argmax(predictions)
    print(f"Predictions: {predictions}")
    print(f"The highest prediction is for tone: {highest_index+1}")

    yt, index = librosa.effects.trim(audio, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
    info = get_mp3_info(yt,sr)
    print("MP3 Trimmed File Information:")
    for key, value in info.items():
        print(f"{key}: {value}")

    plt.figure(figsize=(10, 6))
    get_spectrum(audio=yt,output_file="test2.png",sr=sr,max_lenght=1,normalize=False,plot_axis=False)
    example_image = load_and_preprocess_image("test2.png")
    predictions = model.predict(np.expand_dims(example_image, axis=0))
    highest_index = np.argmax(predictions)
    print(f"Predictions: {predictions}")
    print(f"The highest prediction is for tone: {highest_index+1}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default="output.csv", help="CSV file name")
    parser.add_argument('--outDir', type=str, default="spectrum_data", help="Path to img dir")
    parser.add_argument('--modelName', type=str, default="tfModelTones", help="Model name")
    args = parser.parse_args()

    trainModel(args.csv_file, args.outDir, args.modelName)

if __name__ == "__main__":
    main()