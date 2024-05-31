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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from processAudio import get_mp3_info, get_spectrum
samplingRate = 22050
EPOCHS = 10
BATCH_SIZE = 128

def plot_learning_curves(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('accuracy.png')

    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss.png')

def addNoiseData(data, noise_csv, N):
    noise_data = pd.read_csv(noise_csv)
    if len(noise_data) > N:
        noise_data = noise_data.sample(n=N, random_state=42)  # Sample N noise entries if more than N exist
    return pd.concat([data, noise_data], ignore_index=True)

def trainModel(csv_file='output.csv', outDir="spectrum_data", modelName='tfModelTones', addNoise=False, noise_csv='noise.csv'):

    # Extract file paths and tone labels
    data = pd.read_csv(csv_file)
    num_entries = data.shape[0]
    print(f'Number of entries in output csv: {num_entries}')

    # Add noise data
    if(addNoise):
        data = addNoiseData(data, noise_csv, int(num_entries/4))
        print(f'Number of entries after adding noise: {data.shape[0]}')

    file_paths = data['img_filename']
    tones = data['tone']
    speakers = data['speaker']

    # Encode tone labels
    label_encoder = LabelEncoder()
    tones_encoded = label_encoder.fit_transform(tones)
    num_classes = len(label_encoder.classes_)

    print("File paths:", file_paths)
    print("\nTones encoded:", tones_encoded)
    print("\nNumber of classes:", num_classes)

    # Split dataset into train+validation and test sets
    X_temp, X_test, y_temp, y_test, speakers_temp, speakers_test = train_test_split(file_paths, tones_encoded, speakers, test_size=0.1, random_state=42)
    # Split train+validation into separate train and validation sets
    X_train, X_val, y_train, y_val, speakers_train, speakers_val = train_test_split(X_temp, y_temp, speakers_temp, test_size=0.1, random_state=42)
    
    print(f'Number of training events: {X_train.shape[0]}')
    print(f'Number of validation events: {X_val.shape[0]}')
    print(f'Number of test events: {X_test.shape[0]}')

    def check_distribution(name, tones, speakers):
        tones_dist = pd.Series(tones).value_counts(normalize=True)
        speakers_dist = pd.Series(speakers).value_counts(normalize=True)
        print(f"\n{name} Tone Distribution:\n{tones_dist}")
        print(f"\n{name} Speaker Distribution:\n{speakers_dist}")

    check_distribution("Training", y_train, speakers_train)
    check_distribution("Validation", y_val, speakers_val)
    check_distribution("Test", y_test, speakers_test)

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

    # Load training, validation, and test images
    X_train = np.array([load_and_preprocess_image(fp) for fp in X_train])
    X_val = np.array([load_and_preprocess_image(fp) for fp in X_val])
    X_test = np.array([load_and_preprocess_image(fp) for fp in X_test])

    # Define model
    N_hiddenLayers = 3
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(2, 2))
    for _ in range(N_hiddenLayers):
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # model = Sequential([
    #     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    #     MaxPooling2D(2, 2),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dense(num_classes, activation='softmax')
    # ])

    # Train model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    result = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), batch_size=BATCH_SIZE)

    model.save(modelName)
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the converted model to a file
    tflite_model_name = 'my_model.tflite'
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)

    print(f"Model saved as {tflite_model_name}")

    # Print performance
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    plot_learning_curves(result)

    # Predict for example file
    example_image_file = f"{outDir}/ao4_MV3.png"  
    example_image = load_and_preprocess_image(example_image_file)
    predictions = model.predict(np.expand_dims(example_image, axis=0))
    print(f"Predictions: {predictions}")

    # Get probabilities
    label_probabilities = predictions[0]
    class_names = [ '1', '2', '3', '4']  
    if(addNoise): class_names = [ '1', '2', '3', '4', 'noise']  
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

    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tone 1', 'Tone 2', 'Tone 3', 'Tone 4', 'Other'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png')

    # Classification Report
    print(classification_report(y_true, y_pred_classes, target_names=['Tone 1', 'Tone 2', 'Tone 3', 'Tone 4', 'Other']))

    # Extract predictions for the "Other" class
    other_class_index = 4
    other_class_predictions = y_pred[:, other_class_index]
    other_class_true = (y_true == other_class_index).astype(int)

    # Threshold analysis
    thresholds = np.linspace(0, 1, 100)
    true_positives = []
    false_positives = []
    for threshold in thresholds:
        predicted_other = (other_class_predictions >= threshold).astype(int)
        tp = np.sum((predicted_other == 1) & (other_class_true == 1))
        fp = np.sum((predicted_other == 1) & (other_class_true == 0))
        true_positives.append(tp)
        false_positives.append(fp)

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, true_positives, label='True Positives')
    plt.plot(thresholds, false_positives, label='False Positives')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('Threshold Analysis for "Other" Class')
    plt.legend()
    plt.savefig('Threshold.png')

    # Check Class Distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(['Tone 1', 'Tone 2', 'Tone 3', 'Tone 4', 'Other'], counts))

    plt.bar(class_distribution.keys(), class_distribution.values())
    plt.title('Class Distribution in Training Data')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default="output.csv", help="CSV file name")
    parser.add_argument('--outDir', type=str, default="spectrum_data", help="Path to img dir")
    parser.add_argument('--modelName', type=str, default="tfModelTones", help="Model name")
    parser.add_argument('--addNoise', type=bool, default=False, help="Add noise category")
    parser.add_argument('--noise_csv_file', type=str, default="noise.csv", help="Noise CSV file name")

    args = parser.parse_args()

    trainModel(args.csv_file, args.outDir, args.modelName, args.addNoise, args.noise_csv_file)

if __name__ == "__main__":
    main()