import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import seaborn as sns
import time

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from processAudio import get_mp3_info, get_spectrum
samplingRate = 22050

# Probability calibration
class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemperatureScaling, self).__init__(**kwargs)
        self.temperature = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
    
    def call(self, logits):
        return logits / self.temperature

def nll_loss(y_true, y_pred):
    return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred))

def plot_calibration_curve(y_true, y_prob, num_classes, n_bins=10):
    plt.figure(figsize=(10, 5))
    
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_prob_binary = y_prob[:, i]
        
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob_binary, n_bins=n_bins, strategy='uniform')
        
        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')
    
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(num_classes):
        plt.hist(y_prob[:, i], bins=n_bins, range=(0, 1), histtype='step', lw=2, label=f'Class {i}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Probabilities')
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_brier_score(y_true, y_prob, num_classes):
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
    score = brier_score_loss(y_true_onehot.ravel(), y_prob.ravel())
    print(f"Brier score: {score:.4f}")

# Control plots
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

# Add csv files
def addData(data, csv_file):
    noise_data = pd.read_csv(csv_file)
    return pd.concat([data, noise_data], ignore_index=True)

def trainModel(csv_file='output.csv', outDir="spectrum_data", modelName='tfModelTones', addNoise=False, noise_csv='noise.csv', augmentData=False, epochs=10, batch_size=128, N_hiddenLayers = 3, image_resolution=128, runOnGPU=False):

    # Setup GPU 
    print(tf.config.list_physical_devices('GPU'))
    if not runOnGPU: 
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    # Extract file paths and tone labels
    data = pd.read_csv(csv_file)
    print(f'Number of entries in output csv: {data.shape[0]}')
    durations = data['Duration (seconds)']
    print(f'Total duration of audion samples: {durations.sum()/60/60} h')

    # Add noise data
    if(addNoise):
        data = addData(data, noise_csv)
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
    X_temp, X_test, y_temp, y_test, speakers_temp, speakers_test = train_test_split(file_paths, tones_encoded, speakers, test_size=0.1, random_state=42, stratify=tones_encoded)
    # Split train+validation into train and validation sets
    X_train, X_val, y_train, y_val, speakers_train, speakers_val = train_test_split(X_temp, y_temp, speakers_temp, test_size=0.2/0.9, random_state=42, stratify=y_temp)
    
    print(f'Number of training events: {X_train.shape[0]}')
    print(f'Number of validation events: {X_val.shape[0]}')
    print(f'Number of test events: {X_test.shape[0]}')

    # Add augmented data to training set
    if augmentData:
        augmented_csv_file = csv_file.replace('.csv', '_augmented.csv')
        augmented_data = pd.read_csv(augmented_csv_file)
        augmented_noise_csv_file = noise_csv.replace('.csv', '_augmented.csv')
        
        if addNoise:
            augmented_noise_data = pd.read_csv(augmented_noise_csv_file)
            augmented_data = pd.concat([augmented_data, augmented_noise_data], ignore_index=True)
            
        X_train = pd.concat([pd.Series(X_train), augmented_data['img_filename']], ignore_index=True)
        y_train = pd.concat([pd.Series(y_train), pd.Series(label_encoder.transform(augmented_data['tone']))], ignore_index=True)
        speakers_train = pd.concat([pd.Series(speakers_train), pd.Series(augmented_data['speaker'])], ignore_index=True)
        print(f'Number of entries in training set after adding augmented data: {len(X_train)}')

    def check_distribution(name, tones, speakers):
        tones_dist = pd.Series(tones).value_counts(normalize=True)
        speakers_dist = pd.Series(speakers).value_counts(normalize=True)
        print(f"\n{name} Tone Distribution:\n{tones_dist}")
        print(f"\n{name} Speaker Distribution:\n{speakers_dist}")

    check_distribution("Training", y_train, speakers_train)
    check_distribution("Validation", y_val, speakers_val)
    check_distribution("Test", y_test, speakers_test)

    # Preprocess images
    def load_and_preprocess_image(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3) 
        img = tf.image.resize(img, [image_resolution, image_resolution])
        img = tf.cast(img, tf.float32) / 255.0
        return img

    # Test with sample image
    sample_image = load_and_preprocess_image(file_paths[0])
    plt.figure()
    plt.imshow(sample_image)
    plt.axis('off')
    #plt.show()
    plt.savefig('test_image.png')

    # Create datasets 
    print("Create datasets ...")
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def process_path_and_label(file_path, label):
        return load_and_preprocess_image(file_path), label

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = (train_dataset
                    .map(process_path_and_label, num_parallel_calls=AUTOTUNE)
                    .cache()
                    .shuffle(buffer_size=1000)
                    .batch(batch_size)
                    .prefetch(AUTOTUNE))

    val_dataset = (val_dataset
                .map(process_path_and_label, num_parallel_calls=AUTOTUNE)
                .cache()
                .batch(batch_size)
                .prefetch(AUTOTUNE))

    test_dataset = (test_dataset
                    .map(process_path_and_label, num_parallel_calls=AUTOTUNE)
                    .cache()
                    .batch(batch_size)
                    .prefetch(AUTOTUNE))

    # Define model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_resolution, image_resolution, 3)))
    model.add(MaxPooling2D(2, 2))
    for _ in range(N_hiddenLayers):
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(image_resolution, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Train model
    print("Training model ...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    result = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, batch_size=batch_size)

    # Save model
    model.save(modelName)
    print(f"Model saved as {modelName}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_name = 'my_model.tflite'
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved as {tflite_model_name}")

    # Print performance
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    plot_learning_curves(result)

    # Predict for example file
    example_image_file = f"{outDir}/ao4_MV3.png"  
    example_image = load_and_preprocess_image(example_image_file)
    predictions = model.predict(np.expand_dims(example_image, axis=0))
    print(f"Inference for example image: {example_image_file}")
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
    predicted_probabilities = []
    true_labels = []

    for images, labels in test_dataset:
        batch_predictions = model.predict(images)
        predicted_probabilities.append(batch_predictions)
        true_labels.append(labels)

    predicted_probabilities = np.concatenate(predicted_probabilities, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    print("Shape of predicted_probabilities:", predicted_probabilities.shape)
    print("Shape of true_labels:", true_labels.shape)
    print("Unique values in true_labels:", np.unique(true_labels))

    # Compute ROC curve and AUC for each class 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    num_classes = len(class_names)  
    for i in range(num_classes):
        true_labels_binary = (true_labels == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(true_labels_binary, predicted_probabilities[:, i])
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

    # Confusion matrix
    y_pred_classes = np.argmax(predicted_probabilities, axis=1)
    y_true = true_labels

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tone 1', 'Tone 2', 'Tone 3', 'Tone 4', 'Other'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png')

    # Classification Report
    print(classification_report(y_true, y_pred_classes, target_names=['Tone 1', 'Tone 2', 'Tone 3', 'Tone 4', 'Other']))

    # Extract predictions for the "Other" class
    other_class_index = 4
    other_class_predictions = predicted_probabilities[:, other_class_index]
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

    # Extract logits and create a new model with temperature scaling
    logits_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].output)
    temperature_layer = TemperatureScaling()
    scaled_logits = temperature_layer(logits_model.output)
    calibrated_output = tf.keras.layers.Softmax()(scaled_logits)
    calibrated_model = tf.keras.Model(inputs=model.input, outputs=calibrated_output)

    # Compile the model for calibration
    calibrated_model.compile(optimizer='adam', loss=nll_loss, metrics=['accuracy'])

    # Train only the temperature parameter
    calibrated_model.get_layer('temperature_scaling').trainable = True
    for layer in calibrated_model.layers[:-2]:
        layer.trainable = False

    # Train the temperature parameter using validation data
    calibrated_model.fit(val_dataset, epochs=1, verbose=1)

    # Save the calibrated model
    calibrated_model.save(f'{modelName}_calibrated')
    print(f"Calibrated model saved as {modelName}_calibrated")

    #
    calibrated_model.evaluate(test_dataset)
    print(f"Inference for example image: {example_image_file}")
    print(f"Predictions before calibration: {predictions}")
    predictions = calibrated_model.predict(np.expand_dims(example_image, axis=0))
    print(f"Predictions after calibration: {predictions}")

    #
    y_prob = model.predict(val_dataset)
    y_prob = np.array(y_prob)
    plot_calibration_curve(y_val, y_prob, num_classes, n_bins=10)
    print_brier_score(y_val, y_prob, num_classes)

    y_prob_calibrated = calibrated_model.predict(val_dataset)
    y_prob_calibrated = np.array(y_prob_calibrated)
    y_val = np.concatenate([y for _, y in val_dataset], axis=0)
    print("Shape of y_prob_calibrated:", y_prob_calibrated.shape)
    print("Shape of y_val:", y_val.shape)
    plot_calibration_curve(y_val, y_prob_calibrated, num_classes, n_bins=10)
    print_brier_score(y_val, y_prob_calibrated, num_classes)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default="output.csv", help="CSV file name")
    parser.add_argument('--outDir', type=str, default="spectrum_data", help="Path to img dir")
    parser.add_argument('--modelName', type=str, default="tfModelTones", help="Model name")
    parser.add_argument('--addNoise', action='store_true', help="Add noise category")
    parser.add_argument('--noise_csv_file', type=str, default="noise.csv", help="Noise CSV file name")
    parser.add_argument('--augmentData', action='store_true', help="Augment data")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--runOnGPU', action='store_true', help="Run on GPU")
    parser.add_argument('--nHiddenLayers', type=int, default=3, help="Number of hidden layers")
    parser.add_argument('--image_resolution', type=int, default=128, help="Image resolution")

    args = parser.parse_args()
    
    print("Settings: ")
    print(f"CSV File: {args.csv_file}")
    print(f"Output Directory: {args.outDir}")
    print(f"Model Name: {args.modelName}")
    print(f"Add Noise: {args.addNoise}")
    print(f"Noise CSV File: {args.noise_csv_file}")
    print(f"Augment Data: {args.augmentData}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Run on GPU: {args.runOnGPU}")
    print(f"Number of hidden layers: {args.nHiddenLayers}")
    print(f"Image resolution: {args.image_resolution}")
    print("")

    trainModel(args.csv_file, args.outDir, args.modelName, args.addNoise, args.noise_csv_file, args.augmentData, args.epochs, args.batch_size, args.nHiddenLayers, args.image_resolution, args.runOnGPU)

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"This took: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()