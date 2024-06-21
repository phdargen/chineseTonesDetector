import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor, Trainer, TrainingArguments, DefaultDataCollator, DataCollatorWithPadding, TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import requests

import logging
logging.basicConfig(level=logging.INFO)
import time

# Model from https://huggingface.co/google/vit-base-patch16-224
model_name = 'google/vit-base-patch16-224'
image_resolution = 224

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {
        "accuracy": acc,
        "f1": f1,
    }

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = datasets.folder.default_loader(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return {"pixel_values": image, "label": label}

def plot_example_image(dataset, index=0):
    example = dataset[index]
    print("Example data keys:", example.keys())
    image = example["pixel_values"]
    label = example["label"]
    image = image.permute(1, 2, 0)  # Change from CxHxW to HxWxC for plotting
    #image = image * torch.tensor(feature_extractor.image_std).reshape(1, 1, 3) + torch.tensor(feature_extractor.image_mean).reshape(1, 1, 3)  # Unnormalize
    image = image.numpy()
    plt.imshow(image)
    plt.title(f"Label: {label}")
    #plt.show()

# Load the CSV file and preprocess the data
def load_data_from_csv(csv_file, addNoise=False, noise_csv=None, augmentData=False):
    data = pd.read_csv(csv_file)
    print(f'Number of entries in output csv: {data.shape[0]}')

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
    X_temp, X_test, y_temp, y_test, speakers_temp, speakers_test = train_test_split(
        file_paths, tones_encoded, speakers, test_size=0.1, random_state=42, stratify=tones_encoded)
    # Split train+validation into train and validation sets
    X_train, X_val, y_train, y_val, speakers_train, speakers_val = train_test_split(
        X_temp, y_temp, speakers_temp, test_size=0.2/0.9, random_state=42, stratify=y_temp)

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

    return X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_val, y_test, num_classes

def addData(data, csv_file):
    noise_data = pd.read_csv(csv_file)
    return pd.concat([data, noise_data], ignore_index=True)

def print_frozen_params(model):
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params += param.numel()
            print(f"Parameter {name} is frozen.")
        else:
            trainable_params += param.numel()
            print(f"Parameter {name} is trainable.")
    print(f"Total frozen parameters: {frozen_params}")
    print(f"Total trainable parameters: {trainable_params}")

def trainModel(csv_file='output.csv', outDir="spectrum_data", modelName='fineTunedModelTones', addNoise=False, noise_csv='noise.csv', augmentData=False, unfreezeLastBaseLayer=False, epochs=10, batch_size=8):

    # Load feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    # Transform images
    transform = transforms.Compose([
        transforms.Resize((image_resolution, image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    # Create datasets
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = load_data_from_csv(csv_file, addNoise, noise_csv, augmentData)
    train_dataset = ImageDataset(X_train, y_train, transform)
    val_dataset = ImageDataset(X_val, y_val, transform)
    test_dataset = ImageDataset(X_test, y_test, transform)

    print(f"Training dataset size: {len(X_train)}")
    print(f"Validation dataset size: {len(X_val)}")
    print(f"Test dataset size: {len(X_test)}")
    plot_example_image(train_dataset)

    ## Load base model and add new layer to finetune
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)

    # Freeze base model weights
    for name, param in model.named_parameters():
        if 'classifier' not in name:  
            param.requires_grad = False
        if unfreezeLastBaseLayer and 'encoder.layer.11' in name:
            param.requires_grad = True

    # Print parameter info
    print("\nAfter modifying requires_grad:")
    print_frozen_params(model)

    # Select backend
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print (x)
    else:
        device = torch.device("cpu")
    model.to(device)

    # Init training
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=epochs,              # number of training epochs
        per_device_train_batch_size=batch_size,   # batch size for training
        per_device_eval_batch_size=batch_size,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        save_total_limit=2,             # limit the total amount of checkpoints on disk
        remove_unused_columns=False,    # remove unused columns from the dataset
        push_to_hub=False,              # do not push the model to the hub
        report_to='tensorboard',        # report metrics to tensorboard
        load_best_model_at_end=True,    # load the best model at the end of training
        eval_strategy="epoch",          # evaluation strategy to adopt during training
        save_strategy="epoch",
    )

    class CustomDataCollator:
        def __call__(self, features, debug=False):
            pixel_values = torch.stack([feature["pixel_values"] for feature in features])
            labels = torch.tensor([feature["label"] for feature in features])
            if debug:
                print(f"Batch pixel_values shape: {pixel_values.shape}")
                print(f"Batch labels shape: {labels.shape}")
                print(f"Batch labels: {labels}")
            return {"pixel_values": pixel_values, "labels": labels}

    data_collator = CustomDataCollator()

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,    
        data_collator=data_collator, 
        compute_metrics=compute_metrics,           
    )

    # Train model
    training = trainer.train()
    print(training)

    # Evaluate model
    results = trainer.evaluate(train_dataset)
    print("Evaluate train sample ...")
    print(results)

    results = trainer.evaluate(test_dataset)
    print("Evaluate test sample ...")
    print(results)

    trainer.save_model(f"./results/{modelName}")

    # Test predictions
    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        return image

    example_image_paths = ["spectrum_data/hong1_MV2.png", "spectrum_data/cao2_FV1.png", "spectrum_data/sa3_MV1.png", "spectrum_data/kai4_MV2.png"]
    example_images = [load_image(image_path) for image_path in example_image_paths]
    example_images_batch = torch.stack(example_images).to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=example_images_batch)
        predictions = torch.argmax(outputs.logits, dim=-1)

    for i, image_path in enumerate(example_image_paths):
        print(f"Prediction for {image_path}: Class {predictions[i].item()}")


def test():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default="output.csv", help="CSV file name")
    parser.add_argument('--outDir', type=str, default="spectrum_data", help="Path to img dir")
    parser.add_argument('--modelName', type=str, default="fineTunedModelTones", help="Model name")
    parser.add_argument('--addNoise', type=bool, default=False, help="Add noise category")
    parser.add_argument('--noise_csv_file', type=str, default="noise.csv", help="Noise CSV file name")
    parser.add_argument('--augmentData', type=bool, default=False, help="Augment data")
    parser.add_argument('--test', action='store_true', help="Prepare example spectrograms")
    parser.add_argument('--unfreezeLastBaseLayer', type=bool, default=False, help="Unfreeze last base layer")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")

    args = parser.parse_args()

    if args.test:
        test()
    else:
        trainModel(args.csv_file, args.outDir, args.modelName, args.addNoise, args.noise_csv_file, args.augmentData, args.unfreezeLastBaseLayer, args.epochs, args.batch_size)

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"This took: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()