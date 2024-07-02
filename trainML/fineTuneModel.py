import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor, Trainer, TrainingArguments, DefaultDataCollator, TrainerCallback, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from peft import LoraConfig, get_peft_model

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

def plot_roc_curve(labels, probs, num_classes):
    labels_bin = label_binarize(labels, classes=range(num_classes))
    
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig('results/roc.png')

def plot_confusion_matrix(labels, predictions, class_names):
    cm = confusion_matrix(labels, predictions,normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    #plt.show()
    plt.savefig('results/confusion.png')

class TrainingMetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            self.steps.append(step)
            
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
            if 'eval_accuracy' in logs:
                self.eval_accuracies.append(logs['eval_accuracy'])

    def plot_metrics(self):
        if not self.steps:
            print("No training steps recorded. Unable to plot metrics.")
            return

        plt.figure(figsize=(12, 5))

        if self.train_losses:
            plt.subplot(1, 2, 1)
            plt.plot(self.steps[:len(self.train_losses)], self.train_losses, label='Training Loss')
            if self.eval_losses:
                plt.plot(self.steps[:len(self.eval_losses)], self.eval_losses, label='Validation Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')

        if self.eval_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(self.steps[:len(self.eval_accuracies)], self.eval_accuracies, label='Validation Accuracy')
            plt.xlabel('Steps')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Validation Accuracy')

        plt.tight_layout()
        #plt.show()
        plt.savefig('results/metrics.png')

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

class MyModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(MyModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        logits = self.classifier(outputs.mean(dim=1))  
        return logits

class CustomDataCollator:
    def __call__(self, features, debug=False):
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        labels = torch.tensor([feature["label"] for feature in features])
        if debug:
            print(f"Batch pixel_values shape: {pixel_values.shape}")
            print(f"Batch labels shape: {labels.shape}")
            print(f"Batch labels: {labels}")
        return {"pixel_values": pixel_values, "labels": labels}

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

def trainModel(csv_file='output.csv', outDir="spectrum_data", modelName='fineTunedModelTones', addNoise=False, noise_csv='noise.csv', augmentData=False, unfreezeLastBaseLayer=False, epochs=10, batch_size=8, addMoreLayers=False, resume_from_checkpoint=None, doCalibration=False, applyLora=False):

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
    if addMoreLayers:
        base_model = ViTForImageClassification.from_pretrained(model_name)
        model = MyModel(base_model, num_labels=num_classes)
    else:
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

    # Apply Lora
    lora_config = LoraConfig(
        r=4,  
        lora_alpha=32,
        target_modules=["attention.self", "attention.output.dense"],
        lora_dropout=0.1,
        bias="none"
    )
    if applyLora:
        model = get_peft_model(model, lora_config)

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
        output_dir='./results',      
        logging_dir='./logs',            
        
        push_to_hub=False,              
        #report_to='tensorboard',        
        load_best_model_at_end=True,    
        gradient_accumulation_steps=1,
        remove_unused_columns=False,    

        eval_strategy="epoch",          
        save_strategy="epoch",
        #save_total_limit=2,            

        warmup_ratio=0.1,
        lr_scheduler_type="cosine", # or linear?
        metric_for_best_model="accuracy",

        ## Hyperparameters to tune
        per_device_train_batch_size=batch_size,   
        per_device_eval_batch_size=batch_size,  
        num_train_epochs=epochs,         
        learning_rate=1e-5,
        weight_decay=0.01,   # regularization
    )

    data_collator = CustomDataCollator()
    metrics_callback = TrainingMetricsCallback()

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,    
        data_collator=data_collator, 
        compute_metrics=compute_metrics,   
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), metrics_callback]        
    )

    # Train model
    if resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    # Evaluate model
    print("Evaluate validation sample ...")
    results = trainer.evaluate(val_dataset)
    print(results)

    print("Evaluate test sample ...")
    results = trainer.evaluate(test_dataset)
    print(results)

    print(f"Saving models as: ./results/{modelName}")
    trainer.save_model(f"./results/{modelName}")

    # Plot training metrics
    metrics_callback.plot_metrics()

    # Test predictions
    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        return image

    example_image_paths = ["spectrum_data/hong1_MV2.png", "spectrum_data/cao2_FV1.png", "spectrum_data/sa3_MV1.png", "spectrum_data/kai4_MV2.png", "spectrum_noise/noise_sample_28.png"]
    example_images = [load_image(image_path) for image_path in example_image_paths]
    example_images_batch = torch.stack(example_images).to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=example_images_batch)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

    print("Predicting examples ...")
    for i, image_path in enumerate(example_image_paths):
        pred_tone = predictions[i].item() + 1
        probs = probabilities[i].cpu().numpy() * 100 
        print(f"Prediction for {image_path}:")
        print(f"  Predicted Tone: {pred_tone}")
        print("  Class Probabilities:")
        for j, prob in enumerate(probs):
            print(f"    Tone {j+1}: {prob:.2f}%")
        print()

    # Make control plots
    all_labels = []
    all_preds = []
    all_probs = []
    all_logits = []

    for batch in DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator):
        with torch.no_grad():
            outputs = model(pixel_values=batch["pixel_values"].to(device))
            logits = outputs.logits
            logits_converted = logits.cpu().numpy()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

        all_logits.extend(logits_converted)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels)

    all_logits = np.array(all_logits)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    plot_roc_curve(all_labels, all_probs, num_classes)
    plot_confusion_matrix(all_labels, all_preds, [str(i) for i in range(num_classes)])

    if doCalibration:
        # Calibration
        print("Calibrating probabilities...")

        # Fit the Logistic Regression model
        # lr = LogisticRegression(max_iter=1000)
        # lr.fit(all_val_logits, all_val_labels)

        # # Use Platt scaling for calibration
        # calibrator = CalibratedClassifierCV(estimator=lr, method='sigmoid', cv='prefit')
        # calibrator.fit(all_val_logits, all_val_labels)

        from scipy.special import softmax
        class SKLearnWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, model):
                self.model = model
            
            def fit(self, X, y):
                return self
            
            def predict_proba(self, X):
                return softmax(X, axis=1)
        
        base_classifier = SKLearnWrapper(model)
        calibrated_classifier = CalibratedClassifierCV(base_classifier, cv='prefit', method='isotonic')
        calibrated_classifier.fit(all_logits, all_labels)

        # Save calibrated model
        import joblib
        joblib.dump(calibrated_classifier, f"./results/{modelName}_calibrated.joblib")
        print(f"Saved calibrated model as: ./results/{modelName}_calibrated.joblib")

        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values=example_images_batch)
            logits = outputs.logits.cpu().numpy()
            calibrated_probabilities = calibrated_classifier.predict_proba(logits)
            predictions = np.argmax(calibrated_probabilities, axis=-1)

        print("Predicting examples with calibrated probabilities...")
        for i, image_path in enumerate(example_image_paths):
            pred_tone = predictions[i] + 1
            probs = calibrated_probabilities[i] * 100 
            print(f"Prediction for {image_path}:")
            print(f"  Predicted Tone: {pred_tone}")
            print("  Calibrated Class Probabilities:")
            for j, prob in enumerate(probs):
                print(f"    Tone {j+1}: {prob:.2f}%")
            print()

        # Make control plots with calibrated probabilities
        all_labels = []
        all_preds = []
        all_probs = []

        for batch in DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator):
            with torch.no_grad():
                outputs = model(pixel_values=batch["pixel_values"].to(device))
                logits = outputs.logits.cpu().numpy()
                probs = calibrated_classifier.predict_proba(logits)
                preds = np.argmax(probs, axis=-1)
                labels = batch["labels"].cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        plot_roc_curve(all_labels, all_probs, num_classes)
        plot_confusion_matrix(all_labels, all_preds, [str(i) for i in range(num_classes)])

        # Plot calibration curve
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        for i in range(num_classes):
            prob_true, prob_pred = calibration_curve(all_labels == i, all_probs[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i+1}')
        
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration curve')
        plt.legend()
        plt.savefig(f"./results/{modelName}_calibration_curve.png")
        plt.close()

        # # Predict on the test set with calibration
        # all_test_labels = []
        # all_test_logits = []

        # for batch in DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator):
        #     with torch.no_grad():
        #         outputs = model(pixel_values=batch["pixel_values"].to(device))
        #         logits = outputs.logits.cpu().numpy()
        #         labels = batch["labels"].cpu().numpy()

        #     all_test_labels.extend(labels)
        #     all_test_logits.extend(logits)

        # all_test_labels = np.array(all_test_labels)
        # all_test_logits = np.array(all_test_logits)

        # # Apply calibration
        # calibrated_probs = calibrator.predict_proba(all_test_logits)
        # calibrated_preds = np.argmax(calibrated_probs, axis=1)

        # # Evaluate calibrated predictions
        # plot_roc_curve(all_test_labels, calibrated_probs, num_classes)
        # plot_confusion_matrix(all_test_labels, calibrated_preds, [str(i) for i in range(num_classes)])

        # accuracy = accuracy_score(all_test_labels, calibrated_preds)
        # f1 = f1_score(all_test_labels, calibrated_preds, average="macro")
        # print(f"Calibrated Accuracy: {accuracy:.4f}")
        # print(f"Calibrated F1 Score: {f1:.4f}")

        # print("Predicting examples after calibration ...")
        # for i, image_path in enumerate(example_image_paths):
        #     pred_tone = predictions[i].item() + 1
        #     probs = probabilities[i].cpu().numpy() * 100 
        #     print(f"Prediction for {image_path}:")
        #     print(f"  Predicted Tone: {pred_tone}")
        #     print("  Class Probabilities:")
        #     for j, prob in enumerate(probs):
        #         print(f"    Tone {j+1}: {prob:.2f}%")
        #     print()


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
    parser.add_argument('--addNoise', action='store_true', help="Add noise category")
    parser.add_argument('--noise_csv_file', type=str, default="noise.csv", help="Noise CSV file name")
    parser.add_argument('--augmentData', action='store_true', help="Augment data")
    parser.add_argument('--test', action='store_true', help="Prepare example spectrograms")
    parser.add_argument('--unfreezeLastBaseLayer', action='store_true', help="Unfreeze last base layer")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--addMoreLayers', action='store_true', help="Add more layers")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument('--applyLora', action='store_true', help="Apply Lora")

    args = parser.parse_args()

    print("Settings: ")
    print(f"CSV File: {args.csv_file}")
    print(f"Output Directory: {args.outDir}")
    print(f"Model Name: {args.modelName}")
    print(f"Add Noise: {args.addNoise}")
    print(f"Noise CSV File: {args.noise_csv_file}")
    print(f"Augment Data: {args.augmentData}")
    print(f"Test Mode: {args.test}")
    print(f"Unfreeze Last Base Layer: {args.unfreezeLastBaseLayer}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Add More Layers: {args.addMoreLayers}")
    print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    print(f"Apply Lora: {args.applyLora}")
    print("")

    if args.test:
        test()
    else:
        trainModel(args.csv_file, args.outDir, args.modelName, args.addNoise, args.noise_csv_file, args.augmentData, args.unfreezeLastBaseLayer, args.epochs, args.batch_size, args.addMoreLayers, args.resume_from_checkpoint, args.applyLora)

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"This took: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()