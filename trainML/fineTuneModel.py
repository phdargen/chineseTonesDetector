import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import argparse
import pandas as pd

model_name = 'google/vit-base-patch16-224'
image_resolution = 224

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = datasets.folder.default_loader(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# Load the CSV file and preprocess the data
def load_data_from_csv(csv_file, add_noise=False, noise_csv=None):
    data = pd.read_csv(csv_file)
    print(f'Number of entries in output csv: {data.shape[0]}')

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

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes

def addData(data, csv_file):
    noise_data = pd.read_csv(csv_file)
    return pd.concat([data, noise_data], ignore_index=True)

def print_frozen_params(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} is frozen.")
        else:
            print(f"Parameter {name} is trainable.")

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data).logits
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data).logits
            test_loss += criterion(outputs, target).item() 
            pred = outputs.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')
    return accuracy

def trainModel(csv_file='output.csv', outDir="spectrum_data", modelName='tfModelTones', addNoise=False, noise_csv='noise.csv', augmentData=False):

    # Load base model 
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=5,ignore_mismatched_sizes=True)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    # Print frozen parameters before training
    print("Before training:")
    print_frozen_params(model)

    # Freeze base model weights
    for name, param in model.named_parameters():
        if 'classifier' not in name:  
            param.requires_grad = False

    # Print frozen parameters after modifying
    print("\nAfter modifying requires_grad:")
    print_frozen_params(model)

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((image_resolution, image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    # Create datasets
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = load_data_from_csv(csv_file)
    train_dataset = ImageDataset(X_train, y_train, transform)
    val_dataset = ImageDataset(X_val, y_val, transform)
    test_dataset = ImageDataset(X_test, y_test, transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

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

    # Define optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    num_epochs = 10
    # for epoch in range(1, num_epochs + 1):
    #     train(model, device, train_loader, criterion, optimizer, epoch)
    #     test(model, device, test_loader, criterion)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default="output.csv", help="CSV file name")
    parser.add_argument('--outDir', type=str, default="spectrum_data", help="Path to img dir")
    parser.add_argument('--modelName', type=str, default="fineTunedModelTones", help="Model name")
    parser.add_argument('--addNoise', type=bool, default=False, help="Add noise category")
    parser.add_argument('--noise_csv_file', type=str, default="noise.csv", help="Noise CSV file name")
    parser.add_argument('--augmentData', type=bool, default=False, help="Augment data")

    args = parser.parse_args()

    trainModel(args.csv_file, args.outDir, args.modelName, args.addNoise, args.noise_csv_file, args.augmentData)

if __name__ == "__main__":
    main()