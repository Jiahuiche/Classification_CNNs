import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from torchvision.transforms import functional as F
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast,
    Normalize, CoarseDropout, GridDistortion
)
from albumentations.pytorch import ToTensorV2

# Configure PyTorch to use all available cores
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())

class AlbumentationsWrapper:
    """Adapt Albumentations transforms to work with PyTorch Datasets"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)  # PIL to numpy
        
        # Convert RGB to grayscale if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Ensure grayscale has channel dimension
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
            
        augmented = self.transform(image=img)
        return augmented['image']


def get_transforms(img_size=28):
    """Return train and validation transforms"""
    train_transform = Compose([
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.3),
        GridDistortion(p=0.2),
        CoarseDropout(num_holes_range=(1, 1), hole_height_range=(4, 4), hole_width_range=(4, 4), p=0.5),
        Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    val_transform = Compose([
        Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    return AlbumentationsWrapper(train_transform), AlbumentationsWrapper(val_transform)


def balance_classes(dataset, samples_per_class):
    """Return a balanced subset with N samples per class"""
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    balanced_indices = [
        idx
        for indices in class_indices.values()
        for idx in random.sample(indices, samples_per_class)
    ]

    random.shuffle(balanced_indices)
    return Subset(dataset, balanced_indices)


def prepare_dataloaders(data_dir, val_dir, samples_per_class, batch_size=128, num_workers=4):
    # Add grayscale conversion before albumentations
    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1)
    ])
    
    # Albumentations transforms
    train_tf, val_tf = get_transforms()

    # Full dataset with grayscale conversion first, then albumentations
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transforms.Compose([
        base_transform,
        train_tf
    ]))
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transforms.Compose([
        base_transform,
        val_tf
    ]))

    # Balanced subset
    balanced_subset = balance_classes(train_dataset, samples_per_class)

    train_loader = DataLoader(
        balanced_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader

class SimplifiedCNN(nn.Module):
    """Simplified but effective CNN model"""
    def __init__(self, num_classes, dropout_rate=0.3):
        super(SimplifiedCNN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def evaluate(model, loader, name, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    # Set seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    start_time = time.time()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {torch.get_num_threads()} CPU threads for computation")
    
    # Parameters
    lr = 0.001
    batch_size = 128  # Increased for better GPU utilization
    max_total_time = 600
    samples_per_class = 8000  # Reduced for faster training

    # Load and balance dataset using new function
    data_dir = "data/train"
    val_dir = "data/validation"
    num_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid overwhelming
    
    dataloader, val_loader = prepare_dataloaders(
        data_dir, val_dir, samples_per_class, batch_size, num_workers
    )
    
    # Get number of classes from the first dataset
    temp_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(temp_dataset.classes)
    
    print(f"Balanced dataset: {samples_per_class * num_classes} images")
    
    # Model and training setup
    model = SimplifiedCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    # Training loop
    epoch = 0
    best_val_acc = 0.0
    model_save_path = "best_model.pth"
    
    print(f"Starting training with {num_classes} classes")
    print(f"Using {num_workers} workers for data loading")
    
    while True:
        epoch += 1
        model.train()
        
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for images, labels in loop:
            if time.time() - start_time > max_total_time:
                print("Maximum time reached. Ending training")
                break
                
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        if time.time() - start_time > max_total_time:
            break
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Validation (skip first 2 epochs for more training time)
        val_acc = evaluate(model, val_loader, f"Validation (Epoch {epoch})", device)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with {best_val_acc * 100:.2f}% validation accuracy")
        
        remaining_time = max_total_time - (time.time() - start_time)
        print(f"Remaining time: {remaining_time:.2f} seconds")
    
    # Final evaluation
    print("Training completed. Evaluating best model...")
    model.load_state_dict(torch.load(model_save_path))
    
    train_acc = evaluate(model, dataloader, "Train", device)
    val_acc = evaluate(model, val_loader, "Validation", device)
    
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
