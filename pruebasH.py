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

class AdvancedAugmentation:
    """Custom data augmentation for our drawing dataset"""
    
    def __init__(self, prob=0.7):  # Changed default probability to 1.0 (100%)
        self.prob = prob
    
    def __call__(self, img):
        # Apply all transformations with 100% probability by forcing prob to 1.0
        # Random rotation (± 20 degrees) - Always apply
        angle = random.uniform(-20, 20)
        img = F.rotate(img, angle, fill=0)
        
        # Random shifts (± 15%) - Always apply
        shift_x = random.uniform(-0.15, 0.15)
        shift_y = random.uniform(-0.15, 0.15)
        img = F.affine(img, angle=0, translate=[shift_x, shift_y], 
                        scale=1.0, shear=0, fill=0)
        
        # Random erasing (simulates occlusions) - Always apply
        h, w = img.shape[1:]
        area = h * w
        target_area = random.uniform(0.02, 0.15) * area
        aspect_ratio = random.uniform(0.3, 1/0.3)
        
        h_rect = int(np.sqrt(target_area * aspect_ratio))
        w_rect = int(np.sqrt(target_area / aspect_ratio))
        
        if h_rect < h and w_rect < w:
            x1 = random.randint(0, h - h_rect)
            y1 = random.randint(0, w - w_rect)
            mask = torch.ones_like(img)
            mask[:, x1:x1+h_rect, y1:y1+w_rect] = 0
            img = img * mask
        
        # Add slight Gaussian noise - Always apply
        noise = torch.randn_like(img) * 0.05
        img = torch.clamp(img + noise, 0, 1)
        
        # Adjust brightness/contrast - Always apply
        brightness = random.uniform(-0.2, 0.2)
        contrast = random.uniform(0.8, 1.2)
        img = F.adjust_brightness(img, 1 + brightness)
        img = F.adjust_contrast(img, contrast)
        
        return img

# Move AugmentedDataset out of main() function to make it picklable
class AugmentedDataset(Dataset):
    """Dataset wrapper that applies augmentation to images after sampling"""
    def __init__(self, dataset, augmentation):
        self.dataset = dataset
        self.augmentation = augmentation
        
    def __getitem__(self, index):
        img, label = self.dataset[index]
        # Apply augmentation
        img = self.augmentation(img)
        return img, label
        
    def __len__(self):
        return len(self.dataset)

# Reverting to the earlier model with four convolutional blocks
class EnhancedCNNModel(nn.Module):
    """Enhanced CNN model with residual connections and deeper architecture"""
    def __init__(self, num_classes, dropout_rate=0.3, fc_dropout=0.5):
        super(EnhancedCNNModel, self).__init__()
        
        # First convolutional block with residual connection
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Second convolutional block with residual connection
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        # Projection shortcut for residual connection (to match dimensions)
        self.shortcut = nn.Conv2d(32, 64, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Third convolutional block with increased channels
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Fourth convolutional block
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        '''self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(dropout_rate)'''
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc_dropout1 = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc_dropout2 = nn.Dropout(fc_dropout * 0.8)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Attention mechanism for better feature focus
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # First block with residual connection
        x = torch.nn.functional.relu(self.bn1_1(self.conv1_1(x)))
        x = self.bn1_2(self.conv1_2(x))
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with residual connection
        identity2 = x
        x = torch.nn.functional.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        identity2 = self.bn_shortcut(self.shortcut(identity2))
        x = torch.nn.functional.relu(x + identity2)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block with attention mechanism
        x = torch.nn.functional.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.nn.functional.relu(self.bn3_2(self.conv3_2(x)))
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = torch.nn.functional.relu(self.bn4_1(self.conv4_1(x)))
        '''x = torch.nn.functional.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.dropout4(x)'''
        
        # Global pooling
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc_dropout1(x)
        x = torch.nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc_dropout2(x)
        x = self.fc3(x)
        
        return x

def evaluate(model, loader, name, device):
    """Evaluate model performance on a given dataset"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def load_best_model(path, num_classes, device):
    """Load the best saved model"""
    model = EnhancedCNNModel(num_classes)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    start_time = time.time()
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters - adjusted for better performance
    lr = 0.0001  # Slightly lower learning rate for the more complex model
    batch_size = 64
    max_total_time = 600  # 10 minutes
    
    # Set up transformations - separate basic and augmentation steps
    basic_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor()
    ])
    
    # First load the dataset with only basic transforms
    data_dir = "data/train"
    full_dataset = datasets.ImageFolder(root=data_dir, transform=basic_transform)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Reverting back to 10000 samples per class
    samples_per_class = 10000
    balanced_indices = []
    
    # Obtain indices grouped by class
    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Take samples_per_class random samples from each class
    # or all if there are fewer available
    for label, indices in class_indices.items():
        if len(indices) <= samples_per_class:
            balanced_indices.extend(indices)
            print(f"Clase {class_names[label]}: tomando todas las {len(indices)} muestras disponibles")
        else:
            sampled_indices = random.sample(indices, samples_per_class)
            balanced_indices.extend(sampled_indices)
            print(f"Clase {class_names[label]}: tomando {samples_per_class} muestras aleatorias de {len(indices)}")
    
    # Create a subset with balanced classes
    balanced_subset = Subset(full_dataset, balanced_indices)
    
    print(f"Dataset original: {len(full_dataset)} imágenes")
    print(f"Dataset balanceado: {len(balanced_subset)} imágenes")
    
    # Create augmentation function with 100% probability
    augmentation = AdvancedAugmentation(prob=1.0)  # Ensure 100% probability
    
    # Apply augmentation to the balanced subset
    dataset = AugmentedDataset(balanced_subset, augmentation)
    
    # Validation dataset (sin cambios)
    val_dir = "data/validation"
    val_dataset = datasets.ImageFolder(root=val_dir, transform=basic_transform)
    
    # Set up data loaders - usando dataset balanceado con augmentation
    num_core = multiprocessing.cpu_count()
    
    # Reduce num_workers to avoid potential issues
    num_workers = max(1, num_core-2)
    
    dataloader = DataLoader(dataset, 
                           batch_size=batch_size,
                           shuffle=True,  # Shuffle en lugar de sampler
                           num_workers=num_workers,
                           persistent_workers=True if num_workers > 0 else False,
                           prefetch_factor=2 if num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, 
                           batch_size=batch_size, 
                           shuffle=False, 
                           num_workers=num_workers,
                           persistent_workers=True if num_workers > 0 else False,
                           prefetch_factor=2 if num_workers > 0 else None)
    
    # Initialize model with the four-block architecture
    model = EnhancedCNNModel(num_classes).to(device)
    
    # Optimizer settings that worked well in the previous version
    weight_decay = 0.0001
    
    # Set up optimizer with weight decay to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Add learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Add missing criterion definition to avoid error
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    epoch = 0
    best_val_acc = 0.0
    model_save_path = "best_model.pth"
    
    print(f"Preprocess completat. Iniciant l'entrenament del model amb {num_classes} classes.")
    print(f"Les classes han estat balancejades manualment ({samples_per_class} por classe)")
    print("Using enhanced CNN architecture with 4 convolutional blocks and residual connections")
    print("Data augmentation applied with 100% probability to all training samples")
    
    while True:
        start_epoch = time.time()
        epoch += 1
        model.train()
        loop = tqdm(dataloader, desc=f"Època {epoch}", leave=False)
        
        # Track class distribution during training
        epoch_class_samples = [0] * num_classes
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(loop):
            if time.time() - start_time > max_total_time:
                print("Temps màxim assolit. Fi de l'entrenament")
                break
            
            # Track samples per class in this batch
            for label in labels:
                epoch_class_samples[label.item()] += 1
    
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        if time.time() - start_time > max_total_time:
            break
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
            
        # Show class distribution for this epoch (verify balancing)
        if epoch % 5 == 0:  # Only show every 5 epochs to avoid clutter
            print("Class distribution in this epoch:")
            for i, name in enumerate(class_names):
                print(f"{name}: {epoch_class_samples[i]} samples")
                
        # Validation per epoch
        val_acc = evaluate(model, val_loader, f"Validació (després de la època {epoch}[{time.time() - start_epoch:.2f}s])", device)
        
        # Update learning rate based on validation performance
        scheduler.step(val_acc)
        
        print("Temps restant: {:.2f} segons".format(max_total_time - (time.time() - start_time)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Nou millor model guardat amb {best_val_acc * 100:.2f}% de precisió de validació")
    
    # Final evaluation
    print("Entrenament completat. Evaluant el millor model...")
    best_model = load_best_model(model_save_path, num_classes, device)
    
    # Evaluate on training and validation sets
    train_acc = evaluate(best_model, dataloader, "Train (subset)", device)
    val_acc = evaluate(best_model, val_loader, "Validation (final)", device)
    
    print(f"\nFinal Metrics:")
    print(f"   Train Accuracy: {train_acc * 100:.2f}%")
    print(f"   Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Temps total d'entrenament: {time.time() - start_time:.2f} segons")

if __name__ == "__main__":
    # Important for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
