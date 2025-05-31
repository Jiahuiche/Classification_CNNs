import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from multiprocessing import freeze_support

# Definición del modelo
class ElVostreModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ElVostreModel, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2d = nn.Dropout2d(0.2)
        
        # Adaptive pooling for flexibility
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        # Second conv block
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        # Third conv block
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Función de evaluación
def evaluate(model, loader, name, device):
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

# Función para cargar el mejor modelo
def load_best_model(path, num_classes, device):
    """Carga el mejor modelo guardado."""
    model = ElVostreModel(num_classes)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model

def main():
    # Configuración de semillas para reproducibilidad
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    start_time = time.time()

    # Dataset
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    data_dir = "data/train"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    class_names = dataset.classes
    num_classes = len(class_names)

    # Validación
    val_dir = "data/validation"
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    # Hiperparámetros
    lr = 0.001
    batch_size = 64
    max_total_time = 1800 

    # Configuración de DataLoaders
    num_core = multiprocessing.cpu_count()
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_core-1,
                            persistent_workers=True,
                            prefetch_factor=4)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_core-1,
                            persistent_workers=True,
                            prefetch_factor=4)

    # Configuración del dispositivo y modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ElVostreModel(num_classes).to(device)

    # Configuración del optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Entrenamiento del modelo
    epoch = 0
    best_val_acc = 0.0
    model_save_path = "best_model.pth"

    while True:
        start_epoch = time.time()
        epoch += 1
        model.train()
        loop = tqdm(dataloader, desc=f"Època {epoch}", leave=False)

        for i, (images, labels) in enumerate(loop):
            if time.time() - start_time > max_total_time:
                print("Temps màxim assolit. Fi de l'entrenament")
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        
        if time.time() - start_time > max_total_time:
            break
        
        # Validación por época
        val_acc = evaluate(model, val_loader, f"Validació (després de la època {epoch}[{time.time() - start_epoch:.2f}s])", device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Nou millor model guardat amb {best_val_acc * 100:.2f}% de precisió de validació")

    # Evaluación final
    print("\nEvaluando el millor modelo guardado...")
    best_model = load_best_model(model_save_path, num_classes, device)
    train_acc = evaluate(best_model, dataloader, "Train (subset)", device)
    val_acc = evaluate(best_model, val_loader, "Validation (final)", device)

    print(f"\nFinal Metrics:")
    print(f"   Train Accuracy: {train_acc * 100:.2f}%")
    print(f"   Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Temps total d'entrenament: {time.time() - start_time:.2f} segons")

if __name__ == '__main__':
    # Esta línea es necesaria para evitar problemas con multiprocessing en Windows
    freeze_support()
    main()
