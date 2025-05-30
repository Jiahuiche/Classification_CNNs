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
# Paralelize the model
import multiprocessing
# Learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

start_time = time.time()
# ---------- Reproducibilitat ----------
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
max_total_time = 1200 
# ---------- Model ----------
class ElVostreModel(nn.Module):
    def __init__(self, num_classes):
        super(ElVostreModel, self).__init__()
        self.net = nn.Sequential(
            # Bloque 1: 28×28 → 14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2: 14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Full Connection
            nn.Flatten(),
            nn.Linear(3136, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout para evitar overfitting
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
# ----------Funcions----------
def evaluate(model, loader, name):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Para calcular accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy
def load_best_model(path, num_classes):
    """Carga el mejor modelo guardado."""
    model = ElVostreModel(num_classes)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model
device = torch.device("cpu")

# Importante: Código principal dentro de esta condición
if __name__ == '__main__':
    # Añade esto para compatibilidad con aplicaciones congeladas
    multiprocessing.freeze_support()

    # ---------- Dataset ----------
    #---------- Entrenamiento ----------
    # Transforma para entrenamiento (con data augmentation)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(15),
        # Más variedad en traslación y escala
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.15, 0.15),  # Mayor traslación 
            scale=(0.85, 1.15),      # Mayor variación de escala
            shear=5                  # Añadido: distorsión de perspectiva
        ),
        
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    data_dir = "data/train"
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    class_names = dataset.classes
    num_classes = len(class_names)

    # ---------- Validació ----------
    # Transforma para validación (sin augmentation)
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_dir = "data/validation"
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # ----------Paralelización---------------------
    batch_size = 64
    num_core = multiprocessing.cpu_count()
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_core-1,
                            persistent_workers=True,
                            prefetch_factor=4
                            )
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_core-1,
                            persistent_workers=True,
                            prefetch_factor=4
                            )

    # ----------Train----------
    # Hiperparàmetres
    lr = 0.001
    weight_decay = 1e-4
    model = ElVostreModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Añadir el scheduler
    scheduler = ReduceLROnPlateau(optimizer, 
                                mode='max', 
                                factor=0.1, 
                                patience=1,
                                threshold=0.01
                                )

    criterion = nn.CrossEntropyLoss()

    epoch = 0
    best_val_acc = 0.0
    model_save_path = "best_model.pth"

    print("Iniciant l'entrenament...")
    while True:
        start_epoch = time.time()
        epoch += 1
        model.train()
        loop = tqdm(dataloader, desc=f"Època {epoch}", leave=False)

        for i, (images, labels) in enumerate(loop):
            if time.time() - start_time > max_total_time-60:
                print("Temps màxim assolit. Fi de l'entrenament")
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        if time.time() - start_time > max_total_time-60:
            break
        # --- VALIDACIÓN ---
        val_acc = evaluate(model, val_loader, f"Època {epoch}[{time.time() - start_epoch:.2f}s]")
        print("Tiempo restant: {:.2f} segons".format(max_total_time - (time.time() - start_time)))
        if  val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Nou millor model guardat amb {best_val_acc*100:.2f}% de accuracy de validació")
        scheduler.step(val_acc)  # Ajusta el LR basado en la accuracy de validación

    # ---------- Avaluació final ----------
    print("Entrenament completat. Evaluant el millor model...")
    best_model = load_best_model(model_save_path, num_classes)
    # Evaluar el modelo en el conjunto de entrenamiento y validación
    train_acc = evaluate(best_model,dataloader, "Train (subset)")
    val_acc = evaluate(best_model, val_loader, "Validation (final)")

    print(f"\nFinal Metrics:")
    print(f"   Train Accuracy: {train_acc * 100:.2f}%")
    print(f"   Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Temps total: {time.time() - start_time:.2f} segons")
