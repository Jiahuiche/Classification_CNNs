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
# ...

# ---------- Reproducibilitat ----------
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)


# ---------- Dataset ----------
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

data_dir = "data/train"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)
# Mostra la primera imatge per pantalla 
img_tensor, label = dataset[0]
print(f"Image shape: {img_tensor.shape}")
plt.imshow(img_tensor.squeeze(), cmap="gray")
plt.title(f"Label: {class_names[label]}")
plt.axis("off")
plt.show()


eval_subset_size = 1000
indices = random.sample(range(len(dataset)), eval_subset_size)
train_eval_subset = Subset(dataset, indices)

# ---------- Validació ----------
val_dir = "data/validation"
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# ---------- Model ----------
class ElVostreModel(nn.Module):
    def __init__(self, num_classes):
        super(ElVostreModel, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Alguns hiperparàmetres d'exemple----------
lr = 0.001
batch_size = 64

max_total_time = 600  # Maxim temps d'entrenament: 10 minuts

# ---------- Exemple d'optimitzador ----------
device = torch.device("cpu")
model = ElVostreModel(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
# ...
criterion = nn.CrossEntropyLoss()

# ---------- DataLoaders ----------
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_eval_loader = DataLoader(train_eval_subset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------- Avaluació ----------
def evaluate(loader, name):
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

start_time = time.time()
epoch = 0

while True:
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
        # ...



    # Validació per època
    val_acc = evaluate(val_loader, f"Validació (després de la època {epoch})")

# Avaluació final
train_acc = evaluate(train_eval_loader, "Train (subset)")
val_acc = evaluate(val_loader, "Validation (final)")
print(f"\nFinal Metrics:")
print(f"   Train Accuracy: {train_acc * 100:.2f}%")
print(f"   Validation Accuracy: {val_acc * 100:.2f}%")
