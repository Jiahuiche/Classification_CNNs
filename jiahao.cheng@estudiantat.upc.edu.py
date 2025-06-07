import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# ==================== CONFIGURACIÓ INICIAL ====================
if __name__ == "__main__":
    device = torch.device("cpu")
    torch.set_num_threads(torch.get_num_threads())
    print(f"Utilitzant {torch.get_num_threads()} fils de CPU")
    print(f"Utilitzant dispositiu: {device}")
    # ---------- Reproducibilitat ----------
    torch.manual_seed(123)
    np.random.seed(123)
# ==================== DEFINICIÓ DEL MODEL ====================
class ConvNet(nn.Module):
    """
    Model de xarxa neuronal convolucional per a classificació d'imatges 28x28.
    
    Estructura:
    - Dos blocs convolucionals amb BatchNorm i MaxPool
    - Capes completament connectades amb dropout
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(25, 50, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            
            nn.Conv2d(50, 75, kernel_size=3, padding=1),
            nn.BatchNorm2d(75),
            nn.ReLU(inplace=True),
            nn.Conv2d(75, 100, kernel_size=3, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            
            nn.Conv2d(100, 125, kernel_size=3, padding=1),
            nn.BatchNorm2d(125),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
                    
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(125*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)  # Pasa por las capas convolucionales
        x = self.classifier(x)  # Pasa por la MLP
        return x

# ==================== FUNCIONS D'ENTRENAMENT ====================
def train_epoch(model, train_loader, criterion, optimizer, device, start_time, time_limit=600, 
                validation_interval=100, val_loader=None, scheduler=None, global_best_val_acc=0):
    """
    Entrena el model durant una època amb validacions parcials.
    
    Args:
        model: Model de xarxa neuronal
        train_loader: DataLoader per a les dades d'entrenament
        criterion: Funció de pèrdua
        optimizer: Optimitzador per actualitzar els pesos
        device: Dispositiu on s'executa l'entrenament (CPU)
        start_time: Temps d'inici del procés d'entrenament
        time_limit: Temps màxim en segons per l'entrenament complet
        validation_interval: Cada quants batches fer una validació parcial
        val_loader: DataLoader de validació
        
    Returns:
        epoch_loss: Pèrdua mitjana de l'època
        epoch_acc: Precisió de l'època
        time_exceeded: Booleà que indica si s'ha superat el límit de temps
        best_val_acc: Millor precisió de validació aconseguida
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    best_val_acc = 0
    best_val_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # Mostrar progrés a intervals regulars
        if i % 100 == 0:
            print(f"Entrenament lot {i}/{len(train_loader)} - Temps transcorregut: {time.time() - start_time:.1f}s", end="\r")
        
        # Comprovar si s'ha superat el límit de temps
        if time.time() - start_time > time_limit:
            print("\nS'ha arribat al límit de 10 minuts! Aturant l'entrenament.")
            break
            
        images, labels = images.to(device), labels.to(device)
        
        # Pas endavant
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Pas enrere i optimització
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calcular Accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Validació parcial cada 'validation_interval' batches
        if val_loader is not None and (i+1) % validation_interval == 0:
            val_loss, val_acc, _ = validate_partial(model, val_loader, criterion, device, start_time, time_limit)
            # Actualizar el scheduler con la pérdida de validación
            if scheduler is not None:
                scheduler.step(val_loss)
            # Desar el millor model si millora
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                if val_acc > global_best_val_acc:
                    torch.save({
                        'batch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'train_loss': running_loss / (i + 1),
                        'train_acc': 100.0 * correct / total
                    }, 'millor_model.pth')
                    if val_acc > 60:
                        print(f"           Nou millor model guardat amb {val_acc:.2f}% de accuracy i {val_loss:.2f} de loss de validació")
            # Tornar a mode entrenament
            model.train()
    
    print(" " * 80, end="\r")  # Netejar la línia de progrés

    if total > 0:  # En cas d'aturada anticipada
        epoch_loss = running_loss / (i + 1)
        epoch_acc = 100.0 * correct / total
    else:
        epoch_loss, epoch_acc = 0, 0

    return epoch_loss, epoch_acc, time.time() - start_time > time_limit, best_val_acc, best_val_loss

def validate_partial(model, val_loader, criterion, device, start_time, time_limit=600, max_batches=None):
    """
    Valida el model parcialment (només una part del dataset).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            # Limitar número de batches para validación rápida
            if max_batches is not None and i >= max_batches:
                break
                
            # Comprovar si s'ha superat el límit de temps
            if time.time() - start_time > time_limit:
                print("\nS'ha arribat al límit de 10 minuts! Aturant la validació parcial.")
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Pas endavant
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Calcular precisió
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    if total > 0:
        partial_loss = running_loss / (i + 1)
        partial_acc = 100.0 * correct / total
    else:
        partial_loss, partial_acc = 0, 0
    
    return partial_loss, partial_acc, time.time() - start_time > time_limit


def load_best_model(path, num_classes):
        """Carga el mejor modelo guardado."""
        print(f"Cargando el mejor modelo desde {path}...")
        model = ConvNet(num_classes)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])       
        model = model.to(device)
        if "Train Loss" in checkpoint:
            print(f"Train Loss: {checkpoint['train_loss']:.2f}, Accuracy Entrenament:{checkpoint['train_acc']:.2f}%")
        print(f"Val Loss: {checkpoint['val_loss']:.2f}, Accuracy Validació: {checkpoint['val_acc']:.2f}%")
        
        return model

def plot_confusion_matrix(model, test_loader, device, class_names=None):
    """
    Genera i mostra la matriu de confusió.
    
    Args:
        model: Model entrenat
        test_loader: DataLoader amb dades de test
        device: Dispositiu (CPU/CUDA)
        class_names: Noms de les classes (opcional)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Si no es proporcionen noms de classes, utilitzar números
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predit')
    plt.ylabel('Real')
    plt.title('Matriu de Confusió')
    plt.show()

# ==================== FUNCIÓ PRINCIPAL ====================
def main():
    """
    Funció principal que executa l'entrenament i l'avaluació del model.
    """
    # Iniciar cronometratge
    start_time = time.time()
    time_limit = 600  # 10 minuts en segons
    
    # Hiperparàmetres
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    # Definir transformacions
    transform_b = transforms.Compose([
    transforms.Grayscale(),  # Convertir a escala de grises si las imágenes son a color
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalización genérica
])
    transform_data_augmentation = transforms.Compose([
        transforms.Grayscale(),  # Convertir a escala de grises si les imatges són a color
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalització genèrica
    ])

    
    # Carregar datasets
    train_dataset = datasets.ImageFolder(root='data/train', transform=transform_data_augmentation)
    test_dataset = datasets.ImageFolder(root='data/validation', transform=transform_b)
    
    # Crear data loaders sense multiprocessament
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ##############################################################################################
    ########### MODIFICAR (AFEGIR TEST) ##########################################################
    """
    test_dataset = datasets.ImageFolder(root='directori del test', transform=transform_b) # modificar root
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    """
    ##############################################################################################
    ##############################################################################################

    # Inicialitzar model
    noms_classes = train_dataset.classes
    num_classes = len(noms_classes)

    model = ConvNet(num_classes)
    model = model.to(device)
    
    # Funció de pèrdua i optimitzador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Programador de taxa d'aprenentatge
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    
    # Bucle d'entrenament
    global_best_val_acc = 0
    global_best_val_loss = 0.0
    for epoch in range(num_epochs):
        # Comprovar si s'ha superat el límit de temps
        if time.time() - start_time > time_limit:
            print(f"\nS'ha arribat al límit de 10 minuts després de {epoch} èpoques!")
            break
            
        print(f"\nÈpoca {epoch+1}/{num_epochs} (Temps transcorregut: {time.time() - start_time:.1f}s)")
        
        # Entrenar
        train_loss, train_acc, time_exceeded, best_partial_val_acc, best_partial_val_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, start_time, time_limit,
            validation_interval=100, val_loader=val_loader, scheduler=scheduler, global_best_val_acc=global_best_val_acc
            )
        if best_partial_val_acc > global_best_val_acc:
            global_best_val_acc = best_partial_val_acc
            global_best_val_loss = best_partial_val_loss
        
        if time_exceeded:
            break
        
        print(f"Taxa d'aprenentatge actual: {optimizer.param_groups[0]['lr']}")
        print(f"Train Loss: {train_loss:.4f}, Accuracy entrenament: {train_acc:.2f}%")
        print(f"Val Loss: {global_best_val_loss:.4f}, Accuracy validació: {global_best_val_acc:.2f}%")
    
    val_loss, val_acc, _ = validate_partial(model, val_loader, criterion, device, start_time, time_limit, max_batches=100)
    if val_acc > global_best_val_acc:
        global_best_val_acc = val_acc
        global_best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss
        }, 'millor_model.pth')
        print(f"Millor model guardat amb {val_acc:.2f}% de accuracy i {val_loss:.2f} de loss de validació")
    # Desar model final
    model = load_best_model('millor_model.pth', num_classes)
    ##############################################################################################
    ################################## Modificar ###########################################

    """
    # Generar matriu de confusió
    print("Generant matriu de confusió...")
    plot_confusion_matrix(model, test_loader, device, noms_classes)
    """
    test_loss, test_acc, _ = validate_partial(model=model, val_loader=val_loader, 
                                              criterion=criterion, device=device, 
                                              start_time=start_time, time_limit=700, 
                                              max_batches=100)
    print(f"Test Loss: {test_loss:.2f}, Accuracy test: {test_acc:.2f}%")

    ##############################################################################################
    ##############################################################################################
    
    print(f"Temps total d'execució: {time.time() - start_time:.1f} segons")
if __name__ == "__main__":
    main()