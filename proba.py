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

# ==================== DEFINICIÓ DEL MODEL ====================
class ConvNet(nn.Module):
    """
    Model de xarxa neuronal convolucional per a classificació d'imatges 28x28.
    
    Estructura:
    - Dos blocs convolucionals amb BatchNorm i MaxPool
    - Capes completament connectades amb dropout
    Entrades:
    - Imatges de dimensions [batch_size, 1, 28, 28]
    Sortides:
    - Prediccions de classe de dimensions [batch_size, num_classes]
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Primer bloc convolucional
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Segon bloc convolucional
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Capes completament connectades
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """
        Pas endavant del model.
        
        Args:
            x: Tensor d'entrada de dimensions [batch_size, 1, 28, 28]
        
        Returns:
            Tensor de sortida amb les prediccions
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# ==================== FUNCIONS D'ENTRENAMENT ====================
def train_epoch(model, train_loader, criterion, optimizer, device, start_time, time_limit=600):
    """
    Entrena el model durant una època.
    
    Args:
        model: Model de xarxa neuronal
        train_loader: DataLoader per a les dades d'entrenament
        criterion: Funció de pèrdua
        optimizer: Optimitzador per actualitzar els pesos
        device: Dispositiu on s'executa l'entrenament (CPU/CUDA)
        start_time: Temps d'inici del procés d'entrenament
        time_limit: Temps màxim en segons per l'entrenament complet
        
    Returns:
        epoch_loss: Pèrdua mitjana de l'època
        epoch_acc: Precisió de l'època
        time_exceeded: Booleà que indica si s'ha superat el límit de temps
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        # Mostrar progrés a intervals regulars
        if i % 50 == 0:
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
        
        # Calcular precisió
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    print(" " * 80, end="\r")  # Netejar la línia de progrés
    
    if total > 0:  # En cas d'aturada anticipada
        epoch_loss = running_loss / (i + 1)
        epoch_acc = 100.0 * correct / total
    else:
        epoch_loss, epoch_acc = 0, 0
    
    return epoch_loss, epoch_acc, time.time() - start_time > time_limit

def validate_epoch(model, test_loader, criterion, device, start_time, time_limit=600):
    """
    Valida el model amb el conjunt de test.
    
    Args:
        model: Model de xarxa neuronal
        test_loader: DataLoader per a les dades de validació
        criterion: Funció de pèrdua
        device: Dispositiu on s'executa la validació (CPU/CUDA)
        start_time: Temps d'inici del procés
        time_limit: Temps màxim en segons per tot el procés
        
    Returns:
        epoch_loss: Pèrdua mitjana de la validació
        epoch_acc: Precisió de la validació
        time_exceeded: Booleà que indica si s'ha superat el límit de temps
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            # Mostrar progrés a intervals regulars
            if i % 20 == 0:
                print(f"Validació lot {i}/{len(test_loader)} - Temps transcorregut: {time.time() - start_time:.1f}s", end="\r")
            
            # Comprovar si s'ha superat el límit de temps
            if time.time() - start_time > time_limit:
                print("\nS'ha arribat al límit de 10 minuts! Aturant la validació.")
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
    
    print(" " * 80, end="\r")  # Netejar la línia de progrés
    
    if total > 0:  # En cas d'aturada anticipada
        epoch_loss = running_loss / (i + 1)
        epoch_acc = 100.0 * correct / total
    else:
        epoch_loss, epoch_acc = 0, 0
    
    return epoch_loss, epoch_acc, time.time() - start_time > time_limit

def predict(model, image, device):
    """
    Realitza una predicció amb el model entrenat.
    
    Args:
        model: Model de xarxa neuronal entrenat
        image: Imatge a classificar (tensor)
        device: Dispositiu per processar la predicció
        
    Returns:
        Classe predita (enter)
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image.unsqueeze(0))  # Afegir dimensió de batch
        _, predicted = output.max(1)
        return predicted.item()

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
    plt.savefig('matriu_confusio.png')
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mitjana i desviació estàndard de MNIST
    ])
    
    # Carregar datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Crear data loaders sense multiprocessament
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Inicialitzar model
    model = ConvNet()
    model = model.to(device)
    
    # Funció de pèrdua i optimitzador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Programador de taxa d'aprenentatge
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # Variables per desar el millor model
    best_val_acc = 0
    
    # Bucle d'entrenament
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Comprovar si s'ha superat el límit de temps
        if time.time() - start_time > time_limit:
            print(f"\nS'ha arribat al límit de 10 minuts després de {epoch} èpoques!")
            break
            
        print(f"\nÈpoca {epoch+1}/{num_epochs} (Temps transcorregut: {time.time() - start_time:.1f}s)")
        
        # Entrenar
        train_loss, train_acc, time_exceeded = train_epoch(
            model, train_loader, criterion, optimizer, device, start_time, time_limit
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        if time_exceeded:
            break
        
        # Validar
        val_loss, val_acc, time_exceeded = validate_epoch(
            model, test_loader, criterion, device, start_time, time_limit
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if time_exceeded:
            break
        
        # Actualitzar taxa d'aprenentatge
        scheduler.step(val_loss)
        print(f"Taxa d'aprenentatge actual: {optimizer.param_groups[0]['lr']}")
        
        print(f"Pèrdua entrenament: {train_loss:.4f}, Precisió entrenament: {train_acc:.2f}%")
        print(f"Pèrdua validació: {val_loss:.4f}, Precisió validació: {val_acc:.2f}%")
        
        # Desar el millor model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'millor_model.pth')
            print(f"Millor model desat amb precisió de validació: {val_acc:.2f}%")
    
    # Desar model final
    print(f"Temps total d'execució: {time.time() - start_time:.1f} segons")
    torch.save(model.state_dict(), 'model_cnn.pth')
    
    # Generar matriu de confusió
    noms_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Per MNIST
    print("Generant matriu de confusió...")
    plot_confusion_matrix(model, test_loader, device, noms_classes)
    
    # Visualitzar resultats si s'ha completat alguna època
    if train_losses:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Pèrdua Entrenament')
        plt.plot(val_losses, label='Pèrdua Validació')
        plt.xlabel('Època')
        plt.ylabel('Pèrdua')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Precisió Entrenament')
        plt.plot(val_accs, label='Precisió Validació')
        plt.xlabel('Època')
        plt.ylabel('Precisió (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('resultats_entrenament.png')
        plt.show()

if __name__ == "__main__":
    main()