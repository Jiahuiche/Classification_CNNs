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
import itertools
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

# Import your existing model and augmentation
from jiahuichen1estudiantatupcedu import EnhancedCNNModel, AdvancedAugmentation

class GridSearch:
    def __init__(self, data_dir="data/train", val_dir="data/validation"):
        # Set random seeds for reproducibility
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        
        # Configure device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Parameters for timing
        self.max_time_per_config = 600  # 10 minutes per configuration
        self.results_dir = "grid_search_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define hyperparameter grid
        self.param_grid = {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [32, 64, 128],
            'optimizer': ['adam', 'adamw', 'sgd'],
            'dropout_rate': [0.2, 0.3, 0.5],
            'weight_decay': [0, 1e-5, 1e-4],
            'scheduler': ['plateau', 'onecycle', None],
            'augmentation_prob': [0.7, 1.0]
        }
        
        # Load datasets
        self.data_dir = data_dir
        self.val_dir = val_dir
        self._prepare_data()
        
        # Results storage
        self.results = []
        self.best_config = None
        self.best_val_acc = 0.0
        
    def _prepare_data(self):
        """Prepare the datasets for training and validation"""
        # Set up basic transform
        self.basic_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor()
        ])
        
        # Load datasets
        self.full_dataset = datasets.ImageFolder(
            root=self.data_dir, transform=self.basic_transform
        )
        self.val_dataset = datasets.ImageFolder(
            root=self.val_dir, transform=self.basic_transform
        )
        
        self.class_names = self.full_dataset.classes
        self.num_classes = len(self.class_names)
        
        # Create balanced dataset - using 10000 samples per class
        self.samples_per_class = 10000
        balanced_indices = []
        
        # Group indices by class
        class_indices = {}
        for idx, (_, label) in enumerate(self.full_dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Sample from each class
        for label, indices in class_indices.items():
            if len(indices) <= self.samples_per_class:
                balanced_indices.extend(indices)
                print(f"Class {self.class_names[label]}: using all {len(indices)} samples")
            else:
                sampled_indices = random.sample(indices, self.samples_per_class)
                balanced_indices.extend(sampled_indices)
                print(f"Class {self.class_names[label]}: using {self.samples_per_class} samples")
        
        # Create the balanced subset
        self.balanced_subset = Subset(self.full_dataset, balanced_indices)
        print(f"Original dataset: {len(self.full_dataset)} images")
        print(f"Balanced dataset: {len(self.balanced_subset)} images")
    
    def _get_optimizer(self, model, config):
        """Get optimizer based on configuration"""
        if config['optimizer'] == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
    
    def _get_scheduler(self, optimizer, config, steps_per_epoch, epochs=10):
        """Get learning rate scheduler based on configuration"""
        if config['scheduler'] == 'plateau':
            return ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=2
            )
        elif config['scheduler'] == 'onecycle':
            return OneCycleLR(
                optimizer, 
                max_lr=config['learning_rate']*10,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                pct_start=0.3
            )
        else:
            return None  # No scheduler
    
    def train_and_evaluate(self, config):
        """Train and evaluate a model with the given configuration"""
        print(f"\nTesting configuration: {config}")
        config_start_time = time.time()
        
        # Create augmentation
        augmentation = AdvancedAugmentation(prob=config['augmentation_prob'])
        
        # Create dataset with augmentation
        dataset = AugmentedDataset(self.balanced_subset, augmentation)
        
        # Create data loaders
        num_core = multiprocessing.cpu_count()
        num_workers = max(1, num_core-2)
        pin_memory = self.device.type == 'cuda'
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=pin_memory
        )
        
        # Create model
        model = EnhancedCNNModel(
            self.num_classes, 
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        # Get optimizer and scheduler
        optimizer = self._get_optimizer(model, config)
        scheduler = self._get_scheduler(
            optimizer, 
            config, 
            len(dataloader), 
            epochs=10
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epoch = 0
        best_epoch_val_acc = 0.0
        model_path = os.path.join(
            self.results_dir, 
            f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        )
        
        while True:
            epoch += 1
            model.train()
            loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(loop):
                # Check if time limit exceeded
                if time.time() - config_start_time > self.max_time_per_config:
                    print(f"Time limit reached after {epoch-1} full epochs")
                    break
                
                # Forward pass
                images = images.to(self.device, non_blocking=pin_memory)
                labels = labels.to(self.device, non_blocking=pin_memory)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Step OneCycle scheduler if used
                if config['scheduler'] == 'onecycle':
                    scheduler.step()
                
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            # Exit if time limit reached
            if time.time() - config_start_time > self.max_time_per_config:
                break
            
            # Validation
            val_acc = self._evaluate(model, val_loader)
            print(f"Epoch {epoch} - Val Accuracy: {val_acc*100:.2f}%")
            
            # Step plateau scheduler if used
            if config['scheduler'] == 'plateau':
                scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_epoch_val_acc:
                best_epoch_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
        
        # Load best model and evaluate
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        
        # Final evaluation
        val_acc = self._evaluate(model, val_loader)
        
        # Save results
        result = {
            'config': config,
            'val_accuracy': val_acc,
            'num_epochs': epoch,
            'training_time': time.time() - config_start_time,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.results.append(result)
        
        # Update best configuration if necessary
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_config = config
            best_model_path = os.path.join(self.results_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model with validation accuracy: {val_acc*100:.2f}%")
        
        # Clean up temporary model
        if os.path.exists(model_path):
            os.remove(model_path)
        
        return val_acc
    
    def _evaluate(self, model, loader):
        """Evaluate model on the given data loader"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total
    
    def run_grid_search(self, num_configs=None):
        """Run grid search over hyperparameters"""
        # Generate all configurations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        all_configs = list(itertools.product(*values))
        all_configs = [dict(zip(keys, config)) for config in all_configs]
        
        # Use subset of configurations if specified
        if num_configs and num_configs < len(all_configs):
            configs = random.sample(all_configs, num_configs)
        else:
            configs = all_configs
        
        print(f"Running grid search with {len(configs)} configurations")
        print(f"Each configuration will be trained for max {self.max_time_per_config} seconds")
        
        # Train and evaluate each configuration
        for i, config in enumerate(configs):
            print(f"\nConfiguration {i+1}/{len(configs)}")
            self.train_and_evaluate(config)
            
            # Save results after each configuration
            self._save_results()
        
        # Final report
        self._print_summary()
    
    def _save_results(self):
        """Save grid search results to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump({
                'results': self.results,
                'best_config': self.best_config,
                'best_val_acc': self.best_val_acc
            }, f, indent=2)
    
    def _print_summary(self):
        """Print summary of grid search results"""
        print("\n" + "="*50)
        print("Grid Search Results")
        print("="*50)
        
        # Sort by validation accuracy
        sorted_results = sorted(
            self.results, 
            key=lambda x: x['val_accuracy'], 
            reverse=True
        )
        
        # Print top 5 configurations
        print("\nTop 5 Configurations:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"\n{i+1}. Validation Accuracy: {result['val_accuracy']*100:.2f}%")
            for key, value in result['config'].items():
                print(f"   {key}: {value}")
            print(f"   Training time: {result['training_time']:.2f} seconds")
            print(f"   Epochs completed: {result['num_epochs']}")
        
        # Print best configuration
        print("\nBest Configuration:")
        print(f"Validation Accuracy: {self.best_val_acc*100:.2f}%")
        if self.best_config:
            for key, value in self.best_config.items():
                print(f"   {key}: {value}")
        
        # Save path info
        print(f"\nBest model saved to: {os.path.join(self.results_dir, 'best_model.pth')}")
        print(f"All results saved in: {self.results_dir}")

# Required for class import
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

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    
    # Create and run grid search
    grid_search = GridSearch()
    
    # Run grid search with a subset of configurations
    # Adjust this number based on available time
    grid_search.run_grid_search(num_configs=10)


'''Best Configuration:
Validation Accuracy: 74.64%
   learning_rate: 0.0001
   batch_size: 64
   optimizer: adamw
   dropout_rate: 0.3
   weight_decay: 0.0001
   scheduler: onecycle
   augmentation_prob: 0.7'''