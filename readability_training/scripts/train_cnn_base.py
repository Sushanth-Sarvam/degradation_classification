#!/usr/bin/env python3
"""
CNN Base Model Training Script

Trains a simple CNN for document readability classification with:
- 256x256 image input size
- 10 epochs training
- Best model saving based on test accuracy
- Comprehensive metrics tracking and plotting
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ReadabilityDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, image_size=(256, 256)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = self.load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
    
    def load_image(self, image_path):
        try:
            # Try loading with OpenCV first
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path).convert('RGB')
                image = np.array(pil_image)
            else:
                # Convert BGR to RGB for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to 256x256
            image = cv2.resize(image, self.image_size)
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a gray image as fallback
            return np.full((self.image_size[0], self.image_size[1], 3), 128, dtype=np.uint8)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block - 256x256 -> 128x128
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            
            # Second block - 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            
            # Third block - 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            
            # Fourth block - 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            
            # Fifth block - 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
        )
        
        # Adaptive pooling to handle any remaining size variations
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'test_loss': [],
            'test_accuracy': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': []
        }
    
    def update(self, epoch, train_metrics, test_metrics):
        self.history['epoch'].append(epoch)
        
        for key, value in train_metrics.items():
            self.history[f'train_{key}'].append(value)
        
        for key, value in test_metrics.items():
            self.history[f'test_{key}'].append(value)
    
    def save_history(self, filepath):
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
        
        # Also save as JSON
        json_filepath = str(filepath).replace('.csv', '.json')
        with open(json_filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, f1"""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    
    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on given dataloader"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images).squeeze()
            if outputs.dim() == 0:  # Single sample case
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_predictions)
    metrics['loss'] = avg_loss
    
    return metrics

def create_plots(tracker, save_dir):
    """Create comprehensive training plots"""
    plt.style.use('seaborn-v0_8')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CNN Base Training Metrics', fontsize=16, fontweight='bold')
    
    epochs = tracker.history['epoch']
    
    # Accuracy plot
    axes[0, 0].plot(epochs, tracker.history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(epochs, tracker.history['test_accuracy'], 'r-', label='Test Accuracy', linewidth=2)
    axes[0, 0].set_title('Accuracy vs Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Precision plot
    axes[0, 1].plot(epochs, tracker.history['train_precision'], 'b-', label='Train Precision', linewidth=2)
    axes[0, 1].plot(epochs, tracker.history['test_precision'], 'r-', label='Test Precision', linewidth=2)
    axes[0, 1].set_title('Precision vs Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Recall plot
    axes[1, 0].plot(epochs, tracker.history['train_recall'], 'b-', label='Train Recall', linewidth=2)
    axes[1, 0].plot(epochs, tracker.history['test_recall'], 'r-', label='Test Recall', linewidth=2)
    axes[1, 0].set_title('Recall vs Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Loss plot
    axes[1, 1].plot(epochs, tracker.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[1, 1].plot(epochs, tracker.history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[1, 1].set_title('Loss vs Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual metric plots
    metrics = ['accuracy', 'precision', 'recall']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, tracker.history[f'train_{metric}'], 'b-', label=f'Train {metric.title()}', linewidth=2)
        plt.plot(epochs, tracker.history[f'test_{metric}'], 'r-', label=f'Test {metric.title()}', linewidth=2)
        plt.title(f'{metric.title()} vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.savefig(save_dir / f'{metric}_vs_epochs.png', dpi=300, bbox_inches='tight')
        plt.close()

def load_data_from_split():
    """Load training and test data from the train_test_split.json"""
    with open('train_test_split.json', 'r') as f:
        split_data = json.load(f)
    
    # Get book lists
    train_books = split_data['training_book_list']
    test_books = split_data['testing_book_list']
    
    # Load full dataset
    df = pd.read_excel('data/Quality.xlsx')
    
    # Split data based on books
    train_df = df[df['Book Name'].isin(train_books)]
    test_df = df[df['Book Name'].isin(test_books)]
    
    print(f"Training data: {len(train_df)} images from {len(train_books)} books")
    print(f"Test data: {len(test_df)} images from {len(test_books)} books")
    
    return train_df, test_df

def main():
    print("="*70)
    print("🚀 CNN BASE MODEL TRAINING")
    print("="*70)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/cnn_base_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    
    print(f"📁 Experiment directory: {exp_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("📊 Loading data...")
    train_df, test_df = load_data_from_split()
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ReadabilityDataset(
        train_df['Image Path'].tolist(),
        train_df['Readability Bookwise'].tolist(),
        transform=train_transform
    )
    
    test_dataset = ReadabilityDataset(
        test_df['Image Path'].tolist(),
        test_df['Readability Bookwise'].tolist(),
        transform=test_transform
    )
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"🔄 Batch size: {batch_size}")
    print(f"📦 Train batches: {len(train_loader)}")
    print(f"📦 Test batches: {len(test_loader)}")
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize metrics tracker
    tracker = MetricsTracker()
    
    # Training loop
    num_epochs = 10
    best_test_accuracy = 0.0
    best_epoch = 0
    
    print(f"\n🏋️  Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            
            # Handle single sample case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(train_labels, train_predictions)
        train_metrics['loss'] = avg_train_loss
        
        # Evaluation phase
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(test_metrics['loss'])
        
        # Update metrics tracker
        tracker.update(epoch + 1, train_metrics, test_metrics)
        
        # Print epoch results
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}")
        print(f"  Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, "
              f"Prec: {test_metrics['precision']:.4f}, Rec: {test_metrics['recall']:.4f}")
        
        # Save best model
        if test_metrics['accuracy'] > best_test_accuracy:
            best_test_accuracy = test_metrics['accuracy']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), exp_dir / "models" / "best_model.pth")
            print(f"  ✅ New best model saved! Test Accuracy: {best_test_accuracy:.4f}")
    
    print(f"\n🏆 Training completed!")
    print(f"Best test accuracy: {best_test_accuracy:.4f} at epoch {best_epoch}")
    
    # Save training history
    tracker.save_history(exp_dir / "data" / "training_history.csv")
    
    # Create plots
    print("📊 Creating plots...")
    create_plots(tracker, exp_dir / "plots")
    
    # Save experiment config
    config = {
        "model": "SimpleCNN",
        "image_size": [256, 256],
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "criterion": "BCELoss",
        "best_test_accuracy": float(best_test_accuracy),
        "best_epoch": best_epoch,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "training_images": len(train_df),
        "test_images": len(test_df)
    }
    
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ All results saved to: {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    main() 