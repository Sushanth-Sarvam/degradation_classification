#!/usr/bin/env python3
"""
Backbone Feature Extraction Script

Extracts features from multiple backbone models and saves them for later use:
- ResNet50 (ImageNet pretrained)
- EfficientNet-B0 (ImageNet pretrained) 
- YOLOv8n (COCO pretrained)
- LayoutXLM (Multilingual document pretrained)
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import logging

# Transformers for LayoutXLM
try:
    from transformers import LayoutXLMProcessor, LayoutLMv2Model
    LAYOUTXLM_AVAILABLE = True
except ImportError:
    LAYOUTXLM_AVAILABLE = False

# YOLOv8 (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLOv8 not available. Install ultralytics library.")
    YOLO_AVAILABLE = False

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, target_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = self.load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32), image_path
    
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
            
            # Resize to target size if specified
            if self.target_size is not None:
                image = cv2.resize(image, self.target_size)
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a gray image as fallback
            if self.target_size is not None:
                return np.full((self.target_size[0], self.target_size[1], 3), 128, dtype=np.uint8)
            else:
                # For LayoutXLM which doesn't have target_size, return a default size
                return np.full((224, 224, 3), 128, dtype=np.uint8)

class FeatureExtractor:
    def __init__(self, backbone_name, device='cuda'):
        self.backbone_name = backbone_name
        self.device = device
        self.model = None
        self.processor = None
        self.feature_dim = None
        self.input_size = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the specified backbone model"""
        if self.backbone_name == 'resnet50':
            self._load_resnet50()
        elif self.backbone_name == 'efficientnet':
            self._load_efficientnet()
        elif self.backbone_name == 'yolov8n':
            self._load_yolov8n()
        elif self.backbone_name == 'layoutxlm':
            self._load_layoutxlm()
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
    
    def _load_resnet50(self):
        """Load ResNet50 backbone"""
        self.model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        self.feature_dim = 2048
        self.input_size = (224, 224)
        print(f"✅ ResNet50 loaded - Feature dim: {self.feature_dim}")
    
    def _load_efficientnet(self):
        """Load EfficientNet-B0 backbone"""
        from torchvision.models import efficientnet_b0
        self.model = efficientnet_b0(pretrained=True)
        # Remove classifier
        self.model.classifier = nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        self.feature_dim = 1280
        self.input_size = (224, 224)
        print(f"✅ EfficientNet-B0 loaded - Feature dim: {self.feature_dim}")
    
    def _load_yolov8n(self):
        """Load YOLOv8n backbone"""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not available. Install ultralytics.")
        
        # Load YOLOv8n and extract backbone
        yolo_model = YOLO('yolov8n.pt')
        self.model = yolo_model.model.model[:9]  # Extract backbone layers (0-8)
        self.model.eval()
        
        # Add global average pooling
        self.model = nn.Sequential(
            self.model,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.model.to(self.device)
        self.feature_dim = 1024  # YOLOv8n backbone output channels
        self.input_size = (640, 640)
        print(f"✅ YOLOv8n backbone loaded - Feature dim: {self.feature_dim}")
    
    def _load_layoutxlm(self):
        """Load LayoutXLM model"""
        try:
            from transformers import LayoutXLMProcessor, LayoutLMv2Model
        except ImportError:
            raise ImportError("LayoutXLM not available. Install transformers.")
        
        self.processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
        self.model = LayoutLMv2Model.from_pretrained("microsoft/layoutxlm-base")
        self.model.eval()
        self.model.to(self.device)
        self.feature_dim = 768  # LayoutXLM hidden size
        self.input_size = None  # Variable size for LayoutXLM
        print(f"✅ LayoutXLM loaded - Feature dim: {self.feature_dim}")
    
    def get_transforms(self):
        """Get appropriate transforms for the backbone"""
        if self.backbone_name == 'layoutxlm':
            # LayoutXLM handles its own preprocessing
            return None
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, dataloader, output_dir):
        """Extract features from dataloader and save them"""
        features_list = []
        labels_list = []
        paths_list = []
        
        print(f"🔍 Extracting {self.backbone_name} features...")
        
        with torch.no_grad():
            for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc=f"Extracting {self.backbone_name}")):
                if self.backbone_name == 'layoutxlm':
                    # Special handling for LayoutXLM
                    batch_features = self._extract_layoutxlm_features(paths)
                else:
                    # Standard CNN feature extraction
                    images = images.to(self.device)
                    features = self.model(images)
                    
                    # Flatten if needed
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    
                    batch_features = features.cpu().numpy()
                
                features_list.append(batch_features)
                labels_list.extend(labels.numpy())
                paths_list.extend(paths)
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Concatenate all features
        all_features = np.vstack(features_list)
        all_labels = np.array(labels_list)
        
        print(f"✅ Extracted features shape: {all_features.shape}")
        print(f"✅ Labels shape: {all_labels.shape}")
        
        return all_features, all_labels, paths_list
    
    def _extract_layoutxlm_features(self, image_paths):
        """Extract features using LayoutXLM"""
        batch_features = []
        
        for image_path in image_paths:
            try:
                # Load image with PIL for LayoutXLM
                image = Image.open(image_path).convert('RGB')
                
                # Process image (LayoutXLM handles variable sizes)
                encoding = self.processor(image, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                # Move to device
                for key in encoding:
                    if isinstance(encoding[key], torch.Tensor):
                        encoding[key] = encoding[key].to(self.device)
                
                # Extract features
                outputs = self.model(**encoding)
                
                # Use pooler output for document-level features
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    # Fallback to mean of hidden states
                    features = outputs.last_hidden_state.mean(dim=1)
                
                batch_features.append(features.cpu().numpy())
                
            except Exception as e:
                print(f"Error processing {image_path} with LayoutXLM: {e}")
                # Return zero features for failed cases
                batch_features.append(np.zeros((1, self.feature_dim)))
        
        return np.vstack(batch_features)

def load_data_from_split():
    """Load training and test data from the train_test_split.json"""
    # Try to find the split file in data directory with split type
    data_dir = Path('data')
    split_files = list(data_dir.glob('train_test_split_*.json'))
    
    if split_files:
        # Use the most recent split file
        split_file = max(split_files, key=lambda x: x.stat().st_mtime)
        print(f"📄 Loading split data from: {split_file}")
    else:
        # Fallback to legacy location
        split_file = 'data/train_test_split.json'
        print(f"📄 Loading split data from: {split_file}")
    
    with open(split_file, 'r') as f:
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

def setup_logging(output_dir):
    """Setup logging for the extraction process"""
    log_file = output_dir / "extraction_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_backbone_features(backbone_name):
    """Extract features for a specific backbone"""
    print(f"\n{'='*70}")
    print(f"🚀 EXTRACTING {backbone_name.upper()} FEATURES")
    print(f"{'='*70}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"embeddings/{backbone_name}_embeddings_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting {backbone_name} feature extraction")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("📊 Loading data...")
    train_df, test_df = load_data_from_split()
    
    # Initialize feature extractor
    extractor = FeatureExtractor(backbone_name, device)
    transforms_func = extractor.get_transforms()
    
    # Create datasets and dataloaders
    if backbone_name == 'layoutxlm':
        # LayoutXLM doesn't need transforms, uses original images
        batch_size = 1  # Must be 1 for LayoutXLM due to variable image sizes
        target_size = None
    else:
        batch_size = 16
        target_size = extractor.input_size
    
    train_dataset = ImageDataset(
        train_df['Image Path'].tolist(),
        train_df['Readability Bookwise'].tolist(),
        transform=transforms_func,
        target_size=target_size
    )
    
    test_dataset = ImageDataset(
        test_df['Image Path'].tolist(),
        test_df['Readability Bookwise'].tolist(),
        transform=transforms_func,
        target_size=target_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"🔄 Batch size: {batch_size}")
    print(f"📦 Train batches: {len(train_loader)}")
    print(f"📦 Test batches: {len(test_loader)}")
    
    # Extract training features
    print("\n📊 Extracting training features...")
    train_features, train_labels, train_paths = extractor.extract_features(train_loader, output_dir)
    
    # Extract test features
    print("\n📊 Extracting test features...")
    test_features, test_labels, test_paths = extractor.extract_features(test_loader, output_dir)
    
    # Save features
    print("\n💾 Saving features...")
    np.save(output_dir / "train_features.npy", train_features)
    np.save(output_dir / "test_features.npy", test_features)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "test_labels.npy", test_labels)
    
    # Save image paths for proper mapping
    print("💾 Saving image paths...")
    with open(output_dir / "train_paths.json", 'w') as f:
        json.dump(train_paths, f, indent=2)
    with open(output_dir / "test_paths.json", 'w') as f:
        json.dump(test_paths, f, indent=2)
    
    # Save feature info
    feature_info = {
        "backbone_name": backbone_name,
        "feature_dimension": int(extractor.feature_dim),
        "input_size": extractor.input_size,
        "train_samples": int(len(train_features)),
        "test_samples": int(len(test_features)),
        "extraction_date": datetime.now().isoformat(),
        "device": str(device),
        "train_feature_shape": list(train_features.shape),
        "test_feature_shape": list(test_features.shape),
        "train_class_distribution": {
            "readable": int(np.sum(train_labels == 1)),
            "non_readable": int(np.sum(train_labels == 0))
        },
        "test_class_distribution": {
            "readable": int(np.sum(test_labels == 1)),
            "non_readable": int(np.sum(test_labels == 0))
        },
        "files_saved": {
            "train_features": "train_features.npy",
            "test_features": "test_features.npy", 
            "train_labels": "train_labels.npy",
            "test_labels": "test_labels.npy",
            "train_paths": "train_paths.json",
            "test_paths": "test_paths.json"
        }
    }
    
    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\n✅ {backbone_name} feature extraction completed!")
    print(f"📁 Features saved to: {output_dir}")
    print(f"📊 Train features: {train_features.shape}")
    print(f"📊 Test features: {test_features.shape}")
    print(f"📋 Image paths saved: {len(train_paths)} train, {len(test_paths)} test")
    
    logger.info(f"{backbone_name} feature extraction completed successfully")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Extract backbone features for readability classification')
    parser.add_argument('--backbone', choices=['resnet50', 'efficientnet', 'yolov8n', 'layoutxlm', 'all'], 
                       default='all', help='Backbone model to use for feature extraction')
    
    args = parser.parse_args()
    
    if args.backbone == 'all':
        backbones = ['resnet50', 'efficientnet', 'yolov8n', 'layoutxlm']
    else:
        backbones = [args.backbone]
    
    print("="*70)
    print("🚀 BACKBONE FEATURE EXTRACTION PIPELINE")
    print("="*70)
    print(f"📋 Backbones to process: {', '.join(backbones)}")
    
    results = {}
    
    for backbone in backbones:
        try:
            output_dir = extract_backbone_features(backbone)
            results[backbone] = str(output_dir)
        except Exception as e:
            print(f"❌ Failed to extract {backbone} features: {e}")
            results[backbone] = f"FAILED: {e}"
    
    print(f"\n{'='*70}")
    print("📋 EXTRACTION SUMMARY")
    print(f"{'='*70}")
    
    for backbone, result in results.items():
        if "FAILED" in result:
            print(f"❌ {backbone}: {result}")
        else:
            print(f"✅ {backbone}: {result}")
    
    print(f"\n🎉 Feature extraction pipeline completed!")

if __name__ == "__main__":
    main() 