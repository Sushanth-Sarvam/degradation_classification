# 📚 Gujarati Document Readability Classification

A machine learning project for automatically classifying the readability quality of Gujarati documents using computer vision and deep learning techniques.

## 🎯 Project Overview

This project tackles the challenge of **automatically determining document readability** for Gujarati text documents. Using a combination of state-of-the-art feature extraction models and machine learning algorithms, we achieve **78.81% test accuracy** in classifying documents as "readable" or "non-readable".



## 🗂️ Dataset Information

- **Total Images**: 1,158 Gujarati document pages
- **Training Set**: 771 images from 24 books
- **Test Set**: 387 images from 11 books  
- **Split Strategy**: Book-level stratification to prevent data leakage
- **Classes**: Binary (Readable vs Non-readable)
- **Language**: Gujarati script documents

## 🛠️ Project Structure

```
├── 📁 data/                    # Main dataset files
├── 📁 best_experiment/         # Best model results and metrics
├── 📁 readability_training/    # Training pipeline and experiments
│   ├── 📁 scripts/            # Training and evaluation scripts
│   ├── 📁 experiments/        # All experiment results
│   ├── 📁 embeddings/         # Extracted features from different models
│   └── 📁 examples/           # Sample images and model artifacts
├── 📁 reports/                # Dataset analysis reports
└── 📁 scripts/                # Utility scripts
```

## 🚀 Quick Start

### Prerequisites
```bash
# Activate the project environment
conda activate akshar

# Required packages (install if needed)
pip install scikit-learn xgboost opencv-python pillow numpy pandas matplotlib seaborn
pip install transformers torch torchvision ultralytics efficientnet-pytorch
```

### 🏃‍♂️ Running Experiments

#### 1. Extract Features
```bash
# Extract EfficientNet features (recommended)
python readability_training/scripts/extract_backbone_features.py --model efficientnet

# Extract ResNet50 features  
python readability_training/scripts/extract_backbone_features.py --model resnet50
```

#### 2. Train Models
```bash
# Train best performing model (XGBoost + EfficientNet)
python readability_training/scripts/train_backbone_xgboost_best_test.py

# Train SVM with EfficientNet (second best)
python readability_training/scripts/train_backbone_svm_best_test.py

# Train baseline CNN
python readability_training/scripts/train_cnn_base.py
```

#### 3. Generate Reports
```bash
# Create dataset analysis report
python readability_training/scripts/generate_dataset_report.py

# Create train/test split summary
python readability_training/scripts/create_simple_split_summary.py
```

## 📈 Model Performance Analysis

### 🏆 Best Model: XGBoost + EfficientNet
- **Test Accuracy**: 78.81%
- **Test Precision**: 84.81%
- **Test Recall**: 69.79% 
- **Test F1-Score**: 76.57%
- **Feature Dimension**: 1,280
- **Training Time**: ~27 minutes

### 🔧 Optimal Hyperparameters
```python
{
    'learning_rate': 0.2,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 1.0,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1
}
```

## 🧪 Experiment Details

### Feature Extraction Methods Tested
1. **EfficientNet-B0** 🥇 - Best performance (1,280 features)
2. **ResNet50** 🥈 - Solid baseline (2,048 features) 
3. **YOLOv8n** 🥉 - Object detection features (varies)
4. **LayoutXLM** - Document-specific model (768 features)

### ML Algorithms Compared
1. **XGBoost** - Gradient boosting (best for complex features)
2. **SVM** - Support Vector Machine (fast training)
3. **CNN** - Direct deep learning approach

## 📝 Key Insights & Findings

### ✅ What Works Well
- **EfficientNet features** provide the most discriminative representations
- **XGBoost** handles high-dimensional features better than SVM
- **Shallow trees** (depth=3) prevent overfitting better than deep ones
- **Book-level splitting** essential to prevent data leakage

### ⚠️ Challenges Identified  
- **Overfitting**: Most models show 95-100% train accuracy vs 46-78% test accuracy
- **Class imbalance**: Some books are consistently harder to classify
- **LayoutXLM underperformance**: Document-specific model disappoints
- **YOLOv8n inconsistency**: High variance across different books

## 📊 Files and Artifacts

### 🎯 Best Model Files
- `best_experiment/`: Complete best model results and analysis
- `readability_training/examples/models/best_test_model.pkl`: Trained model ready for inference
- `best_model_predictions.xlsx`: Detailed predictions on test set

### 📈 Analysis Reports
- `readability_training/readability_experiments_summary.txt`: Comprehensive results comparison
- `reports/dataset_report_*.txt`: Dataset statistics and insights
- `best_experiment/performance_metrics.txt`: Detailed metrics breakdown

### 🖼️ Sample Data
- `readability_training/examples/train_images/`: 5 sample training images
- `readability_training/examples/test_images/`: 5 sample test images with predictions
- `readability_training/examples/summaries/`: Metadata and prediction details
