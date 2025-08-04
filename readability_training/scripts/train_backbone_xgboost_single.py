#!/usr/bin/env python3
"""
Single Hyperparameter XGBoost Training on Backbone Features
Quick testing with one parameter combination
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class SingleXGBoostTrainer:
    def __init__(self, backbone_name, embeddings_dir, experiment_dir):
        self.backbone_name = backbone_name
        self.embeddings_dir = Path(embeddings_dir)
        self.experiment_dir = Path(experiment_dir)
        self.feature_info = {}
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.experiment_dir / 'training.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_features(self):
        """Load features from embeddings directory"""
        try:
            # Find the latest embeddings directory
            embedding_dirs = list(self.embeddings_dir.glob(f"{self.backbone_name}_embeddings_*"))
            if not embedding_dirs:
                raise FileNotFoundError(f"No embeddings found for {self.backbone_name}")
            
            latest_dir = max(embedding_dirs, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Loading {self.backbone_name} features from {latest_dir}")
            
            # Load features
            X_train = np.load(latest_dir / 'train_features.npy')
            X_test = np.load(latest_dir / 'test_features.npy')
            y_train = np.load(latest_dir / 'train_labels.npy')
            y_test = np.load(latest_dir / 'test_labels.npy')
            
            # Store feature info
            self.feature_info = {
                'feature_dimension': X_train.shape[1],
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0]
            }
            
            self.logger.info(f"✅ Features loaded successfully")
            self.logger.info(f"📊 Train features: {X_train.shape}")
            self.logger.info(f"📊 Test features: {X_test.shape}")
            self.logger.info(f"📊 Feature dimension: {self.feature_info['feature_dimension']}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load features: {e}")
            raise
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost with SINGLE hyperparameter combination"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        self.logger.info("🚀 Starting SINGLE hyperparameter XGBoost training...")
        
        # SINGLE parameter combination for quick testing
        single_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 1
        }
        
        self.logger.info(f"🔧 Using SINGLE hyperparameter combination:")
        for param, value in single_params.items():
            self.logger.info(f"   {param}: {value}")
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(**single_params)
        
        # Train model
        start_time = datetime.now()
        self.logger.info("🚀 Starting training...")
        
        xgb_model.fit(X_train, y_train)
        training_time = datetime.now() - start_time
        
        self.logger.info(f"✅ Training completed!")
        self.logger.info(f"⏱️  Total training time: {training_time}")
        
        # Evaluate on train and test sets
        self.logger.info("📊 Evaluating model...")
        train_metrics = self._evaluate_model(xgb_model, X_train, y_train, "train")
        test_metrics = self._evaluate_model(xgb_model, X_test, y_test, "test")
        
        # Create results
        results = {
            "backbone_name": self.backbone_name,
            "feature_dimension": self.feature_info['feature_dimension'],
            "training_time_seconds": training_time.total_seconds(),
            "hyperparameters": single_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
        
        # Save model and results
        self._save_model_and_results(xgb_model, results, X_test, y_test)
        
        return xgb_model, results
    
    def _evaluate_model(self, model, X, y, dataset_name):
        """Evaluate model and return metrics"""
        self.logger.info(f"🔍 Evaluating on {dataset_name} set...")
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0))
        }
        
        try:
            fpr, tpr, _ = roc_curve(y, y_prob)
            metrics["roc_auc"] = float(auc(fpr, tpr))
        except:
            metrics["roc_auc"] = 0.0
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        # Prediction distribution
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        pred_dist = dict(zip(pred_unique, pred_counts))
        
        self.logger.info(f"📊 {dataset_name.upper()} RESULTS:")
        self.logger.info(f"   📈 Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"   📈 Precision: {metrics['precision']:.4f}")
        self.logger.info(f"   📈 Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"   📈 F1 Score:  {metrics['f1']:.4f}")
        self.logger.info(f"   📈 ROC AUC:   {metrics['roc_auc']:.4f}")
        self.logger.info(f"   📊 True distribution: {class_dist}")
        self.logger.info(f"   📊 Pred distribution: {pred_dist}")
        
        return metrics
    
    def _save_model_and_results(self, model, results, X_test, y_test):
        """Save model, results, and create plots"""
        # Save model
        model_path = self.experiment_dir / 'best_model.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save results
        results_path = self.experiment_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Model saved to: {model_path}")
        self.logger.info(f"💾 Results saved to: {results_path}")
        
        # Create plots
        self._create_plots(model, X_test, y_test, results)
    
    def _create_plots(self, model, X_test, y_test, results):
        """Create comprehensive plots"""
        self.logger.info("📊 Creating visualization plots...")
        
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Feature Importance Plot
        self.logger.info("   📈 Creating feature importance plot...")
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=30)
        plt.title(f'Feature Importance - {self.backbone_name.upper()}', fontsize=16, fontweight='bold')
        plt.xlabel('F Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        self.logger.info("   📈 Creating confusion matrix...")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {self.backbone_name.upper()}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
        self.logger.info("   📈 Creating ROC curve...")
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.backbone_name.upper()}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Prediction Distribution
        self.logger.info("   📈 Creating prediction distribution...")
        plt.figure(figsize=(10, 6))
        
        # Plot probability distribution
        plt.subplot(1, 2, 1)
        plt.hist(y_prob[y_test == 0], bins=30, alpha=0.7, label='Non-readable', color='red')
        plt.hist(y_prob[y_test == 1], bins=30, alpha=0.7, label='Readable', color='green')
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot feature importance top 10
        plt.subplot(1, 2, 2)
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[-10:]
        top_features = [f'Feature {i}' for i in top_indices]
        top_importance = importance[top_indices]
        
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Training Summary
        self.logger.info("   📈 Creating training summary...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Metrics comparison
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
        train_metrics = [results['train_metrics']['accuracy'], 
                        results['train_metrics']['precision'],
                        results['train_metrics']['recall'],
                        results['train_metrics']['f1'],
                        results['train_metrics']['roc_auc']]
        test_metrics = [results['test_metrics']['accuracy'],
                       results['test_metrics']['precision'],
                       results['test_metrics']['recall'],
                       results['test_metrics']['f1'],
                       results['test_metrics']['roc_auc']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax1.bar(x - width/2, train_metrics, width, label='Train', color='skyblue')
        ax1.bar(x + width/2, test_metrics, width, label='Test', color='lightcoral')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature importance bar
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[-15:]
        ax2.barh(range(len(top_indices)), importance[top_indices])
        ax2.set_xlabel('Importance')
        ax2.set_title('Top 15 Feature Importance')
        ax2.grid(True, alpha=0.3)
        
        # ROC curve
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('Confusion Matrix')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')
        
        plt.suptitle(f'{self.backbone_name.upper()} - Training Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ 5 plots created successfully")
        self.logger.info(f"📁 Plots saved to: {plots_dir}")
        self.logger.info(f"📊 Plot files:")
        self.logger.info(f"   - feature_importance.png")
        self.logger.info(f"   - confusion_matrix.png")
        self.logger.info(f"   - roc_curve.png")
        self.logger.info(f"   - prediction_analysis.png")
        self.logger.info(f"   - training_summary.png")

def find_latest_embeddings(backbone_name):
    """Find the latest embeddings directory for a backbone"""
    embeddings_dir = Path('embeddings')
    embedding_dirs = list(embeddings_dir.glob(f"{backbone_name}_embeddings_*"))
    
    if not embedding_dirs:
        raise FileNotFoundError(f"No embeddings found for {backbone_name}")
    
    return max(embedding_dirs, key=lambda x: x.stat().st_mtime)

def train_backbone_xgboost(backbone_name):
    """Train XGBoost on backbone features with single hyperparameter"""
    try:
        # Find embeddings
        embeddings_dir = find_latest_embeddings(backbone_name)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path('experiments') / f'single_xgboost_{backbone_name}_{timestamp}'
        
        # Train
        trainer = SingleXGBoostTrainer(backbone_name, 'embeddings', experiment_dir)
        X_train, X_test, y_train, y_test = trainer.load_features()
        model, results = trainer.train_xgboost(X_train, X_test, y_train, y_test)
        
        print(f"✅ SINGLE hyperparameter XGBoost training completed for {backbone_name}")
        print(f"📁 Results saved to: {experiment_dir}")
        print(f"📊 Test F1 Score: {results['test_metrics']['f1']:.4f}")
        print(f"📊 Test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
        
    except Exception as e:
        print(f"❌ Failed to train XGBoost for {backbone_name}: {e}")
        raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Single Hyperparameter XGBoost Training')
    parser.add_argument('--backbone', type=str, required=True, 
                       choices=['resnet50', 'efficientnet', 'yolov8n', 'layoutxlm'],
                       help='Backbone to train on')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 SINGLE HYPERPARAMETER XGBOOST TRAINING")
    print("=" * 70)
    print(f"📋 Backbone to train: {args.backbone}")
    print("=" * 70)
    
    train_backbone_xgboost(args.backbone)

if __name__ == "__main__":
    main() 