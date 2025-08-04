#!/usr/bin/env python3
"""
SVM Training on Backbone Features
Support Vector Machine with hyperparameter tuning for document readability classification
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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class SVMTrainer:
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
    
    def train_svm(self, X_train, X_test, y_train, y_test):
        """Train SVM with hyperparameter tuning"""
        self.logger.info("🚀 Starting SVM training with hyperparameter tuning...")
        
        # SVM parameter grid - 24 combinations (3×4×2×1)
        param_grid = {
            'svm__kernel': ['linear', 'rbf', 'poly'],          # 3 kernels
            'svm__C': [0.1, 1, 10, 100],                      # 4 C values
            'svm__gamma': ['scale', 'auto'],                   # 2 gamma values
            'svm__probability': [True]                         # Need for ROC curve
        }
        
        # Create pipeline with scaling (important for SVM)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42))
        ])
        
        # Calculate total combinations
        total_combinations = len(param_grid['svm__kernel']) * len(param_grid['svm__C']) * len(param_grid['svm__gamma']) * len(param_grid['svm__probability'])
        
        self.logger.info(f"🔧 SVM hyperparameter tuning...")
        self.logger.info(f"📊 Total parameter combinations: {total_combinations}")
        self.logger.info(f"🔄 Cross-validation folds: 3")
        self.logger.info(f"⏱️  Estimated time: {total_combinations * 1} - {total_combinations * 3} minutes")
        self.logger.info("📈 Using StandardScaler for feature normalization")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,  # 3-fold CV
            scoring='f1',
            n_jobs=-1,
            verbose=2
        )
        
        # Train model with progress tracking
        start_time = datetime.now()
        self.logger.info("🚀 Starting hyperparameter search...")
        
        grid_search.fit(X_train, y_train)
        training_time = datetime.now() - start_time
        
        self.logger.info(f"✅ Hyperparameter search completed!")
        self.logger.info(f"⏱️  Total training time: {training_time}")
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        self.logger.info(f"🏆 Best parameters found:")
        for param, value in grid_search.best_params_.items():
            self.logger.info(f"   {param}: {value}")
        self.logger.info(f"🏆 Best CV F1 score: {grid_search.best_score_:.4f}")
        
        # Show top 5 parameter combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_5 = results_df.nlargest(min(5, len(results_df)), 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
        self.logger.info("🔝 Top parameter combinations:")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            self.logger.info(f"   {i}. F1: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f}) - {row['params']}")
        
        # Evaluate on train and test sets
        self.logger.info("📊 Evaluating best model on training and test sets...")
        train_metrics = self._evaluate_model(best_model, X_train, y_train, "train")
        test_metrics = self._evaluate_model(best_model, X_test, y_test, "test")
        
        # Create results
        results = {
            "backbone_name": self.backbone_name,
            "model_type": "SVM",
            "feature_dimension": self.feature_info['feature_dimension'],
            "training_time_seconds": training_time.total_seconds(),
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "total_combinations_tested": total_combinations,
            "cv_folds": 3,
            "uses_feature_scaling": True
        }
        
        # Save model and results
        self._save_model_and_results(best_model, results, grid_search, X_test, y_test)
        
        return best_model, results
    
    def _evaluate_model(self, model, X, y, dataset_name):
        """Evaluate model and return comprehensive metrics"""
        self.logger.info(f"🔍 Evaluating on {dataset_name} set ({len(X)} samples)...")
        
        # Predictions with progress
        self.logger.info("   Making predictions...")
        y_pred = model.predict(X)
        self.logger.info("   Calculating probabilities...")
        y_prob = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        self.logger.info("   Computing metrics...")
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0))
        }
        
        # ROC AUC
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
        
        # Log detailed metrics
        self.logger.info(f"📊 {dataset_name.upper()} RESULTS:")
        self.logger.info(f"   📈 Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"   📈 Precision: {metrics['precision']:.4f}")
        self.logger.info(f"   📈 Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"   📈 F1 Score:  {metrics['f1']:.4f}")
        self.logger.info(f"   📈 ROC AUC:   {metrics['roc_auc']:.4f}")
        self.logger.info(f"   📊 True distribution: {class_dist}")
        self.logger.info(f"   📊 Pred distribution: {pred_dist}")
        
        return metrics
    
    def _save_model_and_results(self, model, results, grid_search, X_test, y_test):
        """Save model, results, and create comprehensive output"""
        # Save model
        model_path = self.experiment_dir / 'best_model.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save results
        results_path = self.experiment_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save grid search results
        cv_results_path = self.experiment_dir / 'cv_results.csv'
        pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path, index=False)
        
        # Create performance metrics TXT file
        self._create_performance_txt(results)
        
        self.logger.info(f"💾 Model saved to: {model_path}")
        self.logger.info(f"💾 Results saved to: {results_path}")
        self.logger.info(f"💾 CV results saved to: {cv_results_path}")
        
        # Create plots
        self._create_plots(model, X_test, y_test, grid_search, results)
    
    def _create_performance_txt(self, results):
        """Create detailed performance metrics TXT file"""
        txt_path = self.experiment_dir / 'performance_metrics.txt'
        
        with open(txt_path, 'w') as f:
            f.write(f"📈 {self.backbone_name.upper()} SVM Performance Metrics\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Training Summary:\n")
            f.write(f"- Model Type: Support Vector Machine (SVM)\n")
            f.write(f"- Training Time: {results['training_time_seconds']:.1f} seconds\n")
            f.write(f"- Feature Dimension: {results['feature_dimension']:,} features\n")
            f.write(f"- Training Samples: {self.feature_info['train_samples']} images\n")
            f.write(f"- Test Samples: {self.feature_info['test_samples']} images\n")
            f.write(f"- Hyperparameter Combinations Tested: {results['total_combinations_tested']}\n")
            f.write(f"- Cross-validation Folds: {results['cv_folds']}\n")
            f.write(f"- Feature Scaling: {results['uses_feature_scaling']}\n")
            f.write(f"- Best CV F1 Score: {results['best_cv_score']:.4f}\n\n")
            
            f.write("Best Hyperparameters:\n")
            for param, value in results['best_params'].items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
            
            f.write("Performance Metrics:\n")
            f.write("=" * 20 + "\n\n")
            
            f.write("| Metric      | Train Set | Test Set  | Difference |\n")
            f.write("|-------------|-----------|-----------|------------|\n")
            
            train_metrics = results['train_metrics']
            test_metrics = results['test_metrics']
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                train_val = train_metrics[metric] * 100
                test_val = test_metrics[metric] * 100
                diff = test_val - train_val
                
                f.write(f"| {metric.replace('_', ' ').title():<11} | {train_val:8.2f}% | {test_val:8.2f}% | {diff:+8.2f}% |\n")
            
            f.write(f"\n\nSVM Characteristics:\n")
            f.write(f"- Kernel: {results['best_params'].get('svm__kernel', 'N/A')}\n")
            f.write(f"- C (Regularization): {results['best_params'].get('svm__C', 'N/A')}\n")
            f.write(f"- Gamma: {results['best_params'].get('svm__gamma', 'N/A')}\n")
            f.write(f"- Feature scaling applied before training\n")
            f.write(f"- Suitable for high-dimensional feature spaces\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.logger.info(f"📄 Performance metrics saved to: {txt_path}")
    
    def _create_plots(self, model, X_test, y_test, grid_search, results):
        """Create comprehensive plots for analysis"""
        self.logger.info("📊 Creating visualization plots...")
        
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Confusion Matrix
        self.logger.info("   📈 Creating confusion matrix...")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - SVM ({self.backbone_name.upper()})', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        self.logger.info("   📈 Creating ROC curve...")
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - SVM ({self.backbone_name.upper()})')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Hyperparameter Search Results
        self.logger.info("   📈 Creating hyperparameter search visualization...")
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        plt.figure(figsize=(15, 10))
        
        # Plot CV scores for all combinations
        plt.subplot(2, 3, 1)
        plt.plot(range(len(results_df)), results_df['mean_test_score'], 'o-')
        plt.fill_between(range(len(results_df)), 
                        results_df['mean_test_score'] - results_df['std_test_score'],
                        results_df['mean_test_score'] + results_df['std_test_score'], 
                        alpha=0.3)
        plt.xlabel('Parameter Combination')
        plt.ylabel('CV F1 Score')
        plt.title('Cross-Validation Scores')
        plt.grid(True, alpha=0.3)
        
        # Kernel performance comparison
        plt.subplot(2, 3, 2)
        kernel_scores = results_df.groupby('param_svm__kernel')['mean_test_score'].mean()
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        bars = plt.bar(kernel_scores.index, kernel_scores.values, color=colors[:len(kernel_scores)])
        plt.xlabel('Kernel Type')
        plt.ylabel('Mean CV F1 Score')
        plt.title('Kernel Performance Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, kernel_scores.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # C parameter performance
        plt.subplot(2, 3, 3)
        c_scores = results_df.groupby('param_svm__C')['mean_test_score'].mean()
        plt.plot(c_scores.index, c_scores.values, 'o-', color='green', linewidth=2, markersize=8)
        plt.xlabel('C (Regularization Parameter)')
        plt.ylabel('Mean CV F1 Score')
        plt.title('C Parameter Impact')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Gamma parameter performance  
        plt.subplot(2, 3, 4)
        gamma_scores = results_df.groupby('param_svm__gamma')['mean_test_score'].mean()
        plt.bar(gamma_scores.index, gamma_scores.values, color=['orange', 'purple'])
        plt.xlabel('Gamma Parameter')
        plt.ylabel('Mean CV F1 Score')
        plt.title('Gamma Parameter Impact')
        plt.grid(True, alpha=0.3)
        
        # Best vs worst comparison
        plt.subplot(2, 3, 5)
        best_idx = results_df['mean_test_score'].idxmax()
        worst_idx = results_df['mean_test_score'].idxmin()
        
        plt.bar(['Best', 'Worst'], 
               [results_df.loc[best_idx, 'mean_test_score'], 
                results_df.loc[worst_idx, 'mean_test_score']], 
               color=['green', 'red'], alpha=0.7)
        plt.ylabel('CV F1 Score')
        plt.title('Best vs Worst Performance')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        plt.text(0, results_df.loc[best_idx, 'mean_test_score'] + 0.001, 
                f"{results_df.loc[best_idx, 'mean_test_score']:.3f}", ha='center', va='bottom')
        plt.text(1, results_df.loc[worst_idx, 'mean_test_score'] + 0.001, 
                f"{results_df.loc[worst_idx, 'mean_test_score']:.3f}", ha='center', va='bottom')
        
        # Training time analysis
        plt.subplot(2, 3, 6)
        plt.hist(results_df['mean_fit_time'], bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        plt.xlabel('Mean Fit Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Training Time Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'SVM Hyperparameter Search Analysis - {self.backbone_name.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'hyperparameter_search.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Prediction Analysis
        self.logger.info("   📈 Creating prediction analysis...")
        plt.figure(figsize=(12, 8))
        
        # Prediction probability distribution
        plt.subplot(2, 2, 1)
        plt.hist(y_prob[y_test == 0], bins=30, alpha=0.7, label='Non-readable', color='red')
        plt.hist(y_prob[y_test == 1], bins=30, alpha=0.7, label='Readable', color='green')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Decision boundary visualization (for 2D projection)
        plt.subplot(2, 2, 2)
        # Use first 2 principal components for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_test_scaled = model.named_steps['scaler'].transform(X_test)
        X_test_2d = pca.fit_transform(X_test_scaled)
        
        scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='RdYlGn', alpha=0.7)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Data Distribution (PCA Projection)')
        plt.colorbar(scatter)
        
        # Performance metrics comparison
        plt.subplot(2, 2, 3)
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
        
        plt.bar(x - width/2, train_metrics, width, label='Train', color='skyblue')
        plt.bar(x + width/2, test_metrics, width, label='Test', color='lightcoral')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, metrics_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Support vectors info (if available)
        plt.subplot(2, 2, 4)
        svm_model = model.named_steps['svm']
        if hasattr(svm_model, 'n_support_'):
            support_info = svm_model.n_support_
            plt.bar(['Class 0', 'Class 1'], support_info, color=['red', 'green'], alpha=0.7)
            plt.ylabel('Number of Support Vectors')
            plt.title('Support Vectors per Class')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(support_info):
                plt.text(i, v + max(support_info) * 0.01, str(v), ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'Support Vector\nInformation\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Support Vectors Info')
        
        plt.suptitle(f'SVM Prediction Analysis - {self.backbone_name.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("✅ All 4 plots created successfully")
        self.logger.info(f"📁 Plots saved to: {plots_dir}")
        self.logger.info(f"📊 Plot files:")
        self.logger.info(f"   - confusion_matrix.png")
        self.logger.info(f"   - roc_curve.png")
        self.logger.info(f"   - hyperparameter_search.png")
        self.logger.info(f"   - prediction_analysis.png")

def find_latest_embeddings(backbone_name):
    """Find the latest embeddings directory for a backbone"""
    embeddings_dir = Path('embeddings')
    embedding_dirs = list(embeddings_dir.glob(f"{backbone_name}_embeddings_*"))
    
    if not embedding_dirs:
        raise FileNotFoundError(f"No embeddings found for {backbone_name}")
    
    return max(embedding_dirs, key=lambda x: x.stat().st_mtime)

def train_backbone_svm(backbone_name):
    """Train SVM on backbone features with hyperparameter search"""
    try:
        # Find embeddings
        embeddings_dir = find_latest_embeddings(backbone_name)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path('experiments') / f'svm_{backbone_name}_{timestamp}'
        
        # Train
        trainer = SVMTrainer(backbone_name, 'embeddings', experiment_dir)
        X_train, X_test, y_train, y_test = trainer.load_features()
        model, results = trainer.train_svm(X_train, X_test, y_train, y_test)
        
        print(f"✅ SVM training completed for {backbone_name}")
        print(f"📁 Results saved to: {experiment_dir}")
        print(f"📊 Best CV F1 Score: {results['best_cv_score']:.4f}")
        print(f"📊 Test F1 Score: {results['test_metrics']['f1']:.4f}")
        print(f"📊 Test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
        print(f"🔧 Best Kernel: {results['best_params'].get('svm__kernel', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Failed to train SVM for {backbone_name}: {e}")
        raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SVM Training on Backbone Features (24 combinations)')
    parser.add_argument('--backbone', type=str, required=True, 
                       choices=['resnet50', 'efficientnet', 'yolov8n', 'layoutxlm'],
                       help='Backbone to train on')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 SVM TRAINING ON BACKBONE FEATURES")
    print("=" * 70)
    print(f"📋 Backbone to train: {args.backbone}")
    print(f"🔧 Hyperparameter combinations: 24")
    print(f"⏱️  Estimated time: 24-72 minutes")
    print(f"📈 Features will be scaled using StandardScaler")
    print("=" * 70)
    
    train_backbone_svm(args.backbone)

if __name__ == "__main__":
    main() 