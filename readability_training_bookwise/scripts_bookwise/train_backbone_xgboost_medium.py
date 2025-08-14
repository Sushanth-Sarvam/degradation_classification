#!/usr/bin/env python3
"""
Medium Hyperparameter XGBoost Training on Backbone Features
16 hyperparameter combinations for balanced speed vs performance
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class MediumXGBoostTrainer:
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
        """Train XGBoost with MEDIUM hyperparameter search (16 combinations)"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        self.logger.info("🚀 Starting MEDIUM hyperparameter XGBoost training...")
        
        # MEDIUM parameter grid (16 combinations: 2×2×2×2×1×1×1)
        param_grid = {
            'max_depth': [3, 6],              # 2 values
            'learning_rate': [0.1, 0.2],      # 2 values
            'n_estimators': [100, 200],       # 2 values
            'subsample': [0.8, 1.0],          # 2 values
            'colsample_bytree': [0.8],        # 1 value (fixed)
            'reg_alpha': [0],                 # 1 value (fixed)
            'reg_lambda': [1]                 # 1 value (fixed)
        }
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=1
        )
        
        # Calculate total combinations
        total_combinations = len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda'])
        
        self.logger.info(f"🔧 MEDIUM hyperparameter tuning...")
        self.logger.info(f"📊 Total parameter combinations: {total_combinations}")
        self.logger.info(f"🔄 Cross-validation folds: 3")
        self.logger.info(f"⏱️  Estimated time: {total_combinations * 0.5} - {total_combinations * 1.5} minutes")
        
        # Grid search with cross-validation
        # Use fewer parallel jobs for high-dimensional features to avoid memory issues
        n_jobs = 1 if self.feature_info['feature_dimension'] > 1500 else -1
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,  # 3-fold CV for speed
            scoring='f1',
            n_jobs=n_jobs,
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
            "feature_dimension": self.feature_info['feature_dimension'],
            "training_time_seconds": training_time.total_seconds(),
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "total_combinations_tested": total_combinations,
            "cv_folds": 3
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
        self._create_plots(model, X_test, y_test, grid_search)
    
    def _create_performance_txt(self, results):
        """Create detailed performance metrics TXT file"""
        txt_path = self.experiment_dir / 'performance_metrics.txt'
        
        with open(txt_path, 'w') as f:
            f.write(f"📈 {self.backbone_name.upper()} XGBoost Performance Metrics (Medium Search)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Training Summary:\n")
            f.write(f"- Training Time: {results['training_time_seconds']:.1f} seconds\n")
            f.write(f"- Feature Dimension: {results['feature_dimension']:,} features\n")
            f.write(f"- Training Samples: {self.feature_info['train_samples']} images\n")
            f.write(f"- Test Samples: {self.feature_info['test_samples']} images\n")
            f.write(f"- Hyperparameter Combinations Tested: {results['total_combinations_tested']}\n")
            f.write(f"- Cross-validation Folds: {results['cv_folds']}\n")
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
            
            f.write(f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.logger.info(f"📄 Performance metrics saved to: {txt_path}")
    
    def _create_plots(self, model, X_test, y_test, grid_search):
        """Create comprehensive plots for analysis"""
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
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        self.logger.info("   📈 Creating confusion matrix...")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.backbone_name.upper()}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
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
        plt.title(f'ROC Curve - {self.backbone_name.upper()}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Hyperparameter Search Results
        self.logger.info("   📈 Creating hyperparameter search visualization...")
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        plt.figure(figsize=(12, 8))
        
        # Plot CV scores for all combinations
        plt.subplot(2, 2, 1)
        plt.plot(range(len(results_df)), results_df['mean_test_score'], 'o-')
        plt.fill_between(range(len(results_df)), 
                        results_df['mean_test_score'] - results_df['std_test_score'],
                        results_df['mean_test_score'] + results_df['std_test_score'], 
                        alpha=0.3)
        plt.xlabel('Parameter Combination')
        plt.ylabel('CV F1 Score')
        plt.title('Cross-Validation Scores')
        plt.grid(True, alpha=0.3)
        
        # Best vs worst comparison
        plt.subplot(2, 2, 2)
        best_idx = results_df['mean_test_score'].idxmax()
        worst_idx = results_df['mean_test_score'].idxmin()
        
        plt.bar(['Best', 'Worst'], 
               [results_df.loc[best_idx, 'mean_test_score'], 
                results_df.loc[worst_idx, 'mean_test_score']], 
               color=['green', 'red'], alpha=0.7)
        plt.ylabel('CV F1 Score')
        plt.title('Best vs Worst Performance')
        plt.grid(True, alpha=0.3)
        
        # Parameter importance (if enough variations)
        plt.subplot(2, 2, 3)
        if len(set(results_df['param_max_depth'])) > 1:
            depth_scores = results_df.groupby('param_max_depth')['mean_test_score'].mean()
            plt.bar(depth_scores.index.astype(str), depth_scores.values)
            plt.xlabel('Max Depth')
            plt.ylabel('Mean CV F1 Score')
            plt.title('Max Depth Impact')
        
        plt.subplot(2, 2, 4)
        if len(set(results_df['param_learning_rate'])) > 1:
            lr_scores = results_df.groupby('param_learning_rate')['mean_test_score'].mean()
            plt.bar(lr_scores.index.astype(str), lr_scores.values)
            plt.xlabel('Learning Rate')
            plt.ylabel('Mean CV F1 Score')
            plt.title('Learning Rate Impact')
        
        plt.suptitle(f'Hyperparameter Search Analysis - {self.backbone_name.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'hyperparameter_search.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("✅ All 4 plots created successfully")
        self.logger.info(f"📁 Plots saved to: {plots_dir}")

def find_latest_embeddings(backbone_name):
    """Find the latest embeddings directory for a backbone"""
    embeddings_dir = Path('embeddings')
    embedding_dirs = list(embeddings_dir.glob(f"{backbone_name}_embeddings_*"))
    
    if not embedding_dirs:
        raise FileNotFoundError(f"No embeddings found for {backbone_name}")
    
    return max(embedding_dirs, key=lambda x: x.stat().st_mtime)

def train_backbone_xgboost(backbone_name):
    """Train XGBoost on backbone features with medium hyperparameter search"""
    try:
        # Find embeddings
        embeddings_dir = find_latest_embeddings(backbone_name)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path('experiments') / f'medium_xgboost_{backbone_name}_{timestamp}'
        
        # Train
        trainer = MediumXGBoostTrainer(backbone_name, 'embeddings', experiment_dir)
        X_train, X_test, y_train, y_test = trainer.load_features()
        model, results = trainer.train_xgboost(X_train, X_test, y_train, y_test)
        
        print(f"✅ MEDIUM hyperparameter XGBoost training completed for {backbone_name}")
        print(f"📁 Results saved to: {experiment_dir}")
        print(f"📊 Best CV F1 Score: {results['best_cv_score']:.4f}")
        print(f"📊 Test F1 Score: {results['test_metrics']['f1']:.4f}")
        print(f"📊 Test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
        
    except Exception as e:
        print(f"❌ Failed to train XGBoost for {backbone_name}: {e}")
        raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Medium Hyperparameter XGBoost Training (16 combinations)')
    parser.add_argument('--backbone', type=str, required=True, 
                       choices=['resnet50', 'efficientnet', 'yolov8n', 'layoutxlm'],
                       help='Backbone to train on')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 MEDIUM HYPERPARAMETER XGBOOST TRAINING")
    print("=" * 70)
    print(f"📋 Backbone to train: {args.backbone}")
    print(f"🔧 Hyperparameter combinations: 16")
    print(f"⏱️  Estimated time: 8-24 minutes")
    print("=" * 70)
    
    train_backbone_xgboost(args.backbone)

if __name__ == "__main__":
    main() 