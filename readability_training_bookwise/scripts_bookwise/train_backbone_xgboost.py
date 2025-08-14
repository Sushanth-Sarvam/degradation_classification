#!/usr/bin/env python3
"""
XGBoost Training on Backbone Features

Trains XGBoost classifiers on pre-extracted features from backbone models:
- ResNet50, EfficientNet-B0, YOLOv8n, LayoutXLM
- Creates separate experiment directories for each backbone
- Comprehensive hyperparameter tuning and evaluation
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import pickle
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install xgboost library.")
    XGBOOST_AVAILABLE = False

class XGBoostTrainer:
    def __init__(self, backbone_name, embeddings_dir, experiment_dir):
        self.backbone_name = backbone_name
        self.embeddings_dir = Path(embeddings_dir)
        self.experiment_dir = Path(experiment_dir)
        self.logger = None
        
        # Create experiment subdirectories
        (self.experiment_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the training process"""
        log_file = self.experiment_dir / "logs" / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{self.backbone_name}_trainer")
        
    def load_features(self):
        """Load pre-extracted features from embeddings directory"""
        self.logger.info(f"Loading {self.backbone_name} features from {self.embeddings_dir}")
        
        try:
            # Load features and labels
            X_train = np.load(self.embeddings_dir / "train_features.npy")
            X_test = np.load(self.embeddings_dir / "test_features.npy")
            y_train = np.load(self.embeddings_dir / "train_labels.npy")
            y_test = np.load(self.embeddings_dir / "test_labels.npy")
            
            # Load feature info
            with open(self.embeddings_dir / "feature_info.json", 'r') as f:
                self.feature_info = json.load(f)
            
            self.logger.info(f"✅ Features loaded successfully")
            self.logger.info(f"📊 Train features: {X_train.shape}")
            self.logger.info(f"📊 Test features: {X_test.shape}")
            self.logger.info(f"📊 Feature dimension: {self.feature_info['feature_dimension']}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load features: {e}")
            raise
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost with hyperparameter tuning"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        self.logger.info("🚀 Starting XGBoost training with hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }
        
        # Create XGBoost classifier with verbose training
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=1  # Show training progress
        )
        
        # Grid search with cross-validation
        total_combinations = len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda'])
        self.logger.info(f"🔧 Performing hyperparameter tuning...")
        self.logger.info(f"📊 Total parameter combinations to test: {total_combinations}")
        self.logger.info(f"🔄 Cross-validation folds: 5")
        self.logger.info(f"⏱️  Estimated time: {total_combinations * 2} - {total_combinations * 5} minutes")
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=2  # Show progress for each fit
        )
        
        # Train model with progress tracking
        start_time = datetime.now()
        self.logger.info("🚀 Starting hyperparameter search...")
        
        # Custom callback to track progress
        class ProgressCallback:
            def __init__(self, logger, total_fits):
                self.logger = logger
                self.total_fits = total_fits
                self.current_fit = 0
                
            def __call__(self, score, model, params):
                self.current_fit += 1
                progress = (self.current_fit / self.total_fits) * 100
                self.logger.info(f"📈 Progress: {self.current_fit}/{self.total_fits} ({progress:.1f}%) - Current F1: {score:.4f}")
        
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
        top_5 = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
        self.logger.info("🔝 Top 5 parameter combinations:")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            self.logger.info(f"   {i}. F1: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f}) - {row['params']}")
        
        # Evaluate on train and test sets
        self.logger.info("📊 Evaluating best model on training and test sets...")
        train_metrics = self._evaluate_model(best_model, X_train, y_train, "train")
        test_metrics = self._evaluate_model(best_model, X_test, y_test, "test")
        
        # Create detailed results
        results = {
            "backbone_name": self.backbone_name,
            "feature_dimension": self.feature_info['feature_dimension'],
            "training_time_seconds": training_time.total_seconds(),
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "hyperparameter_search": {
                "total_combinations": len(grid_search.cv_results_['params']),
                "cv_folds": 5,
                "scoring_metric": "f1"
            }
        }
        
        # Save model and results
        self._save_model_and_results(best_model, results, grid_search, X_test, y_test)
        
        # Create plots
        self._create_plots(best_model, X_train, X_test, y_train, y_test, grid_search)
        
        return results
    
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
        """Save trained model and all results"""
        # Save model
        model_path = self.experiment_dir / "models" / "xgboost_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.logger.info(f"💾 Model saved to: {model_path}")
        
        # Save main results
        results_path = self.experiment_dir / "data" / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save hyperparameter search details
        hyperparams_path = self.experiment_dir / "data" / "hyperparameter_search.json"
        hyperparams_data = {
            "cv_results": {
                "params": grid_search.cv_results_['params'],
                "mean_test_scores": grid_search.cv_results_['mean_test_score'].tolist(),
                "std_test_scores": grid_search.cv_results_['std_test_score'].tolist(),
                "ranks": grid_search.cv_results_['rank_test_score'].tolist()
            },
            "best_index": int(grid_search.best_index_),
            "best_score": float(grid_search.best_score_),
            "best_params": grid_search.best_params_
        }
        
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams_data, f, indent=2)
        
        # Save predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_probability': y_prob,
            'correct_prediction': (y_test == y_pred)
        })
        
        predictions_path = self.experiment_dir / "data" / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        
        # Save feature statistics
        feature_stats = {
            "feature_dimension": self.feature_info['feature_dimension'],
            "feature_importance": model.feature_importances_.tolist(),
            "top_10_features": [
                {"index": int(idx), "importance": float(model.feature_importances_[idx])}
                for idx in np.argsort(model.feature_importances_)[::-1][:10]
            ]
        }
        
        feature_stats_path = self.experiment_dir / "data" / "feature_stats.json"
        with open(feature_stats_path, 'w') as f:
            json.dump(feature_stats, f, indent=2)
        
        # Save experiment config
        config = {
            "backbone_name": self.backbone_name,
            "embeddings_source": str(self.embeddings_dir),
            "experiment_date": datetime.now().isoformat(),
            "feature_info": self.feature_info,
            "model_config": {
                "algorithm": "XGBoost",
                "hyperparameter_tuning": "GridSearchCV",
                "cross_validation_folds": 5,
                "scoring_metric": "f1"
            },
            "results_summary": results
        }
        
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"📊 All results saved to: {self.experiment_dir}")
    
    def _create_plots(self, model, X_train, X_test, y_train, y_test, grid_search):
        """Create comprehensive plots for analysis"""
        self.logger.info("📊 Creating visualization plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Feature Importance Plot
        self.logger.info("   📈 Creating feature importance plot...")
        self._plot_feature_importance(model)
        
        # 2. Confusion Matrix
        self.logger.info("   📈 Creating confusion matrix...")
        self._plot_confusion_matrix(model, X_test, y_test)
        
        # 3. ROC Curve
        self.logger.info("   📈 Creating ROC curve...")
        self._plot_roc_curve(model, X_test, y_test)
        
        # 4. Hyperparameter Search Results
        self.logger.info("   📈 Creating hyperparameter search visualization...")
        self._plot_hyperparameter_search(grid_search)
        
        self.logger.info("✅ All 4 plots created successfully")
        self.logger.info(f"📁 Plots saved to: {self.experiment_dir / 'plots'}")
    
    def _plot_feature_importance(self, model):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.bar(range(len(indices)), importances[indices])
        plt.title(f'{self.backbone_name.upper()} - Feature Importance (Top 20)', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.xticks(range(len(indices)), indices, rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.experiment_dir / "plots" / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, model, X_test, y_test):
        """Plot confusion matrix"""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Readable', 'Readable'],
                   yticklabels=['Non-Readable', 'Readable'])
        plt.title(f'{self.backbone_name.upper()} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        plt.savefig(self.experiment_dir / "plots" / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, model, X_test, y_test):
        """Plot ROC curve"""
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.backbone_name.upper()} - ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.experiment_dir / "plots" / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hyperparameter_search(self, grid_search):
        """Plot hyperparameter search results"""
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Plot top 10 parameter combinations
        top_10 = results_df.nlargest(10, 'mean_test_score')
        
        plt.figure(figsize=(12, 8))
        plt.errorbar(range(len(top_10)), top_10['mean_test_score'], 
                    yerr=top_10['std_test_score'], marker='o', capsize=5)
        plt.title(f'{self.backbone_name.upper()} - Top 10 Hyperparameter Combinations', fontsize=14, fontweight='bold')
        plt.xlabel('Hyperparameter Combination Rank')
        plt.ylabel('Mean CV F1 Score')
        plt.xticks(range(len(top_10)), [f'Rank {i+1}' for i in range(len(top_10))], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.experiment_dir / "plots" / "hyperparameter_search.png", dpi=300, bbox_inches='tight')
        plt.close()

def find_latest_embeddings(backbone_name):
    """Find the latest embeddings directory for a backbone"""
    embeddings_base = Path("embeddings")
    if not embeddings_base.exists():
        raise FileNotFoundError("No embeddings directory found. Run feature extraction first.")
    
    pattern = f"{backbone_name}_embeddings_*"
    matching_dirs = list(embeddings_base.glob(pattern))
    
    if not matching_dirs:
        raise FileNotFoundError(f"No embeddings found for {backbone_name}")
    
    # Return the most recent directory
    latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
    return latest_dir

def train_backbone_xgboost(backbone_name):
    """Train XGBoost on specific backbone features"""
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING XGBOOST ON {backbone_name.upper()} FEATURES")
    print(f"{'='*70}")
    
    # Find latest embeddings
    embeddings_dir = find_latest_embeddings(backbone_name)
    print(f"📁 Using embeddings from: {embeddings_dir}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiments/{backbone_name}_xgboost_{timestamp}")
    
    # Initialize trainer
    trainer = XGBoostTrainer(backbone_name, embeddings_dir, experiment_dir)
    
    # Load features
    X_train, X_test, y_train, y_test = trainer.load_features()
    
    # Train XGBoost
    results = trainer.train_xgboost(X_train, X_test, y_train, y_test)
    
    print(f"\n✅ {backbone_name} XGBoost training completed!")
    print(f"📁 Results saved to: {experiment_dir}")
    print(f"🏆 Test F1 Score: {results['test_metrics']['f1']:.4f}")
    print(f"🏆 Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    
    return experiment_dir, results

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost on backbone features')
    parser.add_argument('--backbone', choices=['resnet50', 'efficientnet', 'yolov8n', 'layoutxlm', 'all'], 
                       default='all', help='Backbone features to train on')
    
    args = parser.parse_args()
    
    if args.backbone == 'all':
        backbones = ['resnet50', 'efficientnet', 'yolov8n', 'layoutxlm']
    else:
        backbones = [args.backbone]
    
    print("="*70)
    print("🚀 XGBOOST TRAINING ON BACKBONE FEATURES")
    print("="*70)
    print(f"📋 Backbones to train: {', '.join(backbones)}")
    
    results_summary = {}
    
    for backbone in backbones:
        try:
            exp_dir, results = train_backbone_xgboost(backbone)
            results_summary[backbone] = {
                "experiment_dir": str(exp_dir),
                "test_f1": results['test_metrics']['f1'],
                "test_accuracy": results['test_metrics']['accuracy'],
                "feature_dim": results['feature_dimension']
            }
        except Exception as e:
            print(f"❌ Failed to train {backbone}: {e}")
            results_summary[backbone] = {"error": str(e)}
    
    # Print final summary
    print(f"\n{'='*70}")
    print("📋 TRAINING SUMMARY")
    print(f"{'='*70}")
    
    best_f1 = 0
    best_backbone = None
    
    for backbone, result in results_summary.items():
        if "error" in result:
            print(f"❌ {backbone}: {result['error']}")
        else:
            f1_score = result['test_f1']
            accuracy = result['test_accuracy']
            feature_dim = result['feature_dim']
            print(f"✅ {backbone}: F1={f1_score:.4f}, Acc={accuracy:.4f}, Dim={feature_dim}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_backbone = backbone
    
    if best_backbone:
        print(f"\n🏆 BEST MODEL: {best_backbone.upper()} (F1: {best_f1:.4f})")
    
    print(f"\n🎉 XGBoost training pipeline completed!")

if __name__ == "__main__":
    main() 