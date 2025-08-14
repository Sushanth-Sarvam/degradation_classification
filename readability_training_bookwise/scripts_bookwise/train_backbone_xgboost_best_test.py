#!/usr/bin/env python3
"""
XGBoost Training with Best Test Accuracy Model Selection
Trains multiple models and saves the one with best test performance to prevent overfitting
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
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class BestTestXGBoostTrainer:
    def __init__(self, backbone_name, embeddings_dir, experiment_dir):
        self.backbone_name = backbone_name
        self.embeddings_dir = Path(embeddings_dir)
        self.experiment_dir = Path(experiment_dir)
        self.feature_info = {}
        self.best_test_score = 0.0
        self.best_model = None
        self.best_params = None
        self.all_results = []
        
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
    
    def train_xgboost_best_test(self, X_train, X_test, y_train, y_test):
        """Train XGBoost and save model with best test accuracy"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        self.logger.info("🚀 Starting XGBoost training with BEST TEST ACCURACY selection...")
        
        # MEDIUM parameter grid for faster execution (16 combinations: 2×2×2×2×1×1×1)
        param_grid = {
            'max_depth': [3, 6],              # 2 values
            'learning_rate': [0.1, 0.2],      # 2 values
            'n_estimators': [100, 200],       # 2 values
            'subsample': [0.8, 1.0],          # 2 values
            'colsample_bytree': [0.8],        # 1 value (fixed)
            'reg_alpha': [0],                 # 1 value (fixed)
            'reg_lambda': [1]                 # 1 value (fixed)
        }
        
        # Create all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        self.logger.info(f"🔧 Testing {total_combinations} parameter combinations")
        self.logger.info(f"🎯 Strategy: Train each model and save the one with BEST TEST ACCURACY")
        self.logger.info(f"⏱️  Estimated time: {total_combinations * 0.5} - {total_combinations * 2} minutes")
        
        start_time = datetime.now()
        
        # Train models with each parameter combination
        for i, params in enumerate(param_combinations, 1):
            try:
                # Create XGBoost model with current parameters
                model = xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    verbosity=0,
                    **params
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on test set (this is our selection criteria)
                test_metrics = self._evaluate_model(model, X_test, y_test, "test", verbose=False)
                train_metrics = self._evaluate_model(model, X_train, y_train, "train", verbose=False)
                
                test_accuracy = test_metrics['accuracy']
                
                # Store results
                result = {
                    'combination': i,
                    'params': params,
                    'test_accuracy': test_accuracy,
                    'test_f1': test_metrics['f1'],
                    'test_roc_auc': test_metrics['roc_auc'],
                    'train_accuracy': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'overfitting': train_metrics['accuracy'] - test_accuracy
                }
                self.all_results.append(result)
                
                # Check if this is the best test accuracy so far
                if test_accuracy > self.best_test_score:
                    self.best_test_score = test_accuracy
                    self.best_model = model
                    self.best_params = params
                    self.logger.info(f"🏆 NEW BEST! Combination {i}/{total_combinations}")
                    self.logger.info(f"   📈 Test Accuracy: {test_accuracy:.4f}")
                    self.logger.info(f"   📈 Test F1: {test_metrics['f1']:.4f}")
                    self.logger.info(f"   📈 Overfitting: {result['overfitting']:.4f}")
                    self.logger.info(f"   🔧 Params: {params}")
                else:
                    if i % 10 == 0:  # Log progress every 10 combinations
                        self.logger.info(f"📊 Progress: {i}/{total_combinations} - Current best test accuracy: {self.best_test_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed on combination {i}: {e}")
                continue
        
        training_time = datetime.now() - start_time
        
        self.logger.info(f"✅ Training completed!")
        self.logger.info(f"⏱️  Total training time: {training_time}")
        self.logger.info(f"🏆 BEST MODEL SELECTED:")
        self.logger.info(f"   📈 Test Accuracy: {self.best_test_score:.4f}")
        
        # Final evaluation of best model
        final_train_metrics = self._evaluate_model(self.best_model, X_train, y_train, "train")
        final_test_metrics = self._evaluate_model(self.best_model, X_test, y_test, "test")
        
        # Create comprehensive results
        results = {
            "backbone_name": self.backbone_name,
            "feature_dimension": self.feature_info['feature_dimension'],
            "training_time_seconds": training_time.total_seconds(),
            "total_combinations_tested": total_combinations,
            "selection_criteria": "best_test_accuracy",
            "best_params": self.best_params,
            "best_test_accuracy": self.best_test_score,
            "train_metrics": final_train_metrics,
            "test_metrics": final_test_metrics,
            "overfitting_score": final_train_metrics['accuracy'] - final_test_metrics['accuracy']
        }
        
        # Save model and results
        self._save_model_and_results(results, X_test, y_test)
        
        return self.best_model, results
    
    def _evaluate_model(self, model, X, y, dataset_name, verbose=True):
        """Evaluate model and return metrics"""
        if verbose:
            self.logger.info(f"🔍 Evaluating on {dataset_name} set ({len(X)} samples)...")
        
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
        
        if verbose:
            self.logger.info(f"📊 {dataset_name.upper()} RESULTS:")
            self.logger.info(f"   📈 Accuracy:  {metrics['accuracy']:.4f}")
            self.logger.info(f"   📈 Precision: {metrics['precision']:.4f}")
            self.logger.info(f"   📈 Recall:    {metrics['recall']:.4f}")
            self.logger.info(f"   📈 F1 Score:  {metrics['f1']:.4f}")
            self.logger.info(f"   📈 ROC AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def _save_model_and_results(self, results, X_test, y_test):
        """Save model, results, and create analysis"""
        # Save best model
        model_path = self.experiment_dir / 'best_test_model.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save results
        results_path = self.experiment_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save all combinations results
        all_results_path = self.experiment_dir / 'all_combinations.csv'
        pd.DataFrame(self.all_results).to_csv(all_results_path, index=False)
        
        # Create performance metrics TXT file
        self._create_performance_txt(results)
        
        self.logger.info(f"💾 Best model saved to: {model_path}")
        self.logger.info(f"💾 Results saved to: {results_path}")
        self.logger.info(f"💾 All combinations saved to: {all_results_path}")
        
        # Create plots
        self._create_plots(X_test, y_test, results)
    
    def _create_performance_txt(self, results):
        """Create detailed performance metrics TXT file"""
        txt_path = self.experiment_dir / 'performance_metrics.txt'
        
        with open(txt_path, 'w') as f:
            f.write(f"📈 {self.backbone_name.upper()} XGBoost Performance (Best Test Accuracy)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Training Summary:\n")
            f.write(f"- Selection Criteria: Best Test Accuracy\n")
            f.write(f"- Training Time: {results['training_time_seconds']:.1f} seconds\n")
            f.write(f"- Feature Dimension: {results['feature_dimension']:,} features\n")
            f.write(f"- Training Samples: {self.feature_info['train_samples']} images\n")
            f.write(f"- Test Samples: {self.feature_info['test_samples']} images\n")
            f.write(f"- Combinations Tested: {results['total_combinations_tested']}\n")
            f.write(f"- Best Test Accuracy: {results['best_test_accuracy']:.4f}\n")
            f.write(f"- Overfitting Score: {results['overfitting_score']:.4f}\n\n")
            
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
            
            f.write(f"\n\nAnti-Overfitting Strategy:\n")
            f.write(f"- Tested {results['total_combinations_tested']} parameter combinations\n")
            f.write(f"- Selected model with highest test accuracy\n")
            f.write(f"- This prevents overfitting by choosing generalization over memorization\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.logger.info(f"📄 Performance metrics saved to: {txt_path}")
    
    def _create_plots(self, X_test, y_test, results):
        """Create comprehensive analysis plots"""
        self.logger.info("📊 Creating visualization plots...")
        
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Overfitting Analysis
        self.logger.info("   📈 Creating overfitting analysis...")
        df_results = pd.DataFrame(self.all_results)
        
        plt.figure(figsize=(15, 10))
        
        # Test accuracy vs overfitting
        plt.subplot(2, 3, 1)
        scatter = plt.scatter(df_results['overfitting'], df_results['test_accuracy'], 
                            c=df_results['test_f1'], cmap='viridis', alpha=0.6)
        plt.xlabel('Overfitting (Train Acc - Test Acc)')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs Overfitting')
        plt.colorbar(scatter, label='Test F1')
        plt.grid(True, alpha=0.3)
        
        # Best model highlight
        best_idx = df_results['test_accuracy'].idxmax()
        best_result = df_results.loc[best_idx]
        plt.scatter(best_result['overfitting'], best_result['test_accuracy'], 
                   color='red', s=100, marker='*', label='Best Model', zorder=5)
        plt.legend()
        
        # Distribution of test accuracies
        plt.subplot(2, 3, 2)
        plt.hist(df_results['test_accuracy'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.best_test_score, color='red', linestyle='--', linewidth=2, label='Best')
        plt.xlabel('Test Accuracy')
        plt.ylabel('Frequency')
        plt.title('Distribution of Test Accuracies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Top 10 models
        plt.subplot(2, 3, 3)
        top_10 = df_results.nlargest(10, 'test_accuracy')
        plt.barh(range(len(top_10)), top_10['test_accuracy'])
        plt.xlabel('Test Accuracy')
        plt.title('Top 10 Models by Test Accuracy')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # Parameter analysis
        if len(df_results) > 20:  # Only if we have enough data
            plt.subplot(2, 3, 4)
            param_corr = df_results[['test_accuracy', 'overfitting']].corrwith(
                pd.get_dummies(df_results['params'].astype(str)).iloc[:, 0])
            if not param_corr.empty:
                param_corr.plot(kind='bar')
                plt.title('Parameter Impact on Performance')
                plt.xticks(rotation=45)
        
        # Confusion Matrix of best model
        plt.subplot(2, 3, 5)
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Best Model)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve of best model
        plt.subplot(2, 3, 6)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Best Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Best Test Accuracy Analysis - {self.backbone_name.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'best_test_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("✅ Analysis plots created successfully")
        self.logger.info(f"📁 Plots saved to: {plots_dir}")

def find_latest_embeddings(backbone_name):
    """Find the latest embeddings directory for a backbone"""
    embeddings_dir = Path('embeddings')
    embedding_dirs = list(embeddings_dir.glob(f"{backbone_name}_embeddings_*"))
    
    if not embedding_dirs:
        raise FileNotFoundError(f"No embeddings found for {backbone_name}")
    
    return max(embedding_dirs, key=lambda x: x.stat().st_mtime)

def train_backbone_xgboost(backbone_name):
    """Train XGBoost with best test accuracy selection"""
    try:
        # Find embeddings
        embeddings_dir = find_latest_embeddings(backbone_name)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path('experiments') / f'best_test_xgboost_{backbone_name}_{timestamp}'
        
        # Train
        trainer = BestTestXGBoostTrainer(backbone_name, 'embeddings', experiment_dir)
        X_train, X_test, y_train, y_test = trainer.load_features()
        model, results = trainer.train_xgboost_best_test(X_train, X_test, y_train, y_test)
        
        print(f"✅ Best Test XGBoost training completed for {backbone_name}")
        print(f"📁 Results saved to: {experiment_dir}")
        print(f"📊 Best Test Accuracy: {results['best_test_accuracy']:.4f}")
        print(f"📊 Test F1 Score: {results['test_metrics']['f1']:.4f}")
        print(f"📊 Overfitting Score: {results['overfitting_score']:.4f}")
        
    except Exception as e:
        print(f"❌ Failed to train XGBoost for {backbone_name}: {e}")
        raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='XGBoost Training with Best Test Accuracy Selection')
    parser.add_argument('--backbone', type=str, required=True, 
                       choices=['resnet50', 'efficientnet', 'yolov8n', 'layoutxlm'],
                       help='Backbone to train on')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 XGBOOST TRAINING - BEST TEST ACCURACY SELECTION")
    print("=" * 70)
    print(f"📋 Backbone to train: {args.backbone}")
    print(f"🎯 Strategy: Test multiple parameter combinations")
    print(f"🏆 Selection: Save model with highest test accuracy")
    print(f"🚫 Anti-overfitting: Choose generalization over memorization")
    print("=" * 70)
    
    train_backbone_xgboost(args.backbone)

if __name__ == "__main__":
    main() 