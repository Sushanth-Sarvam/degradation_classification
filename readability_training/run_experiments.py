#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run experiments and generate Excel files')
    parser.add_argument('--run-xgboost', action='store_true', 
                       help='Run XGBoost training')
    parser.add_argument('--run-cnn', action='store_true', 
                       help='Run CNN training (requires existing CNN script)')
    parser.add_argument('--generate-excel', action='store_true', 
                       help='Generate Excel files with predictions')
    parser.add_argument('--all', action='store_true', 
                       help='Run all experiments and generate Excel files')
    
    args = parser.parse_args()
    
    if not any([args.run_xgboost, args.run_cnn, args.generate_excel, args.all]):
        print("Please specify what to run. Use --help for options.")
        return
    
    print("="*80)
    print("🎯 Readability Experiments Runner")
    print("="*80)
    
    success_count = 0
    total_tasks = 0
    
    # Run XGBoost
    if args.run_xgboost or args.all:
        total_tasks += 1
        if run_command("python readability_training/scripts/train_xgboost.py", 
                      "XGBoost Training"):
            success_count += 1
    
    # Run CNN
    if args.run_cnn or args.all:
        total_tasks += 1
        cnn_script = "readability_training/scripts/train_simple_cnn.py"
        if Path(cnn_script).exists():
            if run_command(f"python {cnn_script}", "CNN Training"):
                success_count += 1
        else:
            print(f"❌ CNN script not found: {cnn_script}")
    
    # Generate Excel files
    if args.generate_excel or args.all:
        total_tasks += 1
        if run_command("python readability_training/scripts/generate_predictions_excel.py", 
                      "Excel Generation"):
            success_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {success_count}/{total_tasks} tasks")
    
    if success_count == total_tasks:
        print("🎉 All experiments completed successfully!")
    else:
        print(f"⚠️  {total_tasks - success_count} tasks failed")
    
    # Show output directories
    print(f"\n📁 Output Locations:")
    print(f"  • Experiments: readability_training/experiments/")
    print(f"  • Excel files: readability_training/predictions/")

if __name__ == "__main__":
    main() 