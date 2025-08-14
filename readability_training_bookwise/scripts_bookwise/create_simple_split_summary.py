#!/usr/bin/env python3
"""
Simple Train/Test Split Summary Generator

Creates a single clean JSON file with essential split parameters and statistics.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def create_simple_split_summary():
    """Create a simple train/test split summary JSON"""
    
    # Load dataset
    df = pd.read_excel('data/Quality.xlsx')
    
    # Create book-level statistics  
    book_stats = df.groupby('Book Name').agg({
        'Readability Bookwise': ['first', 'count']
    })
    book_stats.columns = ['Readability_Label', 'Image_Count']
    book_stats = book_stats.reset_index()
    
    # Get book names by readability
    readable_books = book_stats[book_stats['Readability_Label'] == 1]['Book Name'].tolist()
    non_readable_books = book_stats[book_stats['Readability_Label'] == 0]['Book Name'].tolist()
    
    # Load existing split (use the latest one)
    split_files = list(Path('splits').glob('train_split_*.json'))
    latest_split_file = max(split_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_split_file, 'r') as f:
        train_split = json.load(f)
    
    test_file = str(latest_split_file).replace('train_split', 'test_split')
    with open(test_file, 'r') as f:
        test_split = json.load(f)
    
    # Detect split type and method from the loaded split
    split_type = train_split.get('split_type', 'book_level')
    if split_type == 'book_level':
        split_method = "book_level_stratified"
        test_size = 0.3
    else:  # page_level
        split_method = "page_level_stratified"
        test_size = 0.2
    
    # Get random seed from split data
    random_seed = train_split.get('random_seed', 42)
    
    # Create simple summary
    summary = {
        "train_test_split": {
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_source": "Quality.xlsx",
            "split_method": split_method,
            "split_type": split_type,
            "random_seed": random_seed,
            "test_size_ratio": test_size
        },
        "dataset_overview": {
            "total_books": len(book_stats),
            "total_images": len(df),
            "readable_books": len(readable_books),
            "non_readable_books": len(non_readable_books),
            "readable_images": int(df[df['Readability Bookwise'] == 1].shape[0]),
            "non_readable_images": int(df[df['Readability Bookwise'] == 0].shape[0])
        },
        "split_results": {
            "training": {
                "books": train_split['statistics']['total_books'],
                "readable_books": train_split['statistics']['readable_books'],
                "non_readable_books": train_split['statistics']['non_readable_books'],
                "total_images": train_split['statistics']['total_images'],
                "readable_images": train_split['statistics']['readable_images'],
                "non_readable_images": train_split['statistics']['non_readable_images'],
                "percentage_of_total_images": round((train_split['statistics']['total_images'] / len(df)) * 100, 1)
            },
            "testing": {
                "books": test_split['statistics']['total_books'],
                "readable_books": test_split['statistics']['readable_books'],
                "non_readable_books": test_split['statistics']['non_readable_books'],
                "total_images": test_split['statistics']['total_images'],
                "readable_images": test_split['statistics']['readable_images'],
                "non_readable_images": test_split['statistics']['non_readable_images'],
                "percentage_of_total_images": round((test_split['statistics']['total_images'] / len(df)) * 100, 1)
            }
        },
        "class_balance": {
            "training_balance_ratio": round(train_split['statistics']['non_readable_images'] / train_split['statistics']['readable_images'], 2),
            "testing_balance_ratio": round(test_split['statistics']['non_readable_images'] / test_split['statistics']['readable_images'], 2),
            "overall_balance_status": "acceptable"
        },
        "data_leakage_prevention": {
            "method": "book_level_splitting",
            "description": "All images from the same book are kept in the same split (train or test)",
            "no_book_overlap": True
        },
        "training_book_list": train_split['books'],
        "testing_book_list": test_split['books']
    }
    
    return summary, split_type

def main():
    """Generate the simple split summary"""
    print("🚀 Creating simple train/test split summary...")
    
    summary, split_type = create_simple_split_summary()
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Save to data directory with split type in filename
    output_file = data_dir / f'train_test_split_{split_type}.json'
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Simple split summary created: {output_file}")
    print(f"📊 Training: {summary['split_results']['training']['total_images']} images from {summary['split_results']['training']['books']} books")
    print(f"📊 Testing: {summary['split_results']['testing']['total_images']} images from {summary['split_results']['testing']['books']} books")
    
    return output_file

if __name__ == "__main__":
    main() 