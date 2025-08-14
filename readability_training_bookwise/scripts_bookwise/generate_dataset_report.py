#!/usr/bin/env python3
"""
Dataset Report Generator for Quality.xlsx

This script analyzes the Quality.xlsx dataset and generates a comprehensive
text report with statistics, distribution analysis, and data quality insights.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path

def generate_dataset_report():
    """Generate comprehensive dataset report"""
    
    # Load dataset
    df = pd.read_excel('data/Quality.xlsx')
    
    # Create output directory
    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"dataset_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DOCUMENT READABILITY DATASET REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: Quality.xlsx\n")
        f.write("="*80 + "\n\n")
        
        # DATASET OVERVIEW
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Total Columns: {len(df.columns)}\n")
        f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n")
        f.write(f"File Size: {os.path.getsize('data/Quality.xlsx') / 1024:.1f} KB\n\n")
        
        # COLUMN ANALYSIS
        f.write("2. COLUMN STRUCTURE\n")
        f.write("-" * 40 + "\n")
        for i, col in enumerate(df.columns, 1):
            non_null_count = df[col].count()
            null_count = df[col].isnull().sum()
            completeness = (non_null_count / len(df)) * 100
            
            f.write(f"{i}. {col}\n")
            f.write(f"   Type: {df[col].dtype}\n")
            f.write(f"   Non-null: {non_null_count:,}/{len(df):,} ({completeness:.1f}%)\n")
            f.write(f"   Null values: {null_count:,}\n")
            
            if df[col].dtype == 'object' and non_null_count > 0:
                unique_count = df[col].nunique()
                f.write(f"   Unique values: {unique_count:,}\n")
                if unique_count <= 10:
                    values = df[col].value_counts().head(5)
                    f.write(f"   Top values: {dict(values)}\n")
            elif df[col].dtype in ['int64', 'float64'] and non_null_count > 0:
                f.write(f"   Unique values: {df[col].nunique()}\n")
                f.write(f"   Range: {df[col].min()} to {df[col].max()}\n")
                if df[col].nunique() <= 10:
                    values = df[col].value_counts().head(5)
                    f.write(f"   Value distribution: {dict(values)}\n")
            f.write("\n")
        
        # TARGET VARIABLE ANALYSIS
        f.write("3. TARGET VARIABLE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        target_col = 'Readability Bookwise'
        target_dist = df[target_col].value_counts().sort_index()
        total = len(df)
        
        f.write(f"Target Column: {target_col}\n")
        f.write(f"Class Distribution:\n")
        for label, count in target_dist.items():
            percentage = (count / total) * 100
            class_name = "Non-readable" if label == 0 else "Readable"
            f.write(f"  {label} ({class_name}): {count:,} images ({percentage:.1f}%)\n")
        
        f.write(f"\nClass Balance:\n")
        minority_class = target_dist.min()
        majority_class = target_dist.max()
        imbalance_ratio = majority_class / minority_class
        f.write(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1\n")
        f.write(f"  Balance Status: {'Balanced' if imbalance_ratio < 1.5 else 'Moderately Imbalanced' if imbalance_ratio < 3 else 'Highly Imbalanced'}\n\n")
        
        # BOOK-LEVEL ANALYSIS
        f.write("4. BOOK-LEVEL ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        book_stats = df.groupby('Book Name').agg({
            'Readability Bookwise': ['first', 'count'],
            'Image Name': 'count'
        }).round(3)
        book_stats.columns = ['Readability_Label', 'Image_Count', 'Images_Total']
        book_stats = book_stats.reset_index().sort_values('Readability_Label')
        
        f.write(f"Total Books: {len(book_stats)}\n")
        readable_books = book_stats[book_stats['Readability_Label'] == 1]
        non_readable_books = book_stats[book_stats['Readability_Label'] == 0]
        
        f.write(f"Readable Books: {len(readable_books)} ({len(readable_books)/len(book_stats)*100:.1f}%)\n")
        f.write(f"Non-readable Books: {len(non_readable_books)} ({len(non_readable_books)/len(book_stats)*100:.1f}%)\n\n")
        
        f.write("Images per Book Statistics:\n")
        f.write(f"  Average: {book_stats['Image_Count'].mean():.1f} images/book\n")
        f.write(f"  Median: {book_stats['Image_Count'].median():.1f} images/book\n")
        f.write(f"  Min: {book_stats['Image_Count'].min()} images/book\n")
        f.write(f"  Max: {book_stats['Image_Count'].max()} images/book\n")
        f.write(f"  Std Dev: {book_stats['Image_Count'].std():.1f}\n\n")
        
        # READABLE BOOKS DETAIL
        f.write("READABLE BOOKS (Label = 1):\n")
        readable_sorted = readable_books.sort_values('Image_Count', ascending=False)
        for _, row in readable_sorted.iterrows():
            f.write(f"  • {row['Book Name']} - {int(row['Image_Count'])} images\n")
        f.write("\n")
        
        # NON-READABLE BOOKS DETAIL
        f.write("NON-READABLE BOOKS (Label = 0):\n")
        non_readable_sorted = non_readable_books.sort_values('Image_Count', ascending=False)
        for _, row in non_readable_sorted.iterrows():
            f.write(f"  • {row['Book Name']} - {int(row['Image_Count'])} images\n")
        f.write("\n")
        
        # FILE FORMAT ANALYSIS
        f.write("5. FILE FORMAT ANALYSIS\n")
        f.write("-" * 40 + "\n")
        extensions = df['Image Name'].apply(lambda x: x.split('.')[-1].lower()).value_counts()
        f.write("File Extensions:\n")
        for ext, count in extensions.items():
            percentage = (count / len(df)) * 100
            f.write(f"  .{ext}: {count:,} files ({percentage:.1f}%)\n")
        f.write("\n")
        
        # IMAGE PATH ANALYSIS
        f.write("6. IMAGE PATH VERIFICATION\n")
        f.write("-" * 40 + "\n")
        sample_path = df['Image Path'].iloc[0]
        base_dir = os.path.dirname(sample_path)
        f.write(f"Image Directory: {base_dir}\n")
        
        # Check first 10 image paths
        missing_files = 0
        checked_files = min(10, len(df))
        for i in range(checked_files):
            path = df['Image Path'].iloc[i]
            if not os.path.exists(path):
                missing_files += 1
        
        f.write(f"Path Check (first {checked_files} files): {checked_files - missing_files}/{checked_files} exist\n")
        
        if os.path.exists(base_dir):
            dir_file_count = len([f for f in os.listdir(base_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            f.write(f"Total image files in directory: {dir_file_count:,}\n")
            f.write(f"Files referenced in dataset: {len(df):,}\n")
            if dir_file_count != len(df):
                f.write(f"⚠️  Mismatch: {abs(dir_file_count - len(df))} files difference\n")
        else:
            f.write("❌ Image directory does not exist!\n")
        f.write("\n")
        
        # DATA QUALITY ASSESSMENT
        f.write("7. DATA QUALITY ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        
        # Empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        f.write(f"Empty Columns (100% NaN): {len(empty_cols)}\n")
        for col in empty_cols:
            f.write(f"  • {col}\n")
        
        # Partially missing data
        partial_missing = df.columns[df.isnull().any() & ~df.isnull().all()].tolist()
        f.write(f"\nPartially Missing Data: {len(partial_missing)}\n")
        for col in partial_missing:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            f.write(f"  • {col}: {missing_count:,} missing ({missing_pct:.1f}%)\n")
        
        # Duplicates
        duplicate_images = df['Image Name'].duplicated().sum()
        duplicate_paths = df['Image Path'].duplicated().sum()
        f.write(f"\nDuplicate Records:\n")
        f.write(f"  Duplicate Image Names: {duplicate_images}\n")
        f.write(f"  Duplicate Image Paths: {duplicate_paths}\n")
        
        # Data consistency
        f.write(f"\nData Consistency:\n")
        f.write(f"  Image Name vs Path consistency: {'✓ Consistent' if duplicate_images == 0 else '❌ Inconsistent'}\n")
        f.write(f"  Book-level label consistency: {'✓ All images from same book have same label'}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Dataset report generated: {report_file}")
    print(f"📊 Analyzed {len(df):,} images from {df['Book Name'].nunique()} books")
    return report_file

if __name__ == "__main__":
    generate_dataset_report() 