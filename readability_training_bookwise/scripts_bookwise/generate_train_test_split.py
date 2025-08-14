#!/usr/bin/env python3
"""
Train-Test Split Generator

Generates train-test splits using either book-level or page-level splitting strategies.
Creates split files that can be read by create_simple_split_summary.py.
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from typing import Dict, List, Tuple


class TrainTestSplitter:
    """Handle train-test splitting with different strategies."""
    
    def __init__(self, excel_path: str, random_seed: int = 42):
        """Initialize splitter with dataset."""
        self.excel_path = excel_path
        self.random_seed = random_seed
        self.df = pd.read_excel(excel_path)
        self.splits_dir = Path('splits')
        self.splits_dir.mkdir(exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        print(f"📊 Loaded dataset: {len(self.df)} images from {self.df['Book Name'].nunique()} books")
    
    def get_book_stats(self) -> pd.DataFrame:
        """Get book-level statistics."""
        book_stats = self.df.groupby('Book Name').agg({
            'Readability Bookwise': ['first', 'count']
        })
        book_stats.columns = ['Readability_Label', 'Image_Count']
        book_stats = book_stats.reset_index()
        return book_stats
    
    def book_level_split(self, test_size: float = 0.3) -> Tuple[List[str], List[str]]:
        """
        Book-level stratified splitting.
        All images from same book go to same split.
        """
        print(f"🔄 Performing book-level stratified split (test_size={test_size})...")
        
        book_stats = self.get_book_stats()
        
        # Separate readable and non-readable books
        readable_books = book_stats[book_stats['Readability_Label'] == 1]['Book Name'].tolist()
        non_readable_books = book_stats[book_stats['Readability_Label'] == 0]['Book Name'].tolist()
        
        print(f"   📚 Readable books: {len(readable_books)}")
        print(f"   📚 Non-readable books: {len(non_readable_books)}")
        
        # Split each category separately to maintain balance
        readable_train, readable_test = train_test_split(
            readable_books, test_size=test_size, random_state=self.random_seed
        )
        
        non_readable_train, non_readable_test = train_test_split(
            non_readable_books, test_size=test_size, random_state=self.random_seed
        )
        
        # Combine splits
        train_books = readable_train + non_readable_train
        test_books = readable_test + non_readable_test
        
        print(f"   ✅ Training books: {len(train_books)} ({len(readable_train)} readable, {len(non_readable_train)} non-readable)")
        print(f"   ✅ Testing books: {len(test_books)} ({len(readable_test)} readable, {len(non_readable_test)} non-readable)")
        
        return train_books, test_books
    
    def page_level_split(self, test_size: float = 0.2) -> Tuple[List[str], List[str]]:
        """
        Page-level splitting within each book.
        For each book, split pages into train/test.
        """
        print(f"🔄 Performing page-level split (test_size={test_size})...")
        
        train_images = []
        test_images = []
        
        book_stats = self.get_book_stats()
        
        for _, book_row in book_stats.iterrows():
            book_name = book_row['Book Name']
            book_images = self.df[self.df['Book Name'] == book_name]['Image Name'].tolist()
            
            # Split images within this book
            if len(book_images) == 1:
                # If only one image, put it in training
                train_images.extend(book_images)
            else:
                book_train, book_test = train_test_split(
                    book_images, test_size=test_size, random_state=self.random_seed
                )
                train_images.extend(book_train)
                test_images.extend(book_test)
            
            print(f"   📖 {book_name}: {len(book_images)} total → {len([img for img in book_images if img in train_images])} train, {len([img for img in book_images if img in test_images])} test")
        
        print(f"   ✅ Total training images: {len(train_images)}")
        print(f"   ✅ Total testing images: {len(test_images)}")
        
        return train_images, test_images
    
    def create_split_data(self, train_items: List[str], test_items: List[str], 
                         split_type: str) -> Tuple[Dict, Dict]:
        """Create split data structures for saving."""
        
        if split_type == "book_level":
            # For book-level, items are book names
            train_df = self.df[self.df['Book Name'].isin(train_items)]
            test_df = self.df[self.df['Book Name'].isin(test_items)]
            
            train_data = {
                "split_type": "book_level",
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "random_seed": self.random_seed,
                "books": train_items,
                "statistics": {
                    "total_books": len(train_items),
                    "readable_books": len([book for book in train_items 
                                         if self.df[self.df['Book Name'] == book]['Readability Bookwise'].iloc[0] == 1]),
                    "non_readable_books": len([book for book in train_items 
                                             if self.df[self.df['Book Name'] == book]['Readability Bookwise'].iloc[0] == 0]),
                    "total_images": len(train_df),
                    "readable_images": int(train_df[train_df['Readability Bookwise'] == 1].shape[0]),
                    "non_readable_images": int(train_df[train_df['Readability Bookwise'] == 0].shape[0])
                }
            }
            
            test_data = {
                "split_type": "book_level",
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "random_seed": self.random_seed,
                "books": test_items,
                "statistics": {
                    "total_books": len(test_items),
                    "readable_books": len([book for book in test_items 
                                         if self.df[self.df['Book Name'] == book]['Readability Bookwise'].iloc[0] == 1]),
                    "non_readable_books": len([book for book in test_items 
                                             if self.df[self.df['Book Name'] == book]['Readability Bookwise'].iloc[0] == 0]),
                    "total_images": len(test_df),
                    "readable_images": int(test_df[test_df['Readability Bookwise'] == 1].shape[0]),
                    "non_readable_images": int(test_df[test_df['Readability Bookwise'] == 0].shape[0])
                }
            }
            
        else:  # page_level
            # For page-level, items are image names
            train_df = self.df[self.df['Image Name'].isin(train_items)]
            test_df = self.df[self.df['Image Name'].isin(test_items)]
            
            # Get books that have images in each split
            train_books = list(train_df['Book Name'].unique())
            test_books = list(test_df['Book Name'].unique())
            
            train_data = {
                "split_type": "page_level",
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "random_seed": self.random_seed,
                "books": train_books,  # Books that have training images
                "images": train_items,
                "statistics": {
                    "total_books": len(train_books),
                    "readable_books": len(train_df[train_df['Readability Bookwise'] == 1]['Book Name'].unique()),
                    "non_readable_books": len(train_df[train_df['Readability Bookwise'] == 0]['Book Name'].unique()),
                    "total_images": len(train_df),
                    "readable_images": int(train_df[train_df['Readability Bookwise'] == 1].shape[0]),
                    "non_readable_images": int(train_df[train_df['Readability Bookwise'] == 0].shape[0])
                }
            }
            
            test_data = {
                "split_type": "page_level",
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "random_seed": self.random_seed,
                "books": test_books,  # Books that have testing images
                "images": test_items,
                "statistics": {
                    "total_books": len(test_books),
                    "readable_books": len(test_df[test_df['Readability Bookwise'] == 1]['Book Name'].unique()),
                    "non_readable_books": len(test_df[test_df['Readability Bookwise'] == 0]['Book Name'].unique()),
                    "total_images": len(test_df),
                    "readable_images": int(test_df[test_df['Readability Bookwise'] == 1].shape[0]),
                    "non_readable_images": int(test_df[test_df['Readability Bookwise'] == 0].shape[0])
                }
            }
        
        return train_data, test_data
    
    def save_splits(self, train_data: Dict, test_data: Dict, split_method: str):
        """Save split data to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_file = self.splits_dir / f"train_split_{split_method}_{timestamp}.json"
        test_file = self.splits_dir / f"test_split_{split_method}_{timestamp}.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved split files:")
        print(f"   📄 {train_file}")
        print(f"   📄 {test_file}")
        
        return train_file, test_file
    
    def generate_split(self, method: str, test_size: float = None):
        """Generate train-test split using specified method."""
        
        if method == "book_level_splitting":
            if test_size is None:
                test_size = 0.3
            train_items, test_items = self.book_level_split(test_size)
            split_type = "book_level"
        elif method == "page_level_splitting":
            if test_size is None:
                test_size = 0.2
            train_items, test_items = self.page_level_split(test_size)
            split_type = "page_level"
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        # Create split data structures
        train_data, test_data = self.create_split_data(train_items, test_items, split_type)
        
        # Save split files
        train_file, test_file = self.save_splits(train_data, test_data, method)
        
        return train_file, test_file


def generate_summary(splits_exist: bool = True):
    """Generate summary using the existing create_simple_split_summary.py script."""
    if not splits_exist:
        print("⚠️  No split files found to generate summary")
        return None
    
    try:
        # Import and run the existing summary script
        import sys
        sys.path.append('readability_training/scripts')
        from create_simple_split_summary import main as create_summary
        
        print("📋 Generating summary using create_simple_split_summary.py...")
        output_file = create_summary()
        print(f"💾 Summary saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return None


def main():
    """Main function to handle command line arguments and execute splitting."""
    parser = argparse.ArgumentParser(description='Generate train-test splits for readability classification')
    parser.add_argument('method', choices=['book_level_splitting', 'page_level_splitting'],
                       help='Splitting method to use')
    parser.add_argument('--excel_path', default='data/Quality.xlsx',
                       help='Path to Quality.xlsx file (default: data/Quality.xlsx)')
    parser.add_argument('--test_size', type=float, default=None,
                       help='Test size ratio (default: 0.3 for book_level, 0.2 for page_level)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_summary', action='store_true',
                       help='Skip generating final summary')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 TRAIN-TEST SPLIT GENERATOR")
    print("="*80)
    print(f"📊 Method: {args.method}")
    print(f"📁 Excel file: {args.excel_path}")
    print(f"🎲 Random seed: {args.random_seed}")
    
    # Create splitter
    splitter = TrainTestSplitter(args.excel_path, args.random_seed)
    
    # Generate split
    train_file, test_file = splitter.generate_split(args.method, args.test_size)
    
    print("\n✅ Split generation completed!")
    
    # Generate summary unless disabled
    if not args.no_summary:
        print("\n" + "="*80)
        summary_file = generate_summary(splits_exist=True)
        if summary_file:
            print(f"✅ Summary generated: {summary_file}")
        print("="*80)
    
    return train_file, test_file


if __name__ == "__main__":
    main() 