#!/usr/bin/env python3
"""
Extract Ground Truth from All PDF Files
======================================

This script processes all PDF files in the data directory and extracts
ground truth text using the enhanced PDF GT extraction script.

Outputs:
- Individual page JSON files
- Consolidated ground truth JSON per book
- Raw and cleaned text files per book
"""

import os
import sys
from pathlib import Path

# Add the gt_extraction path
sys.path.append('/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy_experiments/gt_extraction')

from pdf_gt_extraction import extract_ground_truth_from_pdf

def find_pdf_files(data_dir):
    """Find all PDF files in the data directory"""
    pdf_files = []
    data_path = Path(data_dir)
    
    for book_dir in data_path.iterdir():
        if book_dir.is_dir() and not book_dir.name.startswith('.'):
            for pdf_file in book_dir.glob("*.pdf"):
                pdf_files.append({
                    'book_name': book_dir.name,
                    'pdf_path': pdf_file,
                    'output_dir': f"ground_truth_{book_dir.name}"
                })
    
    return pdf_files

def extract_all_ground_truth():
    """Extract ground truth from all PDF files"""
    data_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/data"
    base_output_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy_experiments/pdf_ground_truth"
    
    # Find all PDF files
    pdf_files = find_pdf_files(data_dir)
    
    if not pdf_files:
        print("❌ No PDF files found in data directory!")
        return
    
    print(f"🔍 Found {len(pdf_files)} PDF files to process")
    print("=" * 60)
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, file_info in enumerate(pdf_files, 1):
        book_name = file_info['book_name']
        pdf_path = file_info['pdf_path']
        
        print(f"\n[{i:2d}/{len(pdf_files)}] Processing: {book_name}")
        print(f"   📄 PDF: {pdf_path.name}")
        
        # Create output directory for this book
        output_folder = os.path.join(base_output_dir, file_info['output_dir'])
        
        try:
            # Extract ground truth
            # Using default parameters - adjust as needed for specific books
            start_page = 1  # Start from page 1
            end_page = None  # Process all pages
            pages_to_exclude = []  # No exclusions by default
            
            extract_ground_truth_from_pdf(
                str(pdf_path), 
                output_folder, 
                start_page, 
                end_page, 
                pages_to_exclude
            )
            
            successful += 1
            print(f"   ✅ Success: {book_name}")
            
        except Exception as e:
            failed += 1
            print(f"   ❌ Failed: {book_name} - {str(e)}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"📚 Total PDF files: {len(pdf_files)}")
    print(f"✅ Successful extractions: {successful}")
    print(f"❌ Failed extractions: {failed}")
    print(f"📈 Success rate: {(successful/len(pdf_files)*100):.1f}%")
    
    print(f"\n📁 Output directory: {base_output_dir}")
    
    # List generated directories
    if os.path.exists(base_output_dir):
        print(f"\n📂 Generated directories:")
        for item in sorted(os.listdir(base_output_dir)):
            item_path = os.path.join(base_output_dir, item)
            if os.path.isdir(item_path):
                # Count files in each directory
                json_files = len([f for f in os.listdir(item_path) if f.endswith('.json')])
                txt_files = len([f for f in os.listdir(item_path) if f.endswith('.txt')])
                print(f"   📖 {item}: {json_files} JSON, {txt_files} text files")

def extract_specific_book(book_name, start_page=1, end_page=None, pages_to_exclude=None):
    """Extract ground truth from a specific book with custom parameters"""
    data_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/data"
    base_output_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy_experiments/pdf_ground_truth"
    
    # Find the specific book
    book_dir = Path(data_dir) / book_name
    if not book_dir.exists():
        print(f"❌ Book directory not found: {book_name}")
        return
    
    pdf_files = list(book_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {book_name}")
        return
    
    pdf_path = pdf_files[0]  # Take the first PDF file
    output_folder = os.path.join(base_output_dir, f"ground_truth_{book_name}")
    
    print(f"📖 Extracting ground truth from: {book_name}")
    print(f"   📄 PDF: {pdf_path.name}")
    print(f"   📃 Pages: {start_page} to {end_page if end_page else 'end'}")
    if pages_to_exclude:
        print(f"   ⏭️ Excluding pages: {pages_to_exclude}")
    
    try:
        extract_ground_truth_from_pdf(
            str(pdf_path), 
            output_folder, 
            start_page, 
            end_page, 
            pages_to_exclude or []
        )
        print(f"✅ Successfully extracted ground truth for {book_name}")
        print(f"📁 Output: {output_folder}")
        
    except Exception as e:
        print(f"❌ Failed to extract ground truth: {str(e)}")

if __name__ == "__main__":
    print("🚀 PDF Ground Truth Extractor")
    print("📐 Extracting text from right column of PDF pages")
    print("💾 Generating JSON and text files for each book")
    
    # Check if specific book is requested
    if len(sys.argv) > 1:
        book_name = sys.argv[1]
        
        # Parse additional arguments
        start_page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        end_page = int(sys.argv[3]) if len(sys.argv) > 3 else None
        pages_to_exclude = []
        if len(sys.argv) > 4:
            pages_to_exclude = [int(x) for x in sys.argv[4].split(',') if x.strip()]
        
        extract_specific_book(book_name, start_page, end_page, pages_to_exclude)
    else:
        # Extract from all books
        extract_all_ground_truth()
