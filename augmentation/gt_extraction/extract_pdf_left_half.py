#!/usr/bin/env python3
"""
Extract Left Half from PDF Pages
===============================

This script:
1. Converts PDF pages to images
2. Crops the left half of each page image
3. Saves the left half images

This complements the right half extraction script.
"""

import os
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

class PDFLeftHalfExtractor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.success_count = 0
        self.failed_count = 0
        self.total_pages = 0
    
    def find_pdf_files(self):
        """Find all PDF files"""
        pdf_files = []
        for book_dir in self.data_dir.iterdir():
            if book_dir.is_dir() and not book_dir.name.startswith('.'):
                for pdf_file in book_dir.glob("*.pdf"):
                    pdf_files.append({
                        'book_name': book_dir.name,
                        'pdf_path': pdf_file
                    })
        return pdf_files
    
    def extract_left_half_from_image(self, image_path, output_path):
        """Extract left half from an image"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Calculate left half coordinates (0 to width//2)
                right = width // 2
                left_half = img.crop((0, 0, right, height))
                
                # Convert to RGB if necessary
                if left_half.mode != 'RGB':
                    left_half = left_half.convert('RGB')
                
                # Save with high quality
                left_half.save(output_path, 'PNG', quality=95, optimize=True)
                return True
                
        except Exception as e:
            print(f"   ❌ Failed to crop: {e}")
            return False
    
    def process_pdf_file(self, file_info):
        """Process a single PDF file"""
        book_name = file_info['book_name']
        pdf_path = file_info['pdf_path']
        
        print(f"\n📖 Processing PDF: {book_name}")
        print(f"   📄 File: {pdf_path.name}")
        
        # Create output directory for this book
        book_output_dir = self.output_dir / book_name
        book_output_dir.mkdir(exist_ok=True)
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            print(f"   📊 Found {total_pages} pages")
            
            page_success = 0
            page_failed = 0
            
            # Process each page
            for page_num in range(total_pages):
                try:
                    # Load page
                    page = doc.load_page(page_num)
                    
                    # Convert to image with high resolution
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Save full page as temporary image
                    temp_image_path = book_output_dir / f"temp_page_{page_num + 1:03d}.png"
                    pix.save(temp_image_path)
                    
                    # Extract left half
                    output_image_path = book_output_dir / f"{book_name}_page_{page_num + 1:03d}_left.png"
                    
                    if self.extract_left_half_from_image(temp_image_path, output_image_path):
                        # Remove temporary file
                        temp_image_path.unlink()
                        
                        print(f"   ✅ Page {page_num + 1:3d}: {output_image_path.name}")
                        page_success += 1
                        self.success_count += 1
                    else:
                        print(f"   ❌ Page {page_num + 1:3d}: Failed to extract left half")
                        page_failed += 1
                        self.failed_count += 1
                        
                        # Clean up temp file
                        if temp_image_path.exists():
                            temp_image_path.unlink()
                
                except Exception as e:
                    print(f"   ❌ Page {page_num + 1:3d}: Error - {e}")
                    page_failed += 1
                    self.failed_count += 1
            
            # Close PDF
            doc.close()
            
            self.total_pages += total_pages
            print(f"   📊 {book_name}: {page_success} success, {page_failed} failed")
            
        except Exception as e:
            print(f"   ❌ Failed to process PDF: {e}")
            self.failed_count += 1
    
    def process_all_files(self):
        """Process all PDF files"""
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            print("❌ No PDF files found!")
            return
        
        print(f"🔍 Found {len(pdf_files)} PDF files to process")
        print("=" * 60)
        
        for i, file_info in enumerate(pdf_files, 1):
            print(f"\n[{i:2d}/{len(pdf_files)}]", end="")
            self.process_pdf_file(file_info)
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"📚 PDF files processed: {len(pdf_files)}")
        print(f"📄 Total pages processed: {self.total_pages}")
        print(f"✅ Successful extractions: {self.success_count}")
        print(f"❌ Failed extractions: {self.failed_count}")
        print(f"📈 Success rate: {(self.success_count/max(1,self.total_pages)*100):.1f}%")
        
        print(f"\n📁 Output directory: {self.output_dir}")
        
        # List books and their page counts
        print(f"\n📚 Books and page counts:")
        total_images = 0
        for book_dir in sorted(self.output_dir.iterdir()):
            if book_dir.is_dir():
                page_count = len(list(book_dir.glob("*_left.png")))
                total_images += page_count
                print(f"   📖 {book_dir.name}: {page_count} pages")
        
        print(f"\n🎉 Total left-half images generated: {total_images}")

def main():
    """Main function"""
    data_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/data"
    output_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy_experiments/pdf_left_half_images"
    
    print("🚀 PDF Left Half Extractor")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Output directory: {output_dir}")
    print("🎯 Process: PDF → Page Images → Left Half Crop")
    print("📐 Extracting left half of actual PDF pages")
    
    extractor = PDFLeftHalfExtractor(data_dir, output_dir)
    extractor.process_all_files()

if __name__ == "__main__":
    main()
