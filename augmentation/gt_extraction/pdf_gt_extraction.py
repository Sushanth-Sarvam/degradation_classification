import fitz  # PyMuPDF
import os
import json
import re

#####################################
# Global Configuration
#####################################
class Config:
    # Patterns to identify header artefacts in the first 5 lines
    ARTEFACT_PATTERNS = [
        r'^Sardar_Vallabhabhai.*$',  # Document header (e.g., 'Sardar_Vallabhabhai_Part-1')
        r'^\d+rd Proof.*$',          # Proof text like '1rd Proof'
        r'^[A-Za-z]+ \d{1,2}, \d{4}',  # Date line (e.g., 'February 6, 2025')
    ]

#####################################
# Artefact Cleaning
#####################################
class ArtefactCleaner:
    @staticmethod
    def remove_top_artefacts(lines):
        """
        Remove artefacts (e.g., headers, page numbers, dates) only from the first 5 lines.
        """
        cleaned_lines = []
        for idx, line in enumerate(lines):
            # Remove Byte Order Mark if present
            line = line.replace('\ufeff', '')
            if idx < 5:
                # Skip empty or backspace lines
                if line.strip() in {'\b', ''}:
                    continue
                # Skip if line matches any artefact pattern
                if any(re.match(pattern, line.strip()) for pattern in Config.ARTEFACT_PATTERNS):
                    continue
            cleaned_lines.append(line)
        return cleaned_lines

#####################################
# Numeral Conversion
#####################################
class NumeralConverter:
    @staticmethod
    def convert_arabic_to_gujarati_numerals(text):
        """
        Converts Arabic numerals in a text to Gujarati numerals.
        """
        arabic_to_gujarati = str.maketrans('0123456789', '૦૧૨૩૪૫૬૭૮૯')
        return text.translate(arabic_to_gujarati)

#####################################
# Right Column Text Extraction
#####################################
class RightColumnExtractor:
    @staticmethod
    def extract_blocks(page):
        """
        Extracts text from the right column of a PDF page.
        Splits each block into lines, converts numerals, and removes header artefacts.
        """
        page_width = page.rect.width
        # Get all text blocks from the page (each block is a tuple)
        blocks = page.get_text("blocks")
        # Keep only blocks in the right half (x0 > mid-page)
        right_blocks = [b for b in blocks if b[0] > page_width / 2]
        # Sort blocks top-to-bottom using the y-coordinate (b[1])
        right_blocks = sorted(right_blocks, key=lambda b: b[1])
        extracted_blocks = []

        for block in right_blocks:
            block_text = block[4].strip()  # The text is at index 4
            # Convert any Arabic numerals to Gujarati at the block level
            block_text = NumeralConverter.convert_arabic_to_gujarati_numerals(block_text)
            # Split into individual lines
            lines = block_text.split("\n")
            # Clean the first five lines for header artefacts
            lines = ArtefactCleaner.remove_top_artefacts(lines)
            if lines:
                extracted_blocks.append({
                    "block_text": block_text,
                    "lines": lines
                })
        return extracted_blocks

#####################################
# JSON Output Saving
#####################################
class JSONOutputSaver:
    @staticmethod
    def save(mapped_page_number, content, output_folder, book_name="Unknown_Book"):
        """
        Saves extracted content (blocks and lines) to JSON.
        The output filename is of the form: {book_name}_page_X.json
        """
        output_filename = f"{book_name}_page_{mapped_page_number}.json"
        output_path = os.path.join(output_folder, output_filename)
        page_data = {
            "page_number": mapped_page_number,
            "book_name": book_name,
            "right_column": {
                "blocks": content
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=4)
    
    @staticmethod
    def save_consolidated_ground_truth(all_pages_data, output_folder, book_name="Unknown_Book"):
        """
        Saves consolidated ground truth data as a single JSON file.
        This includes all pages and extracted text for easy access.
        """
        # Extract pure text content for each page
        consolidated_data = {
            "book_name": book_name,
            "total_pages": len(all_pages_data),
            "extraction_metadata": {
                "extractor": "PDF Ground Truth Extractor",
                "version": "1.0",
                "extraction_date": None,
                "source": "PDF right column extraction"
            },
            "pages": [],
            "full_text": {
                "raw": "",
                "cleaned": "",
                "by_page": {}
            }
        }
        
        # Process each page
        full_raw_text = []
        full_cleaned_text = []
        
        for page_data in all_pages_data:
            page_num = page_data["page_number"]
            blocks = page_data["right_column"]["blocks"]
            
            # Extract text from blocks
            page_raw_text = []
            page_cleaned_text = []
            
            for block in blocks:
                # Handle both dict and string cases
                if isinstance(block, dict):
                    # If block is a dictionary, extract lines
                    for line in block.get("lines", []):
                        if isinstance(line, dict):
                            text = line.get("text", "").strip()
                        else:
                            text = str(line).strip()
                        
                        if text:
                            page_raw_text.append(text)
                            # Apply cleaning and numeral conversion
                            cleaned = ArtefactCleaner.remove_top_artefacts([text])
                            if cleaned:
                                converted = NumeralConverter.convert_arabic_to_gujarati_numerals(cleaned[0])
                                page_cleaned_text.append(converted)
                else:
                    # If block is a string, treat it as text directly
                    text = str(block).strip()
                    if text:
                        page_raw_text.append(text)
                        # Apply cleaning and numeral conversion
                        cleaned = ArtefactCleaner.remove_top_artefacts([text])
                        if cleaned:
                            converted = NumeralConverter.convert_arabic_to_gujarati_numerals(cleaned[0])
                            page_cleaned_text.append(converted)
            
            page_raw_str = "\n".join(page_raw_text)
            page_cleaned_str = "\n".join(page_cleaned_text)
            
            # Add to consolidated data
            consolidated_data["pages"].append({
                "page_number": page_num,
                "text_raw": page_raw_str,
                "text_cleaned": page_cleaned_str,
                "line_count": len(page_raw_text),
                "block_count": len(blocks)
            })
            
            consolidated_data["full_text"]["by_page"][str(page_num)] = {
                "raw": page_raw_str,
                "cleaned": page_cleaned_str
            }
            
            full_raw_text.append(page_raw_str)
            full_cleaned_text.append(page_cleaned_str)
        
        # Combine all text
        consolidated_data["full_text"]["raw"] = "\n\n".join(full_raw_text)
        consolidated_data["full_text"]["cleaned"] = "\n\n".join(full_cleaned_text)
        
        # Add extraction date
        from datetime import datetime
        consolidated_data["extraction_metadata"]["extraction_date"] = datetime.now().isoformat()
        
        # Save consolidated JSON
        consolidated_filename = f"{book_name}_ground_truth.json"
        consolidated_path = os.path.join(output_folder, consolidated_filename)
        
        with open(consolidated_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
        
        # Also save as plain text files
        raw_text_path = os.path.join(output_folder, f"{book_name}_text_raw.txt")
        cleaned_text_path = os.path.join(output_folder, f"{book_name}_text_cleaned.txt")
        
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            f.write(consolidated_data["full_text"]["raw"])
        
        with open(cleaned_text_path, 'w', encoding='utf-8') as f:
            f.write(consolidated_data["full_text"]["cleaned"])
        
        return consolidated_path, raw_text_path, cleaned_text_path

#####################################
# PDF Filename Parsing
#####################################
class FilenameParser:
    @staticmethod
    def parse_pdf_filename(filename):
        """
        Parses a PDF filename like 'Sardar_Vallabhabhai_Part-1_Narhari Parikh.pdf'
        or 'Sardar_Vallabhabhai_473-500.pdf' to extract the base name and page range.
        In this case, we assume the filename ends with _<start>-<end>.pdf.
        """
        match = re.match(r'(.+?)_(\d+)-(\d+)\.pdf$', filename)
        if not match:
            raise ValueError(f"Filename '{filename}' does not match expected pattern")
        base_name = match.group(1)
        start_page = int(match.group(2))
        end_page = int(match.group(3))
        return base_name, start_page, end_page

#####################################
# PDF Processing
#####################################
class PDFProcessor:
    def __init__(self, pdf_path, output_folder, start_page=15, end_page=None, pages_to_exclude=None):
        """
        Initializes the PDFProcessor.
        
        Args:
            pdf_path (str): Path to the PDF file.
            output_folder (str): Folder to save the JSON output.
            start_page (int): The first page (1-indexed) to process.
            end_page (int or None): The last page (1-indexed) to process. If None, process till the end.
            pages_to_exclude (list or None): List of page numbers (1-indexed) to exclude. If None, exclude no pages.
        """
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        self.start_page = start_page
        self.end_page = end_page
        self.pages_to_exclude = pages_to_exclude if pages_to_exclude is not None else []
        self.book_name = self._extract_book_name()

    def _extract_book_name(self):
        """Extract book name from PDF filename"""
        filename = os.path.basename(self.pdf_path)
        # Remove extension and clean up name
        book_name = os.path.splitext(filename)[0]
        # Replace spaces and special characters with underscores
        book_name = re.sub(r'[^\w\-_.]', '_', book_name)
        return book_name

    def process(self):
        """
        Opens the PDF, processes the specified page range (skipping excluded pages),
        maps processed pages to output page numbers, saves JSON outputs, and creates consolidated ground truth.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        all_pages_data = []
        
        with fitz.open(self.pdf_path) as pdf:
            total_pages = len(pdf)
            # Convert the 1-indexed start_page to 0-indexed
            start_index = self.start_page - 1
            # Determine the end index (0-indexed)
            if self.end_page is not None:
                end_index = min(self.end_page, total_pages) - 1
            else:
                end_index = total_pages - 1

            processed_count = 0
            for i in range(start_index, end_index + 1):
                current_page_num = i + 1  # 1-indexed page number
                # Skip if this page is in the exclusion list
                if current_page_num in self.pages_to_exclude:
                    if current_page_num == 16:
                        processed_count += 1
                    print(f"Skipping page {current_page_num} as it is in the excluded list.")
                    continue
                
                page = pdf.load_page(i)
                blocks = RightColumnExtractor.extract_blocks(page)
                processed_count += 1
                mapped_page_number = processed_count  # Mapped output page number starts at 1
                
                # Save individual page JSON
                JSONOutputSaver.save(mapped_page_number, blocks, self.output_folder, self.book_name)
                
                # Collect page data for consolidated output
                page_data = {
                    "page_number": mapped_page_number,
                    "book_name": self.book_name,
                    "original_page_number": current_page_num,
                    "right_column": {
                        "blocks": blocks
                    }
                }
                all_pages_data.append(page_data)
                
                print(f"✅ Processed page {current_page_num} → output page {mapped_page_number}")
            
            # Generate consolidated ground truth
            if all_pages_data:
                print(f"\n📊 Generating consolidated ground truth...")
                consolidated_path, raw_text_path, cleaned_text_path = JSONOutputSaver.save_consolidated_ground_truth(
                    all_pages_data, self.output_folder, self.book_name
                )
                
                print(f"✅ Consolidated JSON: {os.path.basename(consolidated_path)}")
                print(f"✅ Raw text file: {os.path.basename(raw_text_path)}")
                print(f"✅ Cleaned text file: {os.path.basename(cleaned_text_path)}")
            
            print(f"\n📈 Summary:")
            print(f"   📄 Total pages processed: {processed_count}")
            print(f"   📁 Individual page JSONs: {processed_count}")
            print(f"   📋 Consolidated files: 3 (JSON + 2 text files)")
            print(f"   📂 Output folder: '{self.output_folder}'")

#####################################
# Main Entry Point
#####################################
def extract_ground_truth_from_pdf(pdf_path, output_folder, start_page, end_page, pages_to_exclude):
    """
    Extracts ground truth from the PDF using the specified parameters.
    """
    processor = PDFProcessor(pdf_path, output_folder, start_page, end_page, pages_to_exclude)
    processor.process()

if __name__ == "__main__":
    pdf_path = '/Users/swapn/Downloads/Books/Sardar Vallabhai/Sample3/Sardar_Vallabhabhai_Part-1_Narhari Parikh.pdf'
    output_folder = '/Users/swapn/Downloads/OCR Experiments/Sardar Vallabhai/ground_truth_extracted'
    
    # Set the desired parameters (1-indexed):
    start_page = 15         # Start processing from page 15
    end_page = 48           # Process up to (and including) page 20
    pages_to_exclude = [23, 24, 33, 34] # For example, exclude page 17
    
    extract_ground_truth_from_pdf(pdf_path, output_folder, start_page, end_page, pages_to_exclude)