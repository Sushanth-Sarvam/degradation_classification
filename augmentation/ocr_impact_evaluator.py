#!/usr/bin/env python3
"""
OCR Impact Evaluator
Evaluates the impact of different augmentation parameters on OCR accuracy
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json
import time

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    print("Warning: pytesseract not available. Install with: pip install pytesseract")
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("Warning: easyocr not available. Install with: pip install easyocr")
    EASYOCR_AVAILABLE = False

def setup_ocr_engines():
    """Setup available OCR engines"""
    engines = {}
    
    if TESSERACT_AVAILABLE:
        engines['tesseract'] = {
            'name': 'Tesseract',
            'engine': pytesseract,
            'config': '--oem 3 --psm 6'  # Default config for documents
        }
    
    if EASYOCR_AVAILABLE:
        try:
            reader = easyocr.Reader(['en', 'hi'])  # English and Hindi for Gujarati support
            engines['easyocr'] = {
                'name': 'EasyOCR',
                'engine': reader,
                'config': None
            }
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR: {e}")
    
    return engines

def extract_text_tesseract(image, config="--oem 3 --psm 6"):
    """Extract text using Tesseract"""
    try:
        text = pytesseract.image_to_string(image, config=config, lang='eng+hin')
        confidence_data = pytesseract.image_to_data(image, config=config, lang='eng+hin', output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence
        confidences = [int(conf) for conf in confidence_data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'text': text.strip(),
            'confidence': avg_confidence,
            'word_count': len(text.split()),
            'char_count': len(text.replace(' ', ''))
        }
    except Exception as e:
        return {
            'text': '',
            'confidence': 0,
            'word_count': 0,
            'char_count': 0,
            'error': str(e)
        }

def extract_text_easyocr(reader, image):
    """Extract text using EasyOCR"""
    try:
        results = reader.readtext(image)
        
        text_parts = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            text_parts.append(text)
            confidences.append(confidence)
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'text': full_text.strip(),
            'confidence': avg_confidence * 100,  # Convert to percentage
            'word_count': len(full_text.split()),
            'char_count': len(full_text.replace(' ', ''))
        }
    except Exception as e:
        return {
            'text': '',
            'confidence': 0,
            'word_count': 0,
            'char_count': 0,
            'error': str(e)
        }

def calculate_text_similarity(text1, text2):
    """Calculate simple text similarity metrics"""
    if not text1 or not text2:
        return 0.0
    
    # Simple character-level similarity
    text1_clean = text1.lower().replace(' ', '').replace('\n', '')
    text2_clean = text2.lower().replace(' ', '').replace('\n', '')
    
    if not text1_clean or not text2_clean:
        return 0.0
    
    # Calculate Levenshtein distance (simplified)
    len1, len2 = len(text1_clean), len(text2_clean)
    
    # Create matrix
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize first row and column
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    
    # Fill matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if text1_clean[i-1] == text2_clean[j-1]:
                cost = 0
            else:
                cost = 1
            
            matrix[i][j] = min(
                matrix[i-1][j] + 1,      # deletion
                matrix[i][j-1] + 1,      # insertion
                matrix[i-1][j-1] + cost  # substitution
            )
    
    # Calculate similarity percentage
    max_len = max(len1, len2)
    if max_len == 0:
        return 100.0
    
    distance = matrix[len1][len2]
    similarity = ((max_len - distance) / max_len) * 100
    return max(0.0, similarity)

def evaluate_image_ocr(image_path, engines, baseline_text=None):
    """Evaluate OCR performance on a single image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    results = {
        'image_path': image_path,
        'image_size': f"{image.shape[1]}x{image.shape[0]}",
        'engines': {}
    }
    
    for engine_name, engine_config in engines.items():
        print(f"  Processing with {engine_config['name']}...")
        
        start_time = time.time()
        
        if engine_name == 'tesseract':
            ocr_result = extract_text_tesseract(image, engine_config['config'])
        elif engine_name == 'easyocr':
            ocr_result = extract_text_easyocr(engine_config['engine'], image)
        
        processing_time = time.time() - start_time
        ocr_result['processing_time'] = processing_time
        
        # Calculate similarity to baseline if available
        if baseline_text and ocr_result.get('text'):
            similarity = calculate_text_similarity(baseline_text, ocr_result['text'])
            ocr_result['similarity_to_baseline'] = similarity
        
        results['engines'][engine_name] = ocr_result
    
    return results

def evaluate_augmentation_impact(base_dir="augraphy_experiments"):
    """Evaluate OCR impact across all augmentation types"""
    
    engines = setup_ocr_engines()
    if not engines:
        print("No OCR engines available. Please install pytesseract or easyocr.")
        return
    
    print(f"Available OCR engines: {list(engines.keys())}")
    
    # Get baseline text from original image
    original_path = f"{base_dir}/original/images/06_Harishchandra_Natak_001_original.png"
    baseline_text = None
    
    if os.path.exists(original_path):
        print("Extracting baseline text from original image...")
        baseline_result = evaluate_image_ocr(original_path, engines)
        if baseline_result and 'tesseract' in baseline_result['engines']:
            baseline_text = baseline_result['engines']['tesseract'].get('text', '')
        print(f"Baseline text length: {len(baseline_text) if baseline_text else 0} characters")
    
    # Evaluate ink phase augmentations
    ink_phase_dir = f"{base_dir}/ink_phase"
    results = {
        'baseline': baseline_result,
        'augmentations': {}
    }
    
    if os.path.exists(ink_phase_dir):
        print("\nEvaluating Ink Phase augmentations...")
        
        for augmentation_type in os.listdir(ink_phase_dir):
            aug_dir = os.path.join(ink_phase_dir, augmentation_type)
            if os.path.isdir(aug_dir):
                print(f"\nProcessing {augmentation_type}...")
                results['augmentations'][augmentation_type] = {}
                
                for image_file in os.listdir(aug_dir):
                    if image_file.endswith('.png'):
                        image_path = os.path.join(aug_dir, image_file)
                        print(f"  Evaluating {image_file}...")
                        
                        ocr_result = evaluate_image_ocr(image_path, engines, baseline_text)
                        if ocr_result:
                            results['augmentations'][augmentation_type][image_file] = ocr_result
    
    return results

def save_results(results, output_file="augraphy_experiments/ocr_evaluation_results.json"):
    """Save evaluation results to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")

def generate_summary_report(results):
    """Generate a summary report of OCR performance"""
    print("\n" + "="*80)
    print("OCR IMPACT ANALYSIS SUMMARY")
    print("="*80)
    
    if 'baseline' in results and results['baseline']:
        baseline = results['baseline']['engines'].get('tesseract', {})
        print(f"\nBASELINE (Original Image):")
        print(f"  Text length: {baseline.get('char_count', 0)} characters")
        print(f"  Word count: {baseline.get('word_count', 0)} words")
        print(f"  Tesseract confidence: {baseline.get('confidence', 0):.1f}%")
    
    if 'augmentations' in results:
        print(f"\nAUGMENTATION IMPACT:")
        
        for aug_type, aug_results in results['augmentations'].items():
            print(f"\n{aug_type.upper()}:")
            
            confidences = []
            similarities = []
            
            for image_name, image_result in aug_results.items():
                tesseract_result = image_result.get('engines', {}).get('tesseract', {})
                confidence = tesseract_result.get('confidence', 0)
                similarity = tesseract_result.get('similarity_to_baseline', 0)
                
                confidences.append(confidence)
                similarities.append(similarity)
                
                print(f"  {image_name}:")
                print(f"    Confidence: {confidence:.1f}%")
                print(f"    Similarity: {similarity:.1f}%")
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                avg_similarity = sum(similarities) / len(similarities)
                print(f"  AVERAGE - Confidence: {avg_confidence:.1f}%, Similarity: {avg_similarity:.1f}%")

def main():
    """Main function to run OCR evaluation"""
    print("Starting OCR Impact Evaluation...")
    
    # Check if images exist
    base_dir = "augraphy_experiments"
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory not found.")
        print("Please run the augmentation generation script first.")
        return
    
    try:
        # Run evaluation
        results = evaluate_augmentation_impact(base_dir)
        
        if results:
            # Save detailed results
            save_results(results)
            
            # Generate summary report
            generate_summary_report(results)
            
            print(f"\n{'='*60}")
            print("EVALUATION COMPLETE")
            print(f"{'='*60}")
            print("Use the results to understand how different augmentation parameters")
            print("affect OCR accuracy and confidence scores.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()