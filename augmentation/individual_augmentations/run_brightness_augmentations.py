#!/usr/bin/env python3
"""
Brightness Augmentation Generator
Generates systematic brightness variations with exact parameter values in filenames
"""

import cv2
import numpy as np
import os
import sys
import random
from pathlib import Path

# Add the augraphy path
sys.path.append('/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy')

from augraphy import *
from augraphy.base import AugraphyPipeline

def setup_environment():
    """Setup reproducible environment"""
    random.seed(42)
    np.random.seed(42)

def load_image():
    """Load the source image"""
    image_path = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/DQA_data/sampled_images/06 Harishchandra Natak_001.png"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    print(f"Loaded image: {image.shape}, dtype: {image.dtype}")
    return image

def create_param_filename(aug_type, **params):
    """Create filename with exact parameter values"""
    param_parts = []
    for key, value in params.items():
        if isinstance(value, tuple):
            param_parts.append(f"{key}_{value[0]}to{value[1]}")
        else:
            param_parts.append(f"{key}_{value}")
    
    param_string = "_".join(param_parts)
    return f"{aug_type}_{param_string}"

def save_augmented_image(image, output_dir, filename):
    """Save augmented image with proper directory structure"""
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{filename}.png")
    
    success = cv2.imwrite(filepath, image)
    if success:
        print(f"✓ Saved: {filepath}")
        return filepath
    else:
        print(f"✗ Failed: {filepath}")
        return None

def save_original_image(image, output_dir):
    """Save original image for comparison"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "original.png")
    success = cv2.imwrite(filepath, image)
    if success:
        print(f"✓ Saved original: {filepath}")
    return filepath

def generate_brightness_augmentations(image, base_output_dir):
    """
    Generate comprehensive brightness augmentations with exact parameter values
    """
    print("\n" + "="*80)
    print("BRIGHTNESS AUGMENTATION GENERATION")
    print("="*80)
    
    # Create output directory structure
    brightness_dir = os.path.join(base_output_dir, "brightness_augmentations")
    os.makedirs(brightness_dir, exist_ok=True)
    
    # Save original image
    save_original_image(image, brightness_dir)
    
    # Comprehensive brightness configurations
    brightness_configs = [
        # Basic brightness range variations
        ((0.3, 0.5), 0, (0, 0), "very_dark_range"),
        ((0.5, 0.7), 0, (0, 0), "dark_range"),
        ((0.7, 0.9), 0, (0, 0), "slightly_dark_range"),
        ((0.9, 1.1), 0, (0, 0), "normal_range"),
        ((1.1, 1.3), 0, (0, 0), "slightly_bright_range"),
        ((1.3, 1.5), 0, (0, 0), "bright_range"),
        ((1.5, 1.8), 0, (0, 0), "very_bright_range"),
        ((1.8, 2.2), 0, (0, 0), "extremely_bright_range"),
        
        # Fixed brightness values for precise control
        ((0.4, 0.4), 0, (0, 0), "fixed_0.4"),
        ((0.6, 0.6), 0, (0, 0), "fixed_0.6"),
        ((0.8, 0.8), 0, (0, 0), "fixed_0.8"),
        ((1.0, 1.0), 0, (0, 0), "fixed_1.0_baseline"),
        ((1.2, 1.2), 0, (0, 0), "fixed_1.2"),
        ((1.4, 1.4), 0, (0, 0), "fixed_1.4"),
        ((1.6, 1.6), 0, (0, 0), "fixed_1.6"),
        
        # With minimum brightness enforcement - prevents complete darkness
        ((0.4, 0.6), 1, (30, 50), "dark_with_min_brightness_30to50"),
        ((0.5, 0.7), 1, (50, 80), "dark_with_min_brightness_50to80"),
        ((0.6, 0.8), 1, (80, 120), "dark_with_min_brightness_80to120"),
        ((0.3, 0.5), 1, (100, 150), "very_dark_with_high_min_brightness"),
        
        # Edge cases for stress testing
        ((0.1, 0.3), 0, (0, 0), "extreme_dark_range"),
        ((2.0, 2.5), 0, (0, 0), "extreme_bright_range"),
        ((0.2, 0.2), 1, (200, 200), "extreme_dark_with_high_min"),
    ]
    
    print(f"Generating {len(brightness_configs)} brightness variations...")
    
    for i, (brightness_range, min_brightness, min_brightness_value, description) in enumerate(brightness_configs):
        print(f"\n[{i+1}/{len(brightness_configs)}] Processing: {description}")
        print(f"  Parameters: brightness_range={brightness_range}, min_brightness={min_brightness}, min_brightness_value={min_brightness_value}")
        
        # Create Augraphy pipeline
        pipeline = AugraphyPipeline(
            ink_phase=[Brightness(
                brightness_range=brightness_range,
                min_brightness=min_brightness,
                min_brightness_value=min_brightness_value,
                p=1.0
            )],
            paper_phase=[],
            post_phase=[]
        )
        
        # Apply augmentation
        try:
            augmented = pipeline(image)
            
            # Create filename with exact parameters
            filename = create_param_filename("brightness", 
                                           brightness_range=brightness_range,
                                           min_brightness=min_brightness,
                                           min_brightness_value=min_brightness_value)
            
            # Save image
            save_augmented_image(augmented, brightness_dir, filename)
            
        except Exception as e:
            print(f"  ✗ Error processing {description}: {e}")
            continue
    
    print(f"\n✅ Brightness augmentation complete!")
    print(f"📁 Output directory: {brightness_dir}")
    print(f"📊 Generated {len(brightness_configs)} variations")

def generate_analysis_summary(base_output_dir):
    """Generate analysis summary file"""
    brightness_dir = os.path.join(base_output_dir, "brightness_augmentations")
    summary_file = os.path.join(brightness_dir, "brightness_analysis_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("BRIGHTNESS AUGMENTATION ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("Generated Variations:\n")
        f.write("1. Basic Brightness Ranges: 8 variations (0.3 to 2.2 range)\n")
        f.write("2. Fixed Brightness Values: 7 precise control points\n")
        f.write("3. With Min Brightness: 4 variations preventing complete darkness\n")
        f.write("4. Edge Cases: 3 extreme testing scenarios\n\n")
        
        f.write("OCR Testing Recommendations:\n")
        f.write("- LOW IMPACT: fixed_1.0_baseline, normal_range, slightly_dark/bright_range\n")
        f.write("- MEDIUM IMPACT: dark_range, bright_range, with_min_brightness variations\n")
        f.write("- HIGH IMPACT: very_dark_range, very_bright_range, extreme variations\n\n")
        
        f.write("Parameter Analysis:\n")
        f.write("- brightness_range < 0.7: May cause text merging and loss of detail\n")
        f.write("- brightness_range > 1.4: May cause text fading and thin appearance\n")
        f.write("- min_brightness=1: Helps preserve text in dark regions\n")
        f.write("- min_brightness_value: Higher values prevent complete black areas\n\n")
        
        f.write("Next Steps:\n")
        f.write("1. Run OCR evaluation on all generated images\n")
        f.write("2. Compare accuracy against original baseline\n")
        f.write("3. Identify optimal parameter ranges for training data\n")
        f.write("4. Document character-level error patterns\n")
    
    print(f"📋 Analysis summary saved: {summary_file}")

def main():
    """Main execution function"""
    print("🚀 Starting Brightness Augmentation Generation")
    print("="*80)
    
    # Setup
    setup_environment()
    base_output_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy_experiments"
    
    # Load image
    print("📂 Loading source image...")
    image = load_image()
    
    # Generate brightness augmentations
    generate_brightness_augmentations(image, base_output_dir)
    
    # Generate analysis summary
    generate_analysis_summary(base_output_dir)
    
    print("\n" + "="*80)
    print("✅ BRIGHTNESS AUGMENTATION GENERATION COMPLETE!")
    print("="*80)
    print(f"📁 All outputs saved to: {base_output_dir}/brightness_augmentations/")
    print("🔍 Ready for OCR impact analysis!")

if __name__ == "__main__":
    main()
