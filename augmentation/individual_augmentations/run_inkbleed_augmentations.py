#!/usr/bin/env python3
"""
InkBleed Augmentation Generator
Generates systematic ink bleeding variations with exact parameter values in filenames
Parts 1, 2, 3: Intensity, Kernel Size, and Severity variations
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
            if len(value) == 2 and isinstance(value[0], (int, float)):
                param_parts.append(f"{key}_{value[0]}to{value[1]}")
            else:
                param_parts.append(f"{key}_{value[0]}x{value[1]}")
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

def generate_inkbleed_augmentations(image, base_output_dir):
    """
    Generate MODERATE-TO-HIGH InkBleed augmentations with exact parameter values
    Balanced ranges: Intensity (0.5-1.0), Severity (0.5-1.0), Kernel (5-10)
    """
    print("\n" + "="*80)
    print("MODERATE-TO-HIGH INKBLEED AUGMENTATION GENERATION")
    print("="*80)
    
    # Delete old directory if it exists and create new one
    inkbleed_dir = os.path.join(base_output_dir, "inkbleed_augmentations")
    if os.path.exists(inkbleed_dir):
        import shutil
        print(f"🗑️  Deleting old directory: {inkbleed_dir}")
        shutil.rmtree(inkbleed_dir)
    os.makedirs(inkbleed_dir, exist_ok=True)
    print(f"📁 Created fresh directory: {inkbleed_dir}")
    
    # Save original image
    save_original_image(image, inkbleed_dir)
    
    # All configurations to test
    all_configs = []
    
    # PART 1: Intensity Variations (5 combinations) - 0.5 to 1.0
    print("\n📊 PART 1: Intensity Variations (0.5-1.0)")
    intensity_configs = [
        ((0.5, 0.6), (7, 7), (0.7, 0.8), "intensity_0.5to0.6"),
        ((0.6, 0.7), (7, 7), (0.7, 0.8), "intensity_0.6to0.7"),
        ((0.7, 0.8), (7, 7), (0.7, 0.8), "intensity_0.7to0.8"),
        ((0.8, 0.9), (7, 7), (0.7, 0.8), "intensity_0.8to0.9"),
        ((0.9, 1.0), (7, 7), (0.7, 0.8), "intensity_0.9to1.0"),
    ]
    all_configs.extend(intensity_configs)
    
    # PART 2: Kernel Size Variations (6 combinations) - 5 to 10
    print("📊 PART 2: Kernel Size Variations (5-10)")
    kernel_configs = [
        ((0.75, 0.85), (5, 5), (0.75, 0.85), "kernel_5x5"),
        ((0.75, 0.85), (6, 6), (0.75, 0.85), "kernel_6x6"),
        ((0.75, 0.85), (7, 7), (0.75, 0.85), "kernel_7x7"),
        ((0.75, 0.85), (8, 8), (0.75, 0.85), "kernel_8x8"),
        ((0.75, 0.85), (9, 9), (0.75, 0.85), "kernel_9x9"),
        ((0.75, 0.85), (10, 10), (0.75, 0.85), "kernel_10x10"),
    ]
    all_configs.extend(kernel_configs)
    
    # PART 3: Severity Variations (5 combinations) - 0.5 to 1.0
    print("📊 PART 3: Severity Variations (0.5-1.0)")
    severity_configs = [
        ((0.75, 0.85), (7, 7), (0.5, 0.6), "severity_0.5to0.6"),
        ((0.75, 0.85), (7, 7), (0.6, 0.7), "severity_0.6to0.7"),
        ((0.75, 0.85), (7, 7), (0.7, 0.8), "severity_0.7to0.8"),
        ((0.75, 0.85), (7, 7), (0.8, 0.9), "severity_0.8to0.9"),
        ((0.75, 0.85), (7, 7), (0.9, 1.0), "severity_0.9to1.0"),
    ]
    all_configs.extend(severity_configs)
    
    total_configs = len(all_configs)
    print(f"\n🎯 MODERATE-TO-HIGH configurations to generate: {total_configs}")
    print("  - Part 1 (Intensity 0.5-1.0): 5 variations")
    print("  - Part 2 (Kernels 5-10): 6 variations")
    print("  - Part 3 (Severity 0.5-1.0): 5 variations")
    print("  📊 These parameters will create moderate to strong bleeding effects")
    
    # Generate all configurations
    for i, (intensity_range, kernel_size, severity, description) in enumerate(all_configs):
        print(f"\n[{i+1}/{total_configs}] Processing: {description}")
        print(f"  Parameters: intensity={intensity_range}, kernel={kernel_size}, severity={severity}")
        
        # Create Augraphy pipeline
        pipeline = AugraphyPipeline(
            ink_phase=[InkBleed(
                intensity_range=intensity_range,
                kernel_size=kernel_size,
                severity=severity,
                p=1.0
            )],
            paper_phase=[],
            post_phase=[]
        )
        
        # Apply augmentation
        try:
            augmented = pipeline(image)
            
            # Create filename with exact parameters
            filename = create_param_filename("inkbleed", 
                                           intensity_range=intensity_range,
                                           kernel_size=kernel_size,
                                           severity=severity)
            
            # Save image
            save_augmented_image(augmented, inkbleed_dir, filename)
            
        except Exception as e:
            print(f"  ✗ Error processing {description}: {e}")
            continue
    
    print(f"\n✅ InkBleed augmentation complete!")
    print(f"📁 Output directory: {inkbleed_dir}")
    print(f"📊 Generated {total_configs} variations")

def generate_analysis_summary(base_output_dir):
    """Generate analysis summary file"""
    inkbleed_dir = os.path.join(base_output_dir, "inkbleed_augmentations")
    summary_file = os.path.join(inkbleed_dir, "inkbleed_analysis_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("INKBLEED AUGMENTATION ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("Generated Variations:\n")
        f.write("1. Intensity Range Variations: 6 variations (0.05 to 1.0 range)\n")
        f.write("   - Tests bleeding strength from minimal to extreme\n")
        f.write("2. Kernel Size Variations: 7 variations (1x1 to 11x11)\n")
        f.write("   - Tests bleeding area from tiny to huge\n")
        f.write("3. Severity Variations: 5 variations (0.1 to 1.0 range)\n")
        f.write("   - Tests bleeding concentration from low to extreme\n\n")
        
        f.write("OCR Testing Recommendations:\n")
        f.write("- LOW IMPACT: minimal_intensity, light_intensity, tiny_kernel, small_kernel, low_severity\n")
        f.write("- MEDIUM IMPACT: moderate_intensity, medium_kernel, medium_severity\n")
        f.write("- HIGH IMPACT: heavy_intensity, large_kernel, very_large_kernel, high_severity\n")
        f.write("- VERY HIGH IMPACT: extreme_intensity, huge_kernel, extreme_severity\n\n")
        
        f.write("Parameter Analysis:\n")
        f.write("- intensity_range < 0.2: Minimal OCR impact, slight edge softening\n")
        f.write("- intensity_range 0.3-0.6: Moderate impact, noticeable character thickening\n")
        f.write("- intensity_range > 0.7: High impact, significant character merging\n")
        f.write("- kernel_size (1,1)-(3,3): Localized bleeding, minimal impact\n")
        f.write("- kernel_size (5,5)-(7,7): Standard bleeding, moderate impact\n")
        f.write("- kernel_size > (9,9): Wide bleeding, high impact on small text\n")
        f.write("- severity > 0.6: Heavy concentration, may obscure fine details\n\n")
        
        f.write("Expected Character-Level Effects:\n")
        f.write("- Low parameters: Slight edge blur, maintains character integrity\n")
        f.write("- Medium parameters: Character thickening, some detail loss\n")
        f.write("- High parameters: Character merging, significant shape distortion\n")
        f.write("- Extreme parameters: Heavy bleeding may make text illegible\n\n")
        
        f.write("Next Steps:\n")
        f.write("1. Run OCR evaluation on all generated images\n")
        f.write("2. Measure character-level accuracy degradation\n")
        f.write("3. Identify threshold parameters for acceptable OCR performance\n")
        f.write("4. Test combinations (Parts 4-5) based on these results\n")
        f.write("5. Document bleeding effects on Gujarati conjunct characters\n")
    
    print(f"📋 Analysis summary saved: {summary_file}")

def main():
    """Main execution function"""
    print("🚀 Starting InkBleed Augmentation Generation")
    print("🎯 Parts 1-3: Intensity, Kernel Size, and Severity Variations")
    print("="*80)
    
    # Setup
    setup_environment()
    base_output_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy_experiments"
    
    # Load image
    print("📂 Loading source image...")
    image = load_image()
    
    # Generate InkBleed augmentations
    generate_inkbleed_augmentations(image, base_output_dir)
    
    # Generate analysis summary
    generate_analysis_summary(base_output_dir)
    
    print("\n" + "="*80)
    print("✅ INKBLEED AUGMENTATION GENERATION COMPLETE!")
    print("="*80)
    print(f"📁 All outputs saved to: {base_output_dir}/inkbleed_augmentations/")
    print("🔍 Ready for OCR impact analysis!")
    print("\n📋 Summary of generated variations:")
    print("   • 6 Intensity variations (bleeding strength)")
    print("   • 7 Kernel size variations (bleeding area)")  
    print("   • 5 Severity variations (bleeding concentration)")
    print("   • Total: 18 systematic InkBleed variations")

if __name__ == "__main__":
    main()
