#!/usr/bin/env python3
"""
Letterpress Augmentation Generation Script
==========================================

This script generates systematic Letterpress augmentations with exact parameter values
for testing OCR robustness against uneven ink pressure effects.

Letterpress simulates letterpress printing with uneven ink density, creating texture interference.

Key Parameters:
- n_samples: Number of points in each cluster (density of effect)
- n_clusters: Number of effect clusters (distribution of effects)  
- std_range: Standard deviation of blob distribution (size of affected areas)
- value_range: Pixel intensity of letterpress effect (darkness/visibility)
- value_threshold_range: Minimum pixel value to apply effect
- blur: Enable/disable blur on letterpress effect

OCR Impact: Creates local intensity variations that can confuse character recognition
"""

import os
import sys
import cv2
import numpy as np

# Add the augraphy path
sys.path.append('/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy')

from augraphy import *

def setup_random_seed(seed=42):
    """Set random seed for reproducible results"""
    np.random.seed(seed)
    
def load_image(image_path):
    """Load and validate image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
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
    """Save augmented image with error handling"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.png")
    
    success = cv2.imwrite(filepath, image)
    if success:
        print(f"✓ Saved: {filepath}")
    else:
        print(f"✗ Failed to save: {filepath}")
    
    return filepath

def save_original_image(image, output_dir):
    """Save original image for comparison"""
    filepath = os.path.join(output_dir, "original.png")
    success = cv2.imwrite(filepath, image)
    if success:
        print(f"✓ Saved original: {filepath}")

    return filepath

def generate_letterpress_augmentations(image, base_output_dir):
    """
    Generate systematic Letterpress augmentations with exact parameter values
    Parts 1-4: Samples, Clusters, Std Range, and Value Range variations
    """
    print("\n" + "="*80)
    print("LETTERPRESS AUGMENTATION GENERATION")
    print("="*80)
    
    # Delete old directory if it exists and create new one
    letterpress_dir = os.path.join(base_output_dir, "letterpress_augmentations")
    if os.path.exists(letterpress_dir):
        import shutil
        print(f"🗑️  Deleting old directory: {letterpress_dir}")
        shutil.rmtree(letterpress_dir)
    os.makedirs(letterpress_dir, exist_ok=True)
    print(f"📁 Created fresh directory: {letterpress_dir}")
    
    # Save original image
    save_original_image(image, letterpress_dir)
    
    # All configurations to test
    all_configs = []
    
    # PART 1: Sample Count Variations (5 combinations)
    print("\n📊 PART 1: Sample Count Variations")
    sample_configs = [
        ((100, 150), (300, 400), (1500, 2500), (180, 220), 1, "samples_100to150"),
        ((200, 300), (300, 400), (1500, 2500), (180, 220), 1, "samples_200to300"),
        ((300, 450), (300, 400), (1500, 2500), (180, 220), 1, "samples_300to450"),
        ((500, 650), (300, 400), (1500, 2500), (180, 220), 1, "samples_500to650"),
        ((700, 850), (300, 400), (1500, 2500), (180, 220), 1, "samples_700to850"),
    ]
    all_configs.extend(sample_configs)
    
    # PART 2: Cluster Count Variations (5 combinations)
    print("📊 PART 2: Cluster Count Variations")
    cluster_configs = [
        ((400, 500), (100, 200), (1500, 2500), (180, 220), 1, "clusters_100to200"),
        ((400, 500), (250, 350), (1500, 2500), (180, 220), 1, "clusters_250to350"),
        ((400, 500), (400, 500), (1500, 2500), (180, 220), 1, "clusters_400to500"),
        ((400, 500), (600, 700), (1500, 2500), (180, 220), 1, "clusters_600to700"),
        ((400, 500), (800, 900), (1500, 2500), (180, 220), 1, "clusters_800to900"),
    ]
    all_configs.extend(cluster_configs)
    
    # PART 3: Standard Deviation Variations (5 combinations)
    print("📊 PART 3: Standard Deviation (Spread) Variations")
    std_configs = [
        ((400, 500), (350, 450), (500, 1000), (180, 220), 1, "std_500to1000"),
        ((400, 500), (350, 450), (1000, 1500), (180, 220), 1, "std_1000to1500"),
        ((400, 500), (350, 450), (1500, 2500), (180, 220), 1, "std_1500to2500"),
        ((400, 500), (350, 450), (2500, 3500), (180, 220), 1, "std_2500to3500"),
        ((400, 500), (350, 450), (3500, 5000), (180, 220), 1, "std_3500to5000"),
    ]
    all_configs.extend(std_configs)
    
    # PART 4: Value Range Variations (6 combinations)
    print("📊 PART 4: Value Range (Darkness) Variations")
    value_configs = [
        ((400, 500), (350, 450), (1500, 2500), (80, 120), 1, "values_80to120_dark"),
        ((400, 500), (350, 450), (1500, 2500), (120, 160), 1, "values_120to160_medium_dark"),
        ((400, 500), (350, 450), (1500, 2500), (160, 200), 1, "values_160to200_medium"),
        ((400, 500), (350, 450), (1500, 2500), (200, 240), 1, "values_200to240_light"),
        ((400, 500), (350, 450), (1500, 2500), (240, 255), 1, "values_240to255_very_light"),
        ((400, 500), (350, 450), (1500, 2500), (100, 200), 1, "values_100to200_wide_range"),
    ]
    all_configs.extend(value_configs)
    
    # PART 5: Blur Variations (2 combinations)
    print("📊 PART 5: Blur Effect Variations")
    blur_configs = [
        ((400, 500), (350, 450), (1500, 2500), (160, 200), 0, "blur_disabled"),
        ((400, 500), (350, 450), (1500, 2500), (160, 200), 1, "blur_enabled"),
    ]
    all_configs.extend(blur_configs)
    
    total_configs = len(all_configs)
    print(f"\n🎯 LETTERPRESS configurations to generate: {total_configs}")
    print("  - Part 1 (Sample Count): 5 variations")
    print("  - Part 2 (Cluster Count): 5 variations")
    print("  - Part 3 (Std Deviation): 5 variations")
    print("  - Part 4 (Value Range): 6 variations")
    print("  - Part 5 (Blur Effect): 2 variations")
    print("  📊 These parameters will create varying levels of letterpress texture")
    
    # Generate all configurations
    for i, (n_samples, n_clusters, std_range, value_range, blur, description) in enumerate(all_configs):
        print(f"\n[{i+1}/{total_configs}] Processing: {description}")
        print(f"  Parameters: samples={n_samples}, clusters={n_clusters}, std={std_range}, values={value_range}, blur={blur}")
        
        # Create Augraphy pipeline
        pipeline = AugraphyPipeline(
            ink_phase=[Letterpress(
                n_samples=n_samples,
                n_clusters=n_clusters,
                std_range=std_range,
                value_range=value_range,
                value_threshold_range=(128, 128),  # Standard threshold
                blur=blur,
                p=1.0
            )],
            paper_phase=[],
            post_phase=[]
        )
        
        # Apply augmentation
        augmented = pipeline(image)
        
        # Create filename with exact parameters
        filename = create_param_filename("letterpress",
                                       n_samples=n_samples,
                                       n_clusters=n_clusters,
                                       std_range=std_range,
                                       value_range=value_range,
                                       blur=blur)
        
        # Save augmented image
        save_augmented_image(augmented, letterpress_dir, filename)
    
    # Create analysis summary
    create_letterpress_summary(letterpress_dir, total_configs)
    
    print(f"\n✅ Letterpress augmentation complete!")
    print(f"📁 Output directory: {letterpress_dir}")
    print(f"📊 Generated {total_configs} variations")
    print(f"📋 Analysis summary saved: {letterpress_dir}/letterpress_analysis_summary.txt")

def create_letterpress_summary(output_dir, total_configs):
    """Create a summary of generated Letterpress augmentations"""
    summary_content = f"""# Letterpress Augmentation Analysis Summary

## Overview
Generated {total_configs} systematic Letterpress variations to test OCR robustness against 
uneven ink pressure effects and texture interference.

## Parameter Categories

### 1. Sample Count Variations (5 configs)
- Tests effect of point density within each cluster
- Range: 100-850 samples per cluster
- OCR Impact: Higher sample counts create denser texture interference

### 2. Cluster Count Variations (5 configs)  
- Tests effect of number of letterpress effect areas
- Range: 100-900 clusters per image
- OCR Impact: More clusters create more distributed interference

### 3. Standard Deviation Variations (5 configs)
- Tests effect of letterpress blob spread/size
- Range: 500-5000 standard deviation
- OCR Impact: Higher std creates larger affected areas

### 4. Value Range Variations (6 configs)
- Tests effect of letterpress darkness/intensity
- Range: 80-255 pixel values (darker to lighter)
- OCR Impact: Darker values (80-120) create stronger interference

### 5. Blur Effect Variations (2 configs)
- Tests effect of smooth vs sharp letterpress edges
- Options: Blur disabled/enabled
- OCR Impact: Blur creates smoother transitions, potentially less disruptive

## OCR Testing Strategy
1. Test each parameter category separately
2. Identify threshold values for acceptable degradation
3. Focus on dark value ranges (80-160) for stress testing
4. Compare blur vs non-blur effects on character recognition
5. Test interaction between sample density and cluster count

## Key Insights
- Sample count affects local texture density
- Cluster count affects global distribution of effects
- Standard deviation controls size of affected areas
- Value range directly impacts visibility/interference level
- Blur can reduce harsh edges but may create different OCR challenges

Generated: {total_configs} total variations
Location: letterpress_augmentations/
"""
    
    summary_path = os.path.join(output_dir, "letterpress_analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_content)

def main():
    """Main function to generate Letterpress augmentations"""
    print("🚀 Starting Letterpress Augmentation Generation")
    print("🎯 Parts 1-5: Samples, Clusters, Std, Values, and Blur Variations")
    print("="*80)
    
    # Setup
    setup_random_seed(42)
    
    # Paths
    image_path = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/DQA_data/sampled_images/06 Harishchandra Natak_001.png"
    base_output_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy_experiments"
    
    # Load image
    print("📂 Loading source image...")
    image = load_image(image_path)
    
    try:
        # Generate augmentations
        generate_letterpress_augmentations(image, base_output_dir)
        
        print("\n" + "="*80)
        print("✅ LETTERPRESS AUGMENTATION GENERATION COMPLETE!")
        print("="*80)
        print(f"📁 All outputs saved to: {base_output_dir}/letterpress_augmentations/")
        print("🔍 Ready for OCR impact analysis!")
        
        print("\n📋 Summary of generated variations:")
        print("   • 5 Sample count variations (density of effect)")
        print("   • 5 Cluster count variations (distribution)")
        print("   • 5 Standard deviation variations (blob size)")
        print("   • 6 Value range variations (darkness/intensity)")
        print("   • 2 Blur effect variations (edge smoothness)")
        print(f"   • Total: 23 systematic Letterpress variations")
        
    except Exception as e:
        print(f"❌ Error during augmentation generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
