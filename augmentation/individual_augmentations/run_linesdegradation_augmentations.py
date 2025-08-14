#!/usr/bin/env python3
"""
LinesDegradation Augmentation Generation Script
===============================================

This script generates systematic LinesDegradation augmentations with exact parameter values
for testing OCR robustness against line discontinuity and stroke degradation.

LinesDegradation detects lines via image gradients and replaces them with different values,
breaking line continuity in text which can severely impact character recognition.

Key Parameters:
- line_gradient_range: Gradient detection threshold (lower = detects subtle lines)
- line_gradient_direction: 0=horizontal, 1=vertical, 2=both
- line_split_probability: Probability to split long lines into shorter segments
- line_replacement_value: New pixel value for detected lines (255=white/invisible)
- line_min_length: Minimum line length to process
- line_long_to_short_ratio: Aspect ratio threshold for line detection
- line_replacement_probability: Probability to replace detected lines
- line_replacement_thickness: Thickness of replacement lines

OCR Impact: Directly breaks character stroke continuity, making letters incomplete
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

def generate_linesdegradation_augmentations(image, base_output_dir):
    """
    Generate systematic LinesDegradation augmentations with exact parameter values
    Parts 1-6: Gradient Range, Direction, Split Probability, Replacement Value, Min Length, Replacement Probability
    """
    print("\n" + "="*80)
    print("LINESDEGRADATION AUGMENTATION GENERATION")
    print("="*80)
    
    # Delete old directory if it exists and create new one
    lines_dir = os.path.join(base_output_dir, "linesdegradation_augmentations")
    if os.path.exists(lines_dir):
        import shutil
        print(f"🗑️  Deleting old directory: {lines_dir}")
        shutil.rmtree(lines_dir)
    os.makedirs(lines_dir, exist_ok=True)
    print(f"📁 Created fresh directory: {lines_dir}")
    
    # Save original image
    save_original_image(image, lines_dir)
    
    # All configurations to test
    all_configs = []
    
    # PART 1: Gradient Range Variations (5 combinations)
    print("\n📊 PART 1: Gradient Range Variations")
    gradient_configs = [
        ((0.0, 0.0, 1.0, 1.0), (16, 64), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "gradient_16to64_sensitive"),
        ((0.0, 0.0, 1.0, 1.0), (32, 96), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "gradient_32to96_moderate"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "gradient_64to128_normal"),
        ((0.0, 0.0, 1.0, 1.0), (96, 180), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "gradient_96to180_strong"),
        ((0.0, 0.0, 1.0, 1.0), (128, 255), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "gradient_128to255_very_strong"),
    ]
    all_configs.extend(gradient_configs)
    
    # PART 2: Gradient Direction Variations (3 combinations)
    print("📊 PART 2: Gradient Direction Variations")
    direction_configs = [
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 0), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "direction_horizontal_only"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (1, 1), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "direction_vertical_only"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (2, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "direction_both"),
    ]
    all_configs.extend(direction_configs)
    
    # PART 3: Split Probability Variations (4 combinations)
    print("📊 PART 3: Split Probability Variations")
    split_configs = [
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.1, 0.2), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "split_0.1to0.2_low"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.2, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "split_0.2to0.4_moderate"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.4, 0.6), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "split_0.4to0.6_high"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.6, 0.8), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "split_0.6to0.8_very_high"),
    ]
    all_configs.extend(split_configs)
    
    # PART 4: Replacement Value Variations (4 combinations)
    print("📊 PART 4: Replacement Value Variations")
    replacement_configs = [
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (200, 220), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "replace_200to220_visible"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (230, 245), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "replace_230to245_light"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "replace_250to255_invisible"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (100, 150), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "replace_100to150_dark"),
    ]
    all_configs.extend(replacement_configs)
    
    # PART 5: Minimum Length Variations (4 combinations)
    print("📊 PART 5: Minimum Length Variations")
    length_configs = [
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (10, 20), (5, 7), (0.4, 0.5), (1, 2), "length_10to20_short"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (20, 30), (5, 7), (0.4, 0.5), (1, 2), "length_20to30_medium"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (30, 50), (5, 7), (0.4, 0.5), (1, 2), "length_30to50_long"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (50, 80), (5, 7), (0.4, 0.5), (1, 2), "length_50to80_very_long"),
    ]
    all_configs.extend(length_configs)
    
    # PART 6: Replacement Probability Variations (4 combinations)
    print("📊 PART 6: Replacement Probability Variations")
    prob_configs = [
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.2, 0.3), (1, 2), "prob_0.2to0.3_conservative"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.4, 0.5), (1, 2), "prob_0.4to0.5_moderate"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.6, 0.7), (1, 2), "prob_0.6to0.7_aggressive"),
        ((0.0, 0.0, 1.0, 1.0), (64, 128), (0, 2), (0.3, 0.4), (250, 255), (25, 35), (5, 7), (0.8, 0.9), (1, 2), "prob_0.8to0.9_very_aggressive"),
    ]
    all_configs.extend(prob_configs)
    
    total_configs = len(all_configs)
    print(f"\n🎯 LINESDEGRADATION configurations to generate: {total_configs}")
    print("  - Part 1 (Gradient Range): 5 variations")
    print("  - Part 2 (Direction): 3 variations")
    print("  - Part 3 (Split Probability): 4 variations")
    print("  - Part 4 (Replacement Value): 4 variations")
    print("  - Part 5 (Min Length): 4 variations")
    print("  - Part 6 (Replacement Probability): 4 variations")
    print("  📊 These parameters will create varying levels of line degradation")
    
    # Generate all configurations
    for i, (roi, gradient_range, direction, split_prob, replacement_value, min_length, ratio, repl_prob, thickness, description) in enumerate(all_configs):
        print(f"\n[{i+1}/{total_configs}] Processing: {description}")
        print(f"  Parameters: gradient={gradient_range}, direction={direction}, split={split_prob}, replace={replacement_value}")
        
        # Create Augraphy pipeline
        pipeline = AugraphyPipeline(
            ink_phase=[LinesDegradation(
                line_roi=roi,
                line_gradient_range=gradient_range,
                line_gradient_direction=direction,
                line_split_probability=split_prob,
                line_replacement_value=replacement_value,
                line_min_length=min_length,
                line_long_to_short_ratio=ratio,
                line_replacement_probability=repl_prob,
                line_replacement_thickness=thickness,
                p=1.0
            )],
            paper_phase=[],
            post_phase=[]
        )
        
        # Apply augmentation
        augmented = pipeline(image)
        
        # Create filename with exact parameters
        filename = create_param_filename("linesdegradation",
                                       gradient_range=gradient_range,
                                       direction=direction,
                                       split_prob=split_prob,
                                       replacement_value=replacement_value,
                                       min_length=min_length,
                                       repl_prob=repl_prob)
        
        # Save augmented image
        save_augmented_image(augmented, lines_dir, filename)
    
    # Create analysis summary
    create_linesdegradation_summary(lines_dir, total_configs)
    
    print(f"\n✅ LinesDegradation augmentation complete!")
    print(f"📁 Output directory: {lines_dir}")
    print(f"📊 Generated {total_configs} variations")
    print(f"📋 Analysis summary saved: {lines_dir}/linesdegradation_analysis_summary.txt")

def create_linesdegradation_summary(output_dir, total_configs):
    """Create a summary of generated LinesDegradation augmentations"""
    summary_content = f"""# LinesDegradation Augmentation Analysis Summary

## Overview
Generated {total_configs} systematic LinesDegradation variations to test OCR robustness against 
line discontinuity and stroke degradation effects.

## Parameter Categories

### 1. Gradient Range Variations (5 configs)
- Tests sensitivity to different gradient detection thresholds
- Range: 16-255 gradient values
- OCR Impact: Lower values detect subtle lines, higher values only strong edges

### 2. Gradient Direction Variations (3 configs)
- Tests directional sensitivity of line detection
- Options: Horizontal only, Vertical only, Both directions
- OCR Impact: Directional bias can affect specific character features

### 3. Split Probability Variations (4 configs)
- Tests effect of breaking long lines into segments
- Range: 0.1-0.8 probability
- OCR Impact: Higher values create more line discontinuities

### 4. Replacement Value Variations (4 configs)
- Tests visibility of replaced line segments
- Range: 100-255 pixel values
- OCR Impact: Lower values create visible artifacts, higher values erase lines

### 5. Minimum Length Variations (4 configs)
- Tests threshold for line detection based on length
- Range: 10-80 pixel minimum length
- OCR Impact: Shorter thresholds affect more character strokes

### 6. Replacement Probability Variations (4 configs)
- Tests aggressiveness of line replacement
- Range: 0.2-0.9 probability
- OCR Impact: Higher probabilities create more extensive degradation

## OCR Testing Strategy
1. Test each parameter category separately to understand individual effects
2. Focus on gradient range and replacement value for maximum impact
3. Compare directional effects (horizontal vs vertical vs both)
4. Analyze interaction between split probability and replacement probability
5. Test on different character sizes and stroke widths

## Key Insights
- Gradient range controls which lines are detected and degraded
- Direction parameter can create systematic bias affecting specific character features
- Split probability increases fragmentation of character strokes
- Replacement value determines visibility vs invisibility of degraded areas
- Minimum length threshold affects fine details vs major character features
- Replacement probability controls overall severity of degradation

## Expected OCR Challenges
- Character stroke discontinuity making letters appear broken
- Confusion between similar characters with different stroke patterns
- Word segmentation issues when character connections are broken
- Difficulty in character boundary detection
- Reduced accuracy on thin fonts and small text sizes

Generated: {total_configs} total variations
Location: linesdegradation_augmentations/
"""
    
    summary_path = os.path.join(output_dir, "linesdegradation_analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_content)

def main():
    """Main function to generate LinesDegradation augmentations"""
    print("🚀 Starting LinesDegradation Augmentation Generation")
    print("🎯 Parts 1-6: Gradient, Direction, Split, Replacement, Length, and Probability Variations")
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
        generate_linesdegradation_augmentations(image, base_output_dir)
        
        print("\n" + "="*80)
        print("✅ LINESDEGRADATION AUGMENTATION GENERATION COMPLETE!")
        print("="*80)
        print(f"📁 All outputs saved to: {base_output_dir}/linesdegradation_augmentations/")
        print("🔍 Ready for OCR impact analysis!")
        
        print("\n📋 Summary of generated variations:")
        print("   • 5 Gradient range variations (detection sensitivity)")
        print("   • 3 Direction variations (horizontal/vertical/both)")
        print("   • 4 Split probability variations (line fragmentation)")
        print("   • 4 Replacement value variations (visibility/invisibility)")
        print("   • 4 Minimum length variations (detail threshold)")
        print("   • 4 Replacement probability variations (severity)")
        print(f"   • Total: 24 systematic LinesDegradation variations")
        
    except Exception as e:
        print(f"❌ Error during augmentation generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
