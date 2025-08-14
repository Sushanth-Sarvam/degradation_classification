#!/usr/bin/env python3
"""
Augraphy Augmentation Generator for OCR Impact Analysis
Generates various augmentations to understand their impact on OCR accuracy.
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

def setup_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def load_image(image_path):
    """Load and validate image"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    print(f"Loaded image: {image.shape}, dtype: {image.dtype}")
    return image

def save_image(image, output_path, suffix=""):
    """Save image with proper naming"""
    base_name = "06_Harishchandra_Natak_001"
    if suffix:
        filename = f"{base_name}_{suffix}.png"
    else:
        filename = f"{base_name}.png"
    
    full_path = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)
    
    success = cv2.imwrite(full_path, image)
    if success:
        print(f"Saved: {full_path}")
        return full_path
    else:
        print(f"Failed to save: {full_path}")
        return None

def generate_ink_phase_augmentations(image, base_output_dir):
    """Generate Ink Phase only augmentations"""
    print("\n=== Generating Ink Phase Augmentations ===")
    
    # 1. Brightness variations
    print("Generating Brightness variations...")
    brightness_variations = [
        (0.7, 1.0, "dark"),
        (0.9, 1.1, "normal"), 
        (1.2, 1.5, "bright")
    ]
    
    for bright_min, bright_max, label in brightness_variations:
        pipeline = AugraphyPipeline(
            ink_phase=[Brightness(brightness_range=(bright_min, bright_max), p=1.0)],
            paper_phase=[],
            post_phase=[]
        )
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_phase/images/brightness", f"brightness_{label}")
    
    # 2. InkBleed variations
    print("Generating InkBleed variations...")
    inkbleed_variations = [
        (0.1, 0.2, (3, 3), "light"),
        (0.4, 0.6, (5, 5), "medium"),
        (0.7, 0.9, (7, 7), "heavy")
    ]
    
    for intensity_min, intensity_max, kernel, label in inkbleed_variations:
        pipeline = AugraphyPipeline(
            ink_phase=[InkBleed(
                intensity_range=(intensity_min, intensity_max),
                kernel_size=kernel,
                severity=(0.3, 0.5),
                p=1.0
            )],
            paper_phase=[],
            post_phase=[]
        )
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_phase/images/inkbleed", f"inkbleed_{label}")
    
    # 3. LinesDegradation
    print("Generating LinesDegradation variations...")
    lines_variations = [
        ((32, 128), (0.1, 0.2), "light"),
        ((64, 180), (0.3, 0.4), "medium"),
        ((128, 255), (0.5, 0.6), "heavy")
    ]
    
    for gradient_range, prob_range, label in lines_variations:
        pipeline = AugraphyPipeline(
            ink_phase=[LinesDegradation(
                line_gradient_range=gradient_range,
                line_split_probability=prob_range,
                line_replacement_probability=(0.4, 0.5),
                p=1.0
            )],
            paper_phase=[],
            post_phase=[]
        )
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_phase/images/linesdegradation", f"lines_{label}")
    
    # 4. LowInkRandomLines
    print("Generating LowInkRandomLines...")
    lowink_variations = [
        ((2, 5), "few_lines"),
        ((5, 10), "medium_lines"),
        ((10, 15), "many_lines")
    ]
    
    for count_range, label in lowink_variations:
        pipeline = AugraphyPipeline(
            ink_phase=[LowInkRandomLines(
                count_range=count_range,
                use_consistent_lines=False,
                noise_probability=0.1,
                p=1.0
            )],
            paper_phase=[],
            post_phase=[]
        )
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_phase/images/lowinklines", f"lowink_{label}")
    
    # 5. InkColorSwap
    print("Generating InkColorSwap...")
    pipeline = AugraphyPipeline(
        ink_phase=[InkColorSwap(
            ink_swap_color="random",
            ink_swap_sequence_number_range=(3, 8),
            p=1.0
        )],
        paper_phase=[],
        post_phase=[]
    )
    augmented = pipeline(image)
    save_image(augmented, f"{base_output_dir}/ink_phase/images/inkcolorswap", "colorswap")
    
    # 6. Letterpress
    print("Generating Letterpress...")
    letterpress_variations = [
        ((100, 200), (150, 200), "light"),
        ((200, 400), (100, 150), "medium"),
        ((400, 600), (50, 100), "heavy")
    ]
    
    for samples_range, value_range, label in letterpress_variations:
        pipeline = AugraphyPipeline(
            ink_phase=[Letterpress(
                n_samples=samples_range,
                n_clusters=(200, 400),
                value_range=value_range,
                p=1.0
            )],
            paper_phase=[],
            post_phase=[]
        )
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_phase/images/letterpress", f"letterpress_{label}")
    
    # 7. Hollow
    print("Generating Hollow...")
    pipeline = AugraphyPipeline(
        ink_phase=[Hollow(
            hollow_median_kernel_value_range=(71, 101),
            hollow_min_area_range=(10, 50),
            hollow_max_area_range=(1000, 3000),
            p=1.0
        )],
        paper_phase=[],
        post_phase=[]
    )
    augmented = pipeline(image)
    save_image(augmented, f"{base_output_dir}/ink_phase/images/hollow", "hollow")

def generate_paper_phase_augmentations(image, base_output_dir):
    """Generate Paper Phase only augmentations"""
    print("\n=== Generating Paper Phase Augmentations ===")
    
    # 1. ColorPaper variations
    print("Generating ColorPaper variations...")
    color_variations = [
        ((0, 50), (5, 15), "subtle"),
        ((0, 100), (15, 30), "medium"),
        ((0, 255), (30, 50), "strong")
    ]
    
    for hue_range, sat_range, label in color_variations:
        pipeline = AugraphyPipeline(
            ink_phase=[],
            paper_phase=[ColorPaper(
                hue_range=hue_range,
                saturation_range=sat_range,
                p=1.0
            )],
            post_phase=[]
        )
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/paper_phase/images/colorpaper", f"colorpaper_{label}")
    
    # 2. NoiseTexturize
    print("Generating NoiseTexturize variations...")
    noise_variations = [
        ((1, 3), (1, 2), "light"),
        ((3, 6), (2, 4), "medium"),
        ((6, 10), (4, 6), "heavy")
    ]
    
    for sigma_range, turb_range, label in noise_variations:
        pipeline = AugraphyPipeline(
            ink_phase=[],
            paper_phase=[NoiseTexturize(
                sigma_range=sigma_range,
                turbulence_range=turb_range,
                texture_width_range=(200, 400),
                texture_height_range=(200, 400),
                p=1.0
            )],
            post_phase=[]
        )
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/paper_phase/images/noisetexturize", f"noise_{label}")
    
    # 3. BrightnessTexturize
    print("Generating BrightnessTexturize...")
    brightness_tex_variations = [
        ((0.95, 0.99), 0.01, "subtle"),
        ((0.9, 0.98), 0.03, "medium"),
        ((0.8, 0.95), 0.05, "strong")
    ]
    
    for tex_range, deviation, label in brightness_tex_variations:
        pipeline = AugraphyPipeline(
            ink_phase=[],
            paper_phase=[BrightnessTexturize(
                texturize_range=tex_range,
                deviation=deviation,
                p=1.0
            )],
            post_phase=[]
        )
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/paper_phase/images/brightnesstexturize", f"brighttex_{label}")
    
    # 4. WaterMark
    print("Generating WaterMark...")
    pipeline = AugraphyPipeline(
        ink_phase=[],
        paper_phase=[WaterMark(
            watermark_word="SAMPLE",
            watermark_font_size=(15, 25),
            watermark_rotation=(0, 360),
            watermark_location="random",
            watermark_color="random",
            p=1.0
        )],
        post_phase=[]
    )
    augmented = pipeline(image)
    save_image(augmented, f"{base_output_dir}/paper_phase/images/watermark", "watermark")
    
    # 5. PatternGenerator
    print("Generating PatternGenerator...")
    pipeline = AugraphyPipeline(
        ink_phase=[],
        paper_phase=[PatternGenerator(
            imgx=random.randint(256, 512),
            imgy=random.randint(256, 512),
            n_rotation_range=(5, 15),
            alpha_range=(0.1, 0.3),
            p=1.0
        )],
        post_phase=[]
    )
    augmented = pipeline(image)
    save_image(augmented, f"{base_output_dir}/paper_phase/images/patterngen", "pattern")
    
    # 6. VoronoiTessellation
    print("Generating VoronoiTessellation...")
    pipeline = AugraphyPipeline(
        ink_phase=[],
        paper_phase=[VoronoiTessellation(
            mult_range=(30, 60),
            num_cells_range=(200, 500),
            background_value=(220, 255),
            p=1.0
        )],
        post_phase=[]
    )
    augmented = pipeline(image)
    save_image(augmented, f"{base_output_dir}/paper_phase/images/voronoi", "voronoi")

def generate_combined_augmentations(image, base_output_dir):
    """Generate combined Ink + Paper phase augmentations"""
    print("\n=== Generating Combined Ink + Paper Augmentations ===")
    
    # 1. Light degradation (office scan quality)
    print("Generating light degradation...")
    pipeline = AugraphyPipeline(
        ink_phase=[
            Brightness(brightness_range=(0.9, 1.1), p=1.0),
            InkBleed(intensity_range=(0.1, 0.2), kernel_size=(3, 3), p=0.5)
        ],
        paper_phase=[
            ColorPaper(hue_range=(0, 30), saturation_range=(5, 15), p=0.3),
            BrightnessTexturize(texturize_range=(0.95, 0.99), deviation=0.01, p=0.4)
        ],
        post_phase=[]
    )
    for i in range(3):
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_paper_combined/images/light_degradation", f"light_{i+1}")
    
    # 2. Medium degradation
    print("Generating medium degradation...")
    pipeline = AugraphyPipeline(
        ink_phase=[
            Brightness(brightness_range=(0.8, 1.2), p=1.0),
            InkBleed(intensity_range=(0.3, 0.5), kernel_size=(5, 5), p=0.7),
            LinesDegradation(line_gradient_range=(64, 180), p=0.3)
        ],
        paper_phase=[
            ColorPaper(hue_range=(0, 100), saturation_range=(10, 25), p=0.5),
            NoiseTexturize(sigma_range=(2, 5), turbulence_range=(1, 3), p=0.6)
        ],
        post_phase=[]
    )
    for i in range(3):
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_paper_combined/images/medium_degradation", f"medium_{i+1}")
    
    # 3. Heavy degradation
    print("Generating heavy degradation...")
    pipeline = AugraphyPipeline(
        ink_phase=[
            Brightness(brightness_range=(0.6, 1.4), p=1.0),
            InkBleed(intensity_range=(0.6, 0.8), kernel_size=(7, 7), p=0.8),
            LinesDegradation(line_gradient_range=(100, 255), p=0.6),
            LowInkRandomLines(count_range=(8, 15), p=0.4)
        ],
        paper_phase=[
            ColorPaper(hue_range=(0, 255), saturation_range=(20, 40), p=0.7),
            NoiseTexturize(sigma_range=(5, 10), turbulence_range=(3, 6), p=0.8),
            BrightnessTexturize(texturize_range=(0.8, 0.95), deviation=0.05, p=0.6)
        ],
        post_phase=[]
    )
    for i in range(3):
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_paper_combined/images/heavy_degradation", f"heavy_{i+1}")
    
    # 4. Realistic office scanner
    print("Generating realistic office scanner...")
    pipeline = AugraphyPipeline(
        ink_phase=[
            Brightness(brightness_range=(0.9, 1.1), p=1.0),
            InkBleed(intensity_range=(0.1, 0.3), kernel_size=(3, 5), p=0.4)
        ],
        paper_phase=[
            ColorPaper(hue_range=(0, 50), saturation_range=(5, 20), p=0.6),
            NoiseTexturize(sigma_range=(1, 3), turbulence_range=(1, 2), p=0.5),
            BrightnessTexturize(texturize_range=(0.92, 0.98), deviation=0.02, p=0.7)
        ],
        post_phase=[]
    )
    for i in range(3):
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_paper_combined/images/realistic_office", f"office_{i+1}")
    
    # 5. Scanned documents
    print("Generating scanned documents...")
    pipeline = AugraphyPipeline(
        ink_phase=[
            Brightness(brightness_range=(0.85, 1.15), p=1.0),
            InkBleed(intensity_range=(0.2, 0.4), kernel_size=(3, 5), p=0.5),
            Letterpress(n_samples=(100, 300), value_range=(180, 220), p=0.3)
        ],
        paper_phase=[
            ColorPaper(hue_range=(0, 80), saturation_range=(8, 25), p=0.4),
            NoiseTexturize(sigma_range=(2, 4), turbulence_range=(1, 3), p=0.6)
        ],
        post_phase=[]
    )
    for i in range(3):
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_paper_combined/images/scanned_documents", f"scanned_{i+1}")
    
    # 6. Photocopied documents
    print("Generating photocopied documents...")
    pipeline = AugraphyPipeline(
        ink_phase=[
            Brightness(brightness_range=(0.7, 1.3), p=1.0),
            InkBleed(intensity_range=(0.4, 0.7), kernel_size=(5, 7), p=0.7),
            LinesDegradation(line_gradient_range=(80, 200), p=0.5),
            Hollow(hollow_median_kernel_value_range=(71, 101), p=0.3)
        ],
        paper_phase=[
            ColorPaper(hue_range=(0, 120), saturation_range=(10, 30), p=0.6),
            NoiseTexturize(sigma_range=(3, 6), turbulence_range=(2, 4), p=0.7),
            BrightnessTexturize(texturize_range=(0.85, 0.95), deviation=0.04, p=0.5)
        ],
        post_phase=[]
    )
    for i in range(3):
        augmented = pipeline(image)
        save_image(augmented, f"{base_output_dir}/ink_paper_combined/images/photocopied", f"photocopy_{i+1}")

def main():
    """Main function to generate all augmentations"""
    setup_random_seed(42)
    
    # Paths
    image_path = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/DQA_data/sampled_images/06 Harishchandra Natak_001.png"
    base_output_dir = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy_experiments"
    
    # Load image
    print("Loading image...")
    image = load_image(image_path)
    
    # Save original image
    save_image(image, f"{base_output_dir}/original/images", "original")
    
    # Generate different types of augmentations
    try:
        generate_ink_phase_augmentations(image, base_output_dir)
        generate_paper_phase_augmentations(image, base_output_dir)
        generate_combined_augmentations(image, base_output_dir)
        
        print("\n=== Augmentation Generation Complete ===")
        print(f"All augmented images saved to: {base_output_dir}")
        
    except Exception as e:
        print(f"Error during augmentation: {e}")
        raise

if __name__ == "__main__":
    main()
