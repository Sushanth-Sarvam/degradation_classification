#!/usr/bin/env python3
"""
Batch Augmentation Generation Script

This script reads a YAML configuration file and generates augmented images
by applying specified Augraphy augmentations to input images.

Usage:
    python generate_augmentations.py [config_file]
    
Default config: batch_augmentation_config.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import glob
from datetime import datetime

# Add augraphy to path
sys.path.append('/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy')

from augraphy import *
from augraphy.base.augmentationpipeline import AugraphyPipeline
from augraphy.augmentations.brightness import Brightness
from augraphy.augmentations.inkbleed import InkBleed
from augraphy.augmentations.letterpress import Letterpress
from augraphy.augmentations.hollow import Hollow
from augraphy.augmentations.lowinkperiodiclines import LowInkPeriodicLines
from augraphy.augmentations.lowinkrandomlines import LowInkRandomLines
from augraphy.augmentations.colorpaper import ColorPaper
from augraphy.augmentations.noisetexturize import NoiseTexturize
from augraphy.augmentations.subtlenoise import SubtleNoise
from augraphy.augmentations.bookbinding import BookBinding
from augraphy.augmentations.folding import Folding
from augraphy.augmentations.pageborder import PageBorder


class AugmentationGenerator:
    """Main class for generating batch augmentations"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file"""
        self.config = self.load_config(config_path)
        self.validate_config()
        
        # Create output directory
        os.makedirs(self.config['output_directory'], exist_ok=True)
        
        # Statistics
        self.stats = {
            'processed_images': 0,
            'generated_augmentations': 0,
            'skipped_existing': 0,
            'errors': 0
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded configuration from: {config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config file {config_path}: {e}")
            sys.exit(1)
    
    def validate_config(self):
        """Validate configuration structure"""
        required_keys = ['input_directory', 'output_directory', 'augmentation_sets']
        for key in required_keys:
            if key not in self.config:
                print(f"❌ Missing required config key: {key}")
                sys.exit(1)
        
        if not os.path.exists(self.config['input_directory']):
            print(f"❌ Input directory does not exist: {self.config['input_directory']}")
            sys.exit(1)
        
        print(f"✅ Configuration validated")
        print(f"📁 Input directory: {self.config['input_directory']}")
        print(f"📁 Output directory: {self.config['output_directory']}")
        print(f"🎨 Augmentation sets: {len(self.config['augmentation_sets'])}")
    
    def get_input_images(self) -> List[str]:
        """Get list of input images"""
        extensions = self.config.get('supported_extensions', ['.png', '.jpg', '.jpeg'])
        image_files = []
        
        for ext in extensions:
            pattern = os.path.join(self.config['input_directory'], f"*{ext}")
            image_files.extend(glob.glob(pattern))
            # Also check uppercase extensions
            pattern = os.path.join(self.config['input_directory'], f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        image_files.sort()
        print(f"📸 Found {len(image_files)} input images")
        return image_files
    
    def create_augmentation_instance(self, aug_type: str, parameters: Dict[str, Any]):
        """Create an augmentation instance from type and parameters"""
        try:
            if aug_type == 'brightness':
                return Brightness(**parameters)
            elif aug_type == 'inkbleed':
                return InkBleed(**parameters)
            elif aug_type == 'letterpress':
                return Letterpress(**parameters)
            elif aug_type == 'hollow':
                return Hollow(**parameters)
            elif aug_type == 'lowinkperiodiclines':
                return LowInkPeriodicLines(**parameters)
            elif aug_type == 'lowinkrandomlines':
                return LowInkRandomLines(**parameters)
            elif aug_type == 'colorpaper':
                return ColorPaper(**parameters)
            elif aug_type == 'noisetexturize':
                return NoiseTexturize(**parameters)
            elif aug_type == 'subtlenoise':
                return SubtleNoise(**parameters)
            elif aug_type == 'bookbinding':
                return BookBinding(**parameters)
            elif aug_type == 'folding':
                return Folding(**parameters)
            elif aug_type == 'pageborder':
                return PageBorder(**parameters)
            else:
                print(f"⚠️ Unknown augmentation type: {aug_type}")
                return None
        except Exception as e:
            print(f"❌ Error creating {aug_type} augmentation: {e}")
            return None
    
    def format_parameters_for_filename(self, aug_type: str, parameters: Dict[str, Any]) -> str:
        """Format key parameters for inclusion in filename"""
        if not self.config.get('filename_format', {}).get('include_parameters', False):
            return ""
        
        # Extract key parameters based on augmentation type
        param_str = ""
        
        if aug_type == 'brightness':
            brightness = parameters.get('brightness_range', [0, 0])[0]
            param_str = f"br{brightness:.1f}"
        
        elif aug_type == 'inkbleed':
            intensity = parameters.get('intensity_range', [0, 0])[0]
            severity = parameters.get('severity', [0, 0])[0]
            kernel = parameters.get('kernel_size', [0, 0])[0]
            param_str = f"int{intensity:.1f}_sev{severity:.1f}_k{kernel}"
        
        elif aug_type == 'letterpress':
            n_samples = parameters.get('n_samples', [0, 0])[0]
            n_clusters = parameters.get('n_clusters', [0, 0])[0]
            value = parameters.get('value_range', [0, 0])[0]
            param_str = f"samp{n_samples}_clust{n_clusters}_val{value}"
        
        elif aug_type == 'colorpaper':
            hue = parameters.get('hue_range', [0, 0])[0]
            saturation = parameters.get('saturation_range', [0, 0])[0]
            param_str = f"hue{hue}_sat{saturation}"
        
        elif aug_type == 'noisetexturize':
            sigma = parameters.get('sigma_range', [0, 0])[0]
            turbulence = parameters.get('turbulence_range', [0, 0])[0]
            param_str = f"sig{sigma}_turb{turbulence}"
        
        # Add prefix if we have parameters
        if param_str:
            param_str = f"_{param_str}"
        
        return param_str
    
    def generate_output_filename(self, input_path: str, aug_set_name: str, 
                                augmentations: List[Dict[str, Any]]) -> str:
        """Generate output filename with augmentation info"""
        input_file = Path(input_path)
        base_name = input_file.stem
        extension = input_file.suffix
        
        filename_parts = []
        
        # Original name
        if self.config.get('filename_format', {}).get('include_original_name', True):
            filename_parts.append(base_name)
        
        # Augmentation set name
        if self.config.get('filename_format', {}).get('include_augmentation_set', True):
            filename_parts.append(aug_set_name)
        
        # Parameter details for each augmentation
        if self.config.get('filename_format', {}).get('include_parameters', True):
            for aug in augmentations:
                param_str = self.format_parameters_for_filename(aug['type'], aug['parameters'])
                if param_str:
                    filename_parts.append(param_str.lstrip('_'))  # Remove leading underscore
        
        # Join parts
        separator = self.config.get('filename_format', {}).get('separator', '_')
        filename = separator.join(filename_parts)
        
        # Add extension
        if self.config.get('processing', {}).get('preserve_format', True):
            filename += extension
        else:
            filename += '.png'  # Default to PNG
        
        return os.path.join(self.config['output_directory'], filename)
    
    def apply_augmentations(self, image: np.ndarray, augmentation_set: Dict[str, Any]) -> Optional[np.ndarray]:
        """Apply augmentation set to image"""
        try:
            phase = augmentation_set.get('phase', 'ink')
            augmentations = augmentation_set['augmentations']
            
            # Create augmentation instances
            ink_augs = []
            paper_augs = []
            
            for aug_config in augmentations:
                aug_instance = self.create_augmentation_instance(
                    aug_config['type'], 
                    aug_config['parameters']
                )
                
                if aug_instance is None:
                    continue
                
                # Determine phase for this augmentation
                aug_type = aug_config['type']
                if aug_type in ['brightness', 'inkbleed', 'letterpress', 'hollow', 
                               'lowinkperiodiclines', 'lowinkrandomlines']:
                    ink_augs.append(aug_instance)
                elif aug_type in ['colorpaper', 'noisetexturize', 'subtlenoise', 
                                 'bookbinding', 'folding', 'pageborder']:
                    paper_augs.append(aug_instance)
            
            # Create and apply pipeline
            if ink_augs or paper_augs:
                pipeline = AugraphyPipeline(
                    ink_phase=ink_augs,
                    paper_phase=paper_augs
                )
                
                # Apply augmentations
                augmented_image = pipeline.augment(image)
                return augmented_image['output']
            else:
                print("⚠️ No valid augmentations created")
                return None
                
        except Exception as e:
            print(f"❌ Error applying augmentations: {e}")
            return None
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image in BGR format for Augraphy"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            return image
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """Save augmented image"""
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            success = cv2.imwrite(output_path, image)
            if success:
                return True
            else:
                print(f"❌ Failed to save image: {output_path}")
                return False
        except Exception as e:
            print(f"❌ Error saving image {output_path}: {e}")
            return False
    
    def process_images(self):
        """Main processing loop"""
        print(f"\n🚀 Starting batch augmentation generation...")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        input_images = self.get_input_images()
        if not input_images:
            print("❌ No input images found!")
            return
        
        total_combinations = len(input_images) * len(self.config['augmentation_sets'])
        print(f"🎯 Total combinations to process: {total_combinations}")
        
        # Process each augmentation set
        for set_idx, aug_set in enumerate(self.config['augmentation_sets']):
            set_name = aug_set['name']
            print(f"\n📦 Processing augmentation set [{set_idx+1}/{len(self.config['augmentation_sets'])}]: {set_name}")
            
            # Check max images limit
            max_images = self.config.get('processing', {}).get('max_images_per_set')
            images_to_process = input_images[:max_images] if max_images else input_images
            
            for img_idx, image_path in enumerate(images_to_process):
                print(f"  🖼️  Processing image [{img_idx+1}/{len(images_to_process)}]: {os.path.basename(image_path)}")
                
                # Generate output filename
                output_path = self.generate_output_filename(image_path, set_name, aug_set['augmentations'])
                
                # Check if output already exists
                if (self.config.get('processing', {}).get('skip_existing', True) and 
                    os.path.exists(output_path)):
                    print(f"    ⏭️  Skipping (already exists): {os.path.basename(output_path)}")
                    self.stats['skipped_existing'] += 1
                    continue
                
                # Load image
                image = self.load_image(image_path)
                if image is None:
                    self.stats['errors'] += 1
                    continue
                
                # Apply augmentations
                augmented_image = self.apply_augmentations(image, aug_set)
                if augmented_image is None:
                    print(f"    ❌ Failed to apply augmentations")
                    self.stats['errors'] += 1
                    continue
                
                # Save result
                if self.save_image(augmented_image, output_path):
                    print(f"    ✅ Saved: {os.path.basename(output_path)}")
                    self.stats['generated_augmentations'] += 1
                else:
                    self.stats['errors'] += 1
                
                self.stats['processed_images'] += 1
        
        # Print final statistics
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        print(f"\n📊 Processing Summary")
        print(f"=" * 50)
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📸 Images processed: {self.stats['processed_images']}")
        print(f"🎨 Augmentations generated: {self.stats['generated_augmentations']}")
        print(f"⏭️ Existing files skipped: {self.stats['skipped_existing']}")
        print(f"❌ Errors encountered: {self.stats['errors']}")
        print(f"📁 Output directory: {self.config['output_directory']}")
        
        if self.stats['generated_augmentations'] > 0:
            print(f"\n✅ Batch augmentation generation completed successfully!")
        else:
            print(f"\n⚠️ No augmentations were generated. Check configuration and input images.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate batch augmentations from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_augmentations.py
    python generate_augmentations.py custom_config.yaml
    python generate_augmentations.py --config batch_augmentation_config.yaml
        """
    )
    
    parser.add_argument(
        'config_file', 
        nargs='?', 
        default='batch_augmentation_config.yaml',
        help='YAML configuration file (default: batch_augmentation_config.yaml)'
    )
    
    parser.add_argument(
        '--config', 
        dest='config_file_alt',
        help='Alternative way to specify config file'
    )
    
    args = parser.parse_args()
    
    # Use --config if provided, otherwise use positional argument
    config_file = args.config_file_alt or args.config_file
    
    print("🎨 Augraphy Batch Augmentation Generator")
    print("=" * 50)
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        print(f"💡 Create a configuration file or specify a different path")
        sys.exit(1)
    
    # Create and run generator
    try:
        generator = AugmentationGenerator(config_file)
        generator.process_images()
    except KeyboardInterrupt:
        print(f"\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
