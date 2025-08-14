#!/usr/bin/env python3
"""
Augraphy Parameters UI
======================

Interactive web interface for controlling Brightness, InkBleed, and Letterpress
augmentation parameters in real-time with file browsing capabilities.

Run with: streamlit run augmentation_ui.py
"""

import streamlit as st
import cv2
import numpy as np
import sys
import os
from PIL import Image
import io
import glob
from pathlib import Path
import yaml
from typing import Dict, Any, List, Tuple, Union

# Add the augraphy path
sys.path.append('/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augraphy')

from augraphy import *
from augraphy.base import AugraphyPipeline
from augraphy.augmentations import LowInkPeriodicLines, LowInkRandomLines, LowLightNoise, InkColorSwap, ColorPaper, NoiseTexturize, SubtleNoise, BookBinding, Folding, PageBorder

def load_data_paths_config() -> Dict[str, Any]:
    """Load configuration from data_paths.yaml"""
    config_path = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augmentation/data_paths.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading data paths configuration: {e}")
        return None

def load_augmentation_config() -> Dict[str, Any]:
    """Load configuration from augmentation_cfg.yaml"""
    config_path = "/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augmentation/augmentation_cfg.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading augmentation configuration: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="Augraphy Parameter Controller",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def convert_ui_value_to_augraphy_param(param_name: str, ui_value: Union[float, int, bool], aug_config: Dict[str, Any]) -> Union[Tuple, int, float, bool]:
    """Convert UI slider value to appropriate Augraphy parameter format"""
    param_info = aug_config.get('parameter_info', {})
    
    # Check if it's a tuple parameter
    if param_name in param_info.get('ranges', {}):
        # Special case for kernel_size - needs to be (size, size)
        if param_name == 'kernel_size':
            return (int(ui_value), int(ui_value))
        # Special case for Letterpress parameters that use np.random.randint
        elif param_name in ['n_samples', 'n_clusters', 'value_range']:
            # For np.random.randint, high must be > low, so we add 1 to high when values are equal
            return (int(ui_value), int(ui_value) + 1)
        # Special case for Hollow parameters that use random.randint (need integers and high > low)
        elif param_name.startswith('hollow_'):
            # All hollow parameters use random.randint which requires high > low
            return (int(ui_value), int(ui_value) + 1)
        # Special case for ColorPaper parameters that already add +1 internally
        elif param_name in ['hue_range', 'saturation_range']:
            # ColorPaper uses np.random.randint(low, high + 1) internally
            # So we need to provide a range where high > low
            int_value = int(ui_value)
            # Create a small range around the value to avoid low >= high
            result = (max(0, int_value - 2), int_value + 2)
            print(f"🔍 ColorPaper {param_name}: UI value {ui_value} → {result}")
            return result
        # Special case for NoiseTexturize parameters that use random.randint (need integers and high > low)
        elif param_name in ['sigma_range', 'turbulence_range', 'texture_width_range', 'texture_height_range']:
            # NoiseTexturize uses random.randint which requires integers and high > low
            int_value = int(ui_value)
            result = (int_value, int_value + 1)
            print(f"🔍 NoiseTexturize {param_name}: UI value {ui_value} → {result}")
            return result
        # Regular range parameters - convert to (value, value) for fixed values
        elif param_info['ranges'][param_name] == 'tuple':
            return (ui_value, ui_value)
    
    # Single value parameters
    elif param_name in param_info.get('single_values', {}):
        param_type = param_info['single_values'][param_name]
        if param_type == 'int':
            return int(ui_value)
        elif param_type == 'float':
            return float(ui_value)
        elif param_type == 'bool':
            return bool(ui_value)
    
    # Default fallback
    return ui_value

def create_parameter_slider(aug_name: str, param_name: str, param_config: Dict[str, Any], session_key: str) -> Union[float, int, bool]:
    """Create a Streamlit slider for a parameter based on its configuration"""
    valid_range = param_config.get('valid_range', [0, 1])
    recommended_range = param_config.get('recommended_range', valid_range)
    default_value = param_config.get('value', valid_range[0] if isinstance(valid_range, list) else valid_range)
    
    # Create help text with range information
    help_text = f"Recommended: {recommended_range}, Augraphy default: {param_config.get('augraphy_range', 'N/A')}"
    
    # Handle different parameter types
    if isinstance(valid_range, list) and len(valid_range) == 2:
        # Numeric range
        if isinstance(valid_range[0], float) or isinstance(default_value, float):
            return st.slider(
                param_name.replace('_', ' ').title(),
                min_value=float(valid_range[0]),
                max_value=float(valid_range[1]),
                value=float(default_value),
                step=0.01,
                help=help_text,
                key=session_key
            )
        else:
            return st.slider(
                param_name.replace('_', ' ').title(),
                min_value=int(valid_range[0]),
                max_value=int(valid_range[1]),
                value=int(default_value),
                step=1,
                help=help_text,
                key=session_key
            )
    else:
        # Boolean or single value
        if isinstance(default_value, bool):
            return st.checkbox(
                param_name.replace('_', ' ').title(),
                value=default_value,
                help=help_text,
                key=session_key
            )
        else:
            # Single numeric value - create a simple input
            return st.number_input(
                param_name.replace('_', ' ').title(),
                value=default_value,
                help=help_text,
                key=session_key
            )

def get_image_files(directory):
    """Get all image files in a directory"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return sorted(image_files)

def get_pdf_half_directories(config):
    """Get PDF left/right half image directories from config"""
    if not config:
        return {}
        
    directories = {}
    
    for source_key in ['pdf_left_half', 'pdf_right_half']:
        if source_key in config['image_sources']:
            source_info = config['image_sources'][source_key]
            base_path = source_info['base_path']
            
            directories[source_info['name']] = {
                'base_path': base_path,
                'books': []
            }
            
            # Get subdirectories (books)
            if os.path.exists(base_path):
                subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                directories[source_info['name']]['books'] = sorted(subdirs)
    
    return directories

def get_images_from_source(config, source_key):
    """Get list of images from a specific source defined in config"""
    if not config or source_key not in config['image_sources']:
        return []
        
    source_info = config['image_sources'][source_key]
    base_path = source_info['base_path']
    images = []
    
    if os.path.exists(base_path):
        extensions = config.get('supported_extensions', ['.png', '.jpg', '.jpeg'])
        for file in os.listdir(base_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                images.append(file)
    
    return sorted(images)

# Removed get_data_directories function - no longer needed since we use PDF halves only

def load_image(image_path):
    """Load image from given path"""
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, image_rgb
    
    # Fallback: create a sample image with text
    image = np.full((600, 800, 3), 240, dtype=np.uint8)
    cv2.putText(image, "Sample Gujarati Text", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "नमस्ते दुनिया", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "Image not found", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_rgb

def apply_augmentations(image, brightness_params, inkbleed_params, letterpress_params, hollow_params, lowink_periodic_params, lowink_random_params, lowlight_noise_params, inkcolorswap_params, colorpaper_params, noisetexturize_params, subtlenoise_params, bookbinding_params, folding_params, pageborder_params, enabled_augs):
    """Apply selected augmentations with given parameters"""
    # Create list of augmentations to apply
    ink_phase_augs = []
    paper_phase_augs = []
    
    if enabled_augs.get('brightness', False):
        ink_phase_augs.append(Brightness(
            brightness_range=brightness_params['brightness_range'],
            min_brightness=brightness_params['min_brightness'],
            min_brightness_value=brightness_params['min_brightness_value'],
            p=1.0
        ))
    
    if enabled_augs.get('inkbleed', False):
        ink_phase_augs.append(InkBleed(
            intensity_range=inkbleed_params['intensity_range'],
            kernel_size=inkbleed_params['kernel_size'],
            severity=inkbleed_params['severity'],
            p=1.0
        ))
    
    if enabled_augs.get('letterpress', False):
        ink_phase_augs.append(Letterpress(
            n_samples=letterpress_params['n_samples'],
            n_clusters=letterpress_params['n_clusters'],
            std_range=letterpress_params['std_range'],
            value_range=letterpress_params['value_range'],
            value_threshold_range=letterpress_params['value_threshold_range'],
            blur=letterpress_params['blur'],
            p=1.0
        ))
    
    if enabled_augs.get('hollow', False):
        ink_phase_augs.append(Hollow(
            hollow_median_kernel_value_range=hollow_params['hollow_median_kernel_value_range'],
            hollow_min_width_range=hollow_params['hollow_min_width_range'],
            hollow_max_width_range=hollow_params['hollow_max_width_range'],
            hollow_min_height_range=hollow_params['hollow_min_height_range'],
            hollow_max_height_range=hollow_params['hollow_max_height_range'],
            hollow_min_area_range=hollow_params['hollow_min_area_range'],
            hollow_max_area_range=hollow_params['hollow_max_area_range'],
            hollow_dilation_kernel_size_range=hollow_params['hollow_dilation_kernel_size_range'],
            p=1.0
        ))
    
    if enabled_augs.get('lowink_periodic', False):
        ink_phase_augs.append(LowInkPeriodicLines(
            count_range=lowink_periodic_params['count_range'],
            period_range=lowink_periodic_params['period_range'],
            use_consistent_lines=lowink_periodic_params['use_consistent_lines'],
            noise_probability=lowink_periodic_params['noise_probability'],
            p=1.0
        ))
    
    if enabled_augs.get('lowink_random', False):
        ink_phase_augs.append(LowInkRandomLines(
            count_range=lowink_random_params['count_range'],
            use_consistent_lines=lowink_random_params['use_consistent_lines'],
            noise_probability=lowink_random_params['noise_probability'],
            p=1.0
        ))
    
    if enabled_augs.get('lowlight_noise', False):
        ink_phase_augs.append(LowLightNoise(
            num_photons_range=lowlight_noise_params['num_photons_range'],
            alpha_range=lowlight_noise_params['alpha_range'],
            beta_range=lowlight_noise_params['beta_range'],
            gamma_range=lowlight_noise_params['gamma_range'],
            bias_range=lowlight_noise_params['bias_range'],
            dark_current_value=lowlight_noise_params['dark_current_value'],
            exposure_time=lowlight_noise_params['exposure_time'],
            gain=lowlight_noise_params['gain'],
            p=1.0
        ))
    
    if enabled_augs.get('inkcolorswap', False):
        ink_phase_augs.append(InkColorSwap(
            ink_swap_color=inkcolorswap_params['ink_swap_color'],
            ink_swap_sequence_number_range=inkcolorswap_params['ink_swap_sequence_number_range'],
            ink_swap_min_width_range=inkcolorswap_params['ink_swap_min_width_range'],
            ink_swap_max_width_range=inkcolorswap_params['ink_swap_max_width_range'],
            ink_swap_min_height_range=inkcolorswap_params['ink_swap_min_height_range'],
            ink_swap_max_height_range=inkcolorswap_params['ink_swap_max_height_range'],
            ink_swap_min_area_range=inkcolorswap_params['ink_swap_min_area_range'],
            ink_swap_max_area_range=inkcolorswap_params['ink_swap_max_area_range'],
            p=1.0
        ))
    
    # Paper phase augmentations
    if enabled_augs.get('colorpaper', False):
        paper_phase_augs.append(ColorPaper(
            hue_range=colorpaper_params['hue_range'],
            saturation_range=colorpaper_params['saturation_range'],
            p=1.0
        ))
    
    if enabled_augs.get('noisetexturize', False):
        paper_phase_augs.append(NoiseTexturize(
            sigma_range=noisetexturize_params['sigma_range'],
            turbulence_range=noisetexturize_params['turbulence_range'],
            texture_width_range=noisetexturize_params['texture_width_range'],
            texture_height_range=noisetexturize_params['texture_height_range'],
            p=1.0
        ))
    
    if enabled_augs.get('subtlenoise', False):
        paper_phase_augs.append(SubtleNoise(
            subtle_range=subtlenoise_params['subtle_range'],
            p=1.0
        ))
    
    if enabled_augs.get('bookbinding', False):
        paper_phase_augs.append(BookBinding(
            shadow_radius_range=bookbinding_params['shadow_radius_range'],
            curve_range_right=bookbinding_params['curve_range_right'],
            curve_range_left=bookbinding_params['curve_range_left'],
            curve_ratio_right=bookbinding_params['curve_ratio_right'],
            curve_ratio_left=bookbinding_params['curve_ratio_left'],
            mirror_range=bookbinding_params['mirror_range'],
            binding_pages=bookbinding_params['binding_pages'],
            curling_direction=bookbinding_params['curling_direction'],
            backdrop_color=bookbinding_params['backdrop_color'],
            enable_shadow=bookbinding_params['enable_shadow'],
            use_cache_images=bookbinding_params['use_cache_images'],
            p=1.0
        ))
    
    if enabled_augs.get('folding', False):
        paper_phase_augs.append(Folding(
            fold_count=folding_params['fold_count'],
            fold_noise=folding_params['fold_noise'],
            fold_angle_range=folding_params['fold_angle_range'],
            gradient_width=folding_params['gradient_width'],
            gradient_height=folding_params['gradient_height'],
            backdrop_color=folding_params['backdrop_color'],
            p=1.0
        ))
    
    if enabled_augs.get('pageborder', False):
        paper_phase_augs.append(PageBorder(
            page_border_color=pageborder_params['page_border_color'],
            page_border_background_color=pageborder_params['page_border_background_color'],
            page_rotation_angle_range=pageborder_params['page_rotation_angle_range'],
            curve_frequency=pageborder_params['curve_frequency'],
            curve_height=pageborder_params['curve_height'],
            curve_length_one_side=pageborder_params['curve_length_one_side'],
            same_page_border=pageborder_params['same_page_border'],
            p=1.0
        ))
    
    # Apply augmentations if any are enabled
    if ink_phase_augs or paper_phase_augs:
        pipeline = AugraphyPipeline(
            ink_phase=ink_phase_augs,
            paper_phase=paper_phase_augs,
            post_phase=[]
        )
        
        augmented = pipeline(image.copy())
        # Convert BGR to RGB for display
        augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        return augmented_rgb
    else:
        # Return original if no augmentations enabled
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def main():
    st.title("🎨 Augraphy Parameter Controller")
    st.markdown("Control **Augraphy Ink & Paper Phase** augmentations in real-time!")
    st.info("🔧 **Organized Setup**: All images are now extracted PDF halves stored in `gt_extraction/` folder for better quality and organization.")
    
    # Load configuration
    config = load_config()
    if not config:
        st.error("Failed to load configuration. Please check data_paths.yaml")
        return
    
    # File Browser Section
    st.subheader("📁 Image Browser")
    
    # Get available sources from config
    source_options = [config['image_sources'][key]['name'] for key in config['image_sources'].keys()]
    default_index = 0
    
    # Set default based on config
    if config.get('default_source') and config['default_source'] in config['image_sources']:
        default_source_name = config['image_sources'][config['default_source']]['name']
        if default_source_name in source_options:
            default_index = source_options.index(default_source_name)
    
    # Source selection
    col_source1, col_source2, col_source3 = st.columns(3)
    
    with col_source1:
        image_source = st.radio(
            "📂 Image Source:",
            options=source_options,
            index=default_index,
            help="Choose your image source from configured paths"
        )
    
    selected_image_path = None
    
    # Find the config key for the selected source
    source_config_key = None
    for key, source_info in config['image_sources'].items():
        if source_info['name'] == image_source:
            source_config_key = key
            break
    
    # Handle different image sources
    if source_config_key in ['synthetic_data', 'sample_images']:
        # Handle single-directory sources (synthetic data, sample images)
        images = get_images_from_source(config, source_config_key)
        source_info = config['image_sources'][source_config_key]
        
        if images:
            col_img1, col_img2 = st.columns([1, 2])
            
            with col_img1:
                st.markdown(f"**{source_info['name']}**")
                st.info(f"Found {len(images)} images")
                st.markdown(f"*{source_info['description']}*")
            
            with col_img2:
                selected_image = st.selectbox(
                    f"Select {source_info['name']}:",
                    options=images,
                    index=0,
                    help=f"Choose from {len(images)} available images",
                    key=f"{source_config_key}_select"
                )
                
                if selected_image:
                    selected_image_path = os.path.join(source_info['base_path'], selected_image)
                    
                    # Show image type info for sample images
                    if source_config_key == 'sample_images':
                        if selected_image.startswith("gu_"):
                            st.success("🇬🇺 Gujarati Synthetic Image")
                        elif selected_image.startswith("hi_"):
                            st.success("🇮🇳 Hindi Synthetic Image")
                        elif "left" in selected_image:
                            st.success("📖 PDF Left Half Page")
                        elif "right" in selected_image:
                            st.success("📖 PDF Right Half Page")
        else:
            st.error(f"No images found in {source_info['name']}!")
    else:
        # Handle PDF half directories (book-based structure)
        pdf_dirs = get_pdf_half_directories(config)
        
        if image_source in pdf_dirs:
            col_browser1, col_browser2 = st.columns([1, 2])
            
            with col_browser1:
                books = pdf_dirs[image_source]['books']
                if books:
                    selected_book = st.selectbox(
                        f"Select Book ({image_source}):",
                        options=books,
                        index=0,
                        help=f"Choose from {len(books)} available books",
                        key=f"book_select_{image_source.lower().replace(' ', '_')}"
                    )
                else:
                    st.error(f"No books found in {image_source} directory!")
                    selected_book = None
            
            with col_browser2:
                if selected_book:
                    base_path = pdf_dirs[image_source]['base_path']
                    book_path = os.path.join(base_path, selected_book)
                    image_files = get_image_files(book_path)
                    
                    if image_files:
                        selected_idx = st.selectbox(
                            f"Select Page from {selected_book}:",
                            options=range(len(image_files)),
                            format_func=lambda x: f"{os.path.basename(image_files[x])} (Page {x+1}/{len(image_files)})",
                            index=0,
                            help=f"Found {len(image_files)} pages in this book",
                            key=f"page_select_{image_source.lower().replace(' ', '_')}"
                        )
                        selected_image_path = image_files[selected_idx]
                    else:
                        st.warning(f"No images found in {selected_book}")
        else:
            st.error(f"{image_source} images not available!")
            st.info("Run the PDF extraction scripts first to generate half images.")
            st.code("""
# To generate PDF half images:
cd augmentation/gt_extraction
python3 extract_pdf_left_half.py
python3 extract_pdf_right_half.py
            """, language="bash")
    
    # Load the selected image
    if selected_image_path:
        original_bgr, original_rgb = load_image(selected_image_path)
        current_file = os.path.basename(selected_image_path)
        
        # Show image source info
        with col_source2:
            st.info(f"📄 Source: {image_source}")
        
        with col_source3:
            st.info(f"🗂️ File: {current_file}")
    else:
        # Fallback to default
        original_bgr, original_rgb = load_image("")
        current_file = "Sample Image"
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Original Image")
        st.image(original_rgb, caption=f"Source: {current_file}", use_container_width=True)
    
    # Sidebar for parameter controls
    st.sidebar.title("🎛️ Parameter Controls")
    st.sidebar.markdown("---")
    
    # Initialize session state for parameters
    if 'reset_triggered' not in st.session_state:
        st.session_state.reset_triggered = False
    
    # Augmentation selection
    st.sidebar.subheader("🎨 Augraphy Ink Phase Augmentations")
    enabled_augs = {
        'brightness': st.sidebar.checkbox("✨ Brightness", value=False, key="enabled_brightness"),
        'inkbleed': st.sidebar.checkbox("💧 InkBleed", value=False, key="enabled_inkbleed"),
        'letterpress': st.sidebar.checkbox("🖨️ Letterpress", value=False, key="enabled_letterpress"),
        'hollow': st.sidebar.checkbox("🕳️ Hollow", value=False, key="enabled_hollow"),
        'lowink_periodic': st.sidebar.checkbox("📏 Low Ink Periodic Lines", value=False, key="enabled_lowink_periodic"),
        'lowink_random': st.sidebar.checkbox("🎲 Low Ink Random Lines", value=False, key="enabled_lowink_random"),
        'lowlight_noise': st.sidebar.checkbox("🌙 Low Light Noise", value=False, key="enabled_lowlight_noise"),
        'inkcolorswap': st.sidebar.checkbox("🎭 Ink Color Swap", value=False, key="enabled_inkcolorswap")
    }
    
    st.sidebar.markdown("---")
    
    # Paper phase augmentation selection
    st.sidebar.subheader("📄 Augraphy Paper Phase Augmentations")
    paper_enabled_augs = {
        'colorpaper': st.sidebar.checkbox("🎨 Color Paper", value=False, key="enabled_colorpaper"),
        'noisetexturize': st.sidebar.checkbox("🌫️ Noise Texturize", value=False, key="enabled_noisetexturize"),
        'subtlenoise': st.sidebar.checkbox("✨ Subtle Noise", value=False, key="enabled_subtlenoise"),
        'bookbinding': st.sidebar.checkbox("📚 Book Binding", value=False, key="enabled_bookbinding"),
        'folding': st.sidebar.checkbox("📄 Page Folding", value=False, key="enabled_folding"),
        'pageborder': st.sidebar.checkbox("🖼️ Page Border", value=False, key="enabled_pageborder")
    }
    
    # Combine all enabled augmentations
    enabled_augs.update(paper_enabled_augs)
    
    st.sidebar.markdown("---")
    
    # Initialize parameters
    brightness_params = {}
    inkbleed_params = {}
    letterpress_params = {}
    hollow_params = {}
    lowink_periodic_params = {}
    lowink_random_params = {}
    lowlight_noise_params = {}
    inkcolorswap_params = {}
    colorpaper_params = {}
    noisetexturize_params = {}
    subtlenoise_params = {}
    bookbinding_params = {}
    folding_params = {}
    pageborder_params = {}
    
    # Brightness Controls
    if enabled_augs['brightness']:
        st.sidebar.subheader("✨ Brightness Parameters")
        
        brightness_value = st.sidebar.slider(
            "Brightness Range",
            min_value=0.1, max_value=2.0, value=0.3, step=0.1,
            help="Range: 0.1-0.5",
            key="brightness_value"
        )
        brightness_params['brightness_range'] = (brightness_value, brightness_value)
        
        brightness_params['min_brightness'] = st.sidebar.selectbox(
            "Min Brightness Mode",
            options=[0, 1],
            index=1,
            help="Your baseline: 10 (enabled)",
            key="brightness_min_mode"
        )
        
        if brightness_params['min_brightness'] == 1:
            min_brightness_value = st.sidebar.slider(
                "Min Brightness Value",
                min_value=0, max_value=100, value=10, step=5,
                help="📋 Recommended: 10 | Full range: 0-100",
                key="brightness_min_value"
            )
            brightness_params['min_brightness_value'] = (min_brightness_value, min_brightness_value)
        else:
            brightness_params['min_brightness_value'] = (0, 0)
        
        st.sidebar.markdown("---")
    
    # InkBleed Controls
    if enabled_augs['inkbleed']:
        st.sidebar.subheader("💧 InkBleed Parameters")
        
        intensity_value = st.sidebar.slider(
            "Intensity",
            min_value=0.1, max_value=1.0, value=0.8, step=0.05,
            help="Range: 0.8 ±0.2 = [0.6-1.0]",
            key="inkbleed_intensity"
        )
        inkbleed_params['intensity_range'] = (intensity_value, intensity_value)
        
        kernel_value = st.sidebar.slider(
            "Kernel Size",
            min_value=1, max_value=15, value=6, step=1,
            help="Range: 6 ±2 = [4-8]",
            key="inkbleed_kernel"
        )
        inkbleed_params['kernel_size'] = (kernel_value, kernel_value)
        
        severity_value = st.sidebar.slider(
            "Severity",
            min_value=0.1, max_value=1.0, value=0.8, step=0.05,
            help="Range: 0.8 ±0.2 = [0.6-1.0]",
            key="inkbleed_severity"
        )
        inkbleed_params['severity'] = (severity_value, severity_value)
        
        st.sidebar.markdown("---")
    
    # Letterpress Controls
    if enabled_augs['letterpress']:
        st.sidebar.subheader("🖨️ Letterpress Parameters")
        
        sample_count = st.sidebar.slider(
            "Sample Count",
            min_value=50, max_value=200, value=125, step=25,
            help="Fixed value: 125",
            key="letterpress_samples"
        )
        letterpress_params['n_samples'] = (sample_count, sample_count)
        
        cluster_count = st.sidebar.slider(
            "Cluster Count",
            min_value=50, max_value=200, value=125, step=25,
            help="Fixed value: 125",
            key="letterpress_clusters"
        )
        letterpress_params['n_clusters'] = (cluster_count, cluster_count)
        
        std_value = st.sidebar.slider(
            "Standard Deviation",
            min_value=1500, max_value=2000, value=1750, step=100,
            help="Range: 1500-2000 (mean: 1750)",
            key="letterpress_std"
        )
        letterpress_params['std_range'] = (std_value, std_value)
        
        value_darkness = st.sidebar.slider(
            "Value (Darkness)",
            min_value=100, max_value=200, value=150, step=10,
            help="Range: 100-200 (mean: 150)",
            key="letterpress_values"
        )
        letterpress_params['value_range'] = (value_darkness, value_darkness + 1)  # +1 for np.random.randint
        
        threshold_value = st.sidebar.slider(
            "Value Threshold",
            min_value=100, max_value=150, value=128, step=5,
            help="Fixed value: 128",
            key="letterpress_threshold"
        )
        letterpress_params['value_threshold_range'] = (threshold_value, threshold_value)
        
        letterpress_params['blur'] = st.sidebar.selectbox(
            "Blur Effect",
            options=[0, 1],
            index=1,
            help="0: Sharp edges, 1: Smooth edges",
            key="letterpress_blur"
        )
        
        st.sidebar.markdown("---")
    
    # Hollow Controls
    if enabled_augs['hollow']:
        st.sidebar.subheader("🕳️ Hollow Parameters")
        
        st.sidebar.markdown("**🎯 Updated Ranges: Median (3-5), Dilation (6-10)**")
        
        median_kernel = st.sidebar.slider(
            "Median Kernel Size",
            min_value=3, max_value=5, value=4, step=1,
            help="Range: 3-5 (mean: 4)",
            key="hollow_median"
        )
        hollow_params['hollow_median_kernel_value_range'] = (median_kernel, median_kernel)
        
        dilation_kernel = st.sidebar.slider(
            "Dilation Kernel Size", 
            min_value=6, max_value=10, value=8, step=1,
            help="Range: 6-10 (mean: 8)",
            key="hollow_dilation"
        )
        hollow_params['hollow_dilation_kernel_size_range'] = (dilation_kernel, dilation_kernel)
        
        # Size filtering controls - Using Augraphy's exact pixel units
        st.sidebar.markdown("**📏 Size Filters (Augraphy Pixel Units)**")
        
        min_width = st.sidebar.slider(
            "Min Width (pixels)",
            min_value=1, max_value=10, value=2, step=1,
            help="Range: 1-10 (baseline: 2 pixels)",
            key="hollow_min_width"
        )
        hollow_params['hollow_min_width_range'] = (min_width, min_width)
        
        max_width = st.sidebar.slider(
            "Max Width (pixels)",
            min_value=50, max_value=300, value=175, step=25,
            help="Range: 50-300 (baseline: 175 pixels)",
            key="hollow_max_width"
        )
        hollow_params['hollow_max_width_range'] = (max_width, max_width)
        
        min_height = st.sidebar.slider(
            "Min Height (pixels)", 
            min_value=1, max_value=10, value=2, step=1,
            help="Range: 1-10 (baseline: 2 pixels)",
            key="hollow_min_height"
        )
        hollow_params['hollow_min_height_range'] = (min_height, min_height)
        
        max_height = st.sidebar.slider(
            "Max Height (pixels)",
            min_value=50, max_value=300, value=175, step=25,
            help="Range: 50-300 (baseline: 175 pixels)",
            key="hollow_max_height"
        )
        hollow_params['hollow_max_height_range'] = (max_height, max_height)
        
        min_area = st.sidebar.slider(
            "Min Area (pixels²)",
            min_value=1, max_value=50, value=15, step=1,
            help="Range: 1-50 (baseline: 15 pixels²)",
            key="hollow_min_area"
        )
        hollow_params['hollow_min_area_range'] = (min_area, min_area)
        
        max_area = st.sidebar.slider(
            "Max Area (pixels²)",
            min_value=1000, max_value=8000, value=3500, step=500,
            help="Range: 1000-8000 (baseline: 3500 pixels²)",
            key="hollow_max_area"
        )
        hollow_params['hollow_max_area_range'] = (max_area, max_area)
        
        st.sidebar.info("📝 Hollow works by detecting contours (text shapes) and replacing their interior with edges only")
        
        # Detailed parameter explanations
        st.sidebar.markdown("### 🔍 Parameter Details:")
        
        with st.sidebar.expander("📏 Size Filtering Parameters", expanded=False):
            st.markdown("""
            **Min Width/Height/Area** control which contours get processed:
            
            - **Lower values** = More small text/characters affected
            - **Higher values** = Only larger text/shapes affected
            - **Current settings process contours that are:**
              - Width: {:.1f}% - 100% of image width
              - Height: {:.1f}% - 100% of image height  
              - Area: {:.2f}% - 100% of image area
            """.format(min_width*100, min_height*100, min_area*100))
        
        with st.sidebar.expander("🎛️ Effect Parameters", expanded=False):
            st.markdown("""
            **Median Kernel ({})**: 
            - Removes interior content of selected contours
            - Larger = more content removed (more hollow)
            - Must be odd number (auto-adjusted)
            - Applied via cv2.medianBlur()
            
            **Dilation Kernel ({})**: 
            - Thickens the remaining outline edges
            - Larger = thicker outline around hollow areas
            - Applied after edge detection
            - Creates final visible outline
            """.format(median_kernel, dilation_kernel))
        
        with st.sidebar.expander("⚙️ Process Steps", expanded=False):
            st.markdown("""
            1. **Contour Detection**: Find text/shape boundaries
            2. **Size Filtering**: Apply width/height/area criteria  
            3. **Mask Creation**: Create masks for selected contours
            4. **Edge Detection**: Apply Canny edge detection
            5. **Median Filter**: Remove interior content 
            6. **Dilation**: Thicken remaining edges
            7. **Random Removal**: Remove some edges for texture
            8. **Final Composite**: Combine hollow areas with original
            """)
        st.sidebar.markdown("---")
    
    # Low Ink Periodic Lines Controls
    if enabled_augs['lowink_periodic']:
        st.sidebar.subheader("📏 Low Ink Periodic Lines Parameters")
        
        count_value = st.sidebar.slider(
            "Line Count",
            min_value=1, max_value=10, value=6, step=1,
            help="Fixed value: 6",
            key="lowink_periodic_count"
        )
        lowink_periodic_params['count_range'] = (count_value, count_value)
        
        period_value = st.sidebar.slider(
            "Period Distance",
            min_value=5, max_value=50, value=30, step=5,
            help="Fixed value: 30",
            key="lowink_periodic_period"
        )
        lowink_periodic_params['period_range'] = (period_value, period_value)
        
        lowink_periodic_params['use_consistent_lines'] = st.sidebar.selectbox(
            "Use Consistent Lines",
            options=[True, False],
            index=0,
            help="Whether to vary width and alpha of lines",
            key="lowink_periodic_consistent"
        )
        
        noise_prob = st.sidebar.slider(
            "Noise Probability",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            help="Fixed value: 0.25",
            key="lowink_periodic_noise"
        )
        lowink_periodic_params['noise_probability'] = noise_prob
        
        st.sidebar.markdown("---")
    
    # Low Ink Random Lines Controls
    if enabled_augs['lowink_random']:
        st.sidebar.subheader("🎲 Low Ink Random Lines Parameters")
        
        count_value = st.sidebar.slider(
            "Line Count",
            min_value=1, max_value=20, value=15, step=1,
            help="Fixed value: 15",
            key="lowink_random_count"
        )
        lowink_random_params['count_range'] = (count_value, count_value)
        
        lowink_random_params['use_consistent_lines'] = st.sidebar.selectbox(
            "Use Consistent Lines",
            options=[True, False],
            index=0,
            help="Whether to vary width and alpha of lines",
            key="lowink_random_consistent"
        )
        
        noise_prob = st.sidebar.slider(
            "Noise Probability",
            min_value=0.0, max_value=1.0, value=0.50, step=0.05,
            help="Fixed value: 0.50",
            key="lowink_random_noise"
        )
        lowink_random_params['noise_probability'] = noise_prob
        
        st.sidebar.markdown("---")
    
    # Low Light Noise Controls
    if enabled_augs['lowlight_noise']:
        st.sidebar.subheader("🌙 Low Light Noise Parameters")
        
        num_photons = st.sidebar.slider(
            "Number of Photons",
            min_value=10, max_value=200, value=75, step=10,
            help="Number of photons to simulate (Default: 50-100)",
            key="lowlight_noise_photons"
        )
        lowlight_noise_params['num_photons_range'] = (num_photons, num_photons)
        
        alpha_value = st.sidebar.slider(
            "Alpha (Brightness)",
            min_value=0.1, max_value=2.0, value=0.85, step=0.05,
            help="Alpha for brightness adjustment (Default: 0.7-1.0)",
            key="lowlight_noise_alpha"
        )
        lowlight_noise_params['alpha_range'] = (alpha_value, alpha_value)
        
        beta_value = st.sidebar.slider(
            "Beta (Bias)",
            min_value=0, max_value=50, value=20, step=5,
            help="Beta for brightness adjustment (Default: 10-30)",
            key="lowlight_noise_beta"
        )
        lowlight_noise_params['beta_range'] = (beta_value, beta_value)
        
        gamma_value = st.sidebar.slider(
            "Gamma (Contrast)",
            min_value=0.5, max_value=3.0, value=1.4, step=0.1,
            help="Gamma for contrast adjustment (Default: 1.0-1.8)",
            key="lowlight_noise_gamma"
        )
        lowlight_noise_params['gamma_range'] = (gamma_value, gamma_value)
        
        bias_value = st.sidebar.slider(
            "Bias Range",
            min_value=0, max_value=80, value=30, step=5,
            help="Bias values to add (Default: 20-40)",
            key="lowlight_noise_bias"
        )
        lowlight_noise_params['bias_range'] = (bias_value, bias_value)
        
        lowlight_noise_params['dark_current_value'] = st.sidebar.slider(
            "Dark Current Value",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="Dark current simulation value (Default: 1.0)",
            key="lowlight_noise_dark_current"
        )
        
        lowlight_noise_params['exposure_time'] = st.sidebar.slider(
            "Exposure Time",
            min_value=0.05, max_value=1.0, value=0.2, step=0.05,
            help="Simulated exposure time in seconds (Default: 0.2)",
            key="lowlight_noise_exposure"
        )
        
        lowlight_noise_params['gain'] = st.sidebar.slider(
            "Camera Gain",
            min_value=0.01, max_value=1.0, value=0.1, step=0.01,
            help="Camera gain value (Default: 0.1)",
            key="lowlight_noise_gain"
        )
        
        st.sidebar.markdown("---")
    
    # Ink Color Swap Controls
    if enabled_augs['inkcolorswap']:
        st.sidebar.subheader("🎭 Ink Color Swap Parameters")
        
        # Color selection - simplified to common colors
        color_options = {
            "Random": "random",
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0), 
            "Green": (0, 255, 0),
            "Purple": (255, 0, 255),
            "Orange": (0, 165, 255),
            "Brown": (42, 42, 165)
        }
        
        selected_color = st.sidebar.selectbox(
            "Ink Swap Color",
            options=list(color_options.keys()),
            index=0,
            help="Color to swap ink with (BGR format)",
            key="inkcolorswap_color"
        )
        inkcolorswap_params['ink_swap_color'] = color_options[selected_color]
        
        sequence_num = st.sidebar.slider(
            "Sequence Number",
            min_value=1, max_value=20, value=7, step=1,
            help="Consecutive swapping number in detected contours (Default: 5-10)",
            key="inkcolorswap_sequence"
        )
        inkcolorswap_params['ink_swap_sequence_number_range'] = (sequence_num, sequence_num)
        
        # Size filtering controls
        st.sidebar.markdown("**📏 Contour Size Filtering**")
        
        min_width = st.sidebar.slider(
            "Min Width",
            min_value=1, max_value=10, value=2, step=1,
            help="Minimum width of contours to swap (Default: 2-3)",
            key="inkcolorswap_min_width"
        )
        inkcolorswap_params['ink_swap_min_width_range'] = (min_width, min_width)
        
        max_width = st.sidebar.slider(
            "Max Width",
            min_value=50, max_value=200, value=110, step=10,
            help="Maximum width of contours to swap (Default: 100-120)",
            key="inkcolorswap_max_width"
        )
        inkcolorswap_params['ink_swap_max_width_range'] = (max_width, max_width)
        
        min_height = st.sidebar.slider(
            "Min Height",
            min_value=1, max_value=10, value=2, step=1,
            help="Minimum height of contours to swap (Default: 2-3)",
            key="inkcolorswap_min_height"
        )
        inkcolorswap_params['ink_swap_min_height_range'] = (min_height, min_height)
        
        max_height = st.sidebar.slider(
            "Max Height",
            min_value=50, max_value=200, value=110, step=10,
            help="Maximum height of contours to swap (Default: 100-120)",
            key="inkcolorswap_max_height"
        )
        inkcolorswap_params['ink_swap_max_height_range'] = (max_height, max_height)
        
        min_area = st.sidebar.slider(
            "Min Area",
            min_value=5, max_value=50, value=15, step=5,
            help="Minimum area of contours to swap (Default: 10-20)",
            key="inkcolorswap_min_area"
        )
        inkcolorswap_params['ink_swap_min_area_range'] = (min_area, min_area)
        
        max_area = st.sidebar.slider(
            "Max Area",
            min_value=200, max_value=1000, value=450, step=50,
            help="Maximum area of contours to swap (Default: 400-500)",
            key="inkcolorswap_max_area"
        )
        inkcolorswap_params['ink_swap_max_area_range'] = (max_area, max_area)
        
        st.sidebar.info("🎭 Swaps ink color in detected text contours based on size criteria")
        
        st.sidebar.markdown("---")
    
    # Paper Phase Controls
    # Color Paper Controls
    if enabled_augs['colorpaper']:
        st.sidebar.subheader("🎨 Color Paper Parameters")
        
        hue_value = st.sidebar.slider(
            "Hue",
            min_value=0, max_value=180, value=36, step=5,
            help="Hue value for paper color (Default: 28-45)",
            key="colorpaper_hue"
        )
        colorpaper_params['hue_range'] = (hue_value, hue_value)
        
        saturation_value = st.sidebar.slider(
            "Saturation",
            min_value=0, max_value=100, value=25, step=5,
            help="Saturation value for paper color (Default: 10-40)",
            key="colorpaper_saturation"
        )
        colorpaper_params['saturation_range'] = (saturation_value, saturation_value)
        
        st.sidebar.info("🎨 Changes the background paper color based on hue and saturation")
        st.sidebar.markdown("---")
    
    # Noise Texturize Controls
    if enabled_augs['noisetexturize']:
        st.sidebar.subheader("🌫️ Noise Texturize Parameters")
        
        sigma_value = st.sidebar.slider(
            "Sigma (Noise Strength)",
            min_value=1, max_value=20, value=6, step=1,
            help="Noise fluctuation bounds (Default: 3-10)",
            key="noisetexturize_sigma"
        )
        noisetexturize_params['sigma_range'] = (sigma_value, sigma_value)
        
        turbulence_value = st.sidebar.slider(
            "Turbulence",
            min_value=1, max_value=10, value=3, step=1,
            help="Pattern replacement speed (Default: 2-5)",
            key="noisetexturize_turbulence"
        )
        noisetexturize_params['turbulence_range'] = (turbulence_value, turbulence_value)
        
        texture_width = st.sidebar.slider(
            "Texture Width",
            min_value=50, max_value=800, value=300, step=50,
            help="Width of texture pattern (Default: 100-500)",
            key="noisetexturize_width"
        )
        noisetexturize_params['texture_width_range'] = (texture_width, texture_width)
        
        texture_height = st.sidebar.slider(
            "Texture Height",
            min_value=50, max_value=800, value=300, step=50,
            help="Height of texture pattern (Default: 100-500)",
            key="noisetexturize_height"
        )
        noisetexturize_params['texture_height_range'] = (texture_height, texture_height)
        
        st.sidebar.info("🌫️ Creates random noise patterns to emulate paper textures")
        st.sidebar.markdown("---")
    
    # Subtle Noise Controls
    if enabled_augs['subtlenoise']:
        st.sidebar.subheader("✨ Subtle Noise Parameters")
        
        subtle_range_value = st.sidebar.slider(
            "Subtle Range",
            min_value=1, max_value=30, value=10, step=1,
            help="Range of subtle noise variation (Default: 10)",
            key="subtlenoise_range"
        )
        subtlenoise_params['subtle_range'] = subtle_range_value
        
        st.sidebar.info("✨ Emulates scanning imperfections due to subtle lighting differences")
        st.sidebar.markdown("---")
    
    # Book Binding Controls
    if enabled_augs['bookbinding']:
        st.sidebar.subheader("📚 Book Binding Parameters")
        
        shadow_radius = st.sidebar.slider(
            "Shadow Radius",
            min_value=10, max_value=200, value=65, step=10,
            help="Radius for shadow effect (Default: 30-100)",
            key="bookbinding_shadow_radius"
        )
        bookbinding_params['shadow_radius_range'] = (shadow_radius, shadow_radius)
        
        curve_right = st.sidebar.slider(
            "Right Curve",
            min_value=20, max_value=200, value=75, step=10,
            help="Right page curve amount (Default: 50-100)",
            key="bookbinding_curve_right"
        )
        bookbinding_params['curve_range_right'] = (curve_right, curve_right)
        
        curve_left = st.sidebar.slider(
            "Left Curve",
            min_value=100, max_value=500, value=250, step=25,
            help="Left page curve amount (Default: 200-300)",
            key="bookbinding_curve_left"
        )
        bookbinding_params['curve_range_left'] = (curve_left, curve_left)
        
        ratio_right = st.sidebar.slider(
            "Right Curve Ratio",
            min_value=0.01, max_value=0.2, value=0.075, step=0.01,
            help="Right page squeeze ratio (Default: 0.05-0.1)",
            key="bookbinding_ratio_right"
        )
        bookbinding_params['curve_ratio_right'] = (ratio_right, ratio_right)
        
        ratio_left = st.sidebar.slider(
            "Left Curve Ratio",
            min_value=0.3, max_value=0.8, value=0.55, step=0.05,
            help="Left page squeeze ratio (Default: 0.5-0.6)",
            key="bookbinding_ratio_left"
        )
        bookbinding_params['curve_ratio_left'] = (ratio_left, ratio_left)
        
        mirror_value = st.sidebar.slider(
            "Mirror Range",
            min_value=0.5, max_value=1.0, value=1.0, step=0.1,
            help="Percentage to mirror (Default: 1.0)",
            key="bookbinding_mirror"
        )
        bookbinding_params['mirror_range'] = (mirror_value, mirror_value)
        
        binding_pages = st.sidebar.slider(
            "Binding Pages",
            min_value=3, max_value=20, value=7, step=1,
            help="Number of pages in binding (Default: 5-10)",
            key="bookbinding_pages"
        )
        bookbinding_params['binding_pages'] = (binding_pages, binding_pages)
        
        bookbinding_params['curling_direction'] = st.sidebar.selectbox(
            "Curling Direction",
            options=[-1, 0, 1],
            index=0,
            format_func=lambda x: {-1: "Random", 0: "Up", 1: "Down"}[x],
            help="Page curl direction (-1: random, 0: up, 1: down)",
            key="bookbinding_curl_direction"
        )
        
        # Color selection for backdrop
        backdrop_colors = {
            "Black": (0, 0, 0),
            "White": (255, 255, 255),
            "Gray": (128, 128, 128),
            "Dark Gray": (64, 64, 64)
        }
        selected_backdrop = st.sidebar.selectbox(
            "Backdrop Color",
            options=list(backdrop_colors.keys()),
            index=0,
            help="Background color for binding effect",
            key="bookbinding_backdrop"
        )
        bookbinding_params['backdrop_color'] = backdrop_colors[selected_backdrop]
        
        bookbinding_params['enable_shadow'] = st.sidebar.selectbox(
            "Enable Shadow",
            options=[0, 1],
            index=1,
            format_func=lambda x: {0: "Disabled", 1: "Enabled"}[x],
            help="Enable shadow effect",
            key="bookbinding_shadow"
        )
        
        bookbinding_params['use_cache_images'] = st.sidebar.selectbox(
            "Use Cache Images",
            options=[0, 1],
            index=1,
            format_func=lambda x: {0: "Disabled", 1: "Enabled"}[x],
            help="Use cache images for left page",
            key="bookbinding_cache"
        )
        
        st.sidebar.info("📚 Creates realistic book binding effect with curved pages and shadows")
        st.sidebar.markdown("---")
    
    # Folding Controls
    if enabled_augs['folding']:
        st.sidebar.subheader("📄 Page Folding Parameters")
        
        fold_count = st.sidebar.slider(
            "Fold Count",
            min_value=1, max_value=5, value=2, step=1,
            help="Number of folds to apply (Default: 2)",
            key="folding_count"
        )
        folding_params['fold_count'] = fold_count
        
        fold_noise = st.sidebar.slider(
            "Fold Noise",
            min_value=0.001, max_value=0.1, value=0.01, step=0.001,
            help="Noise level in folding area (Default: 0.01)",
            key="folding_noise"
        )
        folding_params['fold_noise'] = fold_noise
        
        fold_angle = st.sidebar.slider(
            "Fold Angle",
            min_value=0, max_value=45, value=0, step=5,
            help="Rotation angle before folding (Default: 0)",
            key="folding_angle"
        )
        folding_params['fold_angle_range'] = (fold_angle, fold_angle)
        
        gradient_width = st.sidebar.slider(
            "Gradient Width",
            min_value=0.05, max_value=0.5, value=0.15, step=0.05,
            help="Width of affected area (% of page width, Default: 0.1-0.2)",
            key="folding_grad_width"
        )
        folding_params['gradient_width'] = (gradient_width, gradient_width)
        
        gradient_height = st.sidebar.slider(
            "Gradient Height",
            min_value=0.005, max_value=0.1, value=0.015, step=0.005,
            help="Depth of fold (% of page height, Default: 0.01-0.02)",
            key="folding_grad_height"
        )
        folding_params['gradient_height'] = (gradient_height, gradient_height)
        
        # Backdrop color for folding
        fold_backdrop_colors = {
            "Black": (0, 0, 0),
            "White": (255, 255, 255),
            "Gray": (128, 128, 128)
        }
        selected_fold_backdrop = st.sidebar.selectbox(
            "Backdrop Color",
            options=list(fold_backdrop_colors.keys()),
            index=0,
            help="Background color for folding effect",
            key="folding_backdrop"
        )
        folding_params['backdrop_color'] = fold_backdrop_colors[selected_fold_backdrop]
        
        st.sidebar.info("📄 Creates realistic page folding with perspective transformation")
        st.sidebar.markdown("---")
    
    # Page Border Controls
    if enabled_augs['pageborder']:
        st.sidebar.subheader("🖼️ Page Border Parameters")
        
        # Border colors
        border_colors = {
            "Black": (0, 0, 0),
            "White": (255, 255, 255),
            "Gray": (128, 128, 128),
            "Brown": (42, 42, 165)
        }
        
        selected_border_color = st.sidebar.selectbox(
            "Border Color",
            options=list(border_colors.keys()),
            index=0,
            help="Color of the page border",
            key="pageborder_color"
        )
        pageborder_params['page_border_color'] = border_colors[selected_border_color]
        
        selected_bg_color = st.sidebar.selectbox(
            "Background Color",
            options=list(border_colors.keys()),
            index=0,
            help="Background color of border",
            key="pageborder_bg_color"
        )
        pageborder_params['page_border_background_color'] = border_colors[selected_bg_color]
        
        rotation_angle = st.sidebar.slider(
            "Rotation Angle",
            min_value=-10, max_value=10, value=0, step=1,
            help="Page rotation angle (Default: -3 to 3)",
            key="pageborder_rotation"
        )
        pageborder_params['page_rotation_angle_range'] = (rotation_angle, rotation_angle)
        
        curve_freq = st.sidebar.slider(
            "Curve Frequency",
            min_value=0, max_value=5, value=1, step=1,
            help="Number of curves in border (Default: 0-1)",
            key="pageborder_curve_freq"
        )
        pageborder_params['curve_frequency'] = (curve_freq, curve_freq)
        
        curve_height = st.sidebar.slider(
            "Curve Height",
            min_value=1, max_value=10, value=3, step=1,
            help="Height of curves (Default: 2-4)",
            key="pageborder_curve_height"
        )
        pageborder_params['curve_height'] = (curve_height, curve_height)
        
        curve_length = st.sidebar.slider(
            "Curve Length",
            min_value=20, max_value=200, value=75, step=10,
            help="Length of curve sides (Default: 50-100)",
            key="pageborder_curve_length"
        )
        pageborder_params['curve_length_one_side'] = (curve_length, curve_length)
        
        pageborder_params['same_page_border'] = st.sidebar.selectbox(
            "Same Page Border",
            options=[0, 1],
            index=1,
            format_func=lambda x: {0: "Different borders", 1: "Same border"}[x],
            help="Whether borders should be same or different",
            key="pageborder_same"
        )
        
        st.sidebar.info("🖼️ Adds layered page borders with curves and tears")
        st.sidebar.markdown("---")
    
    # Control buttons
    col_btn1, col_btn2 = st.sidebar.columns(2)
    
    with col_btn1:
        reset_button = st.button("🔄 Reset", help="Reset all parameters to defaults", type="secondary")
    
    with col_btn2:
        auto_update = st.checkbox("⚡ Auto", value=True, help="Auto-apply changes")
    
    if reset_button:
        # Reset session state to trigger parameter reset
        for key in list(st.session_state.keys()):
            if key.startswith(('brightness_', 'inkbleed_', 'letterpress_', 'hollow_', 'lowink_', 'lowlight_', 'inkcolorswap_', 'colorpaper_', 'noisetexturize_', 'subtlenoise_', 'bookbinding_', 'folding_', 'pageborder_', 'enabled_')):
                del st.session_state[key]
        st.rerun()
    
    if not auto_update:
        apply_button = st.sidebar.button("🎨 Apply Changes", type="primary")
        should_update = apply_button
    else:
        should_update = True
    
    # Apply augmentations and show result
    if should_update and any(enabled_augs.values()):
        with st.spinner("Applying augmentations..."):
            try:
                # Debug information - Show parameter formats
                debug_params = {}
                if enabled_augs.get('brightness', False):
                    debug_params['Brightness'] = brightness_params
                if enabled_augs.get('inkbleed', False):
                    debug_params['InkBleed'] = inkbleed_params
                if enabled_augs.get('letterpress', False):
                    debug_params['Letterpress'] = letterpress_params
                if enabled_augs.get('hollow', False):
                    debug_params['Hollow'] = {k: v for k, v in hollow_params.items() if k.endswith('_range')}
                if enabled_augs.get('lowink_periodic', False):
                    debug_params['LowInkPeriodic'] = lowink_periodic_params
                if enabled_augs.get('lowink_random', False):
                    debug_params['LowInkRandom'] = lowink_random_params
                if enabled_augs.get('lowlight_noise', False):
                    debug_params['LowLightNoise'] = lowlight_noise_params
                if enabled_augs.get('inkcolorswap', False):
                    debug_params['InkColorSwap'] = inkcolorswap_params
                if enabled_augs.get('colorpaper', False):
                    debug_params['ColorPaper'] = colorpaper_params
                if enabled_augs.get('noisetexturize', False):
                    debug_params['NoiseTexturize'] = noisetexturize_params
                if enabled_augs.get('subtlenoise', False):
                    debug_params['SubtleNoise'] = subtlenoise_params
                if enabled_augs.get('bookbinding', False):
                    debug_params['BookBinding'] = bookbinding_params
                if enabled_augs.get('folding', False):
                    debug_params['Folding'] = folding_params
                if enabled_augs.get('pageborder', False):
                    debug_params['PageBorder'] = pageborder_params
                
                if debug_params:
                    with st.sidebar.expander("🔍 Debug - Parameter Values", expanded=False):
                        st.write("**Exact values being sent to Augraphy:**")
                        for aug_name, params in debug_params.items():
                            st.write(f"**{aug_name}:**")
                            for key, value in params.items():
                                if isinstance(value, tuple) and len(value) == 2:
                                    if value[0] == value[1]:
                                        st.write(f"   `{key}`: {value} → **Exact: {value[0]}**")
                                    elif abs(value[1] - value[0]) == 1:
                                        st.write(f"   `{key}`: {value} → **Exact: {value[0]}** (+1 fix)")
                                    else:
                                        st.write(f"   `{key}`: {value} → **Range**")
                                else:
                                    st.write(f"   `{key}`: {value} → **Single**")
                
                # Create a unique key based on parameters to force re-computation
                param_key = str(letterpress_params) + str(brightness_params) + str(inkbleed_params) + str(hollow_params)
                
                modified_rgb = apply_augmentations(original_bgr, brightness_params, inkbleed_params, letterpress_params, hollow_params, lowink_periodic_params, lowink_random_params, lowlight_noise_params, inkcolorswap_params, colorpaper_params, noisetexturize_params, subtlenoise_params, bookbinding_params, folding_params, pageborder_params, enabled_augs)
                
                with col2:
                    st.subheader("🎨 Modified Image")
                    st.image(modified_rgb, caption="Augmented Image", use_container_width=True)
                    
                    # Show enabled augmentations
                    enabled_list = [name.title() for name, enabled in enabled_augs.items() if enabled]
                    if enabled_list:
                        st.success(f"Applied: {', '.join(enabled_list)}")
                        
            except Exception as e:
                st.error(f"Error applying augmentations: {str(e)}")
                st.error(f"Full error details: {repr(e)}")
                with col2:
                    st.subheader("❌ Error")
                    st.error("Failed to apply augmentations. Check parameter values.")
    
    elif any(enabled_augs.values()) and not should_update:
        with col2:
            st.subheader("⏳ Ready to Apply")
            st.info("Click 'Apply Changes' to see the augmented image")
    
    else:
        with col2:
            st.subheader("🎨 Modified Image")
            st.info("Enable at least one augmentation to see changes")
    
    # Information section
    st.markdown("---")
    st.subheader("📚 Parameter Guide")
    
    # Split into two rows for better layout
    st.markdown("**🎨 Ink Phase Augmentations:**")
    col_ink1, col_ink2, col_ink3, col_ink4, col_ink5, col_ink6, col_ink7, col_ink8 = st.columns(8)
    
    st.markdown("**📄 Paper Phase Augmentations:**")
    col_paper1, col_paper2, col_paper3, col_paper4, col_paper5, col_paper6 = st.columns(6)
    
    with col_ink1:
        st.markdown("""
        **✨ Brightness**
        - Range: 0.1-1.0 (your baseline: 0.1-0.5)
        - Min Brightness: 10 (enabled)
        - Lower values = darker image
        """)
    
    with col_ink2:
        st.markdown("""
        **💧 InkBleed**
        - Intensity: 0.6 ±0.2 = [0.4-0.8]
        - Severity: 0.6 ±0.2 = [0.4-0.8] 
        - Kernel: 5 ±2 = [3-7]
        """)
    
    with col_ink3:
        st.markdown("""
        **🖨️ Letterpress**
        - Samples: 50-200
        - Clusters: 50-200
        - Std Dev: 1500-2000
        - Value: 80-200
        - Threshold: 128
        """)
    
    with col_ink4:
        st.markdown("""
        **🕳️ Hollow**
        - Median: 3-10 (your custom range)
        - Dilation: 5-10 (your custom range)
        - Size filters: 2px min, 175px max
        - Area: 15-3500 pixels²
        """)
    
    with col_ink5:
        st.markdown("""
        **📏 Low Ink Periodic**
        - Count: 1-10 lines (Default: 2-5)
        - Period: 5-50px (Default: 10-30)
        - Consistent lines: True/False
        - Noise prob: 0.0-1.0 (Default: 0.1)
        """)
    
    with col_ink6:
        st.markdown("""
        **🎲 Low Ink Random**
        - Count: 1-20 lines (Default: 5-10)
        - Consistent lines: True/False
        - Noise prob: 0.0-1.0 (Default: 0.1)
        - Random placement throughout image
        """)
    
    with col_ink7:
        st.markdown("""
        **🌙 Low Light Noise**
        - Photons: 10-200 (Default: 50-100)
        - Alpha: 0.1-2.0 (Default: 0.7-1.0)
        - Beta: 0-50 (Default: 10-30)
        - Gamma: 0.5-3.0 (Default: 1.0-1.8)
        """)
    
    with col_ink8:
        st.markdown("""
        **🎭 Ink Color Swap**
        - Colors: Random/Red/Blue/Green/Purple/Orange/Brown
        - Sequence: 1-20 (Default: 5-10)
        - Size filters: Width/Height/Area ranges
        - Contour-based color replacement
        """)
    
    with col_paper1:
        st.markdown("""
        **🎨 Color Paper**
        - Hue: 0-180 (Default: 28-45)
        - Saturation: 0-100 (Default: 10-40)
        - Changes background paper color
        - HSV color space manipulation
        """)
    
    with col_paper2:
        st.markdown("""
        **🌫️ Noise Texturize**
        - Sigma: 1-20 (Default: 3-10)
        - Turbulence: 1-10 (Default: 2-5)
        - Texture Size: 50-800px (Default: 100-500)
        - Random noise patterns for paper texture
        """)
    
    with col_paper3:
        st.markdown("""
        **✨ Subtle Noise**
        - Range: 1-30 (Default: 10)
        - Emulates scanning imperfections
        - Subtle lighting differences
        - Very light noise overlay
        """)
    
    with col_paper4:
        st.markdown("""
        **📚 Book Binding**
        - Shadow Radius: 10-200 (Default: 30-100)
        - Curve effects for left/right pages
        - Multiple pages with realistic shadows
        - Simulates book spine binding
        """)
    
    with col_paper5:
        st.markdown("""
        **📄 Page Folding**
        - Fold Count: 1-5 (Default: 2)
        - Gradient Width/Height controls
        - Perspective transformation
        - Realistic paper creases
        """)
    
    with col_paper6:
        st.markdown("""
        **🖼️ Page Border**
        - Layered page effects
        - Curve frequency and height
        - Multiple border styles
        - Torn/aged page edges
        """)

def main():
    """Main application function using configuration files"""
    # Load configurations
    data_config = load_data_paths_config()
    aug_config = load_augmentation_config()
    
    if not data_config or not aug_config:
        st.error("Failed to load configuration files. Please check data_paths.yaml and augmentation_cfg.yaml")
        return
    
    # App header
    st.markdown("# 🎨 Augraphy Parameter Controller")
    st.markdown("### Dynamic control of document augmentation parameters using YAML configuration")
    
    # Initialize variables
    original_image = None
    selected_image = None
    
    # Image source selection
    st.subheader("📁 Image Selection")
    
    # Get available image sources from config
    image_sources = data_config['image_sources']
    source_options = [(key, info['name']) for key, info in image_sources.items()]
    default_index = 0
    
    # Set default source
    if 'default_source' in data_config:
        for i, (key, _) in enumerate(source_options):
            if key == data_config['default_source']:
                default_index = i
                break
    
    selected_source_key = st.radio(
        "Select image source:",
        options=[key for key, _ in source_options],
        format_func=lambda x: next(name for k, name in source_options if k == x),
        index=default_index,
        key="image_source_selection"
    )
    
    # Load images from selected source
    available_images = get_images_from_source(data_config, selected_source_key)
    
    if available_images:
        selected_image = st.selectbox(
            f"Select image from {image_sources[selected_source_key]['name']}:",
            available_images,
            key="selected_image"
        )
        
        # Load original image
        if selected_image:
            # Build full path
            full_path = os.path.join(image_sources[selected_source_key]['base_path'], selected_image)
            image_result = load_image(full_path)
            
            if isinstance(image_result, tuple) and len(image_result) == 2:
                # The existing load_image function returns (bgr_image, rgb_image)
                original_image = image_result[1]  # Use RGB version for display
            else:
                original_image = image_result
    else:
        st.warning(f"No images found in {image_sources[selected_source_key]['name']} source")
    
    # Display images side by side
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Original Image")
        if original_image is not None:
            # Display image
            if isinstance(original_image, np.ndarray):
                if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                    st.image(original_image, caption="Original Image", use_container_width=True)
                elif len(original_image.shape) == 2:
                    st.image(original_image, caption="Original Image (Grayscale)", use_container_width=True)
                else:
                    st.error(f"Unexpected image format: {original_image.shape}")
            else:
                st.error(f"Unexpected image type: {type(original_image)}")
        else:
            st.info("Select an image to display")
    
    with col2:
        st.subheader("🔧 Augmented Result")
        
        if original_image is not None:
            # Apply augmentations and show result
            augmented_image = apply_configured_augmentations(original_image, aug_config)
            if augmented_image is not None:
                st.image(augmented_image, caption="Augmented Image", use_container_width=True)
                
                # Debug section below the image
                st.markdown("### 🔍 Debug Information")
                show_augmentation_debug_info(aug_config)
            else:
                st.warning("Failed to apply augmentations")
        else:
            st.info("Select an image to see augmented results")
    
    # Sidebar for parameter controls
    with st.sidebar:
        st.markdown("## 🎛️ Augmentation Controls")
        
        # Reset button
        if st.button("🔄 Reset All Parameters", key="reset_button"):
            reset_session_parameters(aug_config)
            st.rerun()
        
        # Generate controls for each augmentation
        create_augmentation_controls(aug_config)

def show_augmentation_debug_info(aug_config: Dict[str, Any]):
    """Display debug information for all enabled augmentations"""
    # Check ink phase augmentations
    ink_phase_augs = ['brightness', 'inkbleed', 'letterpress', 'hollow', 'lowinkperiodiclines', 'lowinkrandomlines']
    paper_phase_augs = ['colorpaper', 'noisetexturize', 'subtlenoise', 'bookbinding', 'folding', 'pageborder']
    
    enabled_ink = []
    enabled_paper = []
    
    # Collect enabled ink phase augmentations
    for aug_name in ink_phase_augs:
        enabled_key = f"{aug_name}_enabled"
        if st.session_state.get(enabled_key, False):
            aug_params = {}
            aug_config_params = aug_config['augmentations'].get(aug_name, {})
            
            for param_name in aug_config_params.keys():
                session_key = f"{aug_name}_{param_name}_augraphy"
                if session_key in st.session_state:
                    aug_params[param_name] = st.session_state[session_key]
            
            enabled_ink.append((aug_name, aug_params))
    
    # Collect enabled paper phase augmentations
    for aug_name in paper_phase_augs:
        enabled_key = f"{aug_name}_enabled"
        if st.session_state.get(enabled_key, False):
            aug_params = {}
            aug_config_params = aug_config['augmentations'].get(aug_name, {})
            
            for param_name in aug_config_params.keys():
                session_key = f"{aug_name}_{param_name}_augraphy"
                if session_key in st.session_state:
                    aug_params[param_name] = st.session_state[session_key]
            
            enabled_paper.append((aug_name, aug_params))
    
    # Display debug information
    if enabled_ink:
        st.markdown("#### 🖋️ Ink Phase Debug")
        for aug_name, params in enabled_ink:
            with st.expander(f"🔍 {aug_name.title()} Parameters", expanded=False):
                st.json(params)
    
    if enabled_paper:
        st.markdown("#### 📄 Paper Phase Debug")
        for aug_name, params in enabled_paper:
            with st.expander(f"🔍 {aug_name.title()} Parameters", expanded=False):
                st.json(params)
    
    if not enabled_ink and not enabled_paper:
        st.info("No augmentations enabled")

def reset_session_parameters(aug_config: Dict[str, Any]):
    """Reset all session state parameters to their default values"""
    augmentations = aug_config.get('augmentations', {})
    for aug_name, aug_params in augmentations.items():
        # Reset parameter values
        for param_name in aug_params.keys():
            session_key = f"{aug_name}_{param_name}"
            if session_key in st.session_state:
                del st.session_state[session_key]
            augraphy_session_key = f"{session_key}_augraphy"
            if augraphy_session_key in st.session_state:
                del st.session_state[augraphy_session_key]
        
        # Reset enable/disable flags
        enabled_key = f"{aug_name}_enabled"
        if enabled_key in st.session_state:
            del st.session_state[enabled_key]
        
        # Reset tracking flags
        was_enabled_key = f"{aug_name}_was_enabled"
        if was_enabled_key in st.session_state:
            del st.session_state[was_enabled_key]
        
        just_enabled_key = f"{aug_name}_just_enabled"
        if just_enabled_key in st.session_state:
            del st.session_state[just_enabled_key]

def create_augmentation_controls(aug_config: Dict[str, Any]):
    """Create UI controls for all augmentations based on configuration"""
    augmentations = aug_config.get('augmentations', {})
    
    # Ink Phase Augmentations
    st.markdown("### 🖋️ Ink Phase Augmentations")
    
    ink_phase_augs = ['brightness', 'inkbleed', 'letterpress', 'hollow', 'lowinkperiodiclines', 'lowinkrandomlines']
    
    for aug_name in ink_phase_augs:
        if aug_name in augmentations:
            with st.expander(f"{aug_name.replace('_', ' ').title()}", expanded=False):
                aug_params = augmentations[aug_name]
                
                # Enable/disable toggle
                enabled_key = f"{aug_name}_enabled"
                enabled = st.checkbox(f"Enable {aug_name.title()}", value=False, key=enabled_key)
                
                # Check if augmentation was just enabled
                was_just_enabled = f"{aug_name}_just_enabled" in st.session_state
                
                if enabled:
                    # Create sliders for each parameter
                    for param_name, param_config in aug_params.items():
                        session_key = f"{aug_name}_{param_name}"
                        ui_value = create_parameter_slider(aug_name, param_name, param_config, session_key)
                        
                        # Store the converted parameter for later use
                        augraphy_param = convert_ui_value_to_augraphy_param(param_name, ui_value, aug_config)
                        st.session_state[f"{session_key}_augraphy"] = augraphy_param
                    
                    # Clear the just_enabled flag after parameters are set
                    if was_just_enabled:
                        del st.session_state[f"{aug_name}_just_enabled"]
                    # Force a rerun to apply the augmentation immediately when first enabled
                    elif f"{aug_name}_was_enabled" not in st.session_state:
                        st.session_state[f"{aug_name}_was_enabled"] = True
                        st.rerun()
                else:
                    # Clean up flags when disabled
                    if f"{aug_name}_was_enabled" in st.session_state:
                        del st.session_state[f"{aug_name}_was_enabled"]
                    if f"{aug_name}_just_enabled" in st.session_state:
                        del st.session_state[f"{aug_name}_just_enabled"]
    
    # Paper Phase Augmentations
    st.markdown("### 📄 Paper Phase Augmentations")
    
    paper_phase_augs = ['colorpaper', 'noisetexturize', 'subtlenoise', 'bookbinding', 'folding', 'pageborder']
    
    for aug_name in paper_phase_augs:
        if aug_name in augmentations:
            with st.expander(f"{aug_name.replace('_', ' ').title()}", expanded=False):
                aug_params = augmentations[aug_name]
                
                # Enable/disable toggle
                enabled_key = f"{aug_name}_enabled"
                enabled = st.checkbox(f"Enable {aug_name.title()}", value=False, key=enabled_key)
                
                # Check if augmentation was just enabled
                was_just_enabled = f"{aug_name}_just_enabled" in st.session_state
                
                if enabled:
                    # Create sliders for each parameter
                    for param_name, param_config in aug_params.items():
                        session_key = f"{aug_name}_{param_name}"
                        ui_value = create_parameter_slider(aug_name, param_name, param_config, session_key)
                        
                        # Store the converted parameter for later use
                        augraphy_param = convert_ui_value_to_augraphy_param(param_name, ui_value, aug_config)
                        st.session_state[f"{session_key}_augraphy"] = augraphy_param
                    
                    # Clear the just_enabled flag after parameters are set
                    if was_just_enabled:
                        del st.session_state[f"{aug_name}_just_enabled"]
                    # Force a rerun to apply the augmentation immediately when first enabled
                    elif f"{aug_name}_was_enabled" not in st.session_state:
                        st.session_state[f"{aug_name}_was_enabled"] = True
                        st.rerun()
                else:
                    # Clean up flags when disabled
                    if f"{aug_name}_was_enabled" in st.session_state:
                        del st.session_state[f"{aug_name}_was_enabled"]
                    if f"{aug_name}_just_enabled" in st.session_state:
                        del st.session_state[f"{aug_name}_just_enabled"]

def apply_configured_augmentations(image: np.ndarray, aug_config: Dict[str, Any]) -> np.ndarray:
    """Apply enabled augmentations with current parameter values"""
    try:
        # Create lists of enabled augmentations for each phase
        enabled_ink_augs = []
        enabled_paper_augs = []
        
        # Check ink phase augmentations
        ink_phase_augs = ['brightness', 'inkbleed', 'letterpress', 'hollow', 'lowinkperiodiclines', 'lowinkrandomlines']
        
        for aug_name in ink_phase_augs:
            enabled_key = f"{aug_name}_enabled"
            if st.session_state.get(enabled_key, False):
                # Collect parameters for this augmentation
                aug_params = {}
                aug_config_params = aug_config['augmentations'].get(aug_name, {})
                
                for param_name in aug_config_params.keys():
                    session_key = f"{aug_name}_{param_name}_augraphy"
                    if session_key in st.session_state:
                        aug_params[param_name] = st.session_state[session_key]
                
                # Create augmentation instance
                aug_instance = None
                try:
                    if aug_name == 'brightness':
                        aug_instance = Brightness(**aug_params)
                    elif aug_name == 'inkbleed':
                        aug_instance = InkBleed(**aug_params)
                    elif aug_name == 'letterpress':
                        aug_instance = Letterpress(**aug_params)
                    elif aug_name == 'hollow':
                        aug_instance = Hollow(**aug_params)
                    elif aug_name == 'lowinkperiodiclines':
                        aug_instance = LowInkPeriodicLines(**aug_params)
                    elif aug_name == 'lowinkrandomlines':
                        aug_instance = LowInkRandomLines(**aug_params)
                except Exception as e:
                    print(f"❌ {aug_name} creation failed: {e}")
                    aug_instance = None
                
                if aug_instance:
                    enabled_ink_augs.append(aug_instance)
        
        # Check paper phase augmentations
        paper_phase_augs = ['colorpaper', 'noisetexturize', 'subtlenoise', 'bookbinding', 'folding', 'pageborder']
        
        for aug_name in paper_phase_augs:
            enabled_key = f"{aug_name}_enabled"
            if st.session_state.get(enabled_key, False):
                # Collect parameters for this augmentation
                aug_params = {}
                aug_config_params = aug_config['augmentations'].get(aug_name, {})
                
                for param_name in aug_config_params.keys():
                    session_key = f"{aug_name}_{param_name}_augraphy"
                    if session_key in st.session_state:
                        aug_params[param_name] = st.session_state[session_key]
                
                # Create augmentation instance
                aug_instance = None
                try:
                    if aug_name == 'colorpaper':
                        aug_instance = ColorPaper(**aug_params)
                    elif aug_name == 'noisetexturize':
                        aug_instance = NoiseTexturize(**aug_params)
                    elif aug_name == 'subtlenoise':
                        aug_instance = SubtleNoise(**aug_params)
                    elif aug_name == 'bookbinding':
                        aug_instance = BookBinding(**aug_params)
                    elif aug_name == 'folding':
                        aug_instance = Folding(**aug_params)
                    elif aug_name == 'pageborder':
                        aug_instance = PageBorder(**aug_params)
                except Exception as e:
                    print(f"❌ {aug_name} creation failed: {e}")
                    aug_instance = None
                
                if aug_instance:
                    enabled_paper_augs.append(aug_instance)
        
        # Apply augmentations if any are enabled
        if enabled_ink_augs or enabled_paper_augs:
            pipeline = AugraphyPipeline(
                ink_phase=enabled_ink_augs if enabled_ink_augs else [],
                paper_phase=enabled_paper_augs if enabled_paper_augs else []
            )
            result = pipeline(image)
            return result
        else:
            return image
            
    except Exception as e:
        st.error(f"Error applying augmentations: {e}")
        return image

if __name__ == "__main__":
    main()
