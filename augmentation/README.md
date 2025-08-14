# Augraphy Document Augmentation System

A system for applying document degradation effects using the Augraphy library, with both an interactive UI and batch processing.

## Quick Start

### Prerequisites

1. **Clone Augraphy Library**
```bash
cd /root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/
git clone https://github.com/sparkfish/augraphy.git
```

2. **Setup Environment**
```bash
conda activate akshar
cd /root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/augmentation
```

### Interactive UI
```bash
streamlit run augmentation_ui.py --server.port 8505
```
Open browser to: http://localhost:8505

### Batch Processing
```bash
python generate_augmentations.py
```

## How Configurations Work

### Interactive UI Configuration

The UI uses two YAML files:

**`data_paths.yaml`** - Image source locations:
```yaml
image_sources:
  sample_images:
    name: "Sample Images"
    base_path: "/path/to/sample_images"
  pdf_right_half:
    name: "PDF Right Half"
    base_path: "/path/to/pdf_right_half_images"
default_source: "sample_images"
```

**`augmentation_cfg.yaml`** - Parameter definitions:
```yaml
augmentations:
  brightness:
    brightness_range:
      valid_range: [0.1, 3.0]      # Full allowable range
      augraphy_range: [0.8, 1.4]   # Augraphy defaults
      recommended_range: [0.1, 0.5] # Recommended range
      value: 0.3                    # Default slider value
```

### Batch Processing Configuration

**`batch_augmentation_config.yaml`** - Defines what to generate:
```yaml
input_directory: "/path/to/input/images"
output_directory: "/path/to/output/images"

augmentation_sets:
  - name: "brightness_low"
    phase: "ink"
    augmentations:
      - type: "brightness"
        parameters:
          brightness_range: [0.2, 0.2]
          min_brightness: 1
          min_brightness_value: [10, 10]
  
  - name: "vintage_effect"
    phase: "combined"
    augmentations:
      - type: "brightness"
        parameters:
          brightness_range: [0.3, 0.3]
      - type: "colorpaper"
        parameters:
          hue_range: [25, 35]
          saturation_range: [20, 30]

filename_format:
  include_parameters: true  # Adds parameter values to filenames
  separator: "_"
```

## File Structure

```
augmentation/
├── README.md                           # This file
├── augmentation_ui.py                  # Interactive UI
├── generate_augmentations.py           # Batch script
├── augmentation_cfg.yaml               # UI parameters
├── data_paths.yaml                     # Image paths
├── batch_augmentation_config.yaml      # Batch config
├── sample_images/                      # Sample images
└── generated_augmentations/            # Batch outputs
```

## Augmentation Types

### Ink Phase (applied to text/content)
- **Brightness**: Overall image brightness
- **InkBleed**: Ink bleeding effects
- **Letterpress**: Letterpress printing effects
- **Hollow**: Hollow/outlined text

### Paper Phase (applied to paper/background)
- **ColorPaper**: Paper color tinting
- **NoiseTexturize**: Paper texture noise
- **SubtleNoise**: Background noise
- **BookBinding**: Book binding effects

## Usage Examples

### Batch Processing
```bash
# Default configuration
python generate_augmentations.py

# Custom configuration
python generate_augmentations.py custom_config.yaml
```

Output files include parameter values:
```
original_image_brightness_low_br0.2.png
original_image_vintage_effect_br0.3_hue30.png
```

### Interactive UI
1. Select image source (configured in `data_paths.yaml`)
2. Choose an image to work with
3. Enable augmentations using checkboxes
4. Adjust parameters with sliders (ranges from `augmentation_cfg.yaml`)
5. View original and augmented images side-by-side
