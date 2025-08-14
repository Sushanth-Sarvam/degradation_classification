# Readability Training Examples

This directory contains sample images and model artifacts from the readability classification experiments.

## Contents

### 📁 `train_images/` (5 samples)
Randomly sampled training images from various readable books:
- **1 GULAB NATAK_044.png** - From "1 GULAB NATAK" (readable)
- **Gujarat Ehtihasic Prasang_161.png** - From "Gujarat Ehtihasic Prasang" (readable)  
- **Gujarat Ehtihasic Prasang_344.png** - From "Gujarat Ehtihasic Prasang" (readable)
- **Statue_009.png** - From "Statue" (readable)
- **Vibhavana_251.png** - From "Vibhavana" (readable)

### 📁 `test_images/` (5 samples)  
Randomly sampled test images from various books with mixed readability:
- **06 Harishchandra Natak_030.png** - Readable (model got it wrong)
- **06 Harishchandra Natak_082.png** - Readable (model got it right)
- **08 HINDUSTANMA MUSAFARI_122.png** - Non-readable (model got it wrong)
- **Pritam_063.png** - Readable (model got it wrong - from worst book)
- **Swarupsannidhan_156.png** - Non-readable (model got it wrong)

### 📁 `models/`
- **best_test_model.pkl** - The best performing model (XGBoost + EfficientNet, 78.81% test accuracy)

### 📁 `summaries/`
- **train_summary.json** - Metadata about training images
- **test_summary.json** - Metadata about test images with model predictions

## Model Performance

- **Best Model**: XGBoost + EfficientNet
- **Test Accuracy**: 78.81%  
- **Train Accuracy**: 100% (shows overfitting)
- **Feature Extractor**: EfficientNet-B0 (1280 features)

## Usage

These examples can be used to:
1. **Understand the dataset** - See what "readable" vs "non-readable" documents look like
2. **Test the model** - Load the model and run predictions on new images  
3. **Debug failures** - Examine cases where the model made mistakes
4. **Demo the system** - Show the readability classification capability

## Dataset Split

Images are categorized by book-level split to prevent data leakage:
- **Training books**: 24 books, 771 images
- **Test books**: 11 books, 387 images
- **Split method**: Stratified by book readability to maintain class balance

## Notes

- All training images achieved 100% accuracy (perfect memorization)
- Test accuracy varies significantly by book (0% to 100%)
- Pritam and Sundaramni books show worst performance
- Model shows clear overfitting patterns 