# VizWiz-VQA-Grounding
2025 1st place (will)

## Project 구조
project/
    README.md
    models/
        __init__.py
        image_encoder.py
        text_encoder.py -> me
        concat.py -> me
        mask_decoder.py -> unet 
        model.py -> me
    data/
        binary_masks_png/
            train/
            val/
        test/
        train/
        val/
        test_grounding.json
        train_grounding.json
        val_grounding.json
    train.py
    dataset.py
    utils.py
    config.yml
    IoU.py
    metrics.py
    visualize_predictions.py

  

