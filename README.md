# VizWiz-VQA-Grounding
2025 1st place (will)  
 This project was developed for the [VizWiz-VQA-Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025. The goal is to return grounded visual evidence for answers to visual questions posed by people with visual impairments. 


## 🎶Task Objective
Given an image-question pair, the task is to predict the region in the image that supports the most common answer. This is known as **answer grounding**, and predictions are evaluated based on **mean Intersection over Union (IoU)** with human-annotated binary masks.


## 📂Project Structure
```project/
├── README.md
├── models/ │
    ├── init.py │
    ├── image_encoder.py │
    ├── text_encoder.py # me │
    ├── concat.py # me │
    ├── mask_decoder.py # unet │
    └── model.py # me
├── data/ │
    ├── binary_masks_png/ │
    │ ├── train/ │
    │ └── val/ │
├── test/ │
├── train/ │
├── val/ │
    ├── test_grounding.json │
    ├── train_grounding.json │
    └── val_grounding.json
├── train.py
├── dataset.py
├── utils.py
├── visualize_predictions.py
├── IoU.py
├── metrics.py
└── config.yml
```
## ⚙️ Installation


## 🚀 Running the Code

## 🧠 Model Design

## 📊 Evaluation

## 🔗 References

## 📝 License

