# 🏆Workshop Spotlight at CVPR 2025
Our work has been selected as a spotlight paper at a workshop in CVPR 2025! ( https://cvpr.thecvf.com/ )
We are honored that our research was recognized and featured among the notable contributions. 

# Tasks: VizWiz-VQA-Grounding
 This project was developed for the [VizWiz-VQA-Grounding Challenge](https://vizwiz.org/tasks-and-datasets/visual-qa/) 2025. The goal is to return grounded visual evidence for answers to visual questions posed by people with visual impairments. 


## 🎶Task Objective
Given an image-question pair, the task is to predict the region in the image that supports the most common answer. This is known as **answer grounding**, and predictions are evaluated based on **mean Intersection over Union (IoU)** with human-annotated binary masks.


## 📂Project Structure
```project/
├── README.md
├── models/
    ├── init.py
    ├── image_encoder.py
    ├── text_encoder.py
    ├── concat.py
    ├── mask_decoder.py
    └── model.py 
├── data/
    ├── binary_masks_png/
    │   ├── train/
    │   └── val/
    ├── test/
    ├── train/
    ├── val/
    ├── test_grounding.json
    ├── train_grounding.json
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
    apt update
    apt install -y git
    git
    git config --global user.name "<yourname>"
    git config --global user.email "<youremail>"
    git clone https://github.com/yjh9929/VizWiz-VQA-Grounding.git
    pip install torch torchvision transformers pyyaml
    pip install tqdm
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## 🚀 Running the Code
    CUDA_VISIBLE_DEVICES=0 python train.py

## 🧠 Model Design

## 📊 Evaluation
    python visualize_predictions.py
    python IoU.py

## 🔗 References

## 📝 License
<a href="http://creativecommons.org/licenses/by/4.0/" rel="license"><img src="https://i.creativecommons.org/l/by/4.0/88x31.png" alt="Creative Commons License"></a>  
This work is licensed under a Creative Commons Attribution 4.0 International License.
