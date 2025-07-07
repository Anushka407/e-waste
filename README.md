## E-WASTE GENERATION CLASSIFICATION

This project is part of the **EDUNET FOUNDATION  |SHELL|  AI/ML Virtual Internship**. The goal is to build a deep learning model using Convolutional Neural Networks (CNNs) to classify e-waste images into distinct categories such as mobile phones, cables, appliances, and more. This model aims to support automated e-waste sorting for sustainable waste management.

## Problem Statement

With the rapid rise in electronics consumption, managing electronic waste (e-waste) has become a global challenge. Manual sorting of e-waste is time-consuming, inaccurate, and hazardous to health. This project addresses the problem by automating the classification of e-waste images using machine learning, specifically deep learning through CNNs.

## Tools & Technologies Used

| Category            | Tool/Library           |
|---------------------|------------------------|
| Language            | Python                 |
| ML/DL Framework     | TensorFlow / Keras     |
| Data Manipulation   | NumPy                  |
| Visualization       | Matplotlib, Seaborn    |
| Evaluation          | Scikit-learn           |
| Deployment          | Gradio                 |
| IDE                 | Jupyter Notebook       |


## Dataset
- **Source**: Kaggle - E-Waste Image Dataset  
- Classes: Battery, Keyboard, Microwave, Mobile, Mouse, PCB, Player, Printer, Television, Washing Machine
- Structure: Organized in folders by class label.
- Size: Balanced with 2400 training images, 300 validation images, 300 test images.



## Data Preparation
- Dataset organized into folders per class: `train`, `val`, `test`
- Each class folder contains images of specific e-waste types
- Loaded using TensorFlow's `image_dataset_from_directory()`


##  Data Augmentation
- Applied preprocessing: resizing to (128x128), normalization
- Used TensorFlow layers to apply:
  - Random flipping
  - Rotation
  - Zoom
  - Contrast adjustment (optional)


##  Model Architecture
A simple **Sequential CNN** model was implemented:
- `Conv2D` layers with ReLU activation
- `MaxPooling2D` for downsampling
- `Flatten` followed by `Dense` layers
- Output layer uses `softmax` for multi-class classification


## Training
- Trained over 15 epochs with early stopping
- Monitored `val_loss` to avoid overfitting
- Batch size: 32
- Validation accuracy plateaued at ~60%, indicating scope for improvement


## Result & Evaluation
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~60%
- Evaluation metrics included:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
- Visualized results using `seaborn` heatmaps and `matplotlib` plots


## Deployment
Model deployment was done using **Gradio**, a Python library that creates interactive web interfaces.


## Key Improvements
- Early stopping to prevent overfitting
- Dropout layers or regularization could further improve generalization
- Optionally switch to pretrained models like EfficientNet or MobileNet

## Future Scope
- Use a larger, more balanced dataset
- Deploy as a web or mobile app for real-world waste classification
- Integrate with IoT sensors for automated hardware solutions
- Add object detection (not just classification)

## 👩‍💻 Author
Anushka Shree,MCA 2nd year
EDUNET |SHELL| Virtual Internship




