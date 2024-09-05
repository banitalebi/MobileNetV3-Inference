# MobileNetV3 Inference using Tract Library

This repository provides a simple example of performing inference using the MobileNetV3 model with the **Tract** library. The model is evaluated using the mobilenetv3_model.onnx file on a test dataset, with performance metrics calculated, including true positives, false positives, true negatives, false negatives, and overall accuracy. The results are compared against a Python implementation.

## Dataset
The dataset used for fine-tuning the model is sourced from [Kaggle](https://www.kaggle.com/datasets/abdulmananraja/real-life-violence-situations/data). This dataset can be organized into three subsets:
- **Training Set**: 70% of the total dataset
- **Validation Set**: 20% of the total dataset
- **Test Set**: 10% of the total dataset

Note that the test dataset is used exclusively for evaluation. and it can be found [here](https://drive.google.com/file/d/1sDwIpnQJD8iaRm8M322pITrD0esdMZZ6/view?usp=sharing).

## Results
The results are listed as follows:
- **True Positives** (Violence): 529
- **True Negatives** (Non-Violence): 328
- **False Positives** (Predicted Violence, Actual Non-Violence): 196
- **False Negatives** (Predicted Non-Violence, Actual Violence): 56
- **Total Images:** 1109
- **Accuracy:** 77.28%
