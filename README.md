# ANN Cancer Classification

An artificial neural network that classifies breast tumors as malignant or benign, built with TensorFlow/Keras.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oliviagcadwell/ann-cancer-classification-deeplearning/blob/main/Neural_network_deep_learning.ipynb)

## Data

[Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic) from the UCI Machine Learning Repository: 569 tumors (212 malignant, 357 benign), each described by 30 features computed from digitized images of fine-needle aspirates, such as radius, texture, concavity, and symmetry. The notebook loads the dataset directly from scikit-learn, so it runs without downloading anything.

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. Licensed under CC BY 4.0.

## Approach

1. **Exploration:** class balance, feature correlations with the diagnosis, and a correlation heatmap.
2. **Preparation:** train / validation / test split (60 / 15 / 25%) and min-max scaling fit on the training data only.
3. **Model:** a feed-forward network (30 → 15 → 1 units, ReLU hidden layers, sigmoid output) trained with binary cross-entropy and Adam.
4. **Reducing overfitting:** a baseline trained for 600 epochs shows validation loss rising as training loss falls. Early stopping on validation loss, then dropout (0.5), closes that gap.
5. **Evaluation:** precision, recall, and a confusion matrix on the held-out test set, which is never used during training or tuning.

## Tools

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn
