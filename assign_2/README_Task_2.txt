# Task 2: Implementing Defense Mechanisms Against Adversarial Attacks

## Overview
This file outlines and explains the implementation of various defense mechanisms against adversarial attacks for multi-label classification models on the IAPRTC-12 dataset.

The code implements and evaluates five different defense strategies to protect machine learning models against adversarial attacks:

1. **Adversarial Training**: Incorporating adversarial examples into the training process.
2. **Feature Squeezing**: Reducing input precision to eliminate adversarial perturbations.
3. **Gaussian Data Augmentation**: Training with noise-augmented data to improve robustness.
4. **Ensemble Methods**: Combining predictions from multiple models to mitigate attacks.
5. **Input Transformation**: Preprocessing inputs to neutralize adversarial perturbations.

These defenses are evaluated on five classification models:

- **Linear SVM**
- **Logistic Regression**
- **Softmax Regression**
- **Decision Tree**
- **Weighted KNN**

## Prerequisites

- Python 3.x
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Dataset
The code uses the IAPRTC-12 dataset, which contains:

- **Training set**: 17,665 samples × 2,048 features, with 291 possible labels.
- **Testing set**: 1,962 samples × 2,048 features, with 291 possible labels.

## Directory Structure
The code automatically creates the following directories:

- `defense_results`: Contains CSV files with evaluation results for each defense method.
- `defense_checkpoints`: Stores trained models and intermediate checkpoints.

## How to Use

### Setup Prerequisites:

### Prepare Dataset:
Make sure the IAPRTC-12 dataset files are available in the `IAPRTC` directory:
- `IAPRTC-12_TrainFeat.mat`
- `IAPRTC-12_TrainLabels.mat`
- `IAPRTC-12_TestFeat.mat`
- `IAPRTC-12_TestLabels.mat`

### Run Previous Tasks:
This notebook assumes that **Task 0 (model training)** and **Task 1 (adversarial attack assessment)** have been completed.
Make sure the trained models and preprocessing objects (`scaler.pkl`, `pca_model.pkl`) are available.

### Run the Notebook:
Execute the notebook cells sequentially to load data, implement defenses, and evaluate results.
Alternatively, the notebook can be run in its entirety to perform all evaluations.

### Interpret Results:
- Results are saved in the `defense_results` directory as CSV files.
- Visualizations are generated to compare defense effectiveness across models.

## Implementation Details

### Data Preparation
- Loads IAPRTC-12 dataset and previously trained models.
- Sets up a test subset of 100 examples for consistent evaluation.

### Adversarial Example Generation
- Uses **Fast Gradient Sign Method (FGSM)** with numerical gradient estimation.
- Creates adversarial examples for each model with a specified epsilon value.

### Defense Methods

#### 1. Adversarial Training
```python
def adversarial_training(model, X_train, y_train, model_name=None, epsilon=0.1, ratio=0.5):
    """Train a model on a mix of original and adversarial examples"""
```
- Creates a mix of clean and adversarial examples for training.
- Retrains the model on this augmented dataset.

#### 2. Feature Squeezing
```python
def feature_squeezing(X, bit_depth=5):
    """Reduce the precision of input features to remove adversarial perturbations"""
```
- Quantizes inputs to a reduced bit depth (3, 5, or 7 bits).
- Returns the squeezed features that eliminate small adversarial perturbations.

#### 3. Gaussian Data Augmentation
```python
def gaussian_augmentation(model, X_train, y_train, model_name=None, sigma=0.1, num_samples=1):
    """Train a model with Gaussian noise augmentation"""
```
- Creates noise-augmented versions of the training data.
- Trains models on the combined clean and noisy examples.

#### 4. Ensemble Methods
```python
def ensemble_defense(models_dict, X, voting='soft'):
    """Create an ensemble of models for defense"""
```
- Combines predictions from multiple models (excluding the attacked model).
- Supports both hard voting (majority) and soft voting (probability averaging).

#### 5. Input Transformation
```python
def input_transformation_defense(X, method='scaling'):
    """Apply input transformations to defend against adversarial attacks"""
```
- Implements three transformation methods:
  - **Robust scaling**: Normalizes features using statistics resistant to outliers.
  - **PCA transformation**: Performs dimensionality reduction and reconstruction.
  - **Quantile transformation**: Transforms feature distributions to be normal.

## Evaluation Framework

- Tests each defense against adversarial examples.
- Measures performance using **accuracy, precision, recall, and F1 score**.
- Calculates improvements relative to non-defended models.

## Resilience Features

- Comprehensive checkpointing to recover from interruptions.
- Exception handling for robustness.
- Incremental result saving to avoid data loss.

## Results Interpretation

The notebook generates visualizations for comparing defense effectiveness:

- **Bar charts** comparing average improvements across defense methods.
- **Heatmaps** showing which defenses work best for each model type.
- **Summary statistics** identifying the optimal defense strategy per model.

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572.
- Xu, W., Evans, D., & Qi, Y. (2017). "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks." arXiv preprint arXiv:1704.01155.
- Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017). "Practical black-box attacks against machine learning." In Proceedings of the 2017 ACM Asia Conference on Computer and Communications Security.
- Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D., & McDaniel, P. (2017). "Ensemble adversarial training: Attacks and defenses." arXiv preprint arXiv:1705.07204.
- Guo, C., Rana, M., Cisse, M., & Van Der Maaten, L. (2018). "Countering adversarial images using input transformations." arXiv preprint arXiv:1711.00117.

## Notes and Limitations

- The code focuses on evaluating defenses against the **FGSM attack** specifically.
- Due to computational constraints, evaluations use a **subset of 100 test examples**.
- Some defenses may be computationally expensive and require significant resources.
- The **optimal defense method varies** across model types, indicating no one-size-fits-all solution.
