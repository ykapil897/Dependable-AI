# Adversarial Attack Analysis for Multi-Label Classification

## Overview
This file provides an overview of the implementation and analysis of various adversarial attacks on multi-label classification models using the IAPRTC-12 dataset.

### Attacks Implemented:
- **White-box vs. Black-box Attacks**: Comparing attacks with full knowledge of the target model versus transfer attacks
- **Targeted vs. Untargeted Attacks**: Evaluating attacks that aim for specific misclassifications versus general errors
- **Sample-specific vs. Sample-agnostic (Universal) Attacks**: Analyzing individual perturbations versus universal perturbations

### Models Evaluated:
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
- Scikit-learn
- Joblib

## Dataset
The code uses the **IAPRTC-12 dataset**, which contains:
- **Training set**: 17,665 samples with 2,048 features and 291 possible labels
- **Testing set**: 1,962 samples with 2,048 features and 291 possible labels

## Directory Structure
The code automatically creates the following directory:
- **attack_results/**: Stores CSV files with results for each attack type

## How to Use
### 1. Setup Prerequisites:
```sh
pip install numpy scipy pandas matplotlib scikit-learn joblib
```

### 2. Prepare Dataset and Models:
- Ensure the IAPRTC-12 dataset files are available in the **IAPRTC** directory.
- Required files:
  - `IAPRTC-12_TrainFeat.mat`
  - `IAPRTC-12_TrainLabels.mat`
  - `IAPRTC-12_TestFeat.mat`
  - `IAPRTC-12_TestLabels.mat`
- Ensure that trained models from Task 0 are available in the current directory:
  - `linear_svm.pkl`
  - `logistic_regression.pkl`
  - `softmax_regression.pkl`
  - `decision_tree.pkl`
  - `weighted_knn.pkl` (optional)
  - `scaler.pkl`
  - `pca_model.pkl`

### 3. Run the Notebook:
- Execute the notebook cells sequentially to load data, implement attacks, and evaluate results.
- Alternatively, run the entire notebook to perform all attacks.

### 4. Interpret Results:
- Results are saved in the **attack_results/** directory as CSV files.
- Visualizations are generated to compare attack effectiveness across models.
- A summary of findings is saved in `task1_summary.txt`.

## Implementation Details
### 1. Data Loading and Preprocessing
- Loads the IAPRTC-12 dataset and previously trained models.
- Applies feature scaling and PCA dimensionality reduction.
- Defines evaluation metrics for model performance.

### 2. Fast Gradient Sign Method (FGSM) Implementation
```python
def fgsm_attack(model, X, y, epsilon=0.1, model_name=None, targeted=False, target_labels=None):
    """
    Fast Gradient Sign Method (FGSM) attack
    Uses numerical gradient estimation for models without direct gradient support
    """
```
- Implements FGSM using numerical gradient estimation for compatibility with all model types.
- Supports both targeted and untargeted attacks.
- Processes data in batches for memory efficiency.

### 3. White-box vs. Black-box Attack Analysis
- **White-box**: Generates adversarial examples specifically for each target model.
- **Black-box**: Generates adversarial examples using **Linear SVM** and transfers to other models.
- Evaluates and compares the effectiveness of both approaches.

### 4. Targeted vs. Untargeted Attack Analysis
```python
def targeted_attack(model, X, y, epsilon=0.1, model_name=None):
    """
    Generate targeted adversarial examples by flipping specific labels
    """
```
- **Targeted**: Creates adversarial examples aimed at specific target labels.
- **Untargeted**: Creates adversarial examples to maximize general classification error.
- Measures attack success rates and performance impact.

### 5. Sample-specific vs. Sample-agnostic Attack Analysis
```python
def universal_perturbation(model, X, y, epsilon=0.1, max_iter=10, model_name=None):
    """
    Generate a universal (sample-agnostic) perturbation that works across multiple inputs
    """
```
- **Sample-specific**: Customized perturbations for each individual input.
- **Sample-agnostic (Universal)**: Single perturbation pattern that works across multiple inputs.
- Compares effectiveness and transferability between attack types.

### 6. Results Analysis and Visualization
- Generates comprehensive performance metrics.
- Creates visualizations to compare attack types and model robustness.
- Ranks models based on their resilience to different attacks.

## Key Findings
### Attack Effectiveness Hierarchy:
- **White-box attacks** are generally more effective than black-box attacks.
- **Sample-specific attacks** typically outperform universal perturbations.
- The effectiveness of **targeted vs. untargeted attacks** varies by model.

### Model Robustness:
- Based on average performance drop across attacks, models can be ranked from most to least robust (see `task1_summary.txt`).

### Attack Transferability:
- Black-box attacks show varying degrees of transferability between models, with linear models typically being more susceptible.

### Perturbation Size Impact:
- Larger perturbation magnitudes (**higher epsilon values**) generally produce stronger attacks, with diminishing returns.

## Output Files
The code generates several output files:
- **Attack results:**
  - `whitebox_attack_results.csv`
  - `blackbox_attack_results.csv`
  - `targeted_attack_results.csv`
  - `untargeted_attack_results.csv`
  - `specific_attack_results.csv`
  - `agnostic_attack_results.csv`
- **Comparative analysis:**
  - `whitebox_vs_blackbox_results.csv`
  - `targeted_vs_untargeted_results.csv`
  - `specific_vs_agnostic_results.csv`
  - `attack_comparison.csv`
- **Summary:**
  - `task1_summary.txt`

### Visualization Files
- `whitebox_vs_blackbox_f1.png`
- `targeted_vs_untargeted_f1.png`
- `targeted_attack_success_rate.png`
- `specific_vs_agnostic_attacks.png`
- `all_models_specific_vs_agnostic.png`
- `attack_performance_drop_comparison_epsilon_0.5.png`
- `attack_effectiveness_ratio.png`
- `comprehensive_attack_comparison.png`

## References
- [Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)
- [Papernot et al., 2016](https://arxiv.org/abs/1605.07277)
- [Moosavi-Dezfooli et al., 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html)
- [Carlini & Wagner, 2017](https://arxiv.org/abs/1707.08945)
- [Kurakin et al., 2016](https://arxiv.org/abs/1607.02533)
- [Tsipras et al., 2018](https://arxiv.org/abs/1805.12152)
