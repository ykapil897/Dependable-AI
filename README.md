# Data Artificial Intelligence (DAI) - Course Projects

This repository contains my work for the Data Analytics and Intelligence course, focused on Deep Learning Interpretability, Explainability, and Adversarial Robustness. The repository includes two assignments and a final project, each exploring different aspects of modern machine learning techniques.

## Repository Structure

```
.
├── assign_1/           # Assignment 1: Model Explainability
│   ├── LIME/           # LIME-based explainability implementation
│   ├── I_Gradients/    # Integrated Gradients implementation
│   └── AnnexML_/       # AnnexML model and supporting files
│
├── assign_2/           # Assignment 2: Adversarial Attacks & Defenses
│   ├── attack_results/ # Results from different attack strategies
│   └── defense_results/# Evaluation of defense mechanisms
│
└── SliceCode/          # Final Project: Enhanced SLICE with Robustness & Explainability
    └── DAI_Project/    # Project implementation files
```

## Assignment 1: Model Explainability

This assignment explores two different methods for model explainability applied to an AnnexML model trained on the IAPRTC-12 dataset:

### LIME (Local Interpretable Model-agnostic Explanations)
- Implementation of LIME to explain AnnexML model predictions
- Approximates the complex model's behavior locally with an interpretable model
- Highlights feature importance for individual predictions

**Key Features:**
- Integration with the AnnexML multi-label classification model
- Feature importance visualization for individual predictions
- Command-line interaction with the AnnexML prediction system

**Technical Approach:**
- Uses `LimeTabularExplainer` to generate local explanations
- Formats and processes data to work with AnnexML's input requirements
- Visualizes feature contributions to understand model decisions

### Integrated Gradients
- Implementation of Integrated Gradients method for attribution
- Assigns importance scores to features by accumulating gradients
- Identifies which features most contribute to model predictions

**Key Features:**
- Gradient-based attribution along interpolation paths
- Robust theoretical foundation with axiom satisfaction
- Analysis of feature importance across the multi-label setup

**Technical Approach:**
- Creates interpolation paths between baseline and input instances
- Calculates gradient accumulations for attribution
- Ranks and visualizes feature contributions to model predictions

## Assignment 2: Adversarial Attacks & Defenses

This assignment investigates the vulnerability of machine learning models to adversarial examples and explores defense mechanisms.

### Task 0: Model Development and Baseline Evaluation
- Implementation and evaluation of various machine learning models
- Establishment of baseline performance metrics

### Task 1: Adversarial Attacks
- Implementation of targeted and untargeted attacks
- Comparison of white-box vs. black-box attack effectiveness
- Analysis of model-specific vs. model-agnostic attack transferability

**Key Findings:**
- Performance drop comparisons across different attack strategies
- Attack effectiveness ratio measurements
- Targeted vs. untargeted attack success rate analysis

### Task 2: Defense Mechanisms
- Implementation of adversarial training and feature preprocessing defenses
- Evaluation of defense effectiveness against various attack types
- Analysis of trade-offs between model robustness and accuracy

**Techniques Implemented:**
- Feature squeezing and adversarial training
- Quantile transformation and feature scaling
- Ensemble defense strategies

## Final Project: Enhanced SLICE Framework

The project enhances the SLICE (Scalable Linear Extreme Classifiers) framework to address limitations in explainability and robustness.

### Features
- **Feature Importance-Based Explainability**: System for identifying and visualizing influential features
- **Adversarial Robustness**: Defense mechanisms against adversarial inputs
- **Enhanced Training Pipeline**: Integration of both explainability and robustness features

**Results:**
- High precision performance on the EURLex-4K dataset
- Exceptional robustness against adversarial examples
- Fast prediction speed with minimal overhead from explainability features
- Comprehensive feature importance visualizations

## Setup and Dependencies

Each component has specific setup instructions and dependencies. Please refer to the individual README files within each directory for detailed instructions.

## Citations

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. Proceedings of the 34th International Conference on Machine Learning, 70, 3319-3328.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. In Proceedings of the International Conference on Learning Representations.

## Author

Kapil Yadav