# Task 0: Classification Algorithms Evaluation for Multi-Label Image Classification

This file provides a detailed explanation of the implementation and evaluation of multiple classification algorithms for multi-label image classification using the IAPRTC-12 dataset.

## Overview
This notebook implements and compares the performance of five classification algorithms and an ensemble approach for multi-label classification:

- **Linear SVM**: One-vs-Rest strategy with LinearSVC  
- **Logistic Regression**: One-vs-Rest strategy with binary logistic regression  
- **Softmax Regression**: Multinomial logistic regression with 'lbfgs' solver  
- **Decision Tree**: Configured with controlled depth to prevent overfitting  
- **Weighted KNN**: Distance-weighted voting with 5 neighbors  
- **Ensemble**: Majority voting across all individual models  

Each model is evaluated using multiple metrics to provide a comprehensive performance assessment for the multi-label classification task.

## Prerequisites
- Python 3.x
- NumPy
- SciPy
- Pandas
- Matplotlib
- Joblib
- Scikit-learn
- tqdm (for progress bars)

## Dataset
The code uses the **IAPRTC-12** dataset:

- **Training set**: 17,665 samples with 2,048 features and 291 possible labels  
- **Testing set**: 1,962 samples with 2,048 features and 291 possible labels  

### Directory Structure
The notebook expects the dataset to be organized as follows:
```
./IAPRTC/
  ├── IAPRTC-12_TrainFeat.mat
  ├── IAPRTC-12_TrainLabels.mat
  ├── IAPRTC-12_TestFeat.mat
  └── IAPRTC-12_TestLabels.mat
```

## How to Use
### Setup Prerequisites:
```sh
pip install numpy scipy pandas matplotlib scikit-learn joblib tqdm
```

### Prepare Dataset:
- Place the **IAPRTC-12 dataset** files in the `IAPRTC` directory.

### Run the Notebook:
- Execute the cells sequentially.
- The notebook will:
  - Load and process the dataset
  - Train classification models
  - Evaluate and compare their performance
  - Generate visualizations of the results

### Output Files:
The notebook creates several files:
- `scaler.pkl`: Fitted StandardScaler for feature normalization
- `pca_model.pkl`: Fitted PCA model for dimensionality reduction
- `linear_svm.pkl`: Trained Linear SVM model
- `logistic_regression.pkl`: Trained Logistic Regression model
- `softmax_regression.pkl`: Trained Softmax Regression model
- `decision_tree.pkl`: Trained Decision Tree model
- `weighted_knn.pkl`: Trained Weighted KNN model
- `classification_results.csv`: Performance metrics for all models

## Implementation Details
### Data Preprocessing
#### Feature Scaling:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- Normalizes features to zero mean and unit variance
- Critical for distance-based methods like KNN and gradient-based optimization in SVM and Logistic Regression

#### Dimensionality Reduction:
```python
pca = PCA(n_components=200)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)
```
- Reduces features from 2,048 to 200 dimensions
- Preserves approximately 72% of the variance
- Significantly reduces computation time and memory requirements

### Model Implementation
#### Linear SVM:
```python
OneVsRestClassifier(LinearSVC(max_iter=1))
```
- Uses One-vs-Rest strategy for multi-label classification
- Effective for high-dimensional data

#### Logistic Regression:
```python
OneVsRestClassifier(LogisticRegression(max_iter=1, n_jobs=1))
```
- Provides probability estimates

#### Softmax Regression:
```python
OneVsRestClassifier(LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1, n_jobs=1))
```
- Uses L-BFGS optimization algorithm

#### Decision Tree:
```python
OneVsRestClassifier(DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
))
```
- Limited depth to prevent overfitting

#### Weighted KNN:
```python
OneVsRestClassifier(KNeighborsClassifier(
    weights='distance',
    n_neighbors=5,
    algorithm='auto',
    p=2  # Euclidean distance
))
```
- Uses distance-weighted voting

#### Ensemble Method:
```python
def ensemble_predict(X):
    predictions = np.array([
        trained_models[name].predict(X) if name != 'Weighted KNN' 
        else trained_models['Weighted KNN'].predict(pca.transform(X)) 
        for name in trained_models
    ])
    ensemble_pred = sum(predictions) >= 3  # Majority voting
    return ensemble_pred.astype(int)
```
- Combines predictions from all individual models
- Uses majority voting

## Evaluation Metrics
- **Accuracy**: Exact match accuracy
- **Precision**: Ratio of correctly predicted positive observations to total predicted positives
  - **Micro-averaged**: Metrics globally calculated
  - **Macro-averaged**: Unweighted mean across labels
- **Recall**: Ratio of correctly predicted positive observations to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Hamming Loss**: Fraction of incorrect labels (lower is better)

## Results and Analysis
### Performance Summary:
- **Weighted KNN** achieves the highest accuracy (0.0607)
- **Ensemble** achieves the highest F1-micro score (0.4884)
- **Weighted KNN** has the lowest Hamming Loss (0.0161)

## Checkpointing and Efficiency Features
- **Model Saving**:
```python
joblib.dump(model, f'{name.replace(" ", "_").lower()}.pkl')
```
- **Preprocessor Saving**:
```python
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca_model.pkl')
```
- **Checkpoint Loading**:
```python
if os.path.exists('scaler.pkl'):
    scaler = joblib.load('scaler.pkl')
```

## References  
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)  
- [Linear SVM (LinearSVC) Guide](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)  
- [Logistic Regression Guide](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [Softmax Regression (Multinomial Logistic Regression)](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)  
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)  
- [K-Nearest Neighbors (KNN) Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)  
- [One-vs-Rest (OvR) Strategy](https://scikit-learn.org/stable/modules/multiclass.html#one-vs-the-rest)  
- [Ensemble Methods in Scikit-Learn](https://scikit-learn.org/stable/modules/ensemble.html)  