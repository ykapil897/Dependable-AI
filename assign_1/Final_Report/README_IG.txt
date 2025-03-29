# Integrated Gradients Explainability for AnnexML Model on IAPRTC-12 Dataset

This directory contains the implementation of the Integrated Gradients (IG) method for explaining the predictions of an AnnexML model trained on the IAPRTC-12 dataset.  Integrated Gradients is a feature attribution method that assigns importance scores to each feature.

## Overview

Integrated Gradients assigns importance scores to each feature by accumulating gradients along a path from a baseline input (usually a zero vector) to the actual input. It helps understand which features most contribute to the model's prediction. The model in question is AnnexML, trained to classify images in the IAPRTC-12 multi-label dataset.

## Directory contents

*   `train_ig.ipynb`: Jupyter Notebook implementing the Integrated Gradients explainability method.
*   `AnnexML/`: Used AnnexML model files, MATLAB-extracted features, and necessary configuration.
*   `I_Gradients/`: A directory that will be created and used to store intermediate files (interpolated inputs, prediction outputs, and the `predictions_df`).
*   `README.md`: This file.

## Setup and Installation

1.  **Install necessary Python packages:**

    It's highly recommended to create a virtual environment.

    ```
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows

    pip install -r requirements.txt
    ```

    Create a `requirements.txt` file with the following dependencies:

    ```
    scikit-learn
    numpy
    scipy
    pandas
    matplotlib
    ```

2.  **AnnexML Setup:**

    *   Ensure that you have AnnexML installed and configured correctly. The scripts interact with AnnexML through command-line calls.
    *   Place the AnnexML folder in the root folder. This folder should include:

        *   `AnnexML/src/annexml`: The AnnexML executable.
        *   `AnnexML/annexml-example.json`: A configuration file for AnnexML.
        *   `AnnexML/iaprtc12_model.bin`: The pre-trained AnnexML model.
        *   `AnnexML/iaprtc12_train.txt`: The training file in the format `label index:value`.
        *   `AnnexML/iaprtc12_test.txt`: The testing file in the format `label index:value`.

## Usage

1.  **Data Preparation:** Ensure the data files (`iaprtc12_train.txt` and `iaprtc12_test.txt`) are correctly placed within the `AnnexML/` directory and the paths specified in the notebook are accurate.
2.  **Running the Notebook:** Open and execute the Jupyter Notebook (`train_ig.ipynb`). The notebook will:

    *   Load and preprocess the data.
    *   Define the `annexml_predict_svm` function to interface with the AnnexML model.
    *   Calculate Integrated Gradients using the `integrated_gradients_annexml` function.
    *   Generate and visualize feature importances.
3.  **AnnexML Configuration:** Carefully verify the paths to the AnnexML executable, model file, and data files within the `annexml_predict_svm` function in the notebook:

    ```
    subprocess.run(["AnnexML/src/annexml", "predict", "AnnexML/annexml-example.json", f"predict_file={input_file}", f"result_file={output_file}", "model_file=AnnexML/iaprtc12_model.bin"], check=True)
    ```
    *   Make sure the `check=True` flag is enabled to check for error.
4.  **Directory:**

    *   The `integrated_gradients_annexml` function saves intermediate data to the `I_Gradients/` directory. Ensure this directory exists, or the code will create it if it doesn't.

## Key Functions

*   **`save_instance_as_svm(instance, file_path, label=None)`:** Saves a data instance in SVM-light format for AnnexML.
*   **`annexml_predict_svm(input_file, output_file)`:** Runs the AnnexML prediction tool and parses the output.
*   **`integrated_gradients_annexml(sample_instance, baseline=None, steps=50, delta=1e-3)`:** Implements the Integrated Gradients algorithm.

## Analysis

The Notebook computes feature importances for each class in the IAPRTC-12 dataset.  It identifies the most influential features and ranks the classes based on their maximum feature attribution values.  The bar chart visualization illustrates these results.

## Considerations

*   **AnnexML Dependency:**  This implementation relies heavily on AnnexML.  Ensure that AnnexML is correctly installed and configured.
*   **Computational Intensity:**  Integrated Gradients can be computationally expensive. Adjust the `steps` parameter in `integrated_gradients_annexml` to balance accuracy and runtime.
*   **File I/O:** The implementation involves frequent file reads and writes to interact with AnnexML.  Optimize file paths and I/O operations if necessary.
*   **Multi-label Interpretation:** Interpreting feature attributions in multi-label settings requires domain expertise.

## Contributing

Contributions are welcome!  Please submit pull requests with improvements, bug fixes, or new features.

## Citation

* [Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of the 34th International Conference on Machine Learning*, *70*, 3319-3328. ][AxiomaticAttribution]

[AxiomaticAttribution]: https://arxiv.org/abs/1703.01365

* [NeurIPS*2021 - Fast Axiomatic Attributions for Neural Networks][NeurIPS]

* [Integrated Gradients - GitHub][IntegratedGradients]

[NeurIPS]: https://www.youtube.com/watch?v=AHkeeyoNGp8  
[IntegratedGradients]: https://github.com/ankurtaly/Integrated-Gradients
