# LIME Explainability for AnnexML Model on IAPRTC-12 Dataset

This directory contains the implementation of the LIME (Local Interpretable Model-agnostic Explanations) method for explaining the predictions of an AnnexML model trained on the IAPRTC-12 dataset. LIME approximates the behavior of a complex model locally with a more interpretable model.

## Overview

LIME approximates the AnnexML model's behavior locally with a more interpretable model, highlighting feature importance for individual predictions. It helps understand which features most contribute to the model's prediction for a specific instance.

## Repository Contents

*   `train_lime.ipynb`: Jupyter Notebook implementing the LIME explainability method.
*   `AnnexML/`: Contains the AnnexML model files, MATLAB-extracted features, and necessary configuration.
*   `README.md`: This file.

## Setup and Installation

1.  **Clone the repository:**

    ```
    git clone [repository_url]
    cd [repository_name]/lime
    ```

2.  **Install necessary Python packages:**

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
    lime
    ```

3.  **AnnexML Setup**
    * Ensure that you have AnnexML installed and configured correctly. The scripts interact with AnnexML through command-line calls.
    * Place the annexML folder in the root folder. This folder should include:

        *   `AnnexML/src/annexml`: The AnnexML executable.
        *   `AnnexML/annexml-example.json`: A configuration file for AnnexML.
        *   `AnnexML/iaprtc12_model.bin`: The pre-trained AnnexML model.
        *   `AnnexML/iaprtc12_train.txt`: The training file in the format `label index:value`.
        *   `AnnexML/iaprtc12_test.txt`: The testing file in the format `label index:value`.

## Usage

1.  **Data Preparation:** The Jupyter Notebook loads data from `.mat` files within the `AnnexML/` directory.  Ensure the paths are correct.
2.  **Running the Notebook:** Open and execute the Jupyter Notebook (`train_lime.ipynb`).
3.  **Interacting with AnnexML:**
    *   The Jupyter Notebook interacts with the pre-trained AnnexML model via command-line calls using the `subprocess` module.
4.  **Check the results:**
    *   Check for feature importances for each class or class combination, and for generated graphs.

## Key Functions

*   **`annexml_predict(data)`:** This function takes a data instance, formats it into the AnnexML's required input format, saves it to a temporary file, and then executes the AnnexML prediction command-line tool via `subprocess.run`.
*   **`LimeTabularExplainer`:** Initialized using the training data.

## Considerations

*   **Complexity of AnnexML Interaction:** The need to interact with AnnexML via command-line calls adds complexity. Error handling and ensuring proper data formatting are critical.
*   **Computational Cost:** Generating LIME explanations can be computationally expensive, especially for large datasets or complex models.
*   **Interpretation of Multi-label Explanations:** Interpreting feature importances in a multi-label setting can be more challenging than in single-label classification.

## Contributing

Feel free to fork this repository, open issues, and submit pull requests to contribute to this project.

## Citations

* Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
[Why Should I Trust you?]

[Why Should I Trust you?]: https://arxiv.org/abs/1602.04938

* [Explainable AI explained! | #3 LIME][Explainable]
* [AnnexML - Yahoo Japan][AnnexML]

[Explainable]: https://www.youtube.com/watch?v=d6j6bofhj2M&t=627s&pp=ygUEbGltZQ%3D%3D
[AnnexML]: https://github.com/yahoojapan/AnnexML/tree/master
