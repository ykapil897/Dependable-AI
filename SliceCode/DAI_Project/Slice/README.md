
Key parameters:
- `-explain 1`: Generate explanations for predictions
- `-nfeat 5`: Number of features to include in explanations
- `-def 1`: Enable adversarial defense (feature squeezing)

## Results

The enhanced SLICE model achieves:
- High Precision@1 of 0.773609 on the EURLex-4K dataset
- Exceptional robustness against adversarial examples, with negligible performance drop
- Fast prediction speed (1.51ms per instance)
- Comprehensive explainability through feature importance visualizations

## Documentation

For detailed information about the implementation and experimental results, refer to `DAI_Project.pdf`.

## Requirements

- C++ compiler with C++11 support
- Python 3.x with matplotlib, numpy, pandas, and seaborn libraries
- NMSLIB for approximate nearest neighbor search

## Compilation

The code can be compiled using:


2. Run the training with robustness and explainability:
./slice_train [trn_ft_file] [trn_lbl_file] [model_dir] -adv 1 -explain 1


3. Run prediction with defense and explanation generation:
./slice_predict [tst_ft_file] [model_dir] [score_file] -def 1 -explain 1


4. Generate visualizations:
python Tools/visualize_explanations.py [explanation_file] [output_dir]

5. Evaluate performance:
./Tools/metrics/precision_k [score_file] [tst_lbl_file] 5 ./Tools/metrics/nDCG_k [score_file] [tst_lbl_file] 5


## Parameters

### Training Parameters
- `-m`: HNSW M parameter (default=100)
- `-c`: HNSW efConstruction parameter (default=300)
- `-explain`: Enable explanation generation (default=1)
- `-adv`: Enable adversarial training (default=0)
- `-pert`: Perturbation strength for adversarial training (default=0.1)

### Prediction Parameters
- `-explain`: Enable explanation generation (default=1)
- `-nfeat`: Number of top features to explain per label (default=5)
- `-def`: Enable adversarial defense (default=0)

## Results

Our enhanced SLICE model demonstrates exceptional robustness with virtually no degradation in performance when tested against adversarial examples. The model achieves a Precision@1 of 0.773609 on clean data vs. 0.773868 on adversarial data.

The feature squeezing defense mechanism effectively neutralizes adversarial perturbations with minimal computational overhead (only ~0.2 seconds for 3,865 test instances).

The explanation system successfully identifies the most influential features for each prediction, providing transparency without sacrificing prediction speed (only ~1.51ms per instance).