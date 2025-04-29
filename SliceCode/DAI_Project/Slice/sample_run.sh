#!/bin/bash
set -e

dataset="EURLex-4K"
data_dir="./Sandbox/Data/$dataset"
results_dir="./Sandbox/Results/$dataset"
model_dir="./Sandbox/Results/$dataset/model"
explanation_dir="./Sandbox/Results/$dataset/explanations" #added
mkdir -p $model_dir
mkdir -p $explanation_dir #added

trn_ft_file="${data_dir}/xmlcnn_trn_ft_mat_dense.txt"
trn_lbl_file="${data_dir}/xmlcnn_trn_lbl_mat.txt"
tst_ft_file="${data_dir}/xmlcnn_tst_ft_mat_dense.txt"
tst_lbl_file="${data_dir}/xmlcnn_tst_lbl_mat.txt"
score_file="${results_dir}/score_mat.txt"
explanation_file="${explanation_dir}/explanations.csv" #added


#echo "Converting sparse feature matrices to dense format"
#./Tools/c++/smat_to_dmat [sparse train feature file] $trn_ft_file
#./Tools/c++/smat_to_dmat [sparse test feature file] $tst_ft_file


# echo "----------------Slice--------------------------"
# ./slice_train $trn_ft_file $trn_lbl_file $model_dir -m 100 -c 300 -s 300 -k 300 -o 20 -t 1 -f 0.000001 -siter 20 -b 2 -stype 0 -C 1 -q 0
# ./slice_predict $tst_ft_file $model_dir $score_file
# ./Tools/metrics/precision_k $score_file $tst_lbl_file 5
# ./Tools/metrics/nDCG_k $score_file $tst_lbl_file 5

echo "----------------Robust & Explainable Slice--------------------------"
# Train with adversarial examples and explainability
./slice_train $trn_ft_file $trn_lbl_file $model_dir -m 100 -c 300 -s 300 -k 300 -o 20 -t 1 -f 0.000001 -siter 20 -b 2 -stype 0 -C 1 -q 0 -explain 1 -nfeat 5 -adv 1 -pert 0.1 -debug 0

# Predict with explanations and adversarial defense
./slice_predict $tst_ft_file $model_dir $score_file -explain 1 -nfeat 5 -def 1 -debug 0
# Generate visualizations
python Tools/visualize_explanations.py $explanation_file $explanation_dir

# Evaluate performance
./Tools/metrics/precision_k $score_file $tst_lbl_file 5
./Tools/metrics/nDCG_k $score_file $tst_lbl_file 5

# Evaluate robustness against adversarial examples
echo "Evaluating robustness..."
python Tools/generate_adversarial.py $tst_ft_file $tst_lbl_file "${data_dir}/adversarial_tst_ft.txt"
./slice_predict "${data_dir}/adversarial_tst_ft.txt" $model_dir "${results_dir}/adversarial_score.txt" -def 1
./Tools/metrics/precision_k "${results_dir}/adversarial_score.txt" $tst_lbl_file 5
./Tools/metrics/nDCG_k "${results_dir}/adversarial_score.txt" $tst_lbl_file 5
