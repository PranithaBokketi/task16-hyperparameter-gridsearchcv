Breast Cancer Classification – Hyperparameter Tuning with GridSearchCV
Overview
This project uses the Breast Cancer dataset from sklearn.datasets to build and optimize an SVM classifier using GridSearchCV. The goal is to compare the performance of a default SVC model with a tuned SVC model and understand the impact of hyperparameter tuning on classification performance.

Dataset
Source: sklearn.datasets.load_breast_cancer

Samples: 569

Features: 30 numeric features describing cell nuclei (mean radius, mean texture, etc.)

Target: Binary label – 0 (benign), 1 (malignant)
​

Project Steps
Data loading and exploration

Loaded the Breast Cancer dataset using load_breast_cancer.

Converted data to a Pandas DataFrame and inspected feature distributions and class balance.
​

Preprocessing

Used the raw numerical features without heavy preprocessing (dataset is already clean and numeric).
​

Split data into train and test sets with train_test_split (test_size=0.2, stratify=y, random_state=42).
​

Baseline model (Default SVC)

Trained an SVM classifier with default hyperparameters: SVC().

Evaluated on the test set using accuracy, precision, recall, f1-score, and confusion matrix.

Hyperparameter tuning with GridSearchCV

Chosen model: SVM with RBF kernel (SVC).
​

Defined parameter grid:

python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"]
}
Applied GridSearchCV with:

cv=5 (5-fold cross-validation),

scoring="accuracy",

n_jobs=-1 for parallel computation.
​

Obtained:

Best params: {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}

Best cross‑validation accuracy: 0.9341 (approx).

Tuned model evaluation

Used grid_search.best_estimator_ as the final tuned model.

Evaluated on the same test set with the same metrics.

Results
Classification reports (summary)
Tuned SVC (GridSearchCV)

Test accuracy: 0.8947

Benign (0): precision 0.81, recall 0.93, f1-score 0.87

Malignant (1): precision 0.95, recall 0.88, f1-score 0.91

Default SVC

Test accuracy: 0.9298

Benign (0): precision 0.95, recall 0.86, f1-score 0.90

Malignant (1): precision 0.92, recall 0.97, f1-score 0.95

Performance comparison
Model	Test Accuracy
Default SVC	0.9298
Tuned SVC (GridSearchCV)	0.8947
In this experiment, the default SVC slightly outperformed the tuned SVC in terms of raw test accuracy, which highlights that hyperparameter tuning does not always guarantee better performance on a particular train–test split and may still be affected by variance in the data.
​

Files in this Repository
hyperparameter.ipynb – Main notebook with:

Data loading and exploration.

Train–test split.

Baseline SVC training and evaluation.

GridSearchCV setup, training, and results.

Tuned model evaluation and comparison table.

model_comparison.ipynb (if present) – Additional experiments or comparisons.

README.md – Project documentation (this file).

How to Run
Clone the repository:

bash
git clone https://github.com/PranithaBokketi/task16-hyperparameter-gridsearchcv.git
cd task16-hyperparameter-gridsearchcv
Install dependencies (example):

bash
pip install -r requirements.txt
Minimum packages:

scikit-learn

pandas

numpy

jupyter

Open the notebook:

bash
jupyter notebook
Run all cells in hyperparameter.ipynb.

Key Learnings
Understood what hyperparameters are and how they control model behavior (e.g., C, gamma for SVM).

Learned how to use GridSearchCV with cross-validation to search for the best hyperparameter combination.
​

Observed that tuning can sometimes lead to trade‑offs and that performance should be evaluated carefully using train/validation/test splits rather than relying only on cross‑validation scores.
