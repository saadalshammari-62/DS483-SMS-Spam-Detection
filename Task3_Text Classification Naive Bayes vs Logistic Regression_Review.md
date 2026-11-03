Task 3 Review: Naive Bayes and Logistic Regression Classification
Contributor: Sanad alyafei github: snd-y-al 
Repository: https://github.com/saadalshammari-62/DS483-SMS-Spam-Detection
Date: 2026-11-03

Objective
Review and execute the Task 3 code from DS483_SMS_Spam_Project.ipynb
the text classification pipeline combining TfidfVectorizer with MultinomialNB and LogisticRegression and cross-validation using StratifiedKFold 5-fold hyperparameter tuning via GridSearchCV and performance metrics evaluation
Every metric is verified against results.json and matched with Section 3 of the project report DS483_Project_Reordered.docx

The setup and dataset split and preprocessing stages are reused directly from the core pipeline to maintain strict consistency

1. Model Pipeline and Grid Search Setup

import json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

REF = json.load(open("results.json"))

nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True)),
    ("clf", MultinomialNB())
])

lr_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000))
])

nb_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]
}

lr_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "clf__C": [1, 10, 100],
    "clf__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

2. Reproducibility Check Against results.json

| Model               | Metric / Parameter    | Task 3 Code Value        | Stored results.json Value | Status     |
|:--------------------|:----------------------|:------------------------:|:-------------------------:|:----------:|
| **Naive Bayes**     | Best Hyperparameters  | `{'clf__alpha': 0.1, ...}` | `{'clf__alpha': 0.1, ...}` | ✅ Matched |
| **Naive Bayes**     | Best CV F1-Score      | `0.9321`                 | `0.9321`                  | ✅ Matched |
| **Naive Bayes**     | Test Precision (`spam`) | `0.9667`               | `0.9667`                  | ✅ Matched |
| **Naive Bayes**     | Test Recall (`spam`)  | `0.8878`                 | `0.8878`                  | ✅ Matched |
| **Naive Bayes**     | Test F1-Score (`spam`) | `0.9255`                | `0.9255`                  | ✅ Matched |
| **Logistic Regression** | Best Hyperparameters | `{'clf__C': 100, ...}` | `{'clf__C': 100, ...}`    | ✅ Matched |
| **Logistic Regression** | Best CV F1-Score  | `0.9412`                 | `0.9412`                  | ✅ Matched |
| **Logistic Regression** | Test Precision (`spam`) | `0.9778`           | `0.9778`                  | ✅ Matched |
| **Logistic Regression** | Test Recall (`spam`) | `0.8980`              | `0.8980`                  | ✅ Matched |
| **Logistic Regression** | Test F1-Score (`spam`) | `0.9362`            | `0.9362`                  | ✅ Matched |

3. Key Findings and Technical Observations

No Data Leakage: Integrating TfidfVectorizer directly inside the Pipeline ensures vocabulary construction and IDF computation occur strictly within each training fold of the 5-fold cross-validation

Handling Class Imbalance: Optimizing GridSearchCV explicitly for the f1 score of the positive class spam ~13% of dataset prevented the models from biasing toward the majority class ham

Model Superiority: Logistic Regression ($C=100$) outperformed Naive Bayes $\alpha=0.1$ on both CV F1-score 0.9412 vs 0.9321 and Test F1-score 0.9362 vs 0.9255

Comparison with Lexical Baseline: Both trained linear models significantly surpass the baseline 1-NN lexical approach from Task 2 Test F1: 0.8817 proving that statistical feature weighting improves generalization over simple vector distance matching

