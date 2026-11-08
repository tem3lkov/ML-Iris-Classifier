# Machine Learning Iris Classification Pipeline

An automated machine learning pipeline built in Python to classify Iris flower species based on continuous physical measurements. This project demonstrates a structured approach to data analysis, model comparison, and hyperparameter optimization.

## Core Features

* **Exploratory Data Analysis (EDA):** Generated programmatic correlation heatmaps and pairplots (via Seaborn/Matplotlib) to identify linearly separable classes and feature importance.
* **Robust Preprocessing Pipelines:** Utilized `scikit-learn`'s `ColumnTransformer` and `StandardScaler` within a unified Pipeline architecture to prevent data leakage and ensure uniform feature scaling.
* **Comparative Model Evaluation:** Benchmarked multiple classification algorithms (Logistic Regression, kNN, SVM, Random Forest, Gradient Boosting) using 10-fold Stratified Cross-Validation to guarantee unbiased evaluation on balanced classes.
* **Hyperparameter Tuning:** Applied `GridSearchCV` to exhaustively tune parameters for the top-performing models (e.g., `C` and `gamma` for SVM), ultimately achieving a 97.3% macro F1-score on an unseen test set.
* **Model Persistence:** Serialized the fully trained and tuned pipeline via `joblib` for reproducible predictions.

## Technologies Used
* **Language:** Python
* **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn
* **Concepts:** Data Pipelines, Stratified Cross-Validation, GridSearchCV, Feature Scaling, Model Serialization
