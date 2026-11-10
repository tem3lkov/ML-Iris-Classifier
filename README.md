# Machine Learning Iris Classification Pipeline

An automated machine learning pipeline built in Python to classify Iris flower species based on continuous physical measurements. This project demonstrates a structured approach to data analysis, model comparison, and hyperparameter optimization.

<div align="center">
  <img src="https://github.com/user-attachments/assets/01118116-a205-42bb-a270-3bb4c31983a6" width="600" alt="EDA Pairplot">
  <br>
  <i><b>Fig 1. EDA Pairplot:</b> Feature scatter plot demonstrating that the Iris setosa species (top-left cluster) is linearly separable from the other two classes.</i>
  <br><br>
  <img src="https://github.com/user-attachments/assets/19b8ee28-5f82-4b19-9f33-da9a02399198" width="600" alt="CLI Pipeline Output">
  <br>
  <i><b>Fig 2. Pipeline Execution:</b> 10-fold Cross-Validation summary comparing the accuracy and fit time of 5 models, followed by GridSearchCV hyperparameter tuning results for the top performers.</i>
</div>

## How to Run
1. Clone the repository to your local machine.
2. Install the required dependencies (e.g., `pip install -r requirements.txt`).
3. Execute the main script from your terminal: `python iris_classification.py`.
4. Review the terminal output for cross-validation metrics, and check the `outputs/` folder for generated plots (correlation heatmap, confusion matrix) and the serialized `.joblib` model.

## Core Features

- **Exploratory Data Analysis (EDA):** Generated programmatic correlation heatmaps and pairplots (via Seaborn/Matplotlib) to identify linearly separable classes and feature importance.
- **Robust Preprocessing Pipelines:** Utilized `scikit-learn`'s `ColumnTransformer` and `StandardScaler` within a unified Pipeline architecture to prevent data leakage and ensure uniform feature scaling.
- **Comparative Model Evaluation:** Benchmarked multiple classification algorithms (Logistic Regression, kNN, SVM, Random Forest, Gradient Boosting) using 10-fold Stratified Cross-Validation to guarantee unbiased evaluation on balanced classes.
- **Hyperparameter Tuning:** Applied `GridSearchCV` to exhaustively tune parameters for the top-performing models (e.g., `C` and `gamma` for SVM), ultimately achieving a ~97% macro F1-score on an unseen test set.
- **Model Persistence:** Serialized the fully trained and tuned pipeline via `joblib` for reproducible predictions[cite: 7, 15].

## Technologies Used
- **Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn
- **Concepts:** Data Pipelines, Stratified Cross-Validation, GridSearchCV, Feature Scaling, Model Serialization
