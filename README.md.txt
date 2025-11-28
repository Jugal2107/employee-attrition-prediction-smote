# Employee Attrition Prediction using SMOTE

This repository contains the code and experiments for my research work on **Predicting Employee Attrition Using SMOTE**.  
The project focuses on building machine learning models that can predict whether an employee is likely to leave the organization, while also addressing the **class imbalance problem** using **SMOTE (Synthetic Minority Oversampling Technique)**.

The repository includes experiments on **two different HR attrition datasets**, allowing a comparison of model performance across datasets and between **original vs. SMOTE-balanced data**.

This work is associated with my conference paper **"Predicting Employee Attrition Using SMOTE"**.

---

## 🎯 Objectives

- Predict employee attrition (Yes/No) using machine learning.
- Handle **highly imbalanced target classes** using **SMOTE**.
- Compare:
  - Different ML algorithms
  - Performance on **two datasets**
  - Results **with and without SMOTE**.

---

## 📂 Datasets

Two datasets are used in this project:

- **Dataset 1** – Employee attrition dataset with multiple demographic, job-related, and satisfaction-related features.
- **Dataset 2** – Another employee attrition dataset with a slightly different feature set for comparison.

---

## 🧠 Models Used

The following machine learning models are trained and evaluated:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVC)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes (GaussianNB)**
- **Gradient Boosting Classifier**
- **XGBoost (XGBClassifier)**
- **LightGBM (LGBMClassifier)**

Models are trained on:
- The **original imbalanced data**
- The **SMOTE-balanced data**

This allows a clear comparison of how SMOTE affects performance.

---

## ⚙️ Machine Learning Pipeline

For each dataset, the typical workflow is:

1. **Data Loading & Exploration**
   - Load CSV data into pandas DataFrame
   - Inspect missing values, distribution, and basic statistics

2. **Data Preprocessing**
   - Encode categorical variables using **LabelEncoder / One-Hot Encoding**
   - Scale numerical features using **StandardScaler**

3. **Train–Test Split**
   - Split data into train and test sets  
     (e.g., 80% train, 20% test with stratification on target)

4. **Handling Class Imbalance with SMOTE**
   - Apply **SMOTE** on the **training data** to oversample the minority class
   - Create:
     - `X_train`, `y_train` (original)
     - `X_train_s`, `y_train_s` (SMOTE-balanced)

5. **Model Training**
   - Train all models on:
     - Original data
     - SMOTE data

6. **Evaluation**
   - Metrics computed:
     - **Accuracy**
     - **Confusion Matrix**
     - **Classification Report** (Precision, Recall, F1-score)
   - Visualizations:
     - Confusion matrix heatmaps
     - Correlation heatmaps
     - Boxplots / distributions for key features

---

## 📊 Evaluation Metrics

For each model and dataset, the notebooks compute:

- `accuracy_score`
- `confusion_matrix`
- `classification_report`

This makes it easy to compare:

- Which model works best
- How SMOTE improves minority class predictions
- How performance differs across **Dataset 1 vs Dataset 2**

You can run the notebooks to reproduce all scores and plots.

---

## 📂 Repository Structure

A suggested structure for this repository:

```bash
employee-attrition-prediction-smote/
│
├── Employee_Attrition_Dataset1.ipynb      # Notebook for Dataset 1
├── Employee_Attrition_Dataset2.ipynb      # Notebook for Dataset 2
│
├── README.md                              # Project documentation (this file)
├── requirements.txt                       # Python dependencies
│
├── data/                                  # (Optional) Sample or small datasets
│   ├── dataset1_sample.csv
│   └── dataset2_sample.csv
│
└── results/                               # Plots, confusion matrices, etc.
    ├── dataset1_confusion_matrix.png
    ├── dataset2_confusion_matrix.png
    └── feature_correlation_heatmap.png

▶️ How to Run

🔹 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/employee-attrition-prediction-smote.git
cd employee-attrition-prediction-smote

🔹 2. (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   

🔹 3. Install dependencies
pip install -r requirements.txt

🔹 4. Run the notebooks
jupyter notebook


📦 Requirements

The project uses the following main libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm


🖼 Visualizations

The notebooks include visualizations such as:

Correlation heatmaps between features

Boxplots for numerical features

Confusion matrix heatmaps for each model

🔍 Key Findings

- SMOTE significantly improved recall for the minority (attrition = "Yes") class.

- Tree-based models like Random Forest and Gradient Boosting performed better than simple models like Logistic Regression on most metrics.

- The two datasets showed slightly different behavior, but SMOTE consistently helped in handling imbalance.

📚 Research Context

This project is part of my research work on:

- HR Analytics

- Imbalanced Classification

- Machine Learning for Employee Retention

- It is directly related to my conference paper:

  "Predicting Employee Attrition Using SMOTE"

This repository demonstrates my ability to:

- Work with real-world, imbalanced tabular data

- Apply appropriate preprocessing and resampling techniques

- Experiment with multiple ML algorithms

- Evaluate and interpret model results in a research context.
