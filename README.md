# AI_Code_Completion_Task
Predictive analytics using breast cancer data — Task 3

# Task 3 – Predictive Analytics for Resource Allocation

This project demonstrates a complete machine learning pipeline for classifying issue priority using the **Breast Cancer Wisconsin dataset**. It's part of **Week 4: AI in Software Engineering**, showcasing real-world predictive modeling for resource allocation decisions.

---

## 🧠 Problem Statement

Given diagnostic features in the dataset, the goal is to predict the **priority level** (High/Low) of medical cases to assist in efficient triage and resource management.

---

## 🔍 Dataset

- **Source:** [Kaggle Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Target Mapping:**
  - `diagnosis = M` → `priority = high`
  - `diagnosis = B` → `priority = low`

---

## 🛠️ Workflow

1. **Data Loading & Preprocessing**
   - Cleaned the dataset, dropped unused columns
   - Created a new `priority` column from `diagnosis`
   - Encoded the `priority` label for training

2. **Splitting Data**
   - 80/20 split using `train_test_split`

3. **Model Training**
   - Used `RandomForestClassifier` from `scikit-learn`

4. **Evaluation**
   - Calculated **accuracy**, **F1-score**, and a detailed **classification report**

5. **Model Persistence**
   - Saved the trained model with `joblib` for reuse

---

## 📈 Results

- High accuracy and balanced F1-score achieved
- Effective for binary classification (high vs low priority)

---

## 📦 Files Included

- `task3_predictive_model.ipynb`: Full Jupyter Notebook pipeline
- `priority_classifier_model.pkl`: Trained and saved model file

---

## 🚀 How to Run

1. Clone the repository
2. Open the notebook in VS Code or Jupyter
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib

