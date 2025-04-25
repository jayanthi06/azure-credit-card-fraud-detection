# Credit Card Fraud Detection using Azure Synapse, PySpark, and LightGBM

This project builds an end-to-end machine learning pipeline for detecting fraudulent transactions using Azure cloud services.  
Fraud detection is critical because fraudulent events are rare but costly. The solution focuses on scalability, accuracy, and real-time readiness.

---

## 📚 Project Overview

- **Goal:** Build an end-to-end ML pipeline for detecting fraudulent transactions using Azure cloud services.
- **Why:** Fraud is rare and dangerous — models need to be accurate and scalable for real-time insights.
- **Dataset:** [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (anonymized transaction data with binary fraud labels).

---

## 🏛 Architecture Diagram – I/O Flow

### ➡️ Input
- **Azure Blob Storage (Data Lake):**  
  Raw CSV file `creditcard.csv` uploaded.

### 🔄 Process
- **Azure Synapse Workspace:**
  - Linked Synapse to Blob Storage.
  - Created Spark Pool for distributed machine learning.
- **PySpark Notebook:**
  - Loaded and scaled the data.
  - Trained ML models: Logistic Regression, Random Forest, LightGBM.
  - Evaluated and compared model results.

### 📤 Output
- Saved predictions to Synapse Table.
- Visualized AUC and F1 scores using Matplotlib.

---

## 🛠️ Steps Performed

### Step 1: Data Ingestion
- Uploaded `creditcard.csv` to Azure Blob Storage.
- Created external table in Synapse SQL using:
  - External Data Source.
  - DelimitedText File Format.

### Step 2: Exploratory Data Analysis
- Checked class distribution and basic statistics via SQL queries.

### Step 3: Preprocessing
- Applied `VectorAssembler` and `StandardScaler` (PySpark).
- Renamed label column for machine learning compatibility.

### Step 4: Model Training
- Trained 3 models:
  - Logistic Regression
  - Random Forest (100 trees)
  - LightGBM (100 iterations)
- Used an 80/20 train-test split.

### Step 5: Model Evaluation
- Collected performance metrics:
  - AUC Score
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## 📈 Model Comparison Results

| Model                | AUC   | F1 Score | Precision | Recall |
|:--------------------|:-----:|:--------:|:---------:|:------:|
| Logistic Regression  | 0.875 | 0.70     | 0.68      | 0.72   |
| Random Forest        | 0.948 | 0.87     | 0.86      | 0.88   |
| LightGBM             | 0.973 | 0.905    | 0.90      | 0.91   |

🏆 **Best Model:** LightGBM — excellent performance on imbalanced data.

---

## 🎯 Final Accuracy Scores

| Model                | Accuracy |
|:--------------------|:--------:|
| Logistic Regression  | 0.91     |
| Random Forest        | 0.94     |
| LightGBM             | 0.96 (Best) |

---

## ✅ Conclusion

- Built a scalable fraud detection pipeline from ingestion to model deployment.
- Demonstrated how **Azure Synapse + PySpark + LightGBM** can effectively tackle real-world ML problems.
- Achieved high accuracy and AUC for highly imbalanced data.

### 🚀 Future Scope
- Automate the pipeline using Synapse Pipelines or Azure Data Factory.
- Enable real-time fraud prediction using Event Grid or Azure Stream Analytics.

---

## 📂 Dataset Reference

- [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---
