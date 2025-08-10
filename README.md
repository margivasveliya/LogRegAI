--> Brain Cancer Detection using Logistic Regression

This project demonstrates how to use Logistic Regression to classify whether a brain tumor is malignant (cancerous) or benign (non-cancerous) based on patient data.

We use a Brain Cancer Dataset containing different medical features (e.g., cell shape, texture, and other measurements) and train a logistic regression model to make predictions.

---

--> Objective
The goal is simple:
- Train a machine learning model that can predict if a brain tumor is malignant or benign.
- Understand how Logistic Regression works in classification tasks.

---

--> Dataset
We’re using a **Brain Cancer Dataset (CSV file) which contains:
- Input features: Medical test results and tumor characteristics.
- Target variable: Tumor type (`0` = benign, `1` = malignant).

---



--> How It Works
1. Load the dataset (`brain_cancer.csv`)
2. Explore & clean the data (check missing values, types, etc.)
3. Split the dataset into training and testing sets
4. Train the Logistic Regression model using `scikit-learn`
5. Make predictions on the test set
6. Evaluate performance using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

---

--? Example Output
- Accuracy: ~95% (depending on dataset split)
- Confusion Matrix: Shows how many tumors were correctly classified
- Classification Report: Precision, Recall, and F1-score for each class

---
 How to Run This Project
Clone this repository or download the files  
bash
   git clone https://github.com/yourusername/brain-cancer-logistic-regression.git
   cd brain-cancer-logistic-regression
