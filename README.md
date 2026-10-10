# CodeAlpha Credit Scoring Model

A machine learning project developed as part of the **CodeAlpha Machine Learning Internship – Task 1**.

The goal of this project is to predict whether a person is likely to experience **serious delinquency within the next two years** based on historical financial information.

---

## 1. Project Overview

Credit risk assessment is an important problem in the financial sector. Traditional credit assessment can involve many financial factors such as payment history, debt ratio, income, and credit utilization.

In this project, machine learning classification algorithms are used to learn patterns from historical credit data and predict the risk of serious delinquency.

The project includes:

* Data inspection and cleaning
* Exploratory Data Analysis (EDA)
* Missing-value handling
* Train-test splitting
* Multiple classification models
* Model comparison
* Random Forest hyperparameter tuning
* Decision-threshold optimization
* Final model evaluation
* Feature importance analysis
* A command-line prediction system

---

## 2. CodeAlpha Task

**Task:** Credit Scoring Model

The CodeAlpha task requires building a machine learning model that predicts creditworthiness using historical financial data.

The project uses classification techniques and evaluates the models using metrics such as:

* Precision
* Recall
* F1-Score
* ROC-AUC

---

## 3. Dataset

The project uses the **Give Me Some Credit** dataset.

The dataset contains 150,000 records and 12 columns, including the target variable and an ID column.

### Target Variable

`SeriousDlqin2yrs`

* `0` → No serious delinquency
* `1` → Serious delinquency

The dataset is imbalanced:

* No serious delinquency: **93.32%**
* Serious delinquency: **6.68%**

Because of this class imbalance, accuracy alone was not used to judge the model.

---

## 4. Features

The model uses the following financial and demographic features:

| Feature                                | Description                                               |
| -------------------------------------- | --------------------------------------------------------- |
| `RevolvingUtilizationOfUnsecuredLines` | Credit utilization relative to available unsecured credit |
| `age`                                  | Age of the applicant                                      |
| `NumberOfTime30-59DaysPastDueNotWorse` | Number of times payments were 30–59 days late             |
| `DebtRatio`                            | Debt-to-income related ratio                              |
| `MonthlyIncome`                        | Monthly income                                            |
| `NumberOfOpenCreditLinesAndLoans`      | Number of open credit lines and loans                     |
| `NumberOfTimes90DaysLate`              | Number of times payments were 90+ days late               |
| `NumberRealEstateLoansOrLines`         | Number of real estate loans or lines                      |
| `NumberOfTime60-89DaysPastDueNotWorse` | Number of times payments were 60–89 days late             |
| `NumberOfDependents`                   | Number of dependents                                      |

The dataset ID column `Unnamed: 0` was removed because it does not provide useful predictive information.

---

## 5. Data Preprocessing

The following preprocessing steps were performed:

1. Removed the unnecessary ID column.
2. Separated the target variable from the input features.
3. Split the data into training and testing sets using an 80/20 stratified split.
4. Used median imputation for missing values.
5. Used scaling for Logistic Regression.
6. Used class weighting for the classification models to handle class imbalance.

The training set contains **120,000 records** and the test set contains **30,000 records**.

---

## 6. Exploratory Data Analysis

EDA was performed to understand the dataset before training the models.

Important observations included:

* The target variable is highly imbalanced.
* Monthly income contains a significant number of missing values.
* Number of dependents also contains missing values.
* Credit utilization and debt ratio contain highly skewed and extreme values.
* Previous delinquency-related features show noticeable differences between the two target classes.
* The delinquency-related variables are among the most informative features for prediction.

EDA visualizations are stored in the `outputs` directory.

---

## 7. Models Used

Three classification algorithms were evaluated:

### Logistic Regression

Used as a baseline linear classification model.

### Decision Tree

Used to capture nonlinear relationships between financial features and credit risk.

### Random Forest

Used as the main tree-based ensemble model because it can capture more complex patterns and generally performs well on structured tabular data.

---

## 8. Baseline Model Comparison

The initial models produced the following results:

| Model               |   Accuracy |  Precision | Recall |   F1-Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | -----: | ---------: | ---------: |
| Logistic Regression |     77.62% |     18.16% | 66.98% |     28.58% |     80.21% |
| Decision Tree       |     78.44% |     20.51% | 77.41% |     32.42% |     84.95% |
| Random Forest       | **82.56%** | **23.57%** | 71.72% | **35.48%** | **86.26%** |

The baseline Random Forest performed best overall, especially in terms of ROC-AUC and F1-Score.

---

## 9. Random Forest Tuning

RandomizedSearchCV was used to improve the Random Forest model.

The final selected parameters were:

```text
n_estimators = 150
max_depth = 8
min_samples_split = 5
min_samples_leaf = 1
max_features = log2
class_weight = balanced
```

The tuned model achieved:

```text
ROC-AUC = 86.55%
```

The tuning improved the model's ability to identify risky cases and slightly improved ROC-AUC.

---

## 10. Decision Threshold Optimization

The default classification threshold of 0.50 was not used as the final threshold.

Different thresholds were evaluated to find a better balance between precision and recall.

The threshold **0.75** produced the highest F1-Score among the tested thresholds.

Final threshold:

```text
0.75
```

This means the model predicts serious delinquency when the predicted probability reaches at least 75%.

---

## 11. Final Model Results

The final Random Forest model was evaluated on the held-out test set.

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **91.40%** |
| Precision | **38.78%** |
| Recall    | **49.58%** |
| F1-Score  | **43.52%** |
| ROC-AUC   | **86.55%** |

### Classification Report

```text
                        precision    recall  f1-score   support

No Serious Delinquency       0.96      0.94      0.95     27995
Serious Delinquency          0.39      0.50      0.44      2005

accuracy                                0.91     30000
macro avg                   0.68      0.72      0.69     30000
weighted avg                0.92      0.91      0.92     30000
```

---

## 12. Confusion Matrix

The final confusion matrix was:

```text
[[26426  1569]
 [ 1011   994]]
```

This represents:

* 26,426 correctly predicted non-delinquent cases
* 1,569 false positive predictions
* 1,011 missed delinquent cases
* 994 correctly detected delinquent cases

The confusion matrix visualization is available in:

```text
outputs/final_confusion_matrix.png
```

---

## 13. Feature Importance

The Random Forest model identified the following features as the most important:

| Feature                              | Importance |
| ------------------------------------ | ---------: |
| RevolvingUtilizationOfUnsecuredLines |     38.63% |
| NumberOfTimes90DaysLate              |     20.64% |
| NumberOfTime30-59DaysPastDueNotWorse |     19.66% |
| NumberOfTime60-89DaysPastDueNotWorse |      9.99% |
| age                                  |      4.30% |
| DebtRatio                            |      1.96% |
| NumberOfOpenCreditLinesAndLoans      |      1.73% |
| NumberRealEstateLoansOrLines         |      1.51% |
| MonthlyIncome                        |      1.36% |
| NumberOfDependents                   |      0.23% |

The results indicate that previous payment behavior and credit utilization are particularly important for predicting serious delinquency in this dataset.

---

## 14. Prediction System

A command-line prediction system was created using the trained model.

Run:

```powershell
python src\predict.py
```

The system asks for applicant information such as:

* Age
* Credit utilization
* Payment delays
* Debt ratio
* Monthly income
* Open credit lines
* Number of 90-day late payments
* Real estate loans
* Number of dependents

The trained model then calculates a risk probability and provides a prediction.

Example:

```text
Risk Probability : 40.10%
Decision Threshold: 0.75

Prediction:
NO SERIOUS DELINQUENCY RISK
```

The trained model is stored in:

```text
models/credit_scoring_model.pkl
```

The decision threshold is stored in:

```text
models/decision_threshold.txt
```

---

## 15. Project Structure

```text
CodeAlpha_Credit_Scoring_Model/
│
├── data/
│   └── raw/
│       └── cs-training.csv
│
├── models/
│   ├── credit_scoring_model.pkl
│   └── decision_threshold.txt
│
├── notebooks/
│
├── outputs/
│   ├── final_confusion_matrix.png
│   ├── feature_importance.png
│   └── feature_importance.csv
│
├── src/
│   ├── inspect_data.py
│   ├── eda.py
│   ├── preprocess_data.py
│   ├── train_baseline_models.py
│   ├── tune_random_forest.py
│   ├── optimize_threshold.py
│   ├── final_evaluation.py
│   ├── save_model.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## 16. Installation

Clone the repository and open the project directory.

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## 17. Running the Project

### Inspect the dataset

```powershell
python src\inspect_data.py
```

### Run EDA

```powershell
python src\eda.py
```

### Preprocess the data

```powershell
python src\preprocess_data.py
```

### Train baseline models

```powershell
python src\train_baseline_models.py
```

### Tune Random Forest

```powershell
python src\tune_random_forest.py
```

### Optimize the decision threshold

```powershell
python src\optimize_threshold.py
```

### Evaluate the final model

```powershell
python src\final_evaluation.py
```

### Save the final model

```powershell
python src\save_model.py
```

### Run the prediction system

```powershell
python src\predict.py
```

---

## 18. Limitations

This project is an educational machine learning implementation and should not be treated as a production credit approval system.

Some limitations include:

* The dataset represents historical credit behavior and may not represent every population.
* The target classes are highly imbalanced.
* Model predictions are probabilistic rather than guaranteed outcomes.
* The selected threshold reflects the evaluation performed in this project and may need to change for different business requirements.
* Feature importance from Random Forest indicates model influence, not causation.
* Additional fairness, calibration, validation, security, and regulatory checks would be required for real financial deployment.

---

## 19. Future Improvements

Possible future improvements include:

* Probability calibration
* More advanced ensemble models
* Additional feature engineering
* Cross-validation with more extensive hyperparameter optimization
* Explainable AI techniques such as SHAP
* A web-based prediction interface
* Model monitoring and drift detection
* Fairness and bias evaluation

---

## 20. Conclusion

This project demonstrates a complete machine learning workflow for credit risk prediction, starting from raw financial data and ending with a trained and saved prediction model.

Random Forest provided the strongest overall baseline performance, and further threshold optimization improved the F1-Score to **43.52%** while maintaining a ROC-AUC of **86.55%**.

The final system can accept applicant financial information and generate a probability-based credit risk prediction.

**Developed as part of the CodeAlpha Machine Learning Internship.**
#   c o d e a l p h a _ t a s k s  
 