# CodeAlpha Credit Scoring Model

A machine learning project that predicts the risk of serious credit delinquency using historical financial and credit-related data.

This project was developed as part of the **CodeAlpha Machine Learning Internship**.

## 1. Project Overview

Banks and financial institutions need to understand the potential risks associated with lending money. This project uses machine learning classification algorithms to estimate whether an individual may experience serious credit delinquency.

The model learns patterns from historical credit data and predicts a risk probability based on an applicant's financial information and credit history.

**Project objective:** Build and evaluate a credit-risk classification model using machine learning.

> Note: This project is an educational prototype. Its predictions should not be used as the sole basis for real financial or lending decisions.

## 2. Dataset

This project uses the Give Me Some Credit dataset.

* Total records: 150,000
* Original columns: 12
* Input features used: 10
* Target column: `SeriousDlqin2yrs`

### Target Variable

| Value | Meaning                                                     |
| ----- | ----------------------------------------------------------- |
| 0     | No recorded serious delinquency within the specified period |
| 1     | Serious delinquency within the specified period             |

### Class Distribution

| Class                  | Number of Records | Percentage |
| ---------------------- | ----------------: | ---------: |
| No serious delinquency |           139,974 |     93.32% |
| Serious delinquency    |            10,026 |      6.68% |

The dataset is imbalanced because the majority of records belong to class 0. Therefore, multiple evaluation metrics are used instead of relying only on accuracy.

## 3. Features Used

The model uses the following 10 input features:

| Feature                                | Description                                        |
| -------------------------------------- | -------------------------------------------------- |
| `RevolvingUtilizationOfUnsecuredLines` | Utilization of unsecured revolving credit          |
| `age`                                  | Age of the individual                              |
| `NumberOfTime30-59DaysPastDueNotWorse` | Number of times payments were 30–59 days past due  |
| `DebtRatio`                            | Reported debt ratio                                |
| `MonthlyIncome`                        | Monthly income                                     |
| `NumberOfOpenCreditLinesAndLoans`      | Number of open credit lines and loans              |
| `NumberOfTimes90DaysLate`              | Number of times payments were 90 or more days late |
| `NumberRealEstateLoansOrLines`         | Number of real estate loans or credit lines        |
| `NumberOfTime60-89DaysPastDueNotWorse` | Number of times payments were 60–89 days past due  |
| `NumberOfDependents`                   | Number of dependents                               |

The original `Unnamed: 0` index column is removed during preprocessing.

## 4. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Git and GitHub
* Visual Studio Code

## 5. Project Workflow

The project follows these steps:

1. Inspect and understand the dataset.
2. Perform exploratory data analysis (EDA).
3. Handle missing values and prepare the features.
4. Split the data into training and testing sets.
5. Train and compare classification models.
6. Tune the Random Forest model.
7. Optimize the classification decision threshold.
8. Evaluate the final model.
9. Analyze feature importance.
10. Save the trained model.
11. Predict credit delinquency risk for new input data.

## 6. Data Preprocessing

The following preprocessing steps were performed:

* Removed the unnecessary index column.
* Separated input features and the target variable.
* Split the dataset into training and testing sets.
* Used stratified sampling to preserve the target-class proportions.
* Handled missing numerical values using median imputation.
* Used class weighting to help address the imbalanced target distribution.

### Training and Testing Split

| Dataset      | Records | Percentage |
| ------------ | ------: | ---------: |
| Training set | 120,000 |        80% |
| Testing set  |  30,000 |        20% |
| Total        | 150,000 |       100% |

The split uses a fixed random state to make the results reproducible.

The preprocessing pipeline learns imputation values from the training data and applies them to the test data.

## 7. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify patterns.

The analysis includes:

* Target-class distribution.
* Feature distributions.
* Correlation between numerical features.
* Boxplots for selected numerical variables.
* Comparisons between records with different target labels.

The analysis showed that the dataset contains highly skewed values and a significant imbalance between the two target classes.

EDA visualizations are available in the `outputs/` directory.

## 8. Machine Learning Models

Three classification models were trained and compared.

### Logistic Regression

Used as a baseline classification model to estimate the probability of serious delinquency.

### Decision Tree

Uses feature-based decision rules to classify records.

### Random Forest

Combines multiple decision trees to capture relationships between features and produce predictions.

The models were evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## 9. Baseline Model Comparison

The following results were obtained during baseline evaluation:

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   77.62% |    18.16% | 66.98% |   28.58% |  80.21% |
| Decision Tree       |   78.44% |    20.51% | 77.41% |   32.42% |  84.95% |
| Random Forest       |   82.56% |    23.57% | 71.72% |   35.48% |  86.26% |

The baseline Random Forest achieved the highest F1-score and ROC-AUC among the three models, so it was selected for further experimentation.

## 10. Random Forest Hyperparameter Tuning

Random Forest hyperparameters were tuned using `RandomizedSearchCV` with three-fold cross-validation.

The search evaluated 10 candidate configurations using ROC-AUC as the scoring metric.

The selected parameters were:

* Number of trees: 150
* Maximum tree depth: 8
* Minimum samples required for a split: 5
* Minimum samples required at a leaf: 1
* Maximum features considered at each split: `log2`
* Class weight: `balanced`

The best cross-validation ROC-AUC was approximately **86.01%**.

### Tuned Model Results at Threshold 0.50

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 79.18% |
| Precision | 21.23% |
| Recall    | 78.00% |
| F1-score  | 33.37% |
| ROC-AUC   | 86.55% |

Tuning improved ROC-AUC and recall compared with the baseline Random Forest, but some other metrics decreased.

## 11. Decision Threshold Optimization

The tuned Random Forest produces a probability estimate. A decision threshold determines how that probability is converted into a class prediction.

Different thresholds were tested, and the threshold of **0.75** produced the highest F1-score among the tested values.

### Final Model Results at Threshold 0.75

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 91.40% |
| Precision | 38.78% |
| Recall    | 49.58% |
| F1-score  | 43.52% |
| ROC-AUC   | 86.55% |

The threshold of 0.75 improved precision and F1-score compared with using a threshold of 0.50.

However, recall decreased. The selected threshold is therefore a trade-off, not necessarily the best choice for every lending scenario.

The final threshold was selected based on the F1-score among the tested thresholds.

## 12. Final Model Evaluation

The final model was evaluated on the held-out test set of 30,000 records.

### Confusion Matrix

|                             | Predicted: No Delinquency | Predicted: Serious Delinquency |
| --------------------------- | ------------------------: | -----------------------------: |
| Actual: No Delinquency      |                    26,426 |                          1,569 |
| Actual: Serious Delinquency |                     1,011 |                            994 |

The model correctly identified 994 serious-delinquency cases and missed 1,011 cases.

It also incorrectly classified 1,569 records as serious delinquency.

This illustrates why precision, recall, and the confusion matrix are important when evaluating an imbalanced classification problem.

The final confusion matrix is saved as:

`outputs/final_confusion_matrix.png`

## 13. Feature Importance

The Random Forest feature-importance analysis identified the following important features:

| Feature                                  | Importance |
| ---------------------------------------- | ---------: |
| Revolving Utilization of Unsecured Lines |     0.3863 |
| Number of Times 90 Days Late             |     0.2064 |
| Number of Times 30–59 Days Past Due      |     0.1966 |
| Number of Times 60–89 Days Past Due      |     0.0999 |
| Age                                      |     0.0430 |

Revolving credit utilization had the highest feature importance in this model.

Feature importance describes how features contribute to the Random Forest's decisions. It does not establish causation or guarantee that the model is fair.

The feature-importance chart and CSV file are available in the `outputs/` directory.

## 14. Prediction System

The project includes a command-line prediction program.

The program:

1. Loads the saved model.
2. Reads the saved decision threshold.
3. Collects the applicant's financial information.
4. Estimates the probability of serious delinquency.
5. Compares the probability with the decision threshold.
6. Displays the predicted class.

### Saved Model Files

* `models/credit_scoring_model.pkl`
* `models/decision_threshold.txt`

The saved pipeline includes median imputation and the tuned Random Forest classifier.

### Example Prediction

An earlier test produced the following result:

```text
CREDIT RISK PREDICTION

Risk Probability : 40.10%
Decision Threshold: 0.75

Prediction:
NO SERIOUS DELINQUENCY RISK
```

Since the estimated probability was below the threshold of 0.75, the model predicted class 0.

The exact prediction may vary if the model or input values change.

This prediction is not a guarantee of future financial behaviour.

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
│   ├── baseline_model_results.csv
│   ├── target_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
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
├── .gitignore
├── README.md
└── requirements.txt
```

## 16. Installation

### Prerequisites

* Python installed on your system.
* Visual Studio Code or another Python development environment.
* Git, if you want to clone or contribute to the repository.

### Clone the Repository

```bash
git clone https://github.com/codebyprudhvi-web/codealpha_tasks.git
```

Open the downloaded repository in VS Code.

**Note:** This repository may contain multiple internship tasks. Navigate to the Credit Scoring Model project folder before running the commands below.

### Create and Activate a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## 17. Running the Project

Run these commands from the Credit Scoring Model project directory.

### Step 1: Inspect the Dataset

```powershell
python src\inspect_data.py
```

### Step 2: Perform Exploratory Data Analysis

```powershell
python src\eda.py
```

### Step 3: Preprocess the Data

```powershell
python src\preprocess_data.py
```

### Step 4: Train and Compare Baseline Models

```powershell
python src\train_baseline_models.py
```

### Step 5: Tune the Random Forest

```powershell
python src\tune_random_forest.py
```

### Step 6: Optimize the Decision Threshold

```powershell
python src\optimize_threshold.py
```

### Step 7: Evaluate the Final Model

```powershell
python src\final_evaluation.py
```

### Step 8: Run the Prediction System

```powershell
python src\predict.py
```

The prediction script asks for the applicant's financial information and displays the estimated risk probability and classification.

### Optional: Save or Retrain the Model

```powershell
python src\save_model.py
```

**Warning:** This script retrains and overwrites the existing saved model and decision threshold. Run it only when you intentionally want to regenerate those files.

Run the commands from the project directory so that relative file paths resolve correctly.

## 18. Limitations

* The dataset has an imbalanced target distribution.
* The model can produce false-positive and false-negative predictions.
* The selected threshold may not suit every business objective.
* Historical data may not represent current financial conditions.
* Feature importance does not prove causation.
* The project has not been established as a production-ready or independently validated lending system.

Real-world use would require further validation, fairness testing, probability calibration, security controls, and compliance review.

## 19. Future Improvements

Possible improvements include:

* Evaluating additional classification algorithms.
* Testing probability calibration.
* Conducting fairness and bias analysis.
* Evaluating performance on newer or independent datasets.
* Improving the prediction interface.
* Developing a web interface or API.
* Monitoring performance after deployment.

## 20. Conclusion

This project demonstrates an end-to-end machine learning workflow for credit-risk classification.

It includes data inspection, exploratory data analysis, preprocessing, baseline model comparison, Random Forest tuning, threshold optimization, model evaluation, feature-importance analysis, and a reusable prediction program.

The final tuned Random Forest achieved a test ROC-AUC of 86.55%. At the selected threshold of 0.75, the model achieved an F1-score of 43.52%.

These results demonstrate the potential of machine learning for credit-risk analysis while also highlighting the importance of evaluating class imbalance, prediction errors, and threshold selection.

---

**Project:** CodeAlpha Credit Scoring Model
**Internship:** CodeAlpha Machine Learning Internship
**Language:** Python
