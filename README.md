# 📊 Telecom Customer Churn Prediction & Retention Analytics

An end-to-end data analytics and machine learning solution for predicting customer churn in a subscription-based telecom company. Features exploratory data analysis, machine learning classification models, SHAP explainability, an interactive Streamlit web application, and an executive Power BI dashboard.

---

## 📌 Project Overview
* **Problem Statement:** Subscription-based companies face high revenue leakage due to customer churn. Acquiring new customers costs 5x more than retaining existing ones.
* **Goal:** Build a robust predictive system to identify high-risk churn customers in advance and uncover the primary drivers behind churn, enabling retention teams to deliver targeted, high-ROI interventions.
* **Key Deliverables:**
  * Cleaned dataset and data preprocessing pipeline
  * Exploratory Data Analysis (EDA) uncovering key behavioral patterns
  * Comparative Machine Learning Models (Logistic Regression, Random Forest, XGBoost)
  * Model Explainability via SHAP (SHapley Additive exPlanations)
  * Interactive Streamlit Web Application for real-time customer risk scoring
  * Executive Power BI Dashboard with interactive slicers and KPI tracking
  * Production-ready SQL customer segmentation queries

---

## 📂 Project Structure

```plaintext
churn-prediction/
│
├── app/
│   └── streamlit_app.py           # Streamlit web app for live churn risk prediction
│
├── dashboard/
│   └── churn_dashboard.pbix       # Power BI interactive executive dashboard
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv   # Original IBM Telco dataset
│   └── processed/
│       └── churn_clean.csv            # Cleaned and feature-ready dataset
│
├── models/
│   └── churn_model.joblib         # Serialized best-performing ML model pipeline
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb     # Data audit, type conversions, missing value handling
│   ├── 02_eda.ipynb               # Exploratory data analysis & statistical visualization
│   ├── 03_modeling.ipynb          # Model training, tuning, and evaluation
│   └── 04_explainability_shap.ipynb # Feature importance & local SHAP explanations
│
├── reports/
│   ├── 01_data_cleaning.html      # HTML export of cleaning notebook
│   ├── 02_eda.html                # HTML export of EDA findings
│   ├── 03_modeling.html           # HTML export of model comparisons
│   ├── 04_explainability_shap.html# HTML export of SHAP interpretations
│   └── churn_dashboard.pdf        # Power BI report export
│
├── sql/
│   └── churn_queries.sql          # Customer segmentation & churn cohort SQL queries
│
├── .gitignore                     # Git ignore file (excludes venv and cache)
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## 📋 Dataset Dictionary

The dataset is IBM's Telco Customer Churn benchmark dataset containing **7,043 customer records** and **21 attributes**:

| Category | Features | Description |
| :--- | :--- | :--- |
| **Identifiers** | `customerID` | Unique identifier (removed during modeling) |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Customer personal and family profile |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | Subscription duration, contract term, billing method |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Subscribed product offerings and add-ons |
| **Financials** | `MonthlyCharges`, `TotalCharges` | Current monthly charge and lifetime spend |
| **Target** | `Churn` (`Yes` / `No`) | Whether the customer left within the last month (~26.5% baseline) |

---

## 🔍 Key Findings & Business Insights

1. **Contract Type is the #1 Predictor:** Customers on **Month-to-Month contracts** have a churn rate over **42%**, compared to under **11%** for 1-Year and under **3%** for 2-Year contracts.
2. **Early Tenure Vulnerability:** Over **50% of total churn occurs in the first 12 months** of customer tenure ("onboarding cliff").
3. **Tech Support & Online Security as Retention Anchors:** Customers without security/support add-ons churn at more than double the rate of customers who have them.
4. **Fiber Optic Pricing Sensitivity:** Fiber Optic subscribers exhibit the highest monthly charges (~$80-$100) and elevated churn, indicating unmet service expectations or price sensitivity.
5. **Electronic Check Friction:** Customers paying via electronic check churn significantly more frequently than those using automated credit card or bank transfer payments.

---

## ⚙️ Machine Learning Models & Evaluation

Three models were trained with stratified train/test splits (80/20) and full feature preprocessing pipelines:

| Model | Accuracy | ROC-AUC | F1-Score (Churn) | Primary Strength |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | ~80% | ~0.84 | ~0.60 | High baseline interpretability |
| **Random Forest** | ~79% | ~0.83 | ~0.58 | Non-linear interaction capture |
| **XGBoost (Selected)** | **~81%** | **~0.85** | **~0.63** | Highest discriminatory power & ROC-AUC |

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Mahdi-Nawaz/churn-prediction.git
cd churn-prediction
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Prediction App
```bash
streamlit run app/streamlit_app.py
```
*Open your browser at `http://localhost:8501` to test real-time customer risk scoring.*

### 5. Run the Analysis Notebooks
```bash
jupyter lab
```
*Open and execute notebooks in the `notebooks/` directory in numerical order (01 through 04).*

---

## 📈 Strategic Retention Recommendations

* **Contract Transition Incentives:** Offer a 10-15% discount or bundled feature upgrades for month-to-month customers who transition to an annual contract.
* **First 90 Days Onboarding Program:** Proactively reach out to newly acquired customers during months 1-3 to ensure proper setup and service satisfaction.
* **Bundle Security & Support:** Include basic `OnlineSecurity` and `TechSupport` complimentary in higher-tier plans to improve product stickiness.
* **Auto-Pay Discount Campaign:** Incentivize users paying by Electronic Check to switch to Automated Bank Transfer or Credit Card with a small monthly bill credit.

---

## 👤 Author
* **Mahdi Nawaz**
* Email: [mahdienawaz@gmail.com](mailto:mahdienawaz@gmail.com)
