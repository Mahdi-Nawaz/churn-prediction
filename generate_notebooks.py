import nbformat as nbf

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title
    nb['cells'].append(nbf.v4.new_markdown_cell('# Exploratory Data Analysis (EDA)\nIn this notebook, we analyze the telecom customer churn dataset to uncover patterns and relationships.'))
    
    # Imports
    code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)"""
    nb['cells'].append(nbf.v4.new_code_cell(code_imports))
    
    # Load Data
    code_load = """# Load the cleaned dataset
df = pd.read_csv('../data/processed/churn_clean.csv')
df.head()"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 1. Load Data\nWe load the cleaned dataset produced in the `01_data_cleaning` step.'))
    nb['cells'].append(nbf.v4.new_code_cell(code_load))
    
    # Churn distribution
    code_churn = """# Overall churn rate
plt.figure(figsize=(6, 6))
ax = sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Overall Churn Distribution')
plt.ylabel('Number of Customers')

# Calculate percentages
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_y() + p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
    
plt.show()"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 2. Overall Churn Distribution'))
    nb['cells'].append(nbf.v4.new_code_cell(code_churn))
    
    # Categorical Analysis
    code_cat = """# Churn by Contract
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
plt.title('Churn by Contract Type')
plt.show()

# Churn by Internet Service
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='InternetService', hue='Churn', palette='Set2')
plt.title('Churn by Internet Service')
plt.show()

# Churn by Payment Method
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='PaymentMethod', hue='Churn', palette='Set2')
plt.title('Churn by Payment Method')
plt.xticks(rotation=45)
plt.show()"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 3. Churn by Categorical Features (Contract, Internet Service, Payment Method)'))
    nb['cells'].append(nbf.v4.new_code_cell(code_cat))
    
    # Numerical Analysis
    code_num = """# Distribution of Tenure
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='tenure', hue='Churn', multiple="stack", bins=30, palette='Set2')
plt.title('Distribution of Tenure by Churn')
plt.show()

# Distribution of Monthly Charges
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple="stack", bins=30, palette='Set2')
plt.title('Distribution of Monthly Charges by Churn')
plt.show()"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 4. Churn by Numerical Features (Tenure, Monthly Charges)'))
    nb['cells'].append(nbf.v4.new_code_cell(code_num))
    
    # Correlation Matrix
    code_corr = """# Select numeric features for correlation
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
if 'Churn' in df.columns:
    # Temporarily encode Churn as numeric for correlation
    df['Churn_Num'] = df['Churn'].map({'Yes': 1, 'No': 0})
    numeric_cols.append('Churn_Num')

plt.figure(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

if 'Churn_Num' in df.columns:
    df.drop('Churn_Num', axis=1, inplace=True)"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 5. Correlation Analysis'))
    nb['cells'].append(nbf.v4.new_code_cell(code_corr))
    
    import os
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/02_eda.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def create_modeling_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title
    nb['cells'].append(nbf.v4.new_markdown_cell('# Modeling: Churn Prediction\nIn this notebook, we build, train, and evaluate machine learning models to predict customer churn.'))
    
    # Imports
    code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib"""
    nb['cells'].append(nbf.v4.new_code_cell(code_imports))
    
    # Load Data
    code_load = """# Load the cleaned dataset
df = pd.read_csv('../data/processed/churn_clean.csv')

# The methodology says customerID should be dropped
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})  # Encode target variable
"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 1. Load Data & Prepare Target Variable'))
    nb['cells'].append(nbf.v4.new_code_cell(code_load))
    
    # Split
    code_split = """# Stratified train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training churn rate: {y_train.mean():.3f}")
print(f"Test churn rate: {y_test.mean():.3f}")"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 2. Train / Test Split\nWe use a stratified split to ensure both sets keep the 26.5% churn rate.'))
    nb['cells'].append(nbf.v4.new_code_cell(code_split))
    
    # Preprocessing
    code_prep = """# Identify categorical and numerical columns
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore') # drop first to avoid multicollinearity in linear models

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 3. Preprocessing\nOne-hot encoding for categorical variables and scaling for numeric columns.'))
    nb['cells'].append(nbf.v4.new_code_cell(code_prep))
    
    # Train Logistic Regression
    code_lr = """# Logistic Regression Baseline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("--- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")
"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 4. Model Training: Logistic Regression (Baseline)'))
    nb['cells'].append(nbf.v4.new_code_cell(code_lr))
    
    # Train Random Forest
    code_rf = """# Random Forest Model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("--- Random Forest ---")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 5. Model Training: Random Forest'))
    nb['cells'].append(nbf.v4.new_code_cell(code_rf))
    
    # Train XGBoost
    code_xgb = """# XGBoost Model
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum()))
])

xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)
y_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

print("--- XGBoost ---")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 6. Model Training: XGBoost'))
    nb['cells'].append(nbf.v4.new_code_cell(code_xgb))
    
    # Evaluation
    code_eval = """# Plot Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, y_pred) in enumerate(zip(['Logistic Regression', 'Random Forest', 'XGBoost'], 
                                        [y_pred_lr, y_pred_rf, y_pred_xgb])):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
    axes[idx].set_title(f'{name} Confusion Matrix')

plt.tight_layout()
plt.show()"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 7. Model Evaluation'))
    nb['cells'].append(nbf.v4.new_code_cell(code_eval))
    
    # Save Model
    code_save = """# Let's save the Logistic Regression model as our primary baseline model
# Or you can choose the XGBoost model
import os
os.makedirs('../models', exist_ok=True)
joblib.dump(lr_pipeline, '../models/churn_model.joblib')
print("Model saved to ../models/churn_model.joblib")"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 8. Export Model'))
    nb['cells'].append(nbf.v4.new_code_cell(code_save))

    import os
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/03_modeling.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_eda_notebook()
    print("Created notebooks/02_eda.ipynb")
    create_modeling_notebook()
    print("Created notebooks/03_modeling.ipynb")
