import nbformat as nbf
import os

def create_shap_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title
    nb['cells'].append(nbf.v4.new_markdown_cell('# Explainability: SHAP Values\nIn this notebook, we use SHAP (SHapley Additive exPlanations) to interpret our machine learning model and understand the key drivers of customer churn.'))
    
    # Imports
    code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

# Initialize JS visualization for SHAP
shap.initjs()"""
    nb['cells'].append(nbf.v4.new_code_cell(code_imports))
    
    # Load Model and Data
    code_load = """# Load the cleaned dataset
df = pd.read_csv('../data/processed/churn_clean.csv')

# Drop customerID for consistency with the model
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Load the trained model pipeline (Logistic Regression or XGBoost)
# Adjust the path to load the correct model from the previous step
model_pipeline = joblib.load('../models/churn_model.joblib')

print("Model and Data loaded successfully.")"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 1. Load Data and Trained Model'))
    nb['cells'].append(nbf.v4.new_code_cell(code_load))
    
    # Preprocess Data
    code_prep = """# Get the preprocessor and the actual classifier from the pipeline
preprocessor = model_pipeline.named_steps['preprocessor']
classifier = model_pipeline.named_steps['classifier']

# Transform the input features
X_transformed = preprocessor.transform(X)

# Get feature names after transformation
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
feature_names = np.concatenate([numeric_features, categorical_features])

# Create a DataFrame with the transformed features
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
X_transformed_df.head()"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 2. Preprocess the Data for SHAP'))
    nb['cells'].append(nbf.v4.new_code_cell(code_prep))
    
    # Compute SHAP values
    code_shap = """# Compute SHAP values
# We use the LinearExplainer for Logistic Regression, or TreeExplainer for Random Forest/XGBoost
import warnings
warnings.filterwarnings('ignore')

# Check if the classifier is a Linear Model or Tree-based Model
if hasattr(classifier, 'coef_'):
    explainer = shap.LinearExplainer(classifier, X_transformed_df)
else:
    explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(X_transformed_df)

print("SHAP values computed.")"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 3. Compute SHAP Values'))
    nb['cells'].append(nbf.v4.new_code_cell(code_shap))
    
    # SHAP Summary Plot
    code_summary = """# Summary plot of all features
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_transformed_df, show=False)
plt.title('SHAP Summary Plot - Key Churn Drivers')
plt.show()"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 4. SHAP Summary Plot\nThis plot shows the most important features driving churn. Features pushed to the right increase the likelihood of churn, while features pushed to the left decrease it.'))
    nb['cells'].append(nbf.v4.new_code_cell(code_summary))
    
    # Individual Explanations
    code_ind = """# Explanation for a single customer (e.g., customer at index 0)
customer_idx = 0
shap.force_plot(explainer.expected_value, shap_values[customer_idx,:], X_transformed_df.iloc[customer_idx,:], matplotlib=True)"""
    nb['cells'].append(nbf.v4.new_markdown_cell('## 5. Per-Customer Explanation Plot\nWe can break down why a specific individual was predicted to churn or not.'))
    nb['cells'].append(nbf.v4.new_code_cell(code_ind))
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/04_explainability_shap.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_shap_notebook()
    print("Created notebooks/04_explainability_shap.ipynb")
