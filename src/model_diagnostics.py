import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN

# Load processed data
df = pd.read_csv('data/processed/processed_data.csv')

# Load model and preprocessor
model_bundle = joblib.load('models/customer_churn_model.joblib')
model = model_bundle['model']
preprocessor = model_bundle['preprocessor']

# Prepare features and target
X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
y = df[TARGET_COLUMN]

# Transform features
X_processed = preprocessor.transform(X)

# Make predictions
probs = model.predict_proba(X_processed)[:, 1]
preds = model.predict(X_processed)

# 1. Show sample predictions
print('Sample predictions:')
sample = pd.DataFrame({
    'Actual': y.values,
    'Predicted': preds,
    'Churn Probability': probs
}).head(20)
print(sample)

# 2. Prediction probability distribution
plt.figure(figsize=(6, 4))
sns.histplot(probs, bins=20, kde=True)
plt.title('Predicted Churn Probability Distribution')
plt.xlabel('Churn Probability')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('diagnostics/probability_distribution.png')
plt.close()

# 3. Class balance
print('\nClass balance:')
print(y.value_counts())

# 4. Classification report
print('\nClassification Report:')
print(classification_report(y, preds))

# 5. Confusion matrix
cm = confusion_matrix(y, preds)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('diagnostics/confusion_matrix.png')
plt.close()

# 6. Feature importances (if available)
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig('diagnostics/feature_importances.png')
    plt.close()
    print('\nTop feature importances saved to diagnostics/feature_importances.png')
else:
    print('\nModel does not provide feature importances.')

print('\nDiagnostics complete. Plots saved in diagnostics/.') 