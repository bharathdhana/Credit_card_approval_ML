# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib



# Load the dataset
data = pd.read_csv('credit_card_data.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Handle missing values (if any)
for col in data.columns:
    if data[col].dtype == 'object':  # Categorical columns
        data[col] = data[col].fillna(data[col].mode()[0])
    else:  # Numerical columns
        data[col] = data[col].fillna(data[col].median())

# Check the target variable distribution
print("\nTarget Variable Distribution:")
print(data['STATUS'].value_counts())

# If the target variable is not binary, binarize it
# Example: Convert 'X' and 'C' to 1 (approved) and others to 0 (not approved)
if data['STATUS'].nunique() > 2:
    data['STATUS'] = data['STATUS'].apply(lambda x: 1 if x in ['X', 'C'] else 0)

# Separate features and target variable
X = data.drop(['STATUS', 'ID'], axis=1)  # Drop 'ID' column
y = data['STATUS']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = []  # No categorical features after dropping 'ID'
numerical_features = ['MONTHS_BALANCE']  # Update with your numerical columns

# Create a preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

# Apply SMOTE to handle imbalanced data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# Train a Random Forest Classifier with parallel processing
model = RandomForestClassifier(random_state=42, n_jobs=-1)  # Use all available CPU cores
model.fit(X_train_res, y_train_res)

# Make predictions on the test set
X_test_transformed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feature_importances = model.feature_importances_
feature_names = numerical_features
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Save the model (optional)
joblib.dump(model, 'credit_card_approval_model.pkl')
print("\nModel saved as 'credit_card_approval_model.pkl'")





