import pandas as pd #type:ignore
import numpy as np #type:ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler #type:ignore
from sklearn.linear_model import LogisticRegression #type:ignore
from sklearn.metrics import accuracy_score, classification_report #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from imblearn.combine import SMOTEENN #type:ignore
from sklearn.preprocessing import PolynomialFeatures #type:ignore
 
# Loading Dataset
data = pd.read_csv('loan_data.csv')
category_cols = ['person_gender', 'person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file','loan_status']
# Label Encoding useful Columns 
label_encoder = LabelEncoder() 
# Loop to encode all the columns
for col in category_cols:
    data[col] = label_encoder.fit_transform(data[col]) 
#  Split Data into Features (X) and Target (y)
X = data.drop('loan_status', axis=1)
y = data['loan_status'] 
# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Convert Columns to Numeric and Handle Missing Values
numerical_columns = ['person_age', 'person_income', 'loan_amnt','loan_int_rate','cb_person_cred_hist_length','credit_score']
for col in numerical_columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce') 
    X_train[col] = X_train[col].fillna(X_train[col].mean())
    X_test[col] = X_test[col].fillna(X_test[col].mean()) 
# # Initialize Scaler
scaler = StandardScaler()
# Scaling numerical columns in training and testing sets
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
# # Handle Class Imbalance with SMOTEENN
# # smote_enn = SMOTEENN(random_state=42)
# # X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
# Initialize PolynomialFeatures with degree (choose degree based on data complexity)
poly = PolynomialFeatures(degree=3, include_bias=True)
# Transform the training and test`` sets
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# Initialize and Train Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=3000,penalty='l2', C = 0.001 ,solver='liblinear')
log_reg.fit(X_train_poly, y_train)
# Model Evaluation    
y_train_pred = log_reg.predict(X_train_poly)
y_test_pred = log_reg.predict(X_test_poly)
print(" Training Accuracy:", accuracy_score(y_train, y_train_pred))  
print(" Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))  
