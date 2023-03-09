# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score


# Load data from CSV file
data = pd.read_csv("PDFMalware2022.csv")

# identify non-numeric columns
non_numeric_columns = data.select_dtypes(exclude='number').columns.tolist()

# drop non-numeric columns
data = data.drop(non_numeric_columns, axis=1)

# identify missing values
missing_values = data.isnull().sum()

# drop rows with missing values
data = data.dropna()

# alternatively, impute missing values
imputer = SimpleImputer()
data_imputed = imputer.fit_transform(data)

# Extract features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# y_true and y_pred are the true and predicted labels, respectively
print(classification_report(y_test, y_pred, zero_division=0))


