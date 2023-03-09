import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# load the dataset
df = pd.read_csv('PDFMalware2022.csv')


# handle missing values
imputer = SimpleImputer(strategy='mean')
df[['Class', 'pdfsize']] = imputer.fit_transform(df[['Class', 'pdfsize']])

# encode categorical variables
le = LabelEncoder()
df['isEncrypted'] = le.fit_transform(df['isEncrypted'])

# scale numerical variables
scaler = StandardScaler()
df[['Class', 'pdfsize']] = scaler.fit_transform(df[['metadata size', 'pdfsize']])
