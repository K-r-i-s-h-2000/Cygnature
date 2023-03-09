import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# read data
data = pd.read_csv('PDFMalware2022.csv')

# replace missing values with empty string
data['Class'] = data['Class'].fillna('')

# convert text data to numerical data using TfidfVectorizer
vectorizer = TfidfVectorizer()
numerical_data = vectorizer.fit_transform(data['Class'])
