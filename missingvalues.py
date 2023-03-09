import pandas as pd

# load the dataset
df = pd.read_csv('PDFMalware2022.csv')

# check for missing values
print(df.isnull().sum())
# replace missing values with default value
df.fillna(" ", inplace=True)
