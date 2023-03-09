
import pandas as pd

# load the dataset
df = pd.read_csv('PDFMalware2022.csv')

# Check if the 'metadata' column exists in the DataFrame 'df'
print('Class' in df.columns)
