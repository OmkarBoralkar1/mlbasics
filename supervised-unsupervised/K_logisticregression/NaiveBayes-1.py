import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv'

# Specify the range of rows you want to read (rows 76 to 97)
start_row = 119
end_row = 140
row_range = range(start_row, end_row)

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

# Specify the range of columns you want to keep (columns 1 to 3)
start_column = 0
end_column = 3  # Use end_column + 1 since Python slicing is exclusive on the end
column_range = range(start_column, end_column)

# Keep only the specified columns
data = data.iloc[:, column_range]
model =GaussianNB()
GaussianNB(priors=None,var_smoothing=data-0.9)
