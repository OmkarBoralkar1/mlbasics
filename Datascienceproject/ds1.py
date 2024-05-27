# Import necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from ds2 import featurereduction_dimensionality
# Set plot size
matplotlib.rcParams['figure.figsize'] = (20,10)

# Read the CSV file into a DataFrame
data = pd.read_csv('Bengaluru_House_Data.csv')

# Function for data cleaning
def data_cleaning():
    # Count the occurrences of each 'area_type'
    d1 = data.groupby('area_type')['area_type'].agg('count')

    # Drop columns 'area_type', 'society', 'balcony', 'availability'
    d2 = data.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)

    # Drop rows with missing values
    d3 = d2.dropna()

    # Extract the number of bedrooms ('bhk') from the 'size' column
    d3.loc[:, 'bhk'] = d3['size'].apply(lambda x: int(x.split(' ')[0]))

    # Check for non-numeric values in 'total_sqft'
    def is_float(x):
        try:
            float(x)
        except:
            return False
        return True

    # Print rows where 'total_sqft' is not numeric
    print('Rows with non-numeric total_sqft:\n', d3[~d3['total_sqft'].apply(is_float)])

    # Function to convert 'total_sqft' to numeric values
    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except ValueError:
            return None

    # Create a copy of the DataFrame for further processing
    d4 = d3.copy()

    # Apply the 'convert_sqft_to_num' function to 'total_sqft' column
    d4['total_sqft'] = d4['total_sqft'].apply(convert_sqft_to_num)

    # Print the head of the cleaned DataFrame
    print('Cleaned data head:\n', d4.head())

    # Return the cleaned DataFrame
    return d4

# Perform data cleaning
d4 = data_cleaning()

# Perform feature reduction and dimensionality reduction
featurereduction_dimensionality(d4)
