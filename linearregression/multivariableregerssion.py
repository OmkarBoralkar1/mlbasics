import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Read the CSV file
data = pd.read_csv('C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv', delimiter='\t', nrows=21)

print("Original Data:")
print(data)

# Calculate the median value of the 'Bedrooms' column
median_bedrooms = data['Bedrooms'].median()
print(f"Median Bedrooms: {median_bedrooms}")

# Fill NaN values in all columns
data = data.fillna(median_bedrooms)

print("\nData after filling NaN values:")
print(data)

# Scatter plot
plt.scatter(data['Area1'], data['Price1'], cmap='Blues', label='Actual Data')
plt.xlabel('Area (sqr ft)')
plt.ylabel('Price')
plt.title('Scatter Plot of Area, Bedrooms vs. Price')

# Connect the actual data points with a line
plt.plot(data['Area1'], data['Price1'], linestyle='--', color='brown', linewidth=1.25, label='Connecting Line')

# Create a linear regression model
regression_model = make_pipeline(ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), ['Area1', 'Bedrooms', 'Age of House']),
    ],
    remainder='passthrough'),  # Pass through any non-transformed columns
    StandardScaler(),
    LinearRegression()
)
# Prepare the features (X) and target variable (y)
X = data[['Area1', 'Bedrooms', 'Age of House']]
y = data['Price1']
# Fit the model
regression_model.fit(X, y)

# Get the learned coefficients from the 'LinearRegression' step in the pipeline
coefficients = regression_model.named_steps['linearregression'].coef_
intercept = regression_model.named_steps['linearregression'].intercept_

# Manually set a negative coefficient for the 'Age of House' feature
coefficients[-1] *= -1

# Print coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)
print('the model score is',regression_model.score(X,y))
# Get user input for prediction
input_area = float(input("Enter the area: "))
input_bedrooms = int(input("Enter the number of bedrooms: "))
input_age = float(input("Enter the age of the house:"))

# Predict the price based on user input
input_data = pd.DataFrame({'Area1': [input_area], 'Bedrooms': [input_bedrooms], 'Age of House': [input_age]})
predicted_price = regression_model.predict(input_data)
print(f"Predicted Price:{predicted_price[0]}")

# Display the legend and colorbar
plt.legend()
plt.colorbar(label='Price (INR)')

# Display the plot
plt.show()
