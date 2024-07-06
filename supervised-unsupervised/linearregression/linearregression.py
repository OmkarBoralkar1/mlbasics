# Linear Regression is a fundamental supervised machine learning algorithm used for predicting a
# continuous outcome variable based on one or more predictor features. The algorithm assumes a linear
# relationship between the predictors and the target variable. In the case of simple linear regression,
# there is one predictor variable, while multiple predictor variables are used in multiple linear regression.
# The model aims to find the best-fitting line that minimizes the difference between the predicted and actual values.
# This is achieved by optimizing coefficients through methods like Ordinary Least Squares.
# Linear Regression is widely employed in various fields, such as economics, finance, and biology.


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Linear Regression ek moolak supervised machine learning algorithm hai jo ek ya ek se adhik predictor
# features par adharit ek continuous outcome variable ko predict karne ke liye istemal hota hai.
# Is algorithm mein maana jata hai ki predictors aur target variable ke beech ek linear relationship hota hai.
# Simple linear regression mein ek predictor variable hota hai, jabki multiple linear regression mein kai predictor
# variables ka istemal hota hai. Model ka lakshya sabse achhi fitting line dhundhna hota hai jo predicted aur actual
# values ke beech ke farq ko kam kare. Ye Ordinary Least Squares jaise methods se coefficients ko optimize karke kiya
# jata hai. Linear Regression ko vibhinn kshetron mein, jaise ki arthashastra, vittiya, aur jeev vigyan, mein vyapak
# roop se istemal kiya jata hai.



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from savemodeltofile import savefile
from savemodeltofile import joblibsave
import pickle
# Assuming your CSV file has columns named 'area' and 'prices'
# Replace 'your_file_path' with the actual file path
file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv'

# Specify the range of rows you want to read (rows 24 to 43)
start_row = 22
end_row = 42
row_range = range(start_row, end_row )

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

print(data)
# Scatter plot
plt.scatter(data['area'], data['prices'], color='blue', label='Actual Data')
plt.xlabel('Area (sqr ft)')
plt.ylabel('Price (INR)')
plt.title('Scatter Plot of Area vs. Price')

# Connect the actual data points with a line
plt.plot(data['area'], data['prices'], linestyle='--', color='brown', linewidth=1.25,  label='Connecting Line')

# Fit a linear regression model
regression_model = LinearRegression()
regression_model.fit(data[['area']], data['prices'])
print('the model score is',regression_model.score(data[['area']], data['prices']))
# Get user input for area
input_area = float(input("Enter the area: "))

# Predict the price based on the linear regression model
predicted_price = regression_model.predict([[input_area]])

# Print the predicted price
print(f"Predicted Price for {input_area} square feet: {predicted_price[0]} INR")

# Plot the regression line
plt.plot(data['area'], regression_model.predict(data[['area']]), color='red', linewidth=2, label='Regression Line')

# Display the legend
plt.legend()

# Display the plot
plt.show()
savefile(regression_model)
joblibsave(regression_model)
