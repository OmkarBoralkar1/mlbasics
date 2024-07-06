# In scikit-learn, one-hot encoding is implemented using the OneHotEncoder class.
# For instance, in price prediction, if you have a categorical feature like
# "City" with values such as "New York," "London," and "Tokyo," you can use OneHotEncoder to convert it into binary columns.
# Each column represents a unique category, and the presence or absence of a 1 indicates the category of the input.
# This helps machine learning models better understand categorical data and improves predictive performance.
# /////////////////////////////////////////////////////////////////////////
# Scikit-learn mein, one-hot encoding OneHotEncoder class ka istemal karke kiya jata hai.
# For example, price prediction mein, agar aapke paas "City" jaise categorical feature hai jiska value hai "New York,"
# "London," aur "Tokyo," to aap OneHotEncoder ka istemal karke ise binary columns mein convert kar sakte hain.
# Har column ek unique category ko represent karta hai, aur 1 ki presence ya absence input ki category ko darust
# taur par batati hai. Ye machine learning models ko categorical data ko behtar se samajhne mein madad karta hai
# aur predictive performance ko sudharne mein madad karta hai.
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv'

# Specify the range of rows you want to read (rows 45 to 74)
start_row = 44
end_row = 74
row_range = range(start_row, end_row)

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

# Specify the range of columns you want to keep (columns 0 to 2 for 'city', 'area', and 'prices')
start_column = 0
end_column = 3  # Use end_column + 1 since Python slicing is exclusive on the end
column_range = range(start_column, end_column)

# Keep only the specified columns
data = data.iloc[:, column_range]
print(data)

print(data.columns)

# One-hot encode the 'city' column directly
ohe = OneHotEncoder(sparse=False, drop='first')
X_encoded = ohe.fit_transform(data[['city']])

# Concatenate the one-hot encoded features with the 'area' column
X = pd.concat([pd.DataFrame(X_encoded), data['area']], axis=1)

# Assuming you have the target variable 'prices' in your data
y = data['prices']

model = LinearRegression()

# Fit the model
model.fit(X, y)

# Take user input for city and area
user_city = input("Enter the city(Mumbai,Delhi,Kolkata): ")
user_area = float(input("Enter the area: "))  # Assuming area is a numerical value

# Encode the user input city
user_city_encoded = ohe.transform([[user_city]])

# Concatenate the encoded city with the user input area
user_input = pd.concat([pd.DataFrame(user_city_encoded), pd.DataFrame({'area': [user_area]})], axis=1)

# Predict the price for the user input
predicted_price = model.predict(user_input)

print(f"Predicted Price: {predicted_price[0]}")

# Scatter plot of the original data with different colors for each city
colors = {'Mumbai': 'blue', 'Delhi': 'red', 'Kolkata': 'green'}
for city in data['city'].unique():
    city_data = data[data['city'] == city]
    plt.scatter(city_data['area'], city_data['prices'], color=colors[city], label=city)

# Plot the regression line
x_range = X['area'].values.reshape(-1, 1)
y_pred = model.predict(X)
plt.plot(X['area'], y_pred, color='black', linewidth=2, label='Linear Regression')

# Mark the user input on the plot
plt.scatter(user_area, predicted_price, color='purple', marker='X', label='User Input')

# Add labels and a legend
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Linear Regression on Home Prices')
plt.legend()

# Show the plot
plt.show()
