
# They are used for overfitting issue
# L1 Regularization (Lasso):
# L1 regularization adds a penalty term to the linear regression cost function, forcing some coefficients to be exactly zero.
# It helps in feature selection by encouraging sparse models and prevents overfitting by eliminating less important features.

# L2 Regularization (Ridge):
# L2 regularization adds a penalty term based on the square of the magnitude of coefficients to the cost function.
# It discourages large coefficients and, like L1, helps prevent overfitting by penalizing overly complex models.

# In Summary:
# L1 and L2 regularization are techniques used to prevent overfitting in machine learning models.
# L1 encourages sparsity by driving some coefficients to zero, while L2 discourages large coefficients, promoting a more balanced model.
# /////////////////////////////////////////////////////////////////////////////////////////////////

# L1 Regularization (Lasso):
# L1 regularization linear regression cost function mein ek penalty term add karta hai,
# jisse kuch coefficients ko bilkul zero banane
# ki koshish hoti hai. Ye feature selection mein madad karta hai aur overfitting ko
# rokne mein madad karta hai kyunki yeh kamzor features
# ko hata deta hai.

# L2 Regularization (Ridge):
# L2 regularization cost function mein coefficients ke magnitude ke square ke adhar par ek penalty term add karta hai.
# Isse bade coefficients ko rokta hai aur L1 ki tarah overfitting ko rokne mein madad karta hai, kyunki ye zyada complex
# models
# ko penalize karta hai.

# Sankshipt Mein:
# L1 aur L2 regularization overfitting ko rokne ke liye istemal hone wale techniques hain.
# L1 sparsity badhakar kuch coefficients ko zero banata hai, jabki L2 bade coefficients ko rokta hai, jisse model ka
# santulit rehta hai.

# Formula for L1 and L2


# L1 Regularization (Lasso):


# J(θ) = (1/2m) * Σ(i=1 to m) (hθ(x^(i)) - y^(i))^2 + λ * Σ(j=1 to n) |θ_j|
#
# In this formula:
# - J(θ) is the regularized cost function.
# - m is the number of training examples.
# - hθ(x^(i)) is the predicted value for the i-th example.
# - y^(i) is the actual output for the i-th example.
# - θ_j are the model parameters.
# - λ is the regularization parameter.


# L2 Regularization (Ridge):


# J(θ) = (1/2m) * Σ(i=1 to m) (hθ(x^(i)) - y^(i))^2 + λ * Σ(j=1 to n) θ_j^2
#
# In this formula:
# - J(θ) is the regularized cost function.
# - m is the number of training examples.
# - hθ(x^(i)) is the predicted value for the i-th example.
# - y^(i) is the actual output for the i-th example.
# - θ_j are the model parameters.
# - λ is the regularization parameter.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import linear_model

# Load the dataset
file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv'

# Specify the range of rows you want to read (rows 76 to 97)
start_row = 168
end_row = 188
row_range = range(start_row, end_row)

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

# Specify the range of columns you want to keep (columns 0 to 3)
start_column = 0
end_column = 16  # Use end_column + 1 since Python slicing is exclusive on the end
column_range = range(start_column, end_column)

# Keep only the specified columns
data2 = data.iloc[:, column_range]

cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'Seller', 'RegionName', 'PropertyCount', 'DistanceFromPublicTransport',
               'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'LandSize', 'BuildingArea', 'Price']
data = data[cols_to_use]

# Fill missing values with 0 for specific columns
cols_to_fill_zero = ['PropertyCount', 'DistanceFromPublicTransport', 'Bedroom2', 'Bathroom', 'Car']
data[cols_to_fill_zero] = data[cols_to_fill_zero].fillna(0)

# Preprocess the 'Price' column
data['Price'] = data['Price'].str.replace(',', '').astype(float)

# Fill missing values with the mean for numeric columns
numeric_cols = ['LandSize', 'BuildingArea', 'Price']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coercing errors to NaN
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())  # Fill NaN with mean

# Drop rows with any remaining NaN values
data.dropna(inplace=True)

# Perform one-hot encoding for categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split the data into features (X) and target (Y)
X = data.drop('Price', axis=1)
Y = data['Price']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=30)

# Train a linear regression model
reg = LinearRegression().fit(X_train, Y_train)

# Print the R-squared score on the test set
print('Train score with Linear Regression:', reg.score(X_train, Y_train))
print('Test score with Linear Regression:', reg.score(X_test, Y_test))

# Train L1 Regularization model (Lasso)
L1_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
L1_reg.fit(X_train, Y_train)
print('Train score with L1 Regularization:', L1_reg.score(X_train, Y_train))
print('Test score with L1 Regularization:', L1_reg.score(X_test, Y_test))

# Train L2 Regularization model (Ridge)
L2_reg = linear_model.Ridge(alpha=50, max_iter=100, tol=0.1)
L2_reg.fit(X_train, Y_train)
print('Train score with L2 Regularization:', L2_reg.score(X_train, Y_train))
print('Test score with L2 Regularization:', L2_reg.score(X_test, Y_test))

# Collect user input for features
user_input = {}
for column in ['Suburb','Type','Rooms', 'RegionName', 'PropertyCount',
               'DistanceFromPublicTransport', 'Bedroom2', 'Bathroom', 'Car', 'LandSize', 'BuildingArea']:
    user_input[column] = input(f"Enter value for {column}: ")

# Convert user input into a DataFrame
user_data = pd.DataFrame([user_input])

# Label encode categorical columns
label_encoder = LabelEncoder()
for column in ['Suburb', 'Type', 'RegionName']:
    user_data[column] = label_encoder.fit_transform(user_data[column])

# Ensure the user data has the same columns as the training data
user_data = user_data.reindex(columns=X.columns, fill_value=0)

# Make predictions using the trained models
linear_regression_prediction = reg.predict(user_data)
l1_prediction = L1_reg.predict(user_data)
l2_prediction = L2_reg.predict(user_data)

# Print the predictions
print(f"\nLinear Regression Prediction: {linear_regression_prediction[0]}")
print(f"L1 Regularization Prediction: {l1_prediction[0]}")
print(f"L2 Regularization Prediction: {l2_prediction[0]}")

