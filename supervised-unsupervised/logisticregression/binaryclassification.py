# Binary classification in machine learning involves predicting one of two classes for a given input.
# For example, in insurance, it can be used to predict whether a customer is likely to make a claim (class 1)
# or not (class 0) based on features like age, history, and coverage. The algorithm learns a decision boundary
# from labeled training data, enabling it to classify new instances into one of the two classes.
# Popular algorithms for binary classification include logistic regression, support vector machines, and decision trees.
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Machine learning mein binary classification ka kaam hota hai ki kisi input ko do classes mein predict karna.
# Jaise ki insurance mein, yeh uske features jaise umar, itihas, aur coverage ke adhar par predict karta hai ki
# ek customer claim karega (class 1) ya nahi karega (class 0). Algorithm labeled training data se decision boundary
# seekhta hai, jisse wo naye instances ko do classes mein classify kar sake. Isme popular algorithms mein logistic
# regression, support vector machines, aur decision trees shaamil hote hain.



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv'

# Specify the range of rows you want to read (rows 76 to 97)
start_row = 76
end_row = 97
row_range = range(start_row, end_row)

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

# Specify the range of columns you want to keep (columns 1 to 3)
start_column = 0
end_column = 2  # Use end_column + 1 since Python slicing is exclusive on the end
column_range = range(start_column, end_column)

# Keep only the specified columns
data = data.iloc[:, column_range]

# Assuming there is a 'Have Insurance' column in your dataset
plt.scatter(data['Age'], data['Have_Insurance'], label='Data Points')
plt.plot(data['Age'], data['Have_Insurance'], color='red', linestyle='-', label='Connected Line')
plt.xlabel('Age')
plt.ylabel('Have Insurance')
plt.title('Scatter Plot of Age vs Have Insurance')

# Take user input for age
user_age = float(input("Enter the age: "))

# Train a logistic regression model
model = LogisticRegression()
model.fit(data[['Age']], data['Have_Insurance'])

# Predict whether the user is likely to buy insurance
user_prediction = model.predict([[user_age]])
user_probability = model.predict_proba([[user_age]])[:, 1]
X_train, X_test, Y_train, Y_test = train_test_split(data[['Age']], data['Have_Insurance'], test_size=0.2)

# Print the prediction for the user
print(f"For the age {user_age}, the model predicts: {'Will Buy Insurance' if user_prediction[0] == 1 else 'Will Not Buy Insurance'}")
print(f"Probability of buying insurance: {user_probability[0]:.2f}")
accuracy = model.score(X_test, Y_test)

# # Print the results
# print("X_train:", X_train)
# print("Y_train:", Y_train)
# print("X_test:", X_test)
# print("Y_test:", Y_test)
print("Accuracy:", accuracy)

# Plot the logistic regression curve
x_values = data[['Age']].values
y_probs = model.predict_proba(x_values)[:, 1]

plt.plot(x_values, y_probs, color='blue', linestyle='-', label='Logistic Regression Curve')
plt.legend()
plt.show()
