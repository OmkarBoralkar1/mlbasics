# In scikit-learn, the train_test_split function is used to split a dataset into training and testing sets.
# This is crucial in machine learning to assess the model's performance on unseen data.
# The function randomly shuffles and divides the dataset, allocating a specified portion for training and the rest for testing.
# The training set is used to train the model, and the testing set is employed to evaluate its generalization on new,
# unseen instances. This separation helps ensure that the model's performance is not biased by the data it was trained on.

# //////////////////////////////////////////////////////////////////////////////////////////////////////

# Scikit-learn mein, train_test_split function ka istemal karna hota hai ek dataset ko training aur
# testing sets mein bantne ke liye. Ye machine learning mein mahatvapurna hai taki model ka performance naye
# data par assess kiya ja sake. Ye function dataset ko random taur par shuffle karta hai aur usse training aur testing
# ke liye alag alag hisson mein bata deta hai. Training set model ko train karne ke liye istemal hota hai,
# aur testing set uske generalization ko evaluate karne ke liye istemal hota hai, especially jab wo naye,
# dekhe gaye instances par kaam karta hai. Ye alag karne se ye ensure hota hai ki model ka performance uss
# data par depend nahi karta jisse use train kiya gaya tha


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv'

# Specify the range of rows you want to read (rows 45 to 74)
start_row = 0
end_row = 21
row_range = range(start_row, end_row)

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

# Specify the range of columns you want to keep (columns 1 to 3)
start_column = 0
end_column = 4  # Use end_column + 1 since Python slicing is exclusive on the end
column_range = range(start_column, end_column)

# Keep only the specified columns
data = data.iloc[:, column_range]

# Scatter plot using 'Area1' and 'Price1' columns
plt.scatter(data['Area1'], data['Price1'], label='Data Points (Area1 vs. Price1)')
plt.plot(data['Area1'], data['Price1'], color='red', linestyle='--', label='Connected Line (Area1 vs. Price1)')

# Scatter plot using 'Age of House' and 'Price1' columns
plt.scatter(data['Age of House'], data['Price1'], label='Data Points (Age of House vs. Price1)')
plt.plot(data['Age of House'], data['Price1'], color='blue', linestyle='-.', label='Connected Line (Age of House vs. Price1)')

plt.xlabel('Area / Age of House')
plt.ylabel('Price1')
plt.title('Home Prices')
plt.legend()

# Show the plot
plt.show()

# Prepare data for train_test_split
X = data[['Area1', 'Age of House']]
Y = data['Price1']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Print the results
print("X_train:", X_train)
print("Y_train:", Y_train)
print("X_test:", X_test)
print("Y_test:", Y_test)
print(len(X_train))
print(len(X_test))

clf=LinearRegression()
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))
print(clf.predict(X_test))
