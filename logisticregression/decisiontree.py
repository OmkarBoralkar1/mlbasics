# n machine learning, a decision tree is a predictive model that maps features to target values.
# For example, in a job profile predicting salary, a decision tree can analyze factors like experience,
# education, and skills to predict whether a candidate will have a high salary or not.
# The tree structure consists of nodes representing decisions based on feature conditions,
# leading to different salary outcomes.
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Machine learning mein, ek decision tree ek aisa predictive model hai jo features ko target values ke saath map karta hai.
# Jaise ki ek job profile mein salary predict karna, ek decision tree experience, education,
# aur skills jaise factors ko analyze karke ye forecast kar sakta hai ki koi candidate ko zyada salary milegi ya nahi.
# Tree structure mein nodes hote hain jo feature conditions par adharit decisions ko represent karte hain,
# jisse alag salary outcomes aate hain.


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv'

# Specify the range of rows you want to read (rows 76 to 97)
start_row = 99
end_row = 118
row_range = range(start_row, end_row)

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

# Specify the range of columns you want to keep (columns 0 to 3)
start_column = 0
end_column = 4  # Use end_column + 1 since Python slicing is exclusive on the end
column_range = range(start_column, end_column)

# Keep only the specified columns
data = data.iloc[:, column_range]

# Fill NaN values in 'Company Name' with a placeholder string (e.g., 'Unknown')
data['Company Name'].fillna('Unknown', inplace=True)

# Use LabelEncoder for encoding categorical variables
label_encoder = LabelEncoder()
mapping_dict = {}

# Apply LabelEncoder consistently across different parts of your code
for column in ['Company Name', 'Job_Profile', 'Degree']:
    data[column] = label_encoder.fit_transform(data[column])
    mapping_dict[column] = label_encoder.classes_

# Print unique values after encoding
# print("Unique values in 'Company Name' after encoding:", mapping_dict['Company Name'])
# print("Unique values in 'Job_Profile' after encoding:", mapping_dict['Job_Profile'])
# print("Unique values in 'Degree' after encoding:", mapping_dict['Degree'])
# print('the data modified is ', data)

# Define colors based on 'Salary > 100k'
colors = ['red' if value == 0 else 'green' for value in data['Salary > 100k']]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['Company Name'], data['Job_Profile'], data['Degree'], c=colors,
           label='Salary > 100k (0=red, 1=green)')
ax.set_xlabel('Company Name')
ax.set_ylabel('Job Profile')
ax.set_zlabel('Degree')
ax.set_title('3D Scatter Plot: Company Name, Job Profile, and Degree vs Salary > 100k')

# Customize tick labels using the mapping dictionary
ax.set_xticks(range(len(mapping_dict['Company Name'])))
ax.set_yticks(range(len(mapping_dict['Job_Profile'])))
ax.set_zticks(range(len(mapping_dict['Degree'])))

ax.set_xticklabels(mapping_dict['Company Name'])
ax.set_yticklabels(mapping_dict['Job_Profile'])
ax.set_zticklabels(mapping_dict['Degree'])


# Assuming you want to use 'inputs' and 'target' for further analysis
inputs = data.drop('Salary > 100k', axis='columns')
# print('the inputs is ', inputs)

target = data['Salary > 100k']

model = tree.DecisionTreeClassifier()
model.fit(inputs, target)
mp=model.predict([[0,0,1]])
print('the predicted model is',mp)
print("Enter the 'Company Name' for  ['Company A' 'Company B' 'Company C'] as :", data['Company Name'].unique(),'respectively')
print("Enter the  'Job_Profile'  ['Business Engineer' 'Computer Programmer' 'Sales Executive'] as:", data['Job_Profile'].unique(),'respectively')
print("Enter the  'Degree'  ['Bachelors' 'Masters'] as:", data['Degree'].unique(),'respectively')

user_company = input("Enter Company Name: ")
user_job = input("Enter Job Profile: ")
user_degree = input("Enter Degree: ")
# Predict the salary
prediction = model.predict([[user_company, user_job, user_degree]])
print(f"Predicted Salary > 100k: {prediction[0]}")
plt.legend()
plt.show()
