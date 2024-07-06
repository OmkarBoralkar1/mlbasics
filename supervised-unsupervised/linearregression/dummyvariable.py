# Dummy Variables, also known as indicator variables, are used in statistical modeling and machine learning
# to represent categorical data with two or more categories. These variables are binary,
# taking values of 0 or 1, where 1 indicates the presence of a particular category, and 0 indicates its absence.
# Dummy variables are created by converting categorical variables into a set of binary variables,
# with each variable representing one category. They are particularly useful in regression analysis and linear models,
# allowing the incorporation of categorical information into the modeling process.
# However, care must be taken to avoid the "dummy variable trap," where the presence of correlated dummy variables
# can lead to multicollinearity issues in the model.

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Dummy Variables, jo ki indicator variables ke roop mein bhi jaane jaate hain, statistics modeling aur
# machine learning mein istemal hote hain categorical data ko darust taur par represent karne ke liye jisme
# do ya do se adhik categories ho. Ye variables binary hote hain, jo 0 ya 1 ke values lete hain,
# jahaan 1 ek khaas category ki presence ko darust karta hai, aur 0 uski absence ko. Dummy variables ko categorical
# variables ko ek set of binary variables mein convert karke banaya jata hai, jisme har variable ek category
# ko represent karta hai. Ye regression analysis aur linear models mein khaas upyogi hote hain, kyunki inka istemal
# categorical information ko modeling process mein shaamil karne mein hota hai. Haa, dhyan rakhna chahiye
# ki "dummy variable trap" se bacha jaaye, jisme correlated dummy variables ki presence model mein
# multicollinearity problems ko laa sakta hai.

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assuming your CSV file has columns named 'area' and 'prices'
# Replace 'your_file_path' with the actual file path
file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Documents/homeprices.csv'

# Specify the range of rows you want to read (rows 1 to 30 for example)
start_row = 44
end_row = 74
row_range = range(start_row, end_row)

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

# Display the loaded data to check for NaN values
print("Loaded Data:")
print(data)

# Drop rows with NaN values
data = data.dropna()
print("\nData after dropping NaN values:")
print(data)

# Create dummy variables for the 'city' column
dummies = pd.get_dummies(data['city'], drop_first=True, prefix='city')

# Concatenate the dummy variables with the original data
result = pd.concat([data, dummies], axis=1)

# Print the first few rows of the final DataFrame to check if it has rows
print('\nThe final DataFrame:\n', result, '\nThe result head is:\n', result.head())

# Check if the final DataFrame has any rows before fitting the model
if result.shape[0] > 0:
    X = result.drop(['prices', 'city'], axis=1)  # Drop 'city' as we have dummy variables
    Y = result.prices
    model = LinearRegression()
    model.fit(X, Y)

    # Plot the data points
    plt.scatter(data['area'], data['prices'], color='blue', label='Actual Data')

    # Plot the linear regression lines for Mumbai, Kolkata, and Delhi
    for city in ['Mumbai', 'Delhi']:
        city_data = result[result[city] == 1]
        plt.plot(city_data['area'], model.predict(city_data.drop(['prices', 'city'], axis=1)),
                 label=f'{city} Regression Line')

    plt.title('Linear Regression Lines for Mumbai and Delhi')
    plt.xlabel('Area')
    plt.ylabel('Prices')
    plt.legend()
    plt.show()

    print('The model score is', model.score(X, Y))

    # Take user input for prediction
    input_area = float(input("Enter the area: "))
    input_city_mumbai = int(input("Enter 1 if the city is Mumbai, otherwise enter 0: "))
    input_city_delhi = int(input("Enter 1 if the city is Delhi, otherwise enter 0: "))

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({'area': [input_area], 'Mumbai': [input_city_mumbai], 'Delhi': [input_city_delhi]})

    # Make prediction
    predicted_price = model.predict(input_data)

    print(f"Predicted Price: {predicted_price[0]}")
else:
    print('No rows in the final DataFrame. Check your data preprocessing steps.')

