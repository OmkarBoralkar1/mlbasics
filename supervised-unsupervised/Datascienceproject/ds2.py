import pandas as pd
import numpy as np
from ds3 import algosearch
def featurereduction_dimensionality(data):
    # Create a copy of the DataFrame for further processing
    data1 = data.copy()

    # Calculate 'price_per_sqft' by dividing 'price' by 'total_sqft' and multiplying by 1,000,000
    data1['price_per_sqft'] = data1['price'] * 1000000 / data1['total_sqft']

    # Uncomment the following lines if you want to print the head of the DataFrame and unique locations
    # print(data1.head())
    # print('The unique location is', data1.location.unique())
    # print(len(data1.location.unique()))

    # Remove leading and trailing whitespaces from 'location'
    data.location = data.location.apply(lambda x: x.strip())

    # Group by 'location' and count occurrences, then sort in descending order
    location_stats = data1.groupby('location')['location'].agg('count').sort_values(ascending=False)

    # Print the location statistics and the count of locations with 10 or fewer occurrences
    print(location_stats)
    print(len(location_stats[location_stats <= 10]))

    # Filter locations with 10 or fewer occurrences
    location_less_than_10 = location_stats[location_stats <= 10]
    # print(location_less_than_10)

    # Assign 'other' to locations with 10 or fewer occurrences
    data1.location = data1.location.apply(lambda x: 'other' if x in location_less_than_10 else x)

    # Print the unique locations after modification
    # print(data1.location.unique())

    # Print the modified DataFrame
    # print(data1)

    # Call the outlier_detection function
    outlier_detection(data1)

    # Return the modified DataFrame
    return data1


def outlier_detection(data1):
    # Placeholder for outlier detection logic

    # Print the input DataFrame
    # print('the outlier_detection', data1)

    # Remove rows where total_sqft/bhk is less than 300 (potential outliers) in this we remove the unusual data like
    # 600sqft home with 8 beadrooms
    data2 = data1[~(data1.total_sqft / data1.bhk < 300)]

    # Print the shape of the DataFrame after the first outlier removal
    # print(data2.shape)

    # Call the remove_pps_outliers function to further filter outliers in this we remove price_per_sqft is
    # very high or very low

    data3 = remove_pps_outliers(data2)

    # Print the shape of the DataFrame after the second outlier removal
    # print('the data 3.shape', data3.shape)

    # Call the remove_bhk_outliers function to remove outliers based on bhk in this we remove that row eg we remove
    #that data whose hase more price for 2 bhk than the 3bhk for the same location
    data4 = remove_bhk_outliers(data3)

    # Print the filtered DataFrame after removing outliers based on bhk
    # print('the data4 is', data4)

    # Filter rows where the number of bathrooms is less than bhk + 2
    data5 = data4[data4.bath < data4.bhk + 2]

    # Print the shape of the DataFrame after filtering based on bathrooms
    # print(data5.shape)

    # Drop unnecessary columns ('size' and 'price_per_sqft')
    data6 = data5.drop(['size', 'price_per_sqft'], axis=1)

    # Print the head of the modified DataFrame
    # print('the new dataframe head is', data6.head)

    # Call the algosearch function with the modified DataFrame
    algosearch(data6)


# Function to remove outliers based on price per square foot for each location
def remove_pps_outliers(df):
    # Initialize an empty DataFrame to store the filtered data
    df_out = pd.DataFrame()

    # Iterate over each group of data grouped by 'location'
    for key, subdf in df.groupby('location'):
        # Calculate mean and standard deviation of 'price_per_sqft' for the current location
        m = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)  # standard deviation

        # Filter out rows with 'price_per_sqft' outside the range (m-sd, m+sd)
        reduced_df = subdf[(subdf.price_per_sqft > (m - sd)) & (subdf.price_per_sqft <= (m + sd))]

        # Concatenate the filtered DataFrame with the overall DataFrame
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)

    # Return the DataFrame without outliers
    return df_out


# Function to remove outliers based on the number of bedrooms (bhk) for each location
def remove_bhk_outliers(df):
    # Initialize an array to store indices of rows to be excluded
    exclude_indices = np.array([])

    # Iterate over each group of data grouped by 'location'
    for location, location_df in df.groupby('location'):
        # Initialize a dictionary to store statistics for each number of bedrooms (bhk)
        bhk_stats = {}

        # Iterate over each group of data grouped by 'bhk' within the current location
        for bhk, bhk_df in location_df.groupby('bhk'):
            # Calculate mean, standard deviation, and count of 'price_per_sqft' for the current bhk
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }

        # Iterate over each group of data grouped by 'bhk' within the current location
        for bhk, bhk_df in location_df.groupby('bhk'):
            # Get the statistics for bhk=1; fix: change 'bhk==1' to 'bhk'
            stats = bhk_stats.get(bhk)

            # Check if statistics exist and the count is greater than 5
            if stats and stats['count'] > 5:
                # Exclude indices of rows with 'price_per_sqft' less than the mean for the current bhk
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values)

    # Return the DataFrame without outliers
    return df.drop(exclude_indices, axis='index')






