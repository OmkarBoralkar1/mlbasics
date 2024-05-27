# K-Means Clustering is an unsupervised machine learning algorithm used for partitioning a dataset
# into K distinct, non-overlapping subsets or clusters. The algorithm assigns data points to
# clusters based on their similarity. It iteratively refines cluster assignments by minimizing
# the sum of squared distances between data points and the centroid of their assigned cluster.
# The number K represents the predetermined number of clusters in the data. The algorithm continues until convergence,
# and the final result is a set of K clusters, each characterized by its centroid. K-Means Clustering is
# widely applied in various fields, such as customer segmentation, image compression, and anomaly detection.

# //////////////////////////////////////////////////////////////////

# K-Means Clustering ek unsupervised machine learning algorithm hai jiska istemal dataset ko K distinct,
# non-overlapping subsets ya clusters mein bhaatne ke liye hota hai. Is algorithm mein data points ko unki
# similarity ke adhar par clusters mein assign kiya jata hai. Ye cluster assignments ko iteratively refine karta hai,
# jisme data points aur unke assigned cluster ke centroid ke beech ke squared distances ka sum minimize hota hai.
# Yahaan, K ek pahle se nirdharit kiya gaya number of clusters ko represent karta hai. Ye algorithm convergence tak
# chalta hai aur ant mein ek set of K clusters hasil hota hai, jisme har ek cluster apne centroid ke adhar par
# characterize hota hai. K-Means Clustering ko vibhinn kshetron mein, jaise ki customer segmentation, image compression,
# aur anomaly detection, mein vyapak roop se istemal kiya jata hai.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

file_path = 'C:/Users/Omkar/OneDrive - somaiya.edu/Desktop/ML/homeprices.csv'

# Specify the range of rows you want to read (rows 76 to 97)
start_row = 119
end_row = 140
row_range = range(start_row, end_row)

# Read the CSV file for the specified row range
data = pd.read_csv(file_path, delimiter='\t', skiprows=lambda x: x not in row_range)

# Specify the range of columns you want to keep (columns 1 to 3)
start_column = 0
end_column = 3  # Use end_column + 1 since Python slicing is exclusive on the end
column_range = range(start_column, end_column)

# Keep only the specified columns
data = data.iloc[:, column_range]

# Create a figure with 2 rows and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(18, 9))

# Subplot 1: Without Cluster
axs[0].scatter(data['Age'], data['Income'], label='Data Points', color='gray')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Income')
axs[0].set_title('Scatter Plot of Age vs Income without cluster')
axs[0].legend()

# Subplot 2: With Cluster

# Scale the 'Income' and 'Age' columns together
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Age', 'Income']])
data[['Age', 'Income']] = data_scaled
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(data[['Age', 'Income']])
data['cluster'] = y_predicted
print('the data after scaling\n', data)
df0 = data[data.cluster == 0]
df1 = data[data.cluster == 1]
df2 = data[data.cluster == 2]
axs[1].scatter(df0['Age'], df0['Income'], color='red', label='Data Points for df0')
axs[1].scatter(df1['Age'], df1['Income'], color='blue', label='Data Points for df1')
axs[1].scatter(df2['Age'], df2['Income'], color='green', label='Data Points for df2')
axs[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='purple', marker='*', label='Centroid')
axs[1].set_xlabel('Age')
axs[1].set_ylabel('Income')
axs[1].set_title('Scatter Plot of Age vs Income including cluster')
axs[1].legend()

# Subplot 3: SSE
k_rng = range(1, 10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(data[['Age', 'Income']])
    sse.append(km.inertia_)

print('the sse value is', sse)
axs[2].set_xlabel('K')
axs[2].set_ylabel('Sum of Squared Error')
axs[2].plot(k_rng, sse)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
