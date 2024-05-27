# English:
# K-Nearest Neighbors (KNN) is a supervised machine learning algorithm for classification.
# It classifies an input by considering the majority class among its k-nearest neighbors in the feature space.
# The choice of k influences the algorithm's sensitivity to local variations. The KNN classification formula involves
# calculating distances, often using Euclidean distance, and selecting the class with the highest frequency among the
# k neighbors.
#
# Hinglish:
# K-Nearest Neighbors (KNN) ek supervised machine learning algorithm hai jo classification ke liye istemal hota hai.
# Ye ek input ko classify karta hai by considering uske kareebi k neighbors mein se majority class.
# K ka chayan algorithm ki sensitivity ko sthapit karta hai. KNN classification formula mein dooriyan calculate hoti hain,
# jisme Euclidean distance aksar istemal hota hai, aur k neighbors mein se jo class adhik baar aati hai,
# wahi chuni jati hai.

# Formula for KNN

# ^y = majority class among the k-nearest neighbors of the new data point
#
# In this formula:
# - ^y represents the predicted class for the new data point.
# - The majority class is determined by considering the class labels of the k-nearest neighbors.
# - The distance metric, often Euclidean distance, measures the distance between data points in the feature space.
# K value should not very high and not very low

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data['flower_name'] = data.target.apply(lambda x: iris.target_names[x])

# Create separate dataframes for each target
data0 = data[data.target == 0]
data1 = data[data.target == 1]
data2 = data[data.target == 2]

# Plotting with Seaborn
plt.figure(figsize=(12, 5))

# Matplotlib Scatter Plots
plt.subplot(1, 2, 1)
plt.scatter(data0['sepal length (cm)'], data0['sepal width (cm)'], c='green', marker='+', label='Setosa')
plt.scatter(data1['sepal length (cm)'], data1['sepal width (cm)'], c='blue', marker='+', label='Versicolor')
plt.scatter(data2['sepal length (cm)'], data2['sepal width (cm)'], c='orange', marker='+', label='Virginica')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()

# Seaborn Heatmap
plt.subplot(1, 2, 2)
X = data.drop(['target', 'flower_name'], axis='columns')
Y = data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
y_predict = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_predict)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')

# Print Confusion Matrix
print("Confusion Matrix:")
print(cm)

# Print Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, y_predict))

# Display both plots
plt.tight_layout()
plt.show()

