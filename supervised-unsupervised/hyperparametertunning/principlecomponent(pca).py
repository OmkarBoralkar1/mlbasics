#PCA(principle component analysis)
# English:
# Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and statistics.
# The primary goal of PCA is to transform high-dimensional data into a lower-dimensional representation,
# capturing the most significant variability in the data. It achieves this by identifying the principal components,
# which are orthogonal vectors pointing in the directions of maximum variance. These components are ordered by
# the amount of variance they capture, allowing for the retention of essential information while reducing dimensionality.
# PCA finds applications in various fields, including image processing, feature extraction, and data visualization.
#
# Hindi:
# Principal Component Analysis (PCA) ek dimensionality reduction technique hai jo machine learning aur statistics
# mein istemal hoti hai. PCA ka mukhya uddeshya hai ki visheshagya data ko kam-dimensional representation mein badalna,
# jo data mein sabse mahatvapurna variabilities ko capture kare. Isko ye karne mein sahayak hote hain principal components
# , jo ki orthogonal vectors hote hain aur maximum variance ke directions mein point karte hain.
# Ye components variance capture kiye gaye amount ke anusaar order mein hote hain,
# jisse ki dimensionality ko kam kiya ja sake bina mahatvapurna information khoaye.
# PCA ko alag-alag kshetron mein istemal kiya jata hai, jaise ki image processing, feature extraction,
# aur data visualization mein.



# Things to keep in mind before using PCA
# 1) Scale Feature Before Applying PCA
# 2) Accuracy might drop

# PCA is called dimensionality reduction technique as it helps us to reduce dimensions
# very useful for dimensionality curse problem
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

dataset = load_digits()

data1 = pd.DataFrame(dataset.data, columns=dataset.feature_names)

X = data1
Y = dataset.target
scalar = StandardScaler()
X_scale = scalar.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.2, random_state=30)

model = RandomForestClassifier(n_estimators=40)
model.fit(X_train, Y_train)
print('The accuracy of that model without pca is', model.score(X_test, Y_test))
accuracy_no_pca = model.score(X_test, Y_test)

pca = PCA(0.98)
X_pca = pca.fit_transform(X_scale)

X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_pca, Y, test_size=0.2, random_state=30)

model_with_pca = LogisticRegression(max_iter=10000)
model_with_pca.fit(X_train_pca, Y_train_pca)
accuracy_with_pca = model_with_pca.score(X_test_pca, Y_test_pca)
print('The accuracy of that model with pca is',accuracy_with_pca)
# Plotting the comparison graph
labels = ['Without PCA', 'With PCA']
accuracies = [accuracy_no_pca, accuracy_with_pca]

plt.bar(labels, accuracies, color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison with and without PCA')
plt.ylim([0, 1])  # Set y-axis limits to match accuracy range (0 to 1)

# Scatter plots for data points with PCA
plt.figure(figsize=(12, 6))

# Plot training data points
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train_pca, cmap='viridis')
plt.title('Training Data Points with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot testing data points
plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=Y_test_pca, cmap='viridis')
plt.title('Testing Data Points with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()