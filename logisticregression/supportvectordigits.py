import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM classifier with RBF kernel
model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_train_scaled, y_train)

# Measure accuracy with RBF kernel
y_pred_rbf = model_rbf.predict(X_test_scaled)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Accuracy with RBF kernel: {accuracy_rbf:.2%}")

# Train SVM classifier with linear kernel
model_linear = SVC(kernel='linear')
model_linear.fit(X_train_scaled, y_train)

# Measure accuracy with linear kernel
y_pred_linear = model_linear.predict(X_test_scaled)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Accuracy with Linear kernel: {accuracy_linear:.2%}")

# Tune the model parameters (C and gamma) - You can experiment with different values
model_tuned = SVC(C=1.0, gamma='scale', kernel='rbf')
model_tuned.fit(X_train_scaled, y_train)

# Measure accuracy with tuned parameters
y_pred_tuned = model_tuned.predict(X_test_scaled)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Accuracy with Tuned parameters: {accuracy_tuned:.2%}")

# Use PCA for dimensionality reduction to visualize digits in 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Scatter plot the reduced data points
plt.figure(figsize=(12, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k', cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar(label='Digit Label', ticks=range(10))
plt.title('2D PCA of Digits Dataset')
plt.xlabel('Principal Component 1 (Width)')
plt.ylabel('Principal Component 2 (Height)')
plt.show()
