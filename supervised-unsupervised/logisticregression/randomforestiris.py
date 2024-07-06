import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Load the Iris dataset
iris = load_iris()

# Plot the first two features for each class
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.scatter(iris.data[iris.target==i, 0], iris.data[iris.target==i, 1], label=f'Class {i}')
    plt.title(f'Iris Features (Class {i})')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.legend()

plt.tight_layout()


# Create a DataFrame for the Iris dataset
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(iris_data.drop(['target'], axis='columns'), iris_data.target, test_size=0.2)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=40)
model.fit(X_train, Y_train)
print("Model Accuracy:", model.score(X_test, Y_test))

# Predictions and confusion matrix
y_prediction = model.predict(X_test)
cm = confusion_matrix(Y_test, y_prediction)

# Create a Seaborn heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix for Iris Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
