# English:
#
# Bagging (Bootstrap Aggregating):
# Bagging is an ensemble learning technique that combines predictions from multiple models
# to improve overall performance and reduce variance. It involves training multiple instances
# of the same model on different subsets of the training data and aggregating their predictions.
# It helps reduce overfitting and variance in the model.
# Process:
#
# Bootstrap Sampling: Randomly select subsets of the training data with replacement.
# Each subset may contain some samples multiple times and exclude others.
#
# Model Training: Train a separate model on each bootstrap sample.
#
# Prediction Aggregation: Combine the predictions of all models, often using averaging
# for regression or voting for classification.
#
# Example:
# Suppose you have 100 samples of data. In bagging:
#
# Randomly select a subset (sample with replacement) of, let's say, 80 samples.
# Train a model on this subset.
# Repeat the process multiple times, creating different models on different subsets.
# Aggregate predictions by averaging (for regression) or voting (for classification).
# Bagging helps improve model stability and generalization by reducing the impact of outliers and overfitting,
# making it effective in scenarios with high variance.
#
# Hinglish:
#
# Bagging (Bootstrap Aggregating):
# Bagging ek aise ensemble learning technique hai jo multiple models ke predictions ko combine karke overall
# performance ko sudharne aur variance ko kam karne mein madad karta hai. Isme ek hi model ko training
# data ke alag-alag subsets par train karna aur unke predictions ko ekatrit karna shamil hai.
#Ye overfitting aur variance ko kam karne mein madad karta hai.
# Process:
#
# Bootstrap Sampling: Training data ke alag-alag subsets ko replacement ke saath randomly select karein.
# Har subset mein kuch samples multiple baar shamil ho sakte hain aur kuch exclude bhi ho sakte hain.
#
# Model Training: Har bootstrap sample par alag model ko train karein.
#
# Prediction Aggregation: Sabhi models ke predictions ko combine karein, jisme regression ke
# liye averaging aur classification ke liye voting ka istemal hota hai.
#
# Example:
# Maan lijiye aapke paas 100 data samples hain. Bagging mein:
#
# Random taur par, replacement ke saath, kuchh 80 samples ka subset chunein.
# Is subset par ek model ko train karein.
# Ye process kai baar repeat karein, alag-alag subsets par alag models banakar.
# Predictions ko averaging (regression ke liye) ya voting (classification ke liye) se combine karein.
# Bagging model ki stability aur generalization ko sudharne mein madad karta hai, outliers aur overfitting
# ke asar ko kam karke, jisse ye high variance wale scenarios mein prabhavi hota hai.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv('C:/Users/Omkar/OneDrive - somaiya.edu/Desktop/ML/diabetes.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Display summary statistics of the dataset
print(data.describe())

# Display the count of each class in the target variable
print(data.Outcome.value_counts())

# Separate features and target variable
X = data.drop('Outcome', axis=1)
Y = data.Outcome

# Standardize the features
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)
print(X_Scaled[:3])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_Scaled, Y, stratify=Y, test_size=0.2, random_state=20)

# ------------------ Decision Tree Classifier ------------------

# Create a Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=0)

# Use cross-validation to evaluate the performance of Decision Tree
scores_dt = cross_val_score(dt_model, X_Scaled, Y, cv=5)
print("Cross-Validation Scores (Decision Tree):", scores_dt)
print("Mean Cross-Validation Score (Decision Tree):", scores_dt.mean())

# ------------------ Random Forest Classifier ------------------

# Create a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

# Use cross-validation to evaluate the performance of Random Forest
scores_rf = cross_val_score(rf_model, X_Scaled, Y, cv=5)
print("Cross-Validation Scores (Random Forest):", scores_rf)
print("Mean Cross-Validation Score (Random Forest):", scores_rf.mean())

# ------------------ Bagging Classifier with Decision Tree ------------------

# Create a Bagging Classifier with Decision Tree as the base model
bag_dt_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)

# Use cross-validation to evaluate the performance of Bagging + Decision Tree
scores_bag_dt = cross_val_score(bag_dt_model, X_Scaled, Y, cv=5)
print("Cross-Validation Scores (Bagging + Decision Tree):", scores_bag_dt)
print("Mean Cross-Validation Score (Bagging + Decision Tree):", scores_bag_dt.mean())

# ------------------ Bagging Classifier with Random Forest ------------------

# Create a Bagging Classifier with Random Forest as the base model
bag_rf_model = BaggingClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)

# Use cross-validation to evaluate the performance of Bagging + Random Forest
scores_bag_rf = cross_val_score(bag_rf_model, X_Scaled, Y, cv=5)
print("Mean Cross-Validation Score (Bagging + Random Forest):", scores_bag_rf.mean())

# Plotting the comparison of mean cross-validation scores
models = ['Decision Tree', 'Random Forest', 'Bagging + Decision Tree', 'Bagging + Random Forest']
mean_scores = [scores_dt.mean(), scores_rf.mean(), scores_bag_dt.mean(), scores_bag_rf.mean()]

plt.figure(figsize=(10, 6))
plt.bar(models, mean_scores, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Models')
plt.ylabel('Mean Cross-Validation Score')
plt.title('Comparison of Mean Cross-Validation Scores')
plt.show()
