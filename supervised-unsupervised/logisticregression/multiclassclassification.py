# Multiclassification in machine learning involves predicting one of several classes for a given input.
# For instance, in a language identification task, the model may predict whether a given text is in English,
# Spanish, or French. Algorithms like logistic regression, support vector machines, and neural networks
# are commonly used for multiclass classification. The model is trained on labeled data with examples
# from each class to learn to distinguish between them.
# /////////////////////////////////////////////////////////////////////////////
# Machine learning mein multiclassification ka kaam hota hai ki kisi input ko kuch se adhik classes mein predict karna.
# Jaise ki language identification mein, model ye predict kar sakta hai ki koi di gayi text English mein hai,
# Spanish mein hai, ya phir French mein hai. Logistic regression, support vector machines, aur neural networks
# jaise algorithms commonly multiclass classification ke liye istemal hote hain.
# Model labeled data par train hota hai jisme har class se related examples hote hain taki wo unhe alag kar sake.
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

# Set a fixed random seed for reproducibility
np.random.seed(42)

# Number of data points
num_samples = 100

# Hypothetical election dataset (replace with your actual data)
data = {
    'Public Fulfillment': np.random.randint(30, 70, num_samples),
    'Incumbency': np.random.choice([0, 1], size=num_samples),
    'Public Opinion': np.random.randint(40, 80, num_samples),
    'Winning Party': np.random.choice(['A', 'B', 'C', 'D'], size=num_samples)
}

election_df = pd.DataFrame(data)

# Split the dataset into features (X) and target variable (y)
X = election_df.drop('Winning Party', axis=1)
y = election_df['Winning Party']

# Ensure you have data for all parties A, B, C, D
# If your actual data doesn't include samples for C and D, you need to provide such data.

# Create individual classifiers
logreg_classifier = LogisticRegression(random_state=42)  # Set random_state for reproducibility
rf_classifier = RandomForestClassifier(random_state=42)  # Set random_state for reproducibility
svm_classifier = SVC(probability=True, random_state=42)  # Set random_state for reproducibility

# Create a voting classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('logreg', logreg_classifier),
        ('random_forest', rf_classifier),
        ('svm', svm_classifier)
    ],
    voting='soft'  # 'hard' or 'soft' voting
)

# Train the voting classifier
voting_classifier.fit(X, y)

# Take user input for party name
user_party_name = input("Enter the party name (A, B, C, D): ")

# Check if the user's party is in the classes
if user_party_name not in voting_classifier.classes_:
    print(f"Invalid party name: {user_party_name}")
else:
    # Take user input for party-specific data
    user_public_fulfillment = float(input("Enter Public Fulfillment: "))
    user_incumbency = int(input("Enter Incumbency (1 for incumbent, 0 otherwise): "))
    user_public_opinion = float(input("Enter Public Opinion: "))

    # Make predictions for the user input
    user_input = pd.DataFrame({
        'Public Fulfillment': [user_public_fulfillment],
        'Incumbency': [user_incumbency],
        'Public Opinion': [user_public_opinion]
    })

    # Display the predicted winning probability for the specified party
    user_probabilities = voting_classifier.predict_proba(user_input)
    party_index = list(voting_classifier.classes_).index(user_party_name)
    predicted_probability = user_probabilities[0][party_index]
    print(f"The predicted winning probability for {user_party_name} is: {predicted_probability:.2%}")
