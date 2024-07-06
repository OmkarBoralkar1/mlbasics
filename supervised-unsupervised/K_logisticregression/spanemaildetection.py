import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from urllib.request import urlopen
from zipfile import ZipFile
import io
# Download the SMS Spam Collection dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
with urlopen(url) as zipresp:
    with ZipFile(io.BytesIO(zipresp.read())) as zfile:
        zfile.extractall()
# Read the dataset into a DataFrame
sms_df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
sms_df['spam'] = sms_df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sms_df['message'], sms_df['spam'], test_size=0.25, random_state=42)

# Create a pipeline with CountVectorizer and MultinomialNB
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train the classifier using the pipeline
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

# Example emails
new_emails = []
while True:
    user_input = input("Enter a new email (type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break
    new_emails.append(user_input)

# Predict whether the new emails are spam or ham using the pipeline
predictions = classifier.predict(new_emails)

# Output the predictions
for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Email {i + 1} is predicted as spam.")
    else:
        print(f"Email {i + 1} is predicted as not spam (ham(good email).")
