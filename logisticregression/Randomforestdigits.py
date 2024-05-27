# English:
#
# Random Forest is a machine learning ensemble algorithm used for classification and regression.
# It builds multiple decision trees during training and merges them together to get a more accurate and stable prediction.
# Each tree in the forest is trained on a random subset of the data,
# and the final prediction is made by averaging or taking a majority vote of the individual tree predictions.
# This method helps reduce overfitting and increases the overall accuracy of the model.
#

# Hinglish:
#
# Random Forest ek machine learning ensemble algorithm hai, jo classification aur regression ke liye istemal hota hai.
# Isme training ke dauran kai decision trees banaye jaate hain,
# jo  saath milakar ek adhik satik aur sthir prediction prapt karne mein madad karte hain.
# Har tree forest mein ek random data subset par train hota hai,
# aur antim prediction individual tree predictions ka average ya majority vote lekar kiya jata hai.
# Yeh tarika overfitting ko kam karta hai aur model ki overall accuracy badhata hai.


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Load the digits dataset
digits = load_digits()

# Plot the digits from 0 to 9
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Digit {i}")
    ax.axis('off')

data = pd.DataFrame(digits.data)
data['target'] = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(data.drop(['target'], axis='columns'), data.target, test_size=0.2)
model = RandomForestClassifier(n_estimators=40)
model.fit(X_train, Y_train)
print("Model Accuracy:", model.score(X_test, Y_test))

y_prediction = model.predict(X_test)
cm = confusion_matrix(Y_test, y_prediction)


# Create a Seaborn heatmap for the confusion matrix
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
