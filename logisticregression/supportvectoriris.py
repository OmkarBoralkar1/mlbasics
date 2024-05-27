# Support Vector Machines (SVM) is a machine learning algorithm for classification and regression tasks.
# SVM seeks a hyperplane in an n-dimensional space that maximizes the margin between classes,
# aiding in effective separation.
# It works by identifying support vectors, the closest data points to the hyperplane,
# which determine its position and orientation.
# SVM handles non-linear relationships using a kernel trick, mapping data to a higher-dimensional space.
# The regularization parameter (C) balances the desire for a large margin with minimizing training errors.


# ///////////////////////////////////////////////
# Support Vector Machines (SVM) ek machine learning algorithm hai jo classification aur regression tasks ke liye istemal
# hota hai.
# SVM ek aise hyperplane ko dhundhta hai jo n-dimensional space mein classes ke beech mein maximum margin create kare,
# jisse effective separation ho sake. Yeh kaam karta hai support vectors ko pehchan kar,
# jo hyperplane ke sabse nazdeek ke data points hote hain, jinse hyperplane ki position aur orientation decide hoti hai.
# SVM non-linear relationships ko handle karne ke liye kernel trick ka istemal karta hai,
# jo data ko ek higher-dimensional space mein map karta hai.
# Regularization parameter (C) margin ko bada karne aur training errors ko kam karne mein madad karta hai.

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris= load_iris()
print(dir(iris))
print(iris.feature_names)

data =pd.DataFrame(iris.data ,columns=iris.feature_names)
data['target']=iris.target
data['flower_name'] = data.target.apply(lambda x: iris.target_names[x])
print(data.head())
print(iris.target_names)
print(data[data.target == 1].head())
print(data[data.target == 2].head())
data0=data[data.target == 0]
data1=data[data.target == 1]
data2=data[data.target == 2]
plt.scatter(data0['sepal length (cm)'], data0['sepal width (cm)'], c='green', marker='+', label='data0')
plt.scatter(data1['sepal length (cm)'], data1['sepal width (cm)'], c='blue', marker='+', label='data1')
# plt.scatter(data2['sepal length (cm)'], data2['sepal width (cm)'], c='orange', marker='+', label='data2')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
X=data.drop(['target','flower_name'],axis='columns')
Y=data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model=SVC()
model.fit(X_train,Y_train)
print('the model score is',model.score( X_test,Y_test))

plt.legend()  # Add this line to show legend
plt.show()