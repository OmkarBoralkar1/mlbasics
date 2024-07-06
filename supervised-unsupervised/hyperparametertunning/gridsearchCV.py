# Hyperparameter tuning optimizes a machine learning model's performance by finding the best hyperparameters.
# GridSearchCV systematically explores a predefined hyperparameter space by evaluating various combinations
# using cross-validation. It helps select the combination with the highest performance, ensuring a robust and
# generalized model without manual tuning. However, it can be computationally expensive for large hyperparameter
# spaces.

# //////////////////////////////////////////////////////////////////////////

# Hyperparameter tuning ek process hai jisme machine learning model ke performance ko behtar banane ke liye behtareen
# hyperparameters ko dhoondha jata hai. GridSearchCV ek predefined hyperparameter space ko systematic taur par explore
# karta hai, various combinations ko cross-validation ke istemal se evaluate karte hue. Ye process manual tuning ko
# rokta hai aur sabse acchi performance wale combination ko chunta hai, jisse ek robust aur generalized model milta hai.
# Lekin, ye bade hyperparameter spaces ke liye computational roop se mehenga ho sakta hai.

from sklearn import svm,datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
iris =datasets.load_iris()
data =pd.DataFrame(iris.data,columns=iris.feature_names)
data['flower'] =iris.target
data['flower'] =data['flower'].apply(lambda x:iris.target_names[x])
print(data[47:52])
X_train,X_test,Y_train,Y_test=train_test_split(iris.data,iris.target,test_size=0.3)

model=svm.SVC(kernel='rbf',C=10,gamma='auto')
model.fit(X_train,Y_train)
print('the model score is using svm',model.score(X_test,Y_test))

# Define the hyperparameter grid
param_grid = {'kernel': ['rbf', 'linear'], 'C': [1, 10, 20]}

# Create an SVM model
svm_model = svm.SVC(gamma='auto')

# Create a KFold object (let's use k=5 for 5-fold cross-validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(svm_model, param_grid, cv=5, return_train_score=False,scoring='accuracy')

# Fit the model to the training data
grid_search.fit(iris.data, iris.target)
d1=pd.DataFrame(grid_search.cv_results_)
d2=d1[['param_C','param_kernel','mean_test_score']]
print(d2)
# Print the best parameters and corresponding accuracy
print("Best Parameters using GridSearchCV: ", grid_search.best_params_)
print("Best Accuracy using GridSearchCV: ", grid_search.best_score_)

# Access the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_model.score(X_test, Y_test)
print("Test Set Accuracy using k fold cross using GridSearchCV: ", test_accuracy)

grid_search1 = RandomizedSearchCV(svm_model, param_grid, cv=5, return_train_score=False,scoring='accuracy',n_iter=2)

grid_search1.fit(iris.data, iris.target)
d3=pd.DataFrame(grid_search1.cv_results_)
d4=d3[['param_C','param_kernel','mean_test_score']]
print(d4)
# Print the best parameters and corresponding accuracy
print("Best Parameters using RandomizedSearchCV: ", grid_search1.best_params_)
print("Best Accuracy using RandomizedSearchCV: ", grid_search1.best_score_)

# Access the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_model.score(X_test, Y_test)
print("Test Set Accuracy using k fold cross using RandomizedSearchCV: ", test_accuracy)

# SVM model parameters
svm_params = {
    'model': svm.SVC(gamma='auto'),
    'params': {
        'C': [1, 10, 20],
        'kernel': ['rbf', 'linear']
    }
}

# RandomForestClassifier parameters
rf_params = {
    'model': RandomForestClassifier(),
    'params': {
        'n_estimators': [1, 5, 10]
    }
}

# LogisticRegression parameters
lr_params = {
    'model': LogisticRegression(solver='liblinear', multi_class='auto'),
    'params': {
        'C': [1, 5, 10]
    }
}

# List of models and their parameters
models_params = {'svm': svm_params, 'random_forest': rf_params, 'logistic_regression': lr_params}

scores=[]

for model_name ,mp in  models_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False) #
    clf.fit(iris.data,iris.target)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_
    })
d5=pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(d5)