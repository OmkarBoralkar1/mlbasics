import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle
import json
def algosearch(data):
    print("Inside algosearch function\n", data)
    dummies = pd.get_dummies(data.location)
    data1 = pd.concat([data, dummies.drop('other', axis=1)], axis=1)
    data2 = data1.drop('location', axis=1)
    X = data2.drop('price', axis=1)
    Y = data2.price
    # print('the y datatype is',Y.dtype)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    lr_clf = LinearRegression()
    lr_clf.fit(X_train, Y_train)
    # print('the lr_clf scores is',lr_clf.score(X_train, Y_train))
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
    print('the linearregression score is',cross_val_score(LinearRegression(), X, Y, cv=cv))
    # Create an SVM model
    svm_model = svm.SVC(gamma='auto')
    algos ={
        'linear_regression':{
            'model':LinearRegression(),
            'params':{
                'normalize':[True,False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'decesion_tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
            }
        },
        # RandomForestClassifier parameters
        # RandomForestRegressor parameters
        'rf_params': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [1, 5, 10]
            }
        }

    }

    scores =[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name ,config in algos.items() :
        gs=GridSearchCV(config['model'],config['params'],cv=cv ,return_train_score=False)
        gs.fit(X,Y)
        scores.append({
            'model':algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
    # print(pd.DataFrame(scores,columns=['model','best_score','best_params']))

    def predict_price(location, sqft, bath, bhk):
        loc_index = location in X.columns

        x = np.zeros(len(X.columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index:
            x[X.columns.get_loc(location)] = 1

        return lr_clf.predict([x])[0]
    print('the predected price for the given location',predict_price('1st phase Jp Nagar',1000,2,3))
    with open('banglore_home_prices_model_pickel','wb') as f:
        pickle.dump(lr_clf,f)
    columns ={
        'data_coloumns' :[col.lower() for col in X.columns]
    }
    with open('coloumns.json','w') as f:
        f.write(json.dumps(columns))

