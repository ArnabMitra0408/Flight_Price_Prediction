import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import time
import pickle


def import_data():
    data=pd.read_csv("Data/Flight_Data_Processed.csv")
    X=data.drop(columns=['totalFare'],axis=1)
    Y=data['totalFare']
    del data
    return X,Y

def transform_data(X):
    standard=StandardScaler()
    standard=standard.fit(X)
    X=standard.transform(X)
    return X

def Linear_Regression(X,Y):
    start_time=time.time()

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize lists to store MSE for each fold
    lr_mse_scores = []
    lr_r2_scores=[]

    lr=LinearRegression()
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        
        # Train the model
        
        lr.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = lr.predict(X_test)
        
        # Calculate MSE and store it
        mse = mean_squared_error(y_test, y_pred)
        lr_mse_scores.append(mse)

        # Calculate r2 score and store it
        r2 = r2_score(y_test, y_pred)
        lr_r2_scores.append(r2)

    # Calculate average MSE across all folds
    avg_mse = np.mean(lr_mse_scores)
    print("Average MSE:", avg_mse)

    # Calculate average r2 across all folds
    avg_r2_lr = np.mean(lr_r2_scores)
    print("Average R2 score", avg_r2_lr)
    end_time=time.time()

    print(f"time taken for linear Regression {end_time-start_time} seconds")

    return lr,avg_mse,avg_r2_lr


def DecisionTree(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


    X_train=X_train.reset_index(drop=True)
    X_test=X_test.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    #Scaling
    standard=StandardScaler()

    standard=standard.fit(X_train)
    X_train=standard.transform(X_train)
    X_test=standard.transform(X_test)


    param_grid = {
        'max_depth': [8,10,12,15,16],
        'min_samples_split': [1,2,3],
        'min_samples_leaf': [2,3,4,5]
    }


    tree_regressor = DecisionTreeRegressor(random_state=42)

    grid_search = GridSearchCV(tree_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)

    best_regressor = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)

    # Evaluate the best model on the test set
    y_pred = best_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Test set MSE:", mse)

    r2_decision_tree = r2_score(y_test, y_pred)
    print("R2 score", r2_decision_tree)


    del X_train
    del X_test
    del y_train
    del y_test

    return best_regressor, mse, r2_decision_tree


def Best_model(dt_r2,r2_lr,dt,lr):
    if dt_r2>r2_lr:
        print('Saving Decision Tree')
        with open('decision_tree.pkl', 'wb') as file:
            pickle.dump(dt, file)
    else:
        print('Saving Linear Regression')
        with open('Linear_Regression.pkl', 'wb') as file:
            pickle.dump(lr, file)  



if __name__=='__main__':

    #importing data
    X,Y=import_data()


    #Scaling
    print("-------------------Applying Standard Scalar--------------")

    X_Std=transform_data(X)


    print("----------------Training Linear Regression----------------")

    lr,lr_mse,r2_lr=Linear_Regression(X_Std,Y)

    #Decision Tree Regressor:
    print("----------------Training Decision Tree Regressor----------------")

    dt,dt_mse,dt_r2=DecisionTree(X,Y)


    #Choosing best model:

    print("----------------Saving best model----------------")
    Best_model(dt_r2,r2_lr,dt,lr)


 



