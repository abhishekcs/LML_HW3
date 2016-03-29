
import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.grid_search import GridSearchCV

def find_index(value, li):
    l = len(li)
    for i in range(l):
        if(li[i] == value):
            return i
    return -1


def find_best_value_for_parameter(X, y, other_parameter_values,
                                  parameter_name,
                                 first_level_values,
                                 second_level_values):
    grid = {parameter_name: first_level_values}
    clf = RandomForestClassifier()
    clf.set_params(**other_parameter_values)
    grid_search = GridSearchCV(estimator = clf, param_grid = grid, scoring='roc_auc', cv=5, verbose=100)
    grid_search.fit(X, y)
    ind = find_index(grid_search.best_params_[parameter_name], first_level_values)
    if(ind == -1):
        return grid_search.best_params_[parameter_name]
    else:
        grid = {parameter_name: second_level_values[ind]}
        grid_search = GridSearchCV(estimator = clf, param_grid = grid, scoring='roc_auc', cv=5, verbose=100)
        grid_search.fit(X, y)
        return grid_search.best_params_[parameter_name]
    

if __name__ == "__main__": 
    # load labeled data
    train_df = pd.read_csv('train.csv')
    train_data = pd.DataFrame.as_matrix(train_df)
    y = train_data[:,0]; X = train_data[:,1:9];


    ## Hyperparameter Tuning
    other_parameter_values = {}
    parameter_name = 'max_features'
    other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                          parameter_name,
                                                                          [1,2,3,4,5,6,7,8],
                                                                          {0:[1],
                                                                           1:[2],
                                                                           2:[3],
                                                                           3:[4],
                                                                           4:[5],
                                                                           5:[6],
                                                                           6:[7],
                                                                           7:[8]})


    print other_parameter_values[parameter_name]
    parameter_name = 'n_estimators'
    other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                          parameter_name,
                                                                          [50,100,200],
                                                                          {0:[50,60,70,80,90],
                                                                           1:[100,120,140,160,180],
                                                                           2:[200,240,280,320,360]})


    print other_parameter_values[parameter_name]
    parameter_name = 'min_samples_leaf'
    other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                          parameter_name,
                                                                          [1,2,4,8,16,32,64,128,256],
                                                                          {0:[1],
                                                                           1:[2,3],
                                                                           2:[4,5,6,7],
                                                                           3:[8,10,12,14],
                                                                           4:[16,20,24,28],
                                                                           5:[32,40,48,56],
                                                                           6:[65,80,96,112],
                                                                           7:[128,160,192,224]})

    parameter_name = 'max_depth'
    other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                              parameter_name,
                                                                              [9,10,11,12,13,14,15,23,24,25],
                                                                              {0:[9],
                                                                               1:[10],
                                                                               2:[11],
                                                                               3:[12],
                                                                               4:[13],
                                                                               5:[14],
                                                                               6:[15],
                                                                               7:[23],
                                                                               8:[24],
                                                                               9:[25]})


    parameter_name = 'min_samples_split'
    other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                              parameter_name,
                                                                              [1,2,4,8,16,32,64,128,256],
                                                                              {0:[1],
                                                                               1:[2,3],
                                                                               2:[4,5,6,7],
                                                                               3:[8,10,12,14],
                                                                               4:[16,20,24,28],
                                                                               5:[32,40,48,56],
                                                                               6:[65,80,96,112],
                                                                               7:[128,160,192,224]})
      

    ## fit training data with best hyperparemters
    other_parameter_values = {'n_estimators': 270, 'max_features': 4, 
                              'max_depth': 23, 'min_samples_leaf': 2,
                              'min_samples_split':8, 'criterion':'entropy'}
    forest = RandomForestClassifier()
    forest.set_params(**other_parameter_values)
    forest.fit(X,y)


    ## load test data
    test_df = pd.read_csv('test.csv')
    test_data = pd.DataFrame.as_matrix(test_df)
    id_test = test_data[:,0]; X_test = test_data[:,1:9];



    # predictions on unlabeled data
    y_test_pred = forest.predict_proba(X_test)
    ans = pd.DataFrame({'Id': id_test, 'Action' : y_test_pred[:,1]})
    ans.to_csv('hw3rf_en.csv', index=False, columns=['Id', 'Action'])

