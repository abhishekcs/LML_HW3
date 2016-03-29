import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb

from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

def find_index(value, li):
    l = len(li)
    for i in range(l):
        if(li[i] == value):
            return i
    return -1


def find_best_value_for_parameter(X, y,
                                  other_parameter_values,
                                  parameter_name,
                                  first_level_values,
                                  second_level_values):
    grid = {parameter_name: first_level_values}
    clf = xgb.XGBClassifier()
    clf.set_params(**other_parameter_values)
    grid_search = GridSearchCV(estimator = clf, param_grid = grid, scoring='roc_auc', cv = 5, verbose = 100)
    grid_search.fit(X, y)
    ind = find_index(grid_search.best_params_[parameter_name], first_level_values)
    if(ind == -1):
        return grid_search.best_params_[parameter_name]
    else:
        grid = {parameter_name: second_level_values[ind]}
        grid_search = GridSearchCV(estimator = clf, param_grid = grid, scoring='roc_auc', cv = 5, verbose = 100)
        grid_search.fit(X, y)
        return grid_search.best_params_[parameter_name]
    
    
if __name__ == "__main__": 
    # load labeled data
    train_df = pd.read_csv('train.csv')
    train_data = pd.DataFrame.as_matrix(train_df)
    y = train_data[:,0]; X = train_data[:,1:9];
    # load unlabeled data
    test_df = pd.read_csv('test.csv')
    test_data = pd.DataFrame.as_matrix(test_df)
    id_test = test_data[:,0]; X_test = test_data[:,1:9];
    
    X_fitting = np.vstack([X, X_test])
    enc = OneHotEncoder(categorical_features='all')
    enc.fit(X_fitting)
    X = enc.transform(X)


other_parameter_values = {'learning_rate': 0.2,
                          'n_estimators':90,
                          'max_depth':9,
                          'objective':'binary:logistic'}

parameter_name = 'n_estimators'
other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                      parameter_name,
                                                                      [50,100,200,400,800],
                                                                      {0:[50,60,70,80,90],
                                                                       1:[100,120,140,160,180],
                                                                       2:[200,240,280,320,360],
                                                                       3:[400,480,560,640,720],
                                                                       4:[880,960, ]})




parameter_name = 'max_depth'
other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                      parameter_name,
                                                                      [4, 8, 16, 32],
                                                                      {0:[4,5,6,7],
                                                                       1:[8,9,10,11,12,13,14,15],
                                                                       2:[16,20,24,28],
                                                                       3:[32, 40, 48, 56]})






parameter_name = 'colsample_bytree'
other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                      parameter_name,
                                                                      [0.1, 0.2],
                                                                      {0:[0.1,0.12,0.14,0.16,0.18],
                                                                       1:[0.2,0.22,0.24,0.26,0.28]})






parameter_name = 'min_child_weight'
other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                      parameter_name,
                                                                      [0.001, 0.01],
                                                                      {0:[0.001, 0.002, 0.004, 0.008],
                                                                       1:[0.01 , 0.02 , 0.04 , 0.08 ]})






parameter_name = 'max_delta_step'
other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                      parameter_name,
                                                                      [1, 2, 4, 8],
                                                                      {0:[1, 1.2, 1.4, 1.6, 1.8],
                                                                       1:[2, 2.4, 2.8, 3.2, 3.6],
                                                                       2:[4, 4.8, 5.6, 6.4, 7.2],
                                                                       3:[8, 9.6, 11.2, 12.8, 14.4]})



# train with best parameters
xgb_clsf = xgb.XGBClassifier(learning_rate =0.2,
                             n_estimators=880, 
                             max_depth=16,
                             objective= 'binary:logistic',
                             min_child_weight = 0.04,
                             colsample_bytree = 0.22,
                             max_delta_step=4)

xgb_clsf.fit(X, y)


# load unlabeled data
test_df = pd.read_csv('test.csv')
test_data = pd.DataFrame.as_matrix(test_df)
id_test = test_data[:,0]; X_test = test_data[:,1:9];
X_test = enc.transform(X_test)
# predictions on unlabeled data
y_test_pred = xgb_clsf.predict_proba(X_test)
ans = pd.DataFrame({'Id': id_test, 'Action' : y_test_pred[:,1]})
ans.to_csv('XGB-One-Hot.csv', index=False, columns=['Id', 'Action'])