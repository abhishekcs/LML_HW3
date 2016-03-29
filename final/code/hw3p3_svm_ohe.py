import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import svm

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
    clf = svm.SVC()
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


other_parameter_values = {'C': 1,
                          'probability':True,
                          'kernel':'linear'}
parameter_name = 'C'
other_parameter_values[parameter_name] = find_best_value_for_parameter(X, y, other_parameter_values,
                                                                      parameter_name,
                                                                      [0.01,0.1,1,10,100],
                                                                      {0:[0.01, 0.02, 0.04, 0.08],
                                                                       1:[0.1, 0.2, 0.4, 0.8],
                                                                       2:[1, 2, 4, 8],
                                                                       3:[10, 20, 40, 80],
                                                                       4:[100, 200, 400, 800]})



# train with best parameters
svm_clsf = svm.SVC(C = 2, kernel = 'linear', probability = True)
svm_clsf.fit(X, y)


# load unlabeled data
test_df = pd.read_csv('test.csv')
test_data = pd.DataFrame.as_matrix(test_df)
id_test = test_data[:,0]; X_test = test_data[:,1:9];
X_test = enc.transform(X_test)
# predictions on unlabeled data
y_test_pred = svm_clsf.predict_proba(X_test)
ans = pd.DataFrame({'Id': id_test, 'Action' : y_test_pred[:,1]})
ans.to_csv('SVM-One-Hot.csv', index=False, columns=['Id', 'Action'])
