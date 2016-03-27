import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb

from sklearn.grid_search import GridSearchCV

if __name__ == "__main__": 
	# load labeled data
	train_df = pd.read_csv('train.csv')
	train_data = pd.DataFrame.as_matrix(train_df)
	y = train_data[:,0]; X = train_data[:,1:9];

	# cross validation
	grid = {'min_child_weight':range(1,6,2), 'colsample_bytree':[i/10.0 for i in range(6,10)]}
	grid_search = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.2, n_estimators=90, max_depth=9, objective= 'binary:logistic', seed=27), param_grid = grid, scoring='roc_auc', cv=5)
	grid_search.fit(X, y)
	print grid_search.best_params_

	# train with best parameters
	xgb_clsf = xgb.XGBClassifier(learning_rate =0.2, n_estimators=90, max_depth=9, objective= 'binary:logistic',
	 min_child_weight = grid_search.best_params_['min_child_weight'], colsample_bytree=grid_search.best_params_['colsample_bytree'])
	xgb_clsf.fit(X, y)

	# load unlabeled data
	test_df = pd.read_csv('test.csv')
	test_data = pd.DataFrame.as_matrix(test_df)
	id_test = test_data[:,0]; X_test = test_data[:,1:9];
	
	# predictions on unlabeled data
	y_test_pred = xgb_clsf.predict_proba(X_test)
	ans = pd.DataFrame({'Id': id_test, 'Action' : y_test_pred[:,1]})
	ans.to_csv('hw3p2.csv', index=False, columns=['Id', 'Action'])
