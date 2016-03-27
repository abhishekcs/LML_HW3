import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

if __name__ == "__main__": 
	# load labeled data
	train_df = pd.read_csv('train.csv')
	train_data = pd.DataFrame.as_matrix(train_df)
	y = train_data[:,0]; X = train_data[:,1:9];

	# train logisitc regression model
	# pick C using cross validation 
	classifier = LogisticRegressionCV(Cs = np.logspace(-4, 4, 20), penalty='l2', scoring = 'roc_auc', solver = 'lbfgs')
	classifier.fit(X, y)

	# load unlabeled data
	test_df = pd.read_csv('test.csv')
	test_data = pd.DataFrame.as_matrix(test_df)
	id_test = test_data[:,0]; X_test = test_data[:,1:9];
	
	# predictions on unlabeled data
	y_test_pred = classifier.predict_proba(X_test)
	ans = pd.DataFrame({'Id': id_test, 'Action' : y_test_pred[:,1]})
	ans.to_csv('hw3p1.csv', index=False, columns=['Id', 'Action'])
