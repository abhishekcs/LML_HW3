import numpy as np
import pandas as pd
import sklearn as sk

from itertools import combinations
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder

# Miroslaw code for grouping
def group_data(data, degree=3, cutoff = 1, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    
    new_data = []
    m,n = data.shape
    for indexes in combinations(range(n), degree):
        new_data.append([hash(tuple(v))% (2**24) for v in data[:,indexes]])
    for z in range(len(new_data)):
        counts = dict()
        useful = dict()
        for item in new_data[z]:
            if item in counts:
                counts[item] += 1
                if counts[item] > cutoff:
                    useful[item] = 1
            else:
                counts[item] = 1
        for j in range(len(new_data[z])):
            if not new_data[z][j] in useful:
                new_data[z][j] = 0
    return np.array(new_data).T

if __name__ == "__main__": 
	# load data
	df_train = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')

	num_train = np.shape(df_train)[0]
	num_test = np.shape(df_test)[0]
	df = pd.concat([df_train, df_test])
	df.index = range(num_train+num_test)
	
	# combined features
	all_data = pd.DataFrame.as_matrix(df)
	all_data = all_data[:,1:]
	dp1 = group_data(all_data, degree=2, cutoff=0) 
	dt1 = group_data(all_data, degree=3, cutoff=0)

	X_new = np.hstack((dp1, dt1))
	num_new_features = X_new.shape[1]
	for i in range(num_new_features):
		col = str(i)
		df[col] = X_new[:,i]

	# get frequencies
	#for col in  df.columns.values:
		#df['f_'+col] = df.groupby(col)[col].transform('count')
	del all_data, dp1, dt1

	# one hot encoding	
	enc = OneHotEncoder(categorical_features=range(num_new_features + 9))
	enc.fit(df.ix[:,1:])

	df_train = df.drop(df.index[range(num_train, num_train+num_test)])
	df_test = df.drop(df.index[range(num_train)])
	del df
	y = df_train.ix[:,0];X = df_train.ix[:,1:]
	y_test = df_test.ix[:,0]; X_test = df_test.ix[:,1:]
	X = enc.transform(X)
	X_test = enc.transform(X_test)
	
	# CV and training
	classifier = LogisticRegressionCV(Cs = np.logspace(-4, 4, 10, base=2), penalty='l1', scoring = 'roc_auc', solver = 'liblinear')
	classifier.fit(X, y)
	
	# testing
	y_test_pred = classifier.predict_proba(X_test)
	ans = pd.DataFrame({'Id': y_test, 'Action' : y_test_pred[:,1]})
	ans.to_csv('hw3p1.csv', index=False, columns=['Id', 'Action'])	

