import numpy as np
import pandas as pd
import sklearn as sk

from itertools import combinations
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

# Miroslaw code for grouping
def group_data(data, degree=3, cutoff = 1, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    
    new_data = []
    m,n = data.shape
    for indexes in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indexes]])
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
	# load labeled data
	df_train = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')

	num_train = np.shape(df_train)[0]
	num_test = np.shape(df_test)[0]
	df = pd.concat([df_train, df_test])
	df.index = range(num_train+num_test)
	
	# combined features
	all_data = pd.DataFrame.as_matrix(df)
	all_data = all_data[:,1:-1]
	dp1 = group_data(all_data, degree=2, cutoff=0) 
	dt1 = group_data(all_data, degree=3, cutoff=0)

	# get frequencies
	for col in  df.columns.values:
		df['f_'+col] = df.groupby(col)[col].transform('count')

	X_new = np.hstack((dp1, dt1))
	num_new_features = X_new.shape[1]

	for i in range(num_new_features):
		col = str(i)
		df[col] = X_new[:,i]
		df['f_'+col] = df.groupby(col)[col].transform('count')
		df.drop(col, axis=1, inplace=True)

	df_train = df.drop(df.index[range(num_train, num_train+num_test)])
	df_test = df.drop(df.index[range(num_train)])

	df_train.to_csv('train_all.csv', index=False)
	df_test.to_csv('test_all.csv', index=False)

	
