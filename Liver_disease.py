import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv('indian_liver_patient.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 10].values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:, 2:10])
X[:, 2:10]=imputer.transform(X[:, 2:10])



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)


from sklearn.ensemble import RandomForestClassifier


classifier =RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='log2', max_leaf_nodes=None,bootstrap=True, oob_score=True, n_jobs=1, random_state=0, verbose=1, warm_start=True,class_weight=None)


classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


pickle.dump(classifier,open('model.pkl', 'wb'))