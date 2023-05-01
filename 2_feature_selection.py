import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



raw_data = pd.read_pickle("data.pkl")
data = raw_data.dropna()
data.set_index(np.arange(len(data)), inplace = True)

scaler = MinMaxScaler()
data_n = scaler.fit_transform(data)
data_n = pd.DataFrame(data_n, columns= data.columns)

x_n = data_n.iloc[:, 0:6]
y_n = data_n.iloc[:, 6:]

X_trval, X_test, y_trval, y_test = train_test_split(x_n, y_n, test_size=0.2, random_state=(0))
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.2, random_state=(0))
     
max_depth = 10

forest_model = ensemble.RandomForestRegressor(verbose = 1, max_depth = max_depth, n_jobs=-1)

sfs = SequentialFeatureSelector(forest_model, n_features_to_select = 5, n_jobs=-1, direction = 'backward')
sfs.fit(X_train, y_train)

Selected = sfs.get_support()

sfs.transform(X_train).shape

sel_feat = []

for i in range(len(Selected)):
    if Selected[i]:
       sel_feat.append(X_test.columns[i]) 

