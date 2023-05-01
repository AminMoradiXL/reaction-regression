import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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
     
##permutation

    
data_perm = X_train.copy()
max_depth = 10
forest_model = RandomForestRegressor(max_depth= max_depth, n_jobs = -1)
forest_model.fit(data_perm, y_train) 
y_pred = forest_model.predict(X_val)
error = y_pred - y_val
error = np.reshape(np.array(error), newshape=(-1,1))
baseline = np.sqrt(np.mean(error**2))
print(f'baseline = {baseline}')

RMSE_permutation_record = []

for i in range(X_train.shape[1]):
    data_perm = X_train.copy()
    data_perm.iloc[:, i] = np.random.permutation(data_perm.iloc[:, i])
    max_depth = 10
    forest_model = RandomForestRegressor(max_depth= max_depth, n_jobs = -1)
    forest_model.fit(data_perm, y_train) 
    y_pred = forest_model.predict(X_val)
    error = y_pred - y_val
    error = np.reshape(np.array(error), newshape=(-1,1))
    RMSE = np.sqrt(np.mean(error**2))
    print(f'permuated feature = {i}, RMSE = {RMSE}')
    RMSE_permutation_record.append(RMSE) 
  
##dropping    
  
    
RMSE_dropped_record = []

for i in range(X_train.shape[1]):
    data_perm = X_train.copy()
    data_perm.drop(columns = data_perm.columns[i], inplace = True)
    max_depth = 10
    forest_model = RandomForestRegressor(max_depth= max_depth, n_jobs = -1)
    forest_model.fit(data_perm, y_train) 
    data_val_perm = X_val.copy()
    A = data_val_perm.drop(columns = data_val_perm.columns[i], inplace = False)
    
    y_pred = forest_model.predict(A)
    error = y_pred - y_val
    error = np.reshape(np.array(error), newshape=(-1,1))
    RMSE = np.sqrt(np.mean(error**2))
    print(f'dropped feature = {i}, RMSE = {RMSE}')
    RMSE_dropped_record.append(RMSE) 

fig, ax = plt.subplots(constrained_layout = True)
ax.bar(X_train.columns, RMSE_permutation_record, width=0.5, align='edge',label='Permuted', color = '#cb4b43')
ax.bar(X_train.columns, RMSE_dropped_record, width=0.5, align='center', label='Dropped', color = '#337eb8')
ax.set_xticklabels([it.split('(')[0].strip() for it in X_train.columns], rotation=75)
ax.hlines(baseline, 0, len(X_train.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Validation RMSE', fontsize = 'large')
ax.set_xlabel('Features', fontsize = 'large')
# ax.tick_params(labelsize = 15)
ax.grid()
ax.set_axisbelow(True)
ax.legend(fontsize = 'large')
fig.savefig('Results/feature_importance.png', dpi = 600)







