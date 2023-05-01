import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ax.service.managed_loop import optimize
from ax.plot.contour import plot_contour
from ax.utils.notebook.plotting import render
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor



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

## training 

RMSE_train_list = []
RMSE_test_list = []

for m in (KNeighborsRegressor(n_neighbors= 3, leaf_size = 31, weights = "distance", algorithm = "ball_tree"),
          DecisionTreeRegressor(max_depth = 17, splitter = "best"),
          RandomForestRegressor(n_estimators = 44, max_depth= 30)):
    model = m
    model.fit(X_trval, y_trval)
    y_pred_train =model.predict(X_trval) 
    y_pred_test = model.predict(X_test) 
    error_train = y_pred_train - y_trval
    error_test = y_pred_test - y_test
    RMSE_train = np.sqrt(np.mean(error_train**2))
    error_train = np.reshape(np.array(error_train), newshape=(-1,1))
    RMSE_train_list.append(RMSE_train)
    RMSE_test = np.sqrt(np.mean(error_test**2))
    error_test = np.reshape(np.array(error_test), newshape=(-1,1))
    RMSE_test_list.append(RMSE_test)


RMSE_train= [t[0] for t in RMSE_train_list]
RMSE_test = [t[0] for t in RMSE_test_list]


model_name = ["KNN", "Decision Tree", "Random Forest"]

# fig, ax = plt.subplots(constrained_layout = True)
# ax.bar(model_name, RMSE_train, width=0.6, align='edge',label='train', color = '#cb4b43')
# ax.bar(model_name,RMSE_test, width=0.4, align='center', label='test', color = '#337eb8')
# # ax.set_xticklabels([it.split('(')[0].strip() for it in range[0, 1, 2], rotation=75)
# # ax.hlines(baseline, 0, len(X_train.columns)-1, linestyles='dashed', label='Baseline')
# ax.set_ylabel('RMSE', fontsize = 'large')
# ax.set_xlabel('Model', fontsize = 'large')
# # ax.tick_params(labelsize = 15)
# ax.grid()
# ax.set_axisbelow(True)
# ax.legend(fontsize = 'large')
# fig.savefig('Results/training_testing.png', dpi = 600)

fig, ax = plt.subplots(constrained_layout = True)
ax.bar(model_name, RMSE_test, width=0.25, align='center', label='Test', color = '#337eb8')
ax.bar(model_name, RMSE_train, width=0.25, align='edge',label='Train', color = '#cb4b43')

# ax.set_xticklabels([it.split('(')[0].strip() for it in X_train.columns], rotation=75)
ax.set_ylabel('RMSE', fontsize = 'large')
ax.set_xlabel('Features', fontsize = 'large')
# ax.tick_params(labelsize = 15)
ax.grid()
ax.set_axisbelow(True)
ax.legend(fontsize = 'large')
fig.savefig('Results/training_testing.png', dpi = 600)


