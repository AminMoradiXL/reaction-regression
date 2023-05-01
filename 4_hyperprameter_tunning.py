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
     

def ML_model(parameterization):
    
        # Random Forest
        # model = RandomForestRegressor(**parameterization, random_state=0, n_jobs=-1)
        # KNN
        # model = KNeighborsRegressor(**parameterization, n_jobs=-1)
        # Decision Tree
        model = DecisionTreeRegressor(**parameterization, random_state=0)
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        error = y_pred - y_val
        error = np.reshape(np.array(error), newshape=(-1,1))
        RMSE = np.sqrt(np.mean(error**2))
        
        return RMSE


best_parameters, values, experiment, model = optimize(
    # Random forest
    # parameters=[
    #     {"name": "n_estimators", "type": "range", "bounds": [10, 100]},
    #     {"name": "max_depth", "type": "range", "bounds": [5, 30]},
    # ],
    # KNN
    # parameters=[
    #     {"name": "n_neighbors", "type": "range", "bounds": [1, 30]},
    #     {"name": "weights", "type": "choice", "values": ['uniform', 'distance']},
    #     {"name": "algorithm", "type": "choice", "values": ['auto', 'ball_tree', 'kd_tree', 'brute']},
    #     {"name": "leaf_size", "type": "range", "bounds": [10, 50]},
    # ],
    #Decision Tree
    parameters=[
        {"name": "splitter", "type": "choice", "values": ["best", "random"]},
        {"name": "max_depth", "type": "range", "bounds": [1, 30]},
    ],
    
    evaluation_function= ML_model,
    objective_name='RMSE',
    minimize = True,
    total_trials = 50, 
    random_seed=0
)

print( best_parameters )
means, covariances = values
print( means, covariances )

# render(plot_contour(model=model, param_x='n_estimators', param_y='max_depth', metric_name='RMSE'))














