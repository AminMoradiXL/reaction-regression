import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from sklearn import decomposition

raw_data = pd.read_pickle("data.pkl")
data = raw_data.dropna()

x = data.iloc[:, 0:6]
y = data.iloc[:, 6:]

X_trval, x_test, y_trval, y_test = train_test_split(x ,y , test_size= 0.2, 
                                                    random_state=(0), shuffle= (True))

x_train, x_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.2,
                                                  random_state=(0), shuffle=(True))



# model = RandomForestRegressor(max_depth=(25))
# model.fit(x_train, y_train)
# y_pred = model.predict(x_val)
# error = y_pred - y_val
# rmse= np.sqrt(np.mean(error**2))
# mse = np.mean(error**2)
# print(rmse)
# print(mse)

# model = LinearRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_val)
# error = y_pred - y_val
# rmse= np.sqrt(np.mean(error**2))
# print(rmse)



# pca = decomposition.PCA().fit(x_train)
# p = pca.transform(x_train)

## hyperprameter optimization

rmse_list = []
mse_list = []
for i in range(3, 30):
    model = RandomForestRegressor(max_depth= i)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    error = y_pred - y_val
    rmse= np.sqrt(np.mean(error**2))
    mse = np.mean(error**2)
    rmse_list.append(rmse)
    mse_list.append(mse)

print(rmse_list)



