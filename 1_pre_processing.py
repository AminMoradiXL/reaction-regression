import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition 
from matplotlib import pyplot as plt
import seaborn as sb

raw_data = pd.read_pickle("data.pkl")
data = raw_data.dropna()
data.set_index(np.arange(len(data)), inplace = True)


## Normalization 
## minmax scaler

# scaler = MinMaxScaler()
# data_n = scaler.fit_transform(data)
# data_n = pd.DataFrame(data_n, columns= data.columns)

# print(data_n.describe())

# ax = data_n.plot(grid = True, fontsize = 16)
# ax.legend(fontsize = 16)
# fig = ax.get_figure()
# fig.savefig('Results/normal_minmax.png', dpi = 600)


## standard scaler

scaler = StandardScaler()
data_n = scaler.fit_transform(data)
data_n = pd.DataFrame(data_n, columns= data.columns)

# print(data_n.describe())

# ax = data_n.plot(grid = True, fontsize = 16)
# ax.legend(fontsize = 16)
# fig = ax.get_figure()
# fig.savefig('Results/normal_standard.png', dpi = 600)


## Correlation

# correlation_selected = data.corr() 
# fig, ax = plt.subplots(figsize=(18,13)) 
# sb.heatmap(correlation_selected, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
# ax.set_title('Correlation of original concentrations, derivatives and rate constants\n', fontsize = 24)
# ax.tick_params(labelsize = 18)
# fig.savefig('Results/correlation_orig.png', dpi = 600)

# correlation_selected = data_n.corr() 
# fig, ax = plt.subplots(figsize=(18,13)) 
# sb.heatmap(correlation_selected, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
# ax.set_title('Correlation of minmax scaled concentrations, derivatives and rate constants\n', fontsize = 24)
# ax.tick_params(labelsize = 18)
# fig.savefig('Results/correlation_minmax.png', dpi = 600)

# correlation_selected = data_n.corr() 
# fig, ax = plt.subplots(figsize=(18,13)) 
# sb.heatmap(correlation_selected, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
# # ax.set_title('Correlation of concentrations, derivatives and rate constants for originla, standard and minmax scaled data\n', fontsize = 24)
# ax.tick_params(labelsize = 18)
# fig.savefig('Results/correlation.png', dpi = 600)


## three of them  reulted the same, normalization does not neccesaarily change the correlation



##PCA

x_n = data_n.iloc[:, 0:6]
y_n = data_n.iloc[:, 6:]

# pca = decomposition.PCA()
# x_p = pca.fit_transform(x_n)


# # plt.scatter(x_p[:, 0], x_p[:, 1], c = y_n["k1"])
# # plt.scatter(x_p[:, 0], x_p[:, 1], c = y_n["k1"])
# plt.scatter(x_p[:, 0], x_p[:, 1], c = x_p[:, 5])
# plt.xlabel("PCA1")
# plt.ylabel("PCA2")
# plt.colorbar(label = "k1")
# plt.show




pca = decomposition.PCA(n_components = 3)
components = pca.fit_transform(x_n)

# 3D

# fig = plt.figure(figsize = (15,15))
# ax = fig.add_subplot(projection = '3d')
# ax.grid()
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.set_axisbelow(True)
# p = ax.scatter(components[:,0], components[:,1], components[:,2], c = y_n['k1'], cmap = 'RdBu')
# ax.set_xlabel(r'$PCA_1$', fontsize = 32)
# ax.set_ylabel(r'$PCA_2$', fontsize = 32)
# ax.set_zlabel(r'$PCA_3$', fontsize = 32, location ="left")
# ax.xaxis.labelpad = 20
# ax.yaxis.labelpad = 20
# ax.zaxis.labelpad = 20
# ax.set_aspect('equal')
# # ax.set_title('PCA components of selected features in 3d', fontsize = 36)
# ax.tick_params(labelsize = 20, pad = 10)
# cb = fig.colorbar(p, shrink = 0.9, location = 'right', pad=0)
# cb.set_label(label = r'$k_2$', fontsize = 32)
# cb.ax.tick_params(which = 'major', labelsize = 20)
# fig.savefig('Results/pca3d.png', dpi = 600)

# 2D

# fig = plt.figure(figsize = (15,15))
# ax = fig.add_subplot()
# ax.grid()
# ax.set_axisbelow(True)
# p = ax.scatter(components[:,0], components[:,1], c = y_n['k2'], cmap = 'RdBu')
# ax.set_xlabel(r'$PCA_1$', fontsize = 32)
# ax.set_ylabel(r'$PCA_2$', fontsize = 32)
# ax.xaxis.labelpad = 10
# ax.yaxis.labelpad = 0
# ax.tick_params(labelsize = 20, pad = 10)
# ax.set_aspect('equal')
# cb = fig.colorbar(p, shrink = 0.9, location = 'right', pad=0.1)
# cb.set_label(label = r'$k_2$', fontsize = 32)
# cb.ax.tick_params(which = 'major', labelsize = 20)
# fig.savefig('Results/pca2d.png', dpi = 600)

print(pca.explained_variance_ratio_)














