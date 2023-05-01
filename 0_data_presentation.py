import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


raw_data = pd.read_pickle("data.pkl")
data = raw_data.dropna()

x = data.iloc[:, 0:6]
y = data.iloc[:, 6:]


data.set_index(np.arange(len(data)), inplace = True)


# plot = data[['k1', 'k2']].plot(grid = True, fontsize = 16)
# plot.legend(fontsize = 16)
# fig = plot.get_figure()
# fig.savefig('Results/data_k.png', dpi = 600)

# plot = data[['c1', 'c2', 'c3']].plot(grid = True, fontsize = 16)
# plot.legend(fontsize = 16)
# fig = plot.get_figure()
# fig.savefig('Results/data_c.png', dpi = 600)

# plot = data[['dc1', 'dc2', 'dc3']].plot(grid = True, fontsize = 16)
# plot.legend(fontsize = 16)
# fig = plot.get_figure()
# fig.savefig('Results/data_dc.png', dpi = 600)

print(data["k2"].describe())




