import scanpy
import pandas as pd
import numpy as np
import os

temp = scanpy.read_h5ad(os.path.dirname(os.path.abspath(__file__)) + '/scDesign2/neurons.h5ad')
# Filter the cells
temp = temp[temp.obs['disease'] == 'normal']
x = list(temp.obs['cell_type'].values)
mat = temp.raw.X
mat.index = x
y = pd.DataFrame(mat.toarray(), index=x, columns=temp.var_names)
temp_temp = set(x)
print(temp_temp)
y.index = x
y = y.loc[y.index != 'neuron']
y = y.loc[y.index != 'GABAergic neuron']
y = y.T

temp = scanpy.read_h5ad(os.path.dirname(os.path.abspath(__file__)) + '/scDesign2/bcells.h5ad')
temp = temp[temp.obs['disease'] == 'normal']
x = list(temp.obs['cell_type'].values)
mat = temp.raw.X
mat.index = x
z = pd.DataFrame(mat.toarray(), index=x, columns=temp.var_names)
temp_temp = set(x)
print(temp_temp)
z.index = x
z = z.loc[z.index != 'IgA plasma cell']
z = z.loc[z.index != 'IgG plasma cell']
z = z.T

y = pd.merge(y, z, left_index=True, right_index=True)

mask_nonzero = y != 0
row_means = mask_nonzero.mean(axis=1)

selected_zero_rows = row_means[row_means < 0.1].sample(n=1795, random_state=42).index
selected_nonzero_rows = row_means[(row_means >= 0.1) & (row_means < 0.2)].sample(n=1800, random_state=42).index
selected_rows_0_2 = row_means[(row_means >= 0.2) & (row_means < 0.5)].sample(n=1200, random_state=42).index
selected_rows_0_5 = row_means[(row_means >= 0.5) & (row_means < 0.9)].sample(n=200, random_state=42).index
selected_rows_0_9 = row_means[row_means >= 0.9].sample(n=5, random_state=42).index

selected_rows = np.concatenate([selected_rows_0_9, selected_rows_0_5, selected_rows_0_2, selected_nonzero_rows, selected_zero_rows])
y = y.loc[selected_rows]
# y = y.iloc[:20, :20]
# print(os.path.dirname(os.path.abspath('')))

y.to_csv(os.path.dirname(os.path.abspath(__file__)) + '/scDesign2/neurons_100_250_1400_1900_1350_09_05_02_01_disease_free.csv', sep=',', index=True)

# y.to_csv('bcells.csv', index=True)
