import os
from teds_tensor_dataset import TEDSTensorDataset

cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, 'data_tensor_cache')

dataset = TEDSTensorDataset(root)

print(dataset[0])
col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info
print(col_dims)