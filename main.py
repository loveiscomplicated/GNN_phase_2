import os
from models.ctmp_gin import CtmpGIN
from teds_tensor_dataset import TEDSDatasetforCtempGIN, TEDSTensorDataset

cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, 'data_tensor_cache')
dataset = TEDSDatasetforCtempGIN(root)

print(dataset[0][0].shape)

dataset2 = TEDSTensorDataset(root)
print(dataset2[0][0].shape)