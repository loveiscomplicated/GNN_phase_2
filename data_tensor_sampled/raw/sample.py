import os
import pandas as pd

sample_size = 100

cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir, '../../data_tensor_cache/raw/missing_corrected.csv')
target_path = os.path.join(cur_dir, 'missing_corrected.csv')
df = pd.read_csv(data_path)

df_sampled = df[:sample_size]
df_sampled.to_csv(target_path)