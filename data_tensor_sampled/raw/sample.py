import os
import pandas as pd

sample_per_class = 100_000  # 원하는 수로 조절

cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir, '../../data_tensor_cache/raw/missing_corrected.csv')
target_path = os.path.join(cur_dir, 'missing_corrected.csv')

df = pd.read_csv(data_path)

# 균등 샘플링
df_sampled = (
    df.groupby('REASONb', group_keys=False)
      .apply(lambda x: x.sample(min(len(x), sample_per_class), random_state=42))
      .reset_index(drop=True)
)

df_sampled.to_csv(target_path, index=False)
print(df_sampled['REASONb'].value_counts())
