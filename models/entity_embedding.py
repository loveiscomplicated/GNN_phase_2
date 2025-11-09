import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.data import Batch, Data

class EntityEmbedding(torch.nn.Module):
    def __init__(self, col_dims: list, col_list: list):
        '''
        Args:
            cat_dims: 변수별 범주의 개수 리스트
            col_list: 원본 데이터프레임에서 변수들의 순서 왼->오
        '''
        super().__init__()
        self.col_dims = col_dims
        self.col_list = col_list
        # proj_dim must be calculated with the final, corrected col_dims,
        # so we do it here with the provided ones, but it will be re-calculated in the test block
        self.proj_dim = int(max(self.col_dims)**0.5) if self.col_dims else 1

        self.embs = nn.ModuleList([
            nn.Embedding(num_categories, self.proj_dim)
            for num_categories in self.col_dims
        ])

    def forward(self, x_cats):
        '''
        This forward method assumes x_cats is a 2-D tensor of shape [N, F]
        where N is batch_size and F is number of features.
        The current data pipeline produces a different shape, so this is not used.
        '''
        # This logic is for a different data shape and is preserved for potential future use.
        x_cats = x_cats.long()
        outs = []
        for i, emb in enumerate(self.embs):
            out = emb(x_cats[:, i])
            outs.append(out)
        outs_tensor = torch.stack(outs, dim = 1)
        return outs_tensor

class EntityEmbeddingBatch(EntityEmbedding):
    def forward(self, batch: Batch):
        '''
        Args:
            batch(Data): torch_geometric의 Data 객체.
                         batch.x is expected to be of shape [num_nodes * num_features, 1]
        Returns:
            outs_tensor(torch.Tensor): [num_nodes * num_features, proj_dim]
        '''
        outs = []
        features = batch.x.long() # type: ignore

        for i, label in enumerate(features):
            col_idx = i % len(self.col_list)
            out = self.embs[col_idx](label)
            outs.append(out)

        # vstack results in shape [num_nodes * num_features, proj_dim]
        outs_tensor = torch.vstack(outs)
        return outs_tensor
        

if __name__ == "__main__":
    import pickle
    import os
    import torch
    dir_path = os.path.dirname(__file__)
    data_path = os.path.join(dir_path, '..', 'data', 'Sampled_temporal_graph_data_fully_connected.pickle')
    with open(data_path, 'rb') as f:
        pickle_dataset = pickle.load(f)
    
    batch_indi = pickle_dataset[0][0][0]
    
    col_list, col_dim = pickle_dataset[3]
    num_features = len(col_list)

    model = EntityEmbeddingBatch(col_dims=col_dim, col_list=col_list)
    
    print("Running forward pass...")
    output = model.forward(batch_indi)
    print("Forward pass successful!")
    print("Output shape:", output.shape)