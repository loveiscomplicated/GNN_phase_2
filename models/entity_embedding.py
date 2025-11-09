import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.data import Batch

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
        self.proj_dim = int(max(self.col_dims)**0.5) # 648 -->> 25, 즉 25차원으로 통일

        self.embs = nn.ModuleList([
            nn.Embedding(num_categories, self.proj_dim)
            for num_categories in self.col_dims
        ])

    def forward(self, x_cats):
        '''
        x_cats: 엔티티 임베딩이 되지 않은, 레이블 인코딩만 되어 있는 @@2-D 텐서@@
        '''
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
            batch(Batch): torch_geometric의 Batch 객체
        Returns:
            outs_tensor(torch.Tensor): [batch_size, feature_dimension]
                                       feature_dimension은 int(max(self.col_dims)**0.5)
        '''
        outs = []
        features = batch.x # type: ignore

        for i, label in enumerate(features):
            col = i % len(self.col_list)
            out = self.embs[col](label)
            outs.append(out)

        outs_tensor = torch.vstack(outs)       
        return outs_tensor
        

if __name__ == "__main__":
    import pickle
    import os
    dir_path = os.path.dirname(__file__)
    data_path = os.path.join(dir_path, '..', 'data', 'Sampled_temporal_graph_data_fully_connected.pickle')
    with open(data_path, 'rb') as f:
        pickle_dataset = pickle.load(f)
    batch_indi = pickle_dataset[0][0][0]
    batch_indi.x = batch_indi.x.long()
    col_list, col_dim = pickle_dataset[3]
    
    model = EntityEmbeddingBatch(col_dims=col_dim, col_list=col_list)
    
    model.forward(batch_indi)
    
