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
        
        # 1. features 텐서 준비
        features = batch.x.long() # 현재 features.device == cpu (문제의 원인)
        TARGET_DEVICE = self.embs[0].weight.device
        features = features.to(TARGET_DEVICE)
        DEVICE = features.device 
        
        # features의 shape가 [X, 1]이면 [X]로 squeeze (nn.Embedding을 위해)
        if features.dim() > 1 and features.size(1) == 1:
            features = features.squeeze(1) # (N_total * F_count)
        
        N_total_F = features.shape[0]
        F_count = len(self.col_list)
        
        # 2. 텐서 분리 및 인덱스 생성
        # col_indices 생성 시 이미 DEVICE를 사용하고 있으므로, 이제 DEVICE는 MPS/GPU입니다.
        col_indices = torch.arange(F_count, device=DEVICE).repeat(N_total_F // F_count)
        
        # 3. 각 임베딩 레이어를 사용하여 해당하는 인덱스를 한 번에 처리
        all_embedded_features = []
        for i, emb in enumerate(self.embs):
            
            mask = (col_indices == i)
            data_to_embed = features[mask]
            
            embedded_data = emb(data_to_embed) # 이제 CPU 텐서가 아닌 GPU 텐서 입력
            
            all_embedded_features.append((embedded_data, torch.where(mask)[0]))

        # 4. 원래의 순서대로 결과를 재조립
        outs_tensor = torch.zeros(N_total_F, self.proj_dim, device=DEVICE) 
        for embedded_data, indices in all_embedded_features:
            outs_tensor[indices] = embedded_data

        return outs_tensor
    

class EntityEmbeddingBatch2(EntityEmbedding):
    def forward(self, batch: Batch):
        '''
        Args:
            batch(torch_geometric.data.Batch): batch instance, shape=[batch_size, num_nodes, 1]
        '''
        pass


if __name__ == "__main__":
    import pickle
    import os
    import torch
    dir_path = os.path.dirname(__file__)
    data_path = os.path.join(dir_path, '..', 'data', 'Sampled_temporal_graph_data_fully_connected.pickle')
    with open(data_path, 'rb') as f:
        pickle_dataset = pickle.load(f)
    
    batch_indi = pickle_dataset[0][0][0].x

    
    col_list, col_dim = pickle_dataset[3]
    num_features = len(col_list)

    model = EntityEmbedding(col_dims=col_dim, col_list=col_list)
    
    print("Running forward pass...")
    output = model.forward(batch_indi)
    print("Forward pass successful!")
    print("Output shape:", output.shape)