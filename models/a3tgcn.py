import torch
import torch.nn as nn

from entity_embedding import EntityEmbedding
from torch_geometric_temporal.nn.recurrent import A3TGCN2

class A3TGCNCat(nn.Module):
    def __init__(self, col_dims, col_list, num_layers, input_channel, hidden_channel, out_channel, batch_size):
        super().__init__()

        # 엔티티 임베딩 레이어 정의
        self.entitiy_embedding = EntityEmbedding(col_dims, col_list)

        # A3TGCN2 레이어 정의
        self.a3tgcn_layers = nn.ModuleList()
        
        # 첫 번째 레이어 정의
        first_layer = A3TGCN2(in_channels=input_channel, 
                              out_channels=hidden_channel, 
                              periods=1, 
                              batch_size=batch_size)
        self.a3tgcn_layers.append(first_layer)

        # 히든 레이어 정의
        for _ in range(1, num_layers):
            hidden_layer = A3TGCN2(in_channels=hidden_channel, 
                                   out_channels=hidden_channel, 
                                   periods=1, 
                                   batch_size=batch_size)
            self.a3tgcn_layers.append(hidden_layer)

        # 분류기 정의
        self.classifier_b = nn.Sequential(
            nn.Linear(hidden_channel * num_layers, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, 2)
        )
        
    def forward(self, input_tensor):
        
        pass

