# CTMP-GIN

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv

from .entity_embedding import EntityEmbeddingBatch3

cur_dir = os.path.dirname(__file__)


class GatedFusion(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = dim or hidden_dim
        self.score = nn.Sequential(
            nn.Linear(dim * 3,  hidden_dim), # type: ignore
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # type: ignore
        )
    def forward(self, A, B, C):
        x = torch.cat([A, B, C], dim=-1) # [batch, dim * 3]
        w = F.softmax(self.score(x), dim=-1) # [batch, 3]
        fused = w[:, 0:1]* A + w[:, 1:2] * B, w[:, 2:3] * C
        return fused, w


class CtmpGIN(nn.Module):
    def __init__(self, col_dims, embedding_dim, gin_hidden_channel, gin_1_layers, gin_hidden_channel_2, gin_2_layers, dropout_ratio, train_eps=True):
        super().__init__()
        self.col_dims = col_dims
        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)

        gin_nn_input = nn.Sequential(
             nn.Linear(embedding_dim, gin_hidden_channel),
             nn.LayerNorm(gin_hidden_channel),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel, gin_hidden_channel) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn = nn.Sequential(
             nn.Linear(gin_hidden_channel, gin_hidden_channel),
             nn.LayerNorm(gin_hidden_channel),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel, gin_hidden_channel) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn_input2 = nn.Sequential(
             nn.Linear(gin_hidden_channel * 2, gin_hidden_channel_2), # ad, dis 동시에 들어가므로, input 차원이 두 배
             nn.LayerNorm(gin_hidden_channel_2),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn2 = nn.Sequential(
             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2),
             nn.LayerNorm(gin_hidden_channel_2),
             nn.ReLU(),

             nn.Linear(gin_hidden_channel_2, gin_hidden_channel_2) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )
        
        # GIN_1
        self.gin_1 = nn.ModuleList()
        gin1_input = GINConv(gin_nn_input, train_eps=train_eps)
        self.gin_1.append(gin1_input)
        for _ in range(gin_1_layers - 1):
            gin_hidden_layer = GINConv(gin_nn, train_eps=train_eps)
            self.gin_1.append(gin_hidden_layer)

        # GIN_2        
        self.gin_2 = nn.ModuleList()
        gin2_input = GINEConv(gin_nn_input2, train_eps=train_eps, edge_dim=1)
        self.gin_2.append(gin2_input)
        for _ in range(gin_2_layers - 1):
            gin_hidden_layer = GINEConv(gin_nn2, train_eps=train_eps, edge_dim=1)
            self.gin_2.append(gin_hidden_layer)
        
        
        
        

        self.dropout = nn.Dropout(dropout_ratio)


    def forward(self, x, edge_index):
        x_embedded = self.entity_embedding_layer(x)
        print(x_embedded)
        pass