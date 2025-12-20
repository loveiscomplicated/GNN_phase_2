# CTMP-GIN

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv

from .gin_gru import seperate_x
from .entity_embedding import EntityEmbeddingBatch3

cur_dir = os.path.dirname(__file__)

def get_edge_index_2(edge_index, num_nodes, batch_size):
    '''
    Args:
        edge_index: ad, dis 이어져 있는 edge_index, gin_1에서 사용했던 것
    '''
    merged_num_nodes = num_nodes * batch_size
    start_node = torch.arange(0, merged_num_nodes) # [batch_size * num_nodes] == [merged_num_nodes]
    end_node = start_node + merged_num_nodes       # [batch_size * num_nodes] == [merged_num_nodes]
    
    start_node = start_node.unsqueeze(dim=0)       # [1, merged_num_nodes]
    end_node = end_node.unsqueeze(dim=0)           # [1, merged_num_nodes]
    new_edge_index = torch.cat((start_node, end_node), dim = 0) # [2, merged_num_nodes]

    return torch.cat((edge_index, new_edge_index), dim=1) # # [2, 원래 edge_index + merged_num_nodes]


class GatedFusion(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = dim or hidden_dim
        self.score = nn.Sequential(
            nn.Linear(dim * 3,  hidden_dim), # type: ignore
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3) # type: ignore
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, A, B, C):
        x = torch.cat([A, B, C], dim=-1) # [batch, dim * 3]
        w = F.softmax(self.score(x), dim=-1) # [batch, 3]
        A = self.dropout(A)
        B = self.dropout(B)
        fused = w[:, 0:1]* A + w[:, 1:2] * B, w[:, 2:3] * C
        return fused, w


class CtmpGIN(nn.Module):
    def __init__(self, 
                 col_dims, 
                 embedding_dim, 
                 gin_hidden_channel, 
                 gin_1_layers, 
                 gin_hidden_channel_2, 
                 gin_2_layers, 
                 dropout_ratio, 
                 device,
                 los_embedding_dim=8, 
                 train_eps=True):
        super().__init__()
        self.col_dims = col_dims
        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        self.embed_los = EntityEmbeddingBatch3(col_dims=[37], embedding_dim=los_embedding_dim)
        self.device = device

        # GIN_1
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

        # GIN_2
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

    def forward(self, x, edge_index, los):
        self.ad_idx_t = self.ad_idx_t.to(self.device)
        self.dis_idx_t = self.dis_idx_t.to(self.device)
        
        batch_size = x.shape[0]
        num_nodes = len(self.ad_idx_t)

        # x_batch shape: [batch_size, num_var(=72)]
        x_embedded = self.entity_embedding_layer(x) # shape: [batch, num_var, feature_dim]

        # process: [batch * 2, num_nodes, feature_dim]으로 변환하기
        # 위와 같이 되어야 함: 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...
        x_seperated = seperate_x(x=x_embedded, # [B*2, 60, 32]
                                 ad_idx_t=self.ad_idx_t, 
                                 dis_idx_t=self.dis_idx_t, 
                                 device=self.device)

        # GIN_1에 입력
        x_flatten = x_seperated.reshape(batch_size * 2 * num_nodes, -1)

        x_after_gin = x_flatten
        for layer in self.gin_1:
            x_after_gin = layer(x_after_gin, edge_index) # [B * 2 * N, F(32)]
            x_graph = x_after_gin.reshape(batch_size * 2, num_nodes, self.hidden_channel) # [B * 2, N, F]
            x_sum = torch.sum(x_graph, dim=1) # [B * 2, N(60), F(32)] --> [B * 2, F(32)]
        
    def get_edge_attr(self, los):
        '''
        원래 los는 range(1, 38) 안에 있는 정수 값
        그렇지만 이에 대해 엔티티 임베딩을 적용하기 위해서는 0-index 형식으로 바꾸어 주어야 한다.
        즉 los - 1한 것을 엔티티 임베딩 룩업 테이블에 넣어야 제대로 작동한다.
        '''
        los -= 1
        los_emb = self.embed_los(los)
        
        pass