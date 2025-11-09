import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from entity_embedding import EntityEmbedding

class GAT(nn.Module):
    def __init__(self, col_dims, col_list, hidden_channels, output_channels, heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        # 엔티티 임베딩
        self.entitiy_embedding = EntityEmbedding(col_dims, col_list)

        # GAT 레이어 구성
        self.gat_layers = nn.ModuleList()
        gat_in_channels = self.entitiy_embedding.proj_dim
        self.gat_layers.append(GATConv(gat_in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.gat_layers.append(GATConv(hidden_channels * heads, output_channels, heads=1, dropout=dropout)) 
        
    
    def forward(self, x_cats):
        pass