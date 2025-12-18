# CTMP-GIN

import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv

from .entity_embedding import EntityEmbeddingBatch3

cur_dir = os.path.dirname(__file__)

class CtmpGIN(nn.Module):
    def __init__(self, col_dims, embedding_dim):
        super().__init__()
        self.col_dims = col_dims
        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
    
    def forward(self, x, edge_index):
        x_embedded = self.entity_embedding_layer(x)
        print(x_embedded)
        pass