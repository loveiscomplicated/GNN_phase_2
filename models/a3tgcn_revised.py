import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)
from .entity_embedding import EntityEmbeddingBatch2
from .attentiontemporalgcn import A3TGCN2

def _to_temporal(x_sliced: torch.Tensor, # shape: [num_var, feature_dim] (=[72, 25])
                ad_col_index: list, 
                dis_col_index: list,
                los,
                device):
    
    ad_idx_t = torch.tensor(ad_col_index).to(device)
    dis_idx_t = torch.tensor(dis_col_index).to(device)
    
    ad_tensor = torch.index_select(x_sliced, dim=0, index=ad_idx_t)
    dis_tensor = torch.index_select(x_sliced, dim=0, index=dis_idx_t)
    
    tensor_list = [ad_tensor for _ in range(los-1)]
    tensor_list.append(dis_tensor) # tensor_listtensor_listtensor_listtensor_listtensor_listtensor_list 33333333 33
    temp_tensor = torch.stack(tensor_list, dim=-1)

    num_nodes = len(ad_col_index)
    num_features = x_sliced.shape[1] # x_sliced shape: [num_var, feature_dim] (=[72, 25])
    
    zero = torch.zeros((num_nodes, num_features, 37-los), device=device) # 37: max los

    return torch.concatenate((temp_tensor, zero), dim=-1)

def to_temporal(x_tensor: torch.Tensor, # shape: [batch_size, num_var, feature_dim] (=[32, 72, 25])
                ad_col_index: list, 
                dis_col_index: list,
                LOS: torch.Tensor,
                device):
    batch_size = x_tensor.shape[0]

    temp_list = []
    
    for i in range(batch_size):
        x_sliced = x_tensor[i, :, :]
        los = LOS[i].item()
        # temp_tensor shape: [60, 25, 37]
        temp_tensor = _to_temporal(x_sliced=x_sliced,
                                   ad_col_index=ad_col_index,
                                   dis_col_index=dis_col_index,
                                   los=los,
                                   device=device)
        temp_list.append(temp_tensor)
    return torch.stack(temp_list, dim=0) # shape: [32, 60, 25, 37]



class A3TGCNCat1(nn.Module):
    '''
    tensor 연산 위주로 수행하는 모델
    '''
    def __init__(self, batch_size, col_list, col_dims, hidden_channel):
        '''
        Args:
            col_info(list): [col_dims, col_list]
                            col_list(list): 데이터에서 나타나는 변수의 순서
                            col_dims(list): 각 변수 별 범주의 개수, 순서는 col_list를 따라야 함
            num_layers(int): TGCN 레이어의 개수
            hidden_channel(int): TGCN의 hidden channel
        '''
        super().__init__()
        self.batch_size = batch_size
        self.col_dims = col_dims
        self.col_list = col_list
        self.hidden_channel = hidden_channel

        # EntityEmbedding 레이어 정의
        self.entity_embedding_layer = EntityEmbeddingBatch2(col_dims=self.col_dims,
                                                 col_list=self.col_list)
        
        # A3TGCN2 레이어 정의
        a3tgcn_input_channel = self.entity_embedding_layer.proj_dim

        self.a3tgcn_layer = A3TGCN2(in_channels=a3tgcn_input_channel,
                        out_channels=hidden_channel,
                        periods=37,
                        batch_size=batch_size)

        # 분류기 레이어 정의
        self.classifier_b = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel * 2),
            nn.ReLU(),
            nn.Linear(hidden_channel * 2, 2)
        )
    
    def forward(self, ad_col_index, dis_col_index, x_batch: torch.Tensor, LOS_batch: torch.Tensor, template_edge_index: torch.Tensor, device):
        '''
        Args:
            batch(torch.Tensor): X, y만 정의되어 있는 Data 객체, 
            template_edge_index(torch.Tensor): edge_index는 동일하므로 template_edge_index로 한꺼번에 전달
        '''
        x_embedded = self.entity_embedding_layer(x_batch)
        x = to_temporal(x_embedded, ad_col_index, dis_col_index, LOS_batch, device)
        after_GNN = self.a3tgcn_layer(x, template_edge_index) # [32, 60, 32]
        
        # global mean pooling [32, 60, 32] -> [32, 32]
        mean_pooled = torch.mean(after_GNN, dim=1)

        return self.classifier_b(mean_pooled)
    
if __name__ == "__main__":
    import sys
    import os
    CURDIR = os.path.dirname(__file__)
    parent_dir = os.path.join(CURDIR, '..')
    sys.path.append(parent_dir)
    
    from teds_tensor_dataset import TEDSTensorDataset
    from train_eval_a3tgcn_revised import train_test_split_customed

    root = os.path.join(parent_dir, 'data_tensor_cache')
    dataset = TEDSTensorDataset(root)

    train_dataloader, val_dataloader, test_dataloader = train_test_split_customed(dataset)
    
    col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info

    BATCH_SIZE = 32

    model = A3TGCNCat1(batch_size=BATCH_SIZE, col_list=col_list,
                       col_dims=col_dims, hidden_channel=32)

    print(model)