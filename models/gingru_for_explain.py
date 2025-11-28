import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch.nn import GRU
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from models.entity_embedding import EntityEmbeddingBatch3

def get_mask(los_batch: torch.Tensor):
    '''
    주어진 배치별 각 케이스의 시간 길이 (sequence length) 정보를 담고 있는 텐서를 사용하여, 
    해당 길이에 해당하는 부분은 1로, 나머지 부분은 0으로 채워진 마스크 텐서를 생성
    '''
    max_los = 37

    indices = torch.arange(max_los, device=los_batch.device) 
    mask = (indices < los_batch.unsqueeze(1))

    return mask.int().unsqueeze(2)


def to_temporal_gingru(x: torch.Tensor, los_batch: torch.Tensor):
    # process:  [batch, T, gin_h_dim]으로 변환하기 (gin_h_dim = F*num_layers)
    # 1. [batch, 2, gin_h_dim]으로 변환하기 --> .reshape(-1, 2, F)
    # 2. [batch, T, gin_h_dim]으로 변환하기
    # process: PackedSequence로 변환하기

    batch_size = int(x.shape[0] / 2) # [batch(twiced), f]

    ad = x[:batch_size].unsqueeze(1) # 브로드캐스팅을 위해 [B, 1, F]
    dis = x[batch_size:]
    
    # los_batch를 가지고 0,1 마스킹 행렬 생성
    mask = get_mask(los_batch=los_batch) # [B, 37, 1]
    padded = ad * mask # [B, 1, F] * [B, 37, 1] --> [B, 37, F] @@ 브로드캐스팅 !! @@

    batch_indices = torch.arange(batch_size, device=ad.device)
    last_time_indices = los_batch - 1

    padded[batch_indices, last_time_indices, :] = dis

    # -----PackedSequence 만들기-----
    lengths = los_batch.cpu() 

    # 내림차순(descending=True)으로 정렬
    lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)

    padded_sorted = padded[sorted_indices]

    packed_data = pack_padded_sequence(
        padded_sorted,
        lengths_sorted,
        batch_first=True,
        enforce_sorted=True
    )

    return packed_data, sorted_indices # padded_sorted의 디바이스를 따라감


def seperate_x(x: torch.Tensor, ad_idx_t, dis_idx_t, device):
    
    ad_tensor =torch.index_select(x, dim=1, index=ad_idx_t) # [B, 60, F]
    dis_tensor = torch.index_select(x, dim=1, index=dis_idx_t) # [B, 60, F]

    return torch.concatenate((ad_tensor, dis_tensor), dim=0) # [B*2, 60, F]



class ExplainerCompatibleGinGru(nn.Module):
    def __init__(self, batch_size, col_list, col_dims, ad_col_index, dis_col_index, embedding_dim, gin_hidden_channel, train_eps, gin_layers, gru_hidden_channel):
        '''
        Args:
            col_info(list): [col_dims, col_list]
                            col_list(list): 데이터에서 나타나는 변수의 순서
                            col_dims(list): 각 변수 별 범주의 개수, 순서는 col_list를 따라야 함
            embedding_dim(int): 엔티티 임베딩 후의 차원
            gin_hidden_channel(int): GIN의 hidden channel, 일반적으로 인풋, 히든, 아웃풋 차원을 동일하게 설정하는 것이 흔하고 효율적임
            train_eps(bool): GIN의 epsilon을 훈련할 것인지, 고정할 것인지

                                Gemini의 당부
                                다만, 모델을 로드하기 위해 새 인스턴스를 만들 때(e.g., model = MyGINModel(...)), 
                                해당 모델이 epsilon 슬롯을 가지고 있도록 반드시 train_eps=True를 동일하게 설정하여 초기화해야 합니다. 
                                그렇지 않으면 저장된 state_dict와 새 모델의 구조가 일치하지 않아 로딩에 실패합니다.

            gin_layers(int): GIN의 레이어 개수
            gru_hidden_channel(int): GRU의 hidden channel, 이는 모델의 기억 용량을 의미함
                                     우리 데이터는 크게 설정할 필요가 없을 것으로 보임 (오직 1번 달라지기 때문)
                                     시간축으로만 보면 그렇지만, 
                                     GIN이 출력하는 특징의 복잡도를 수용할 수 있는 적절한 최소 크기(Medium-sized*로 설정하는 것이 가장 좋다.
                                     gin_hidden_channel과 동일하게 설정하여 시작하는 것이 좋음
        '''
        super().__init__()
        self.batch_size = batch_size
        self.col_dims = col_dims
        self.col_list = col_list
        self.hidden_channel = gin_hidden_channel

        self.ad_idx_t = torch.tensor(ad_col_index)
        self.dis_idx_t = torch.tensor(dis_col_index)

        # EntityEmbedding 레이어 정의
        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        
        # GIN 레이어 정의
        # MLP 구조는 GIN 논문의 권장 사항(2-Layer MLP, 배치 정규화 )
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

        self.gin_layers = nn.ModuleList()

        gin_layer1 = GINConv(nn=gin_nn_input, eps=0, train_eps=train_eps)
        self.gin_layers.append(gin_layer1)
        
        for _ in range(gin_layers - 1):
            gin_layer_hidden = GINConv(nn=gin_nn, eps=0, train_eps=train_eps)
            self.gin_layers.append(gin_layer_hidden)
        
        gru_input_ch = gin_hidden_channel * gin_layers
        self.gru_layer = GRU(input_size=gru_input_ch, hidden_size=gru_hidden_channel)

        # 분류기 레이어 정의
        self.classifier_b = nn.Sequential(
            nn.Linear(gru_hidden_channel, gru_hidden_channel * 2),
            nn.ReLU(),
            nn.Linear(gru_hidden_channel * 2, 1)
        )

    def forward(self, x_embedded, template_edge_index, **kwargs):
        '''
        template_edge_index: supersized edge_index
        **kwargs: LOS_batch, device, graph_index
        '''
        device = kwargs.get("device")
        LOS_batch = kwargs.get("LOS_batch")


        # x_embedded shape: [72, 32]
        x_embedded = x_embedded.unsqueeze(0)
        zero_tensor = torch.zeros([31, 72, 32])
        gin_input = torch.concatenate([x_embedded, zero_tensor], dim=0) # gin_input shape: [32, 72, 32]
        
        
        self.ad_idx_t = self.ad_idx_t.to(device)
        self.dis_idx_t = self.dis_idx_t.to(device)

        # process: [batch * 2, num_nodes, feature_dim]으로 변환하기
        # 이때 시간 축 평탄화는 다음과 같이 되어야 함: 1, 2, 1, 2, 1, 2, ...
        # 위와 같이 될 필요는 없음: 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...
        # 이렇게 되어도 문제는 없다.
        x_seperated = seperate_x(x=gin_input, 
                                 ad_idx_t=self.ad_idx_t, 
                                 dis_idx_t=self.dis_idx_t, 
                                 device=device)

        # GIN에 입력
        x_after_gin = x_seperated
        sum_pooled = []
        for layer in self.gin_layers:
            x_after_gin = layer(x_after_gin, template_edge_index) # [B * 2, N(60), F(32)]
            x_sum = torch.sum(x_after_gin, dim=1) # [B * 2, N(60), F(32)] --> [B * 2, F(32)]
            sum_pooled.append(x_sum)
        gin_result = torch.concatenate(sum_pooled, dim=1) # [B*2, F*num_gin_layers]
            
        # [B*2, F*num_gin_layers] --> [B, 37, F*num_gin_layers]
        temporal_embedding, sorted_indices = to_temporal_gingru(x=gin_result, los_batch=LOS_batch)

        # GRU에 입력
        gru_out, gru_h = self.gru_layer(temporal_embedding)

        gru_h = gru_h.squeeze(0)

        # PackedSequence로 만들었기 때문에 (시간 길이 순 정렬) 역정렬해야 함
        inv_indices = torch.argsort(sorted_indices.to(gru_h.device))
        gru_h = gru_h[inv_indices]                           

        # Classifier에 입력
        classifier_result = self.classifier_b(gru_h)
        return classifier_result[0]