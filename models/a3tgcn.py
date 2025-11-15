import torch
import torch.nn as nn
import numpy as np
# from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch # 사용되지 않으므로 제거
import sys
import os
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)
from torch_geometric.data import Data, Batch # 👈 [추가]
from .entity_embedding import EntityEmbeddingBatch, EntityEmbedding
from .attentiontemporalgcn import A3TGCN, A3TGCN2

class A3TGCNCat1(nn.Module):
    def __init__(self, batch_size, col_list, col_dims, num_layers, hidden_channel):
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
        self.num_layers = num_layers
        self.hidden_channel = hidden_channel

        # EntityEmbedding 레이어 정의
        self.entity_embedding_layer = EntityEmbedding(col_dims=self.col_dims,
                                                 col_list=self.col_list)
        a3tgcn_input_channel = self.entity_embedding_layer.proj_dim
        self.a3tgcn_layers = nn.ModuleList()
        
        # A3TGCN2 레이어 정의
        a3tgcn_input_layer = A3TGCN2(in_channels=a3tgcn_input_channel,
                        out_channels=hidden_channel,
                        periods=37,
                        batch_size=batch_size)
        self.a3tgcn_layers.append(a3tgcn_input_layer)        

        for _ in range(num_layers - 1):
            layer = A3TGCN2(in_channels=hidden_channel,
                            out_channels=hidden_channel,
                            periods=37,
                            batch_size=batch_size)
            self.a3tgcn_layers.append(layer)

        # 분류기 레이어 정의
        self.classifier_b = nn.Sequential(
            nn.Linear(hidden_channel * num_layers, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, 2)
        )
    
    def forward(self, batch: Batch, template_edge_index: torch.Tensor):
        '''
        Args:
            batch(torch_geometric.data.Data): X, y만 정의되어 있는 Data 객체, 
            template_edge_index(torch.Tensor): edge_index는 동일하므로 template_edge_index로 한꺼번에 전달
        '''
        entity_emb = self.entity_embedding_layer(batch.x) # data.x [batch, num_nodes, num_features, max_time]  # type: ignore
        self.a3tgcn_layers(entity_emb)
        pass








class A3TGCNCat2(nn.Module):
    def __init__(self, col_dims, col_list, num_layers, hidden_channel, out_channel=2):
        '''
        input_channel이 없는 이유는 정수 레이블이 인풋으로 들어가기 때문 (항상 1)
        hidden_channel: A3TGCNCat2에서의 히든 차원의 수
        out_channel: A3TGCNCat2에서의 결과 차원의 수
        '''
        super().__init__()
        self.num_layers = num_layers

        # 엔티티 임베딩 레이어 정의
        self.entitiy_embedding = EntityEmbeddingBatch(col_dims, col_list)

        # 🚨 수정: A3TGCN 입력 채널은 최종 임베딩 차원의 합(total_F)이 아닌, 
        #         EntityEmbeddingBatch의 출력 차원인 proj_dim입니다.
        #         (EntityEmbeddingBatch가 노드 특징을 [N_total, F_proj_dim]으로 출력한다고 가정)
        #         만약 EntityEmbeddingBatch가 모든 특징을 Concat하여 
        #         [N_total, F_orig * F_proj_dim]을 출력한다면 아래 코드는 F = X_in.shape[2]를 사용해야 합니다.
        #         여기서는 이전 코드의 의도를 따르되, 다음 레이어의 입력이 F_proj_dim이라고 가정합니다.
        F_orig = len(col_list)
        F_proj_dim = self.entitiy_embedding.proj_dim
        
        # F_total = 첫 번째 레이어의 실제 입력 채널 (e.g., 60 * 25 = 1500)
        a3tgcn_input_channel = F_orig * F_proj_dim
        
        F_proj_dim = self.entitiy_embedding.proj_dim
        
        # F_total = 첫 번째 레이어의 실제 입력 채널 (e.g., 60 * 25 = 1500)
        self.F_orig = F_orig         # 👈 [추가] Store F_orig
        self.F_proj_dim = F_proj_dim # 👈 [추가] Store F_proj_dim
        a3tgcn_input_channel = F_orig * F_proj_dim

        # A3TGCN2 레이어 정의
        self.a3tgcn_layers = nn.ModuleList()
        
        # 첫 번째 레이어 정의
        first_layer = A3TGCN2(in_channels=a3tgcn_input_channel, 
                              out_channels=hidden_channel, 
                              periods=1, 
                              batch_size=None)
        self.a3tgcn_layers.append(first_layer)
        

        # 히든 레이어 정의
        for _ in range(1, num_layers):
            hidden_layer = A3TGCN2(in_channels=hidden_channel, 
                                   out_channels=hidden_channel, 
                                   periods=1, 
                                   batch_size=None)
            self.a3tgcn_layers.append(hidden_layer)

        # 분류기 정의
        self.classifier_b = nn.Sequential(
            nn.Linear(hidden_channel * num_layers, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, out_channel)
        )

    def forward(self, signal_batch: list, template_edge_index: torch.LongTensor):
        
        T_max = len(signal_batch)
        if T_max == 0:
            return torch.empty(0)
        
        batch_snapshot_0 = signal_batch[0]
        B = batch_snapshot_0.batch_size
        
        # 1.1. 시간 마스크와 총 노드 수 계산
        time_mask = [batch_.mask for batch_ in signal_batch]
        # (T_max, N_total)
        time_mask_stacked = torch.stack(time_mask, dim=0) 
        N_total = time_mask_stacked.shape[1]
        N = N_total // B # 단일 그래프 노드 수

        # 🚨 [수정 1] 4D 텐서 X_in 생성 벡터화 🚨
        # T_max개의 PyG Batch 리스트를 하나의 거대한 Batch 객체로 병합
        giant_batch = Batch.from_data_list(signal_batch)
        
        # 임베딩을 T_max번 호출하는 대신, 거대 배치를 *단 한 번* 호출
        # (T_max * N_total * F_orig, F_proj_dim)
        embedded_features_unrolled = self.entitiy_embedding(giant_batch)
        
        # 🚨 [수정] 텐서 재조정: (T*N_total*F_orig, F_proj_dim) -> (T*N_total, F_orig * F_proj_dim)
        embedded_features = embedded_features_unrolled.reshape(
            T_max * N_total, self.F_orig, self.F_proj_dim
        ).reshape(
            T_max * N_total, self.F_orig * self.F_proj_dim
        )
        
        # F_total (전체 특징 차원) 추출
        F = embedded_features.shape[1] # (이제 F_orig * F_proj_dim, e.g., 1500)
        DEVICE = embedded_features.device

        # (T_max * N_total, F_total) -> (T_max, N_total, F_total)
        # ⬇️ 이 라인은 이제 (888,000) .reshape (37, 16, 1500)이 되어 정상 동작합니다.
        X_in_TNF = embedded_features.reshape(T_max, N_total, F)
        
        # (T_max, N_total, F_total) -> (T_max, B, N, F_total)
        X_in_TBNF = X_in_TNF.reshape(T_max, B, N, F)
        
        # (T_max, B, N, F_total) -> (B, N, F_total, T_max) (최종 4D 텐서)
        X_in = X_in_TBNF.permute(1, 2, 3, 0)
        # 🚨 [수정 1 끝] 🚨
        
        
        # 🚨 [수정 2] 마스크 생성 벡터화 🚨
        # (T_max, N_total) -> (T_max, B, N)
        mask_TBN = time_mask_stacked.reshape(T_max, B, N)
        
        # N 차원을 따라 .any() 연산 (T_max, B)
        mask_TB = mask_TBN.any(dim=2).long()
        
        # (T_max, B) -> (B, T_max)
        mask_BT = mask_TB.T 
        # 🚨 [수정 2 끝] 🚨
        
        
        # 3. 마스크 확장: (B, T_max) -> (B, N, F, T_max)
        mask_BNDT = mask_BT.unsqueeze(1).unsqueeze(2).expand(B, N, F, T_max)
        
        # 4. X_in에 마스크 적용
        X_in = X_in * mask_BNDT.float().to(DEVICE)
        
        # 5. A3TGCN2 레이어 시퀀스 (이전 수정 사항 유지)
        h_list = []
        for i, a3tgcn in enumerate(self.a3tgcn_layers):
            
            x_out = a3tgcn(X_in, template_edge_index) # (B, N, H)
            
            # 다음 레이어 입력 (T=1 시퀀스로 변환)
            X_in = x_out.unsqueeze(3) # (B, N, H, 1)

            graph_embedding = torch.mean(x_out, dim=1) # (B, H)
            h_list.append(graph_embedding)
        
        # 6. 분류기 입력 준비
        h_combined = torch.cat(h_list, dim=1) # (B, H * num_layers)

        # 7. 분류기 통과
        logits = self.classifier_b(h_combined)
        
        return logits
    
if __name__ == "__main__":
    # 포워드 잘 되는지만 보는 용도
    import pickle
    import os
    import torch
    from utils.device_set import device_set
    DEVICE = device_set()
    dir_path = os.path.dirname(__file__)
    data_path = os.path.join(dir_path, '..', 'data', 'Sampled_temporal_graph_data_fully_connected.pickle')
    with open(data_path, 'rb') as f:
        pickle_dataset = pickle.load(f)
    
    signal1 = pickle_dataset[0][0] # signal 객체
    
    col_list, col_dim = pickle_dataset[3]
    num_features = len(col_list)
    template_edge_index = pickle_dataset[0][0][0].edge_index.to(DEVICE)
    time_mask = pickle_dataset[0][0][0].mask

    train_dataset = pickle_dataset[0]

    # 모델 준비
    model = A3TGCNCat2(col_dims=col_dim, col_list=col_list, num_layers=3, hidden_channel=64, out_channel=2)
    model.to(DEVICE)

    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)

    # 최적화 알고리즘
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 포워드
    for signal in train_dataset:
        batch_list = [batch.to(DEVICE) for batch in signal]
        logits = model.forward(batch_list, template_edge_index)
        print(logits)
        loss = criterion(logits, signal[0].y.to(DEVICE))
    

        
            