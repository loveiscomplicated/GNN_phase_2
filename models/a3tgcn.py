import torch
import torch.nn as nn
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch

import sys
import os
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)
from .entity_embedding import EntityEmbeddingBatch
from .attentiontemporalgcn import A3TGCN2

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

        a3tgcn_input_channel = self.entitiy_embedding.proj_dim
        # A3TGCN2 레이어 정의
        self.a3tgcn_layers = nn.ModuleList()
        
        # 첫 번째 레이어 정의
        first_layer = A3TGCN2(in_channels=a3tgcn_input_channel, 
                              out_channels=hidden_channel, 
                              periods=1, 
                              batch_size=None) # type: ignore
        self.a3tgcn_layers.append(first_layer)
        

        # 히든 레이어 정의
        for _ in range(1, num_layers):
            hidden_layer = A3TGCN2(in_channels=hidden_channel, 
                                   out_channels=hidden_channel, 
                                   periods=1, 
                                   batch_size=None) # type: ignore
            self.a3tgcn_layers.append(hidden_layer)

        # 분류기 정의
        self.classifier_b = nn.Sequential(
            nn.Linear(hidden_channel * num_layers, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, out_channel)
        )

    def forward(self, signal_batch: list, template_edge_index: torch.LongTensor):
        """

        입력으로 T개의 시점을 가진 시계열 그래프 배치 리스트를 받고,
        이를 4D 텐서(B, N_max, F, T)로 변환하고 다층(L) A3TGCN2 레이어를
        통과시켜 시공간적 특징을 추출
        각 레이어의 최종 은닉 상태를 풀링(Readout)하여 그래프 임베딩을 얻고,
        이를 모두 연결(concatenate)하여 최종 이진 분류(REASONb) 로짓을 반환

        Args:
            signal_batch (list): 
                                T (시점 수)개의 `torch_geometric.data.Batch` 객체 리스트.
                                [Batch_t1, Batch_t2, ..., Batch_tT]
            time_mask(list):
                                넘파이 벡터들의 리스트. 각각의 요소는 매 시점 별 batch 안에서 누가 살아남았는지

            template_edge_index(torch.LongTensor):
                                어차피 모든 그래프의 구조가 동일하므로 처음 인풋으로 한 번만 입력
        Returns:
            torch.Tensor: 최종 이진 분류 로짓 (Batch_Size, Out_Channel=2)
        """
        # 1. 4D 텐서를 만들기 위한 준비
        features_3d_list = []

        # 엣지 인덱스 및 배치 벡터는 시점 불변이므로 첫 번째 스냅샷에서 추출
        # 즉, batch_snapshot_0은 Batch 객체
        batch_snapshot_0 = signal_batch[0]
        B = batch_snapshot_0.batch_size  # 배치 크기
        N = batch_snapshot_0.x.shape[0] // B # N = N_total / B
        
        for batch_snapshot in signal_batch:
            # 엔티티 임베딩: (N_total, F)
            embedded_features = self.entitiy_embedding(batch_snapshot)

            # (B * N, F) -> (B, N, F)
            features_dense = embedded_features.reshape(B, N, -1)
            features_3d_list.append(features_dense)

        # 1.3. 시간 축 T로 스택
        # X_in 셰이프: (B, N, F, T)
        T_max = len(features_3d_list)
        X_in = torch.stack(features_3d_list, dim=3)
        
        DEVICE = X_in.device
        F = X_in.shape[2]
        
        # 1. 마스크 축 변환: (T, B) -> (B, T)
        # PyTorch 텐서 셰이프 순서에 맞춤
        time_mask = [batch_.mask for batch_ in signal_batch]
        mask_BT = torch.as_tensor(torch.stack(time_mask, dim=0).T, dtype=torch.long)
        
        # 2. 마스크 확장: (B, T) -> (B, N, F, T)
        # N, D 축에 맞춰 확장 (unsqueeze(1) for N, unsqueeze(2) for F)
        mask_BNDT = mask_BT.unsqueeze(1).unsqueeze(2).expand(B, N, F, T_max)
        
        # 3. X_in에 마스크 적용
        # False(0)인 위치의 특징 값은 0이 되고, True(1)인 위치는 유지됨.
        X_in = X_in * mask_BNDT.float().to(DEVICE)


        # 2. A3TGCN2 시퀀스 처리 및 배치 Readout
        h_list = []
        
        for i, a3tgcn in enumerate(self.a3tgcn_layers):
            
            # x_out: 최종 은닉 상태 (B, N, H)
            # A3TGCN2가 (B,N,F,T) 입력을 받아 (B,N,H) 출력을 반환한다고 가정
            x_out = a3tgcn(X_in, template_edge_index) 
            
            # 다음 레이어 입력 구성: (B, N, H, T)
            X_in = x_out.unsqueeze(3).repeat(1, 1, 1, T_max)
            
            # ⭐️ Global Pooling (Readout) 간소화
            # N 차원 (dim=1)에 대해 평균을 계산
            graph_embedding = torch.mean(x_out, dim=1) # (B, H)
            h_list.append(graph_embedding)

        # 3. 그래프 임베딩 연결 및 분류
        
        # combined_h: (B, L * H)
        combined_h = torch.cat(h_list, dim=1)
        
        # logits: (B, 2)
        logits = self.classifier_b(combined_h)
        
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
    

        
            