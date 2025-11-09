import torch
import torch.nn as nn

from entity_embedding import EntityEmbeddingBatch
from .attentiontemporalgcn import A3TGCN2

class A3TGCNCat(nn.Module):
    def __init__(self, col_dims, col_list, num_layers, input_channel, hidden_channel, out_channel=2):
        super().__init__()
        self.num_layers = num_layers

        # 엔티티 임베딩 레이어 정의
        self.entitiy_embedding = EntityEmbeddingBatch(col_dims, col_list)

        # A3TGCN2 레이어 정의
        self.a3tgcn_layers = nn.ModuleList()
        
        # 첫 번째 레이어 정의
        first_layer = A3TGCN2(in_channels=input_channel, 
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
        """

        입력으로 T개의 시점을 가진 시계열 그래프 배치 리스트를 받고,
        이를 4D 텐서(B, N_max, D, T)로 변환하고 다층(L) A3TGCN2 레이어를
        통과시켜 시공간적 특징을 추출
        각 레이어의 최종 은닉 상태를 풀링(Readout)하여 그래프 임베딩을 얻고,
        이를 모두 연결(concatenate)하여 최종 이진 분류(REASONb) 로짓을 반환

        Args:
            signal_batch (StaticGraphTemporalSignalBatch): 
                                T (시점 수)개의 `torch_geometric.data.Batch` 객체 리스트.
                                [Batch_t1, Batch_t2, ..., Batch_tT]
                                이거 자료형이 신기하긴 한데 리스트처럼 사용해도 완전 무방 (애초에 그렇게 설계됨)
            template_edge_index(torch.LongTensor):
                                어차피 모든 그래프의 구조가 동일하므로 처음 인풋으로 한 번만 입력
        Returns:
            torch.Tensor: 최종 이진 분류 로짓 (Batch_Size, Out_Channel=2)
        """
        # 1. 4D 텐서를 만들기 위한 준비
        features_3d_list = []

        # 엣지 인덱스 및 배치 벡터는 시점 불변이므로 첫 번째 스냅샷에서 추출
        batch_snapshot_0 = signal_batch[0]
        B = batch_snapshot_0.num_graphs  # 배치 크기
        N = batch_snapshot_0.num_nodes // B # N = N_total / B
        
        
        for batch_snapshot in signal_batch:
            # 엔티티 임베딩: (N_total, D)
            embedded_features = self.entitiy_embedding(batch_snapshot.x)

            # (B * N, D) -> (B, N, D)
            features_dense = embedded_features.reshape(B, N, -1)
            features_3d_list.append(features_dense)
        # 1.3. 시간 축 T로 스택
        # X_in 셰이프: (B, N, D, T)
        T = len(features_3d_list)
        X_in = torch.stack(features_3d_list, dim=3)
        
        # 2. A3TGCN2 시퀀스 처리 및 배치 Readout
        h_list = []
        
        for i, a3tgcn in enumerate(self.a3tgcn_layers):
            
            # x_out: 최종 은닉 상태 (B, N, H)
            # A3TGCN2가 (B,N,F,T) 입력을 받아 (B,N,H) 출력을 반환한다고 가정
            x_out = a3tgcn(X_in, template_edge_index) 
            
            # 다음 레이어 입력 구성: (B, N, H, T)
            X_in = x_out.unsqueeze(3).repeat(1, 1, 1, T)
            
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