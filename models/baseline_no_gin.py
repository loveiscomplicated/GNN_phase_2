import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .entity_embedding import EntityEmbeddingBatch3


def get_mask(los_batch: torch.Tensor):
    max_los = 37
    indices = torch.arange(max_los, device=los_batch.device)
    mask = (indices < los_batch.unsqueeze(1))
    return mask.int().unsqueeze(2)


def to_temporal_gingru(x: torch.Tensor, los_batch: torch.Tensor):
    batch_size = int(x.shape[0] / 2)
    ad = x[:batch_size].unsqueeze(1)     # [B, 1, F]
    dis = x[batch_size:]                 # [B, F]

    mask = get_mask(los_batch)           # [B, 37, 1]
    padded = ad * mask                   # [B, 37, F]

    batch_indices = torch.arange(batch_size, device=x.device)
    last_time_indices = los_batch - 1
    padded[batch_indices, last_time_indices, :] = dis

    lengths = los_batch.cpu()
    lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
    padded_sorted = padded[sorted_indices]

    packed = pack_padded_sequence(
        padded_sorted,
        lengths_sorted,
        batch_first=True,
        enforce_sorted=True
    )

    return packed, sorted_indices


class BaselineNoGIN(nn.Module):
    """
    GIN 없이 EntityEmbedding + AD/DIS + GRU 기반 Baseline 모델
    """
    def __init__(self, col_dims, embedding_dim, gru_hidden_channel, classifier_hidden=128):
        super().__init__()

        self.embedding_layer = EntityEmbeddingBatch3(
            col_dims=col_dims,
            embedding_dim=embedding_dim
        )

        # Baseline pooling 결과 차원 = embedding_dim 그대로
        self.feature_dim = embedding_dim

        # GRU 정의
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden_channel
        )

        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_channel, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, 1)
        )

    def forward(self, x_batch, LOS_batch, ad_idx_t, dis_idx_t, device):
        """
        x_batch: [B, num_var]
        """
        ad_idx_t = ad_idx_t.to(device)
        dis_idx_t = dis_idx_t.to(device)

        # 1) 엔티티 임베딩
        x_emb = self.embedding_layer(x_batch)          # [B, 72, emb_dim]

        # 2) 변수 차원 sum pooling (GIN 제거)
        ad_tensor = torch.index_select(x_emb, dim=1, index=ad_idx_t)     # [B, 60, F]
        dis_tensor = torch.index_select(x_emb, dim=1, index=dis_idx_t)   # [B, 60, F]

        ad_pooled = ad_tensor.sum(dim=1)   # [B, F]
        dis_pooled = dis_tensor.sum(dim=1) # [B, F]

        # 3) temporal 변환
        x_concat = torch.cat((ad_pooled, dis_pooled), dim=0)  # [B*2, F]

        temporal_packed, sorted_indices = to_temporal_gingru(
            x_concat,
            los_batch=LOS_batch
        )

        # 4) GRU
        gru_out, gru_h = self.gru(temporal_packed)
        gru_h = gru_h.squeeze(0)

        # 5) 정렬 복구
        inv_idx = torch.argsort(sorted_indices.to(gru_h.device))
        gru_h = gru_h[inv_idx]

        # 6) classifier
        out = self.classifier(gru_h)
        return out
