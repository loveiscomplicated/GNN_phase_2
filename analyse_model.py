import os
import pickle
import csv

import torch
import torch.nn as nn
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from torch_geometric.explain import Explainer, GNNExplainer

# 프로젝트 내부 모듈
from models.a3tgcn import A3TGCNCat2
from utils.device_set import device_set


# -------------------------------------------------------------------
# 1. 체크포인트 로더
# -------------------------------------------------------------------
def load_checkpoint(model, optimizer, lr_scheduler, filepath, device):
    """
    학습된 체크포인트를 로드하여 모델 파라미터를 복원한다.
    optimizer / scheduler는 분석에는 직접 쓰지 않지만, 형태만 맞춰 둔다.
    """
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return None, None

    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except KeyError:
            print("Warning: Optimizer/Scheduler state not found in checkpoint. Skipping.")

        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(
            f"Checkpoint loaded from {filepath}. "
            f"Epoch: {start_epoch}, Best Loss: {best_val_loss:.4f}"
        )
        return start_epoch, best_val_loss

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None


# -------------------------------------------------------------------
# 2. (B>1) Batch에서 B=0 하나만 뽑아서 시계열 리스트로 만드는 함수
# -------------------------------------------------------------------
def extract_single_sample_list(original_batch_list, device):
    """
    StaticGraphTemporalSignalBatch 스타일의 하나의 signal(리스트)을 받아,
    그 안에서 배치 인덱스 0번에 해당하는 그래프만 뽑아서
    시간축 리스트(single_sample_list)로 반환한다.
    """
    single_sample_list = []

    for batch_obj in original_batch_list:
        batch_obj = batch_obj.to(device)

        # batch == 0 인 노드만 남긴다.
        node_mask = batch_obj.batch == 0
        subset = torch.where(node_mask)[0]

        edge_index_0, _ = subgraph(
            subset=subset,
            edge_index=batch_obj.edge_index,
            relabel_nodes=True,
            num_nodes=batch_obj.num_nodes,
        )

        x_0 = batch_obj.x[node_mask]
        y_0 = batch_obj.y[0].unsqueeze(0)  # 그래프 레이블 하나

        graph_mask_value = batch_obj.mask[0]
        num_nodes_in_graph_0 = x_0.shape[0]
        mask_0 = graph_mask_value.repeat(num_nodes_in_graph_0)

        data_obj = Data(x=x_0, edge_index=edge_index_0, y=y_0, mask=mask_0).to(device)
        single_sample_list.append(data_obj)

    return single_sample_list


# -------------------------------------------------------------------
# 3. GNNExplainer용 래퍼
#    - t_to_explain 시점의 임베딩만 GNNExplainer가 마스킹할 수 있도록 한다.
# -------------------------------------------------------------------
class A3TGCNGNNExplainerWrapper(nn.Module):
    """
    A3TGCNCat2를 GNNExplainer에 물릴 수 있도록 래핑한 클래스.

    - GNNExplainer는 x(노드 특징)에 node_mask를 곱해가면서
      "설명 가능한 입력"을 만들고,
    - 우리는 그 x를 "t_to_explain 시점의 임베딩"으로 사용한다.
    - 나머지 시점의 임베딩은 모델의 원래 엔티티 임베딩을 사용한다.
    """

    def __init__(self, original_model, template_edge_index, full_sample_list, t_to_explain=0):
        super().__init__()
        self.model = original_model
        self.template_edge_index = template_edge_index
        self.full_sample_list = full_sample_list  # [Data_t0, Data_t1, ...]
        self.t_to_explain = t_to_explain
        self.T_max = len(full_sample_list)

        # 원본 모델의 임베딩 레이어
        self.embedding_layer = original_model.entitiy_embedding

    def forward(self, x, edge_index):
        """
        x: (N_t, F_emb) - GNNExplainer가 마스킹을 적용한 임베딩 (t_to_explain 시점용)
        edge_index: t_to_explain 시점 그래프의 edge_index (사용은 안 하지만 시그니처용)
        """
        device = x.device
        new_batch_list = []

        all_embedded_features = []
        current_index = 0

        for t in range(self.T_max):
            original_data = self.full_sample_list[t].to(device)
            batch_data = Batch.from_data_list([original_data])
            new_batch_list.append(batch_data)

            # 원래 임베딩
            embedded_t = self.embedding_layer(batch_data)  # (N_t, F_emb)

            if t == self.t_to_explain:
                # 이 시점의 임베딩은 GNNExplainer가 조작한 x로 대체
                all_embedded_features.append(x)
            else:
                all_embedded_features.append(embedded_t)

            current_index += embedded_t.shape[0]

        # (T_max * N_t, F_emb)
        final_embedded_features = torch.cat(all_embedded_features, dim=0)

        # A3TGCNCat2.forward가 pre_embedded_features를 받을 수 있다는 가정
        logits = self.model(
            new_batch_list,
            self.template_edge_index,
            pre_embedded_features=final_embedded_features,
        )

        return logits


# -------------------------------------------------------------------
# 4. 하나의 시계열 샘플에 대해 feature-level 중요도 계산
# -------------------------------------------------------------------
def get_explanation_scores(model, device, template_edge_index, single_sample_list, t_to_explain=0):
    """
    하나의 시계열 샘플(single_sample_list)을 받아,
    GNNExplainer를 이용해 t_to_explain 시점의 '입력 변수(노드)' 중요도를 계산한다.

    반환값:
        node_importance: shape (num_nodes,) 의 numpy array
                         각 노드는 col_list[i]에 대응하는 하나의 feature로 해석.
    """
    try:
        data_to_explain = single_sample_list[t_to_explain].to(device)

        model.eval()
        with torch.no_grad():
            temp_batch_for_emb = Batch.from_data_list([data_to_explain])
            embedded_x_input = model.entitiy_embedding(temp_batch_for_emb)  # (N, F_emb)

        x_input = embedded_x_input
        edge_index_input = data_to_explain.edge_index

        wrapped_model = A3TGCNGNNExplainerWrapper(
            original_model=model,            # ✅ 이름 맞춰주기
            template_edge_index=template_edge_index,
            full_sample_list=single_sample_list,
            t_to_explain=t_to_explain,
        ).to(device)

        # node_mask_type='object' → 노드(=feature) 단위로 스칼라 마스크
        explainer = Explainer(
            model=wrapped_model,
            algorithm=GNNExplainer(epochs=200, lr=0.01),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type=None,
            model_config=dict(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )

        explanation = explainer(
            x=x_input,
            edge_index=edge_index_input,
        )

        # node_mask: (num_nodes,) 또는 (num_nodes, 1) → 1D로 평탄화
        node_importance = explanation.node_mask.view(-1).cpu().detach().numpy()
        return node_importance

    except Exception as e:
        print(f"Error during explanation: {e}")
        return None


# -------------------------------------------------------------------
# 5. 메인: 전역 + 시점별 feature importance 분석 & CSV 저장
# -------------------------------------------------------------------
if __name__ == "__main__":
    from tqdm import tqdm

    DEVICE = device_set()
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))

    # ✅ 학습 때 사용한 fully_connected 데이터 기준으로 맞춤
    DATA_PATH = os.path.join(CUR_DIR, "data", "Sampled_temporal_graph_data_fully_connected.pickle")

    MODEL_SAVE_PATH = os.path.join(CUR_DIR, "a3tgcn_checkpoints")
    BEST_MODEL_FILENAME = "best_model_epoch_4_loss_0.9801.pth"  # 실제 파일명에 맞게 조정
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_FILENAME)

    print(f"DEVICE: {DEVICE}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"BEST_MODEL_PATH: {BEST_MODEL_PATH}")

    # 1. 데이터 로드
    if not os.path.exists(DATA_PATH):
        print(f"🚨 Data file not found: {DATA_PATH}")
        exit()

    with open(DATA_PATH, "rb") as f:
        pickle_dataset = pickle.load(f)

    train_dataset, val_dataset, test_dataset, col_info = pickle_dataset
    col_list, col_dim = col_info

    # 2. Edge index (템플릿 그래프) 준비
    #    - train_dataset[0][0] : 첫 번째 샘플의 첫 번째 시점 Batch
    template_edge_index = torch.as_tensor(
        train_dataset[0][0].edge_index, dtype=torch.long
    ).to(DEVICE)

    # 3. 모델 생성 및 체크포인트 로드
    model = A3TGCNCat2(
        col_dims=col_dim,
        col_list=col_list,
        num_layers=2,
        hidden_channel=64,
        out_channel=2,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"🚨 Checkpoint not found: {BEST_MODEL_PATH}")
        exit()

    start_epoch, best_val_loss = load_checkpoint(
        model, optimizer, lr_scheduler, BEST_MODEL_PATH, DEVICE
    )
    if start_epoch is None:
        print("🚨 Failed to load model. Exiting.")
        exit()

    model.eval()

    # 4. GNNExplainer 반복 실행: 시점별 importance 수집
    N_SAMPLES_TO_ANALYZE = 50
    num_samples = min(N_SAMPLES_TO_ANALYZE, len(test_dataset))

    print(f"\n--- 🔬 {num_samples}개 샘플에 대해 시점별/전역 feature 중요도 분석 시작 ---\n")

    time_importance_scores = None
    T_max = None

    for i in tqdm(range(num_samples), desc="Analyzing Samples"):
        original_batch_list = test_dataset[i]

        single_sample_list = extract_single_sample_list(original_batch_list, DEVICE)
        if not single_sample_list:
            print(f"Warning: sample {i} → no data extracted. Skipping.")
            continue

        if time_importance_scores is None:
            T_max = len(single_sample_list)
            time_importance_scores = [[] for _ in range(T_max)]
            print(f"⏱ 시점 수(T_max): {T_max}")

        for t in range(T_max):
            scores_t = get_explanation_scores(
                model=model,
                device=DEVICE,
                template_edge_index=template_edge_index,
                single_sample_list=single_sample_list,
                t_to_explain=t,
            )

            if scores_t is not None:
                time_importance_scores[t].append(scores_t)
            else:
                print(f"Warning: sample {i}, time {t} → importance 계산 실패")

    # 5. 시점별 / 전역 importance 집계 + 출력 + CSV 저장
    if time_importance_scores is None:
        print("🚨 분석 실패: 유효한 중요도 점수를 얻지 못했습니다.")
        exit()

    mean_abs_scores_per_time = []

    for t in range(T_max):
        if not time_importance_scores[t]:
            print(f"⚠️ 시점 t={t}에 대해 수집된 점수가 없습니다. 0 벡터로 대체합니다.")
            mean_abs_scores_per_time.append(np.zeros(ref_num_nodes))
            continue

    stacked_t = np.vstack(time_importance_scores[t])  # (N_samples_t, num_nodes)
    mean_abs_scores_t = np.mean(np.abs(stacked_t), axis=0)
    mean_abs_scores_per_time.append(mean_abs_scores_t)


    # (T_max, num_nodes)
    mean_abs_scores_per_time = np.stack(mean_abs_scores_per_time, axis=0)

    # 전역(시점 평균) 중요도: (num_nodes,)
    overall_mean_abs_scores = mean_abs_scores_per_time.mean(axis=0)

    num_nodes = len(overall_mean_abs_scores)
    max_len = min(len(col_list), num_nodes)

    # 5-1. 전역 중요도 기준 정렬 & 출력
    sorted_indices = np.argsort(overall_mean_abs_scores)[::-1]

    print("\n--- 📊 전역 입력 변수 중요도 (모든 시점 평균, GNNExplainer 기반) ---")
    print(f"[샘플 {num_samples}개, 시점 {T_max}개, 절댓값 평균 기준, 내림차순]\n")

    for idx in sorted_indices[:max_len]:
        col_name = col_list[idx]
        score = overall_mean_abs_scores[idx]
        print(f"  - {col_name:>20s} : {score:.6f}")

    # 5-2. 시점별 Top-K 출력
    TOP_K = 5
    print("\n--- ⏱ 시점별 Top-5 입력 변수 중요도 ---")
    for t in range(T_max):
        scores_t = mean_abs_scores_per_time[t]
        sorted_idx_t = np.argsort(scores_t)[::-1][:TOP_K]

        print(f"\n[Time t={t}]")
        for idx in sorted_idx_t:
            col_name = col_list[idx]
            print(f"  - {col_name:>20s} : {scores_t[idx]:.6f}")

    # 5-3. CSV 저장
    OUTPUT_DIR = os.path.join(CUR_DIR, "analysis_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    overall_csv_path = os.path.join(OUTPUT_DIR, "feature_importance_overall.csv")
    per_time_csv_path = os.path.join(OUTPUT_DIR, "feature_importance_by_time.csv")

    # 전체 평균 중요도
    with open(overall_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance_overall"])
        for idx in range(max_len):
            writer.writerow([col_list[idx], float(overall_mean_abs_scores[idx])])

    # 시점별 중요도 (wide format)
    with open(per_time_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["feature"] + [f"t{t}" for t in range(T_max)]
        writer.writerow(header)

        for idx in range(max_len):
            row = [col_list[idx]] + [
                float(mean_abs_scores_per_time[t, idx]) for t in range(T_max)
            ]
            writer.writerow(row)

    print("\n✅ CSV 저장 완료:")
    print(f" - 전체 평균 중요도: {overall_csv_path}")
    print(f" - 시점별 중요도:   {per_time_csv_path}")
    print("\n--- 전역 Feature-level + 시점별 분석 완료 ---")
