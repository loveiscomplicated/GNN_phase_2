import os
import sys

cur_dir = os.path.dirname(__file__)
par_dir = os.path.join(cur_dir, '..')
sys.path.append(par_dir)

from tqdm import tqdm
import torch
from teds_tensor_dataset import TEDSTensorDataset
from models.gin_gru import GinGru
from models.gingru_for_explain import GinGruForExplain2
from utils.processing_utils import train_test_split_customed_dataset, mi_edge_index_improved, train_test_split_customed, mi_edge_index_batched

def load_checkpoint(model, optimizer, scheduler, filename, device):
    """
    저장된 체크포인트(.pth)를 불러와서 
    model, optimizer, scheduler 상태를 복구합니다.

    Parameters:
        model (nn.Module): 모델 객체
        optimizer (torch.optim.Optimizer): 옵티마이저 객체
        scheduler: 스케줄러 객체
        filename (str): 저장된 체크포인트 경로
        device: CPU로 로드하고 싶으면 torch.device('cpu')

    Returns:
        start_epoch (int): 다음 훈련을 시작할 epoch 번호
        best_loss (float): 저장된 최소 validation loss
    """
    checkpoint = torch.load(filename, map_location=device)

    # --- Load states ---
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']

    return start_epoch, best_loss

root = os.path.join(par_dir, 'data_tensor_cache')
checkpoint_path = os.path.join(par_dir, 'checkpoints', 'gingru', 'real', 'best_gingru_epoch_6_loss_0.3761.pth')

dataset = TEDSTensorDataset(root)
_, _, test_dataset = train_test_split_customed_dataset(dataset=dataset, seed=42)
col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info

batch_size = 1
embedding_dim = 32
gin_hidden_channel = 32
gin_layers = 2
gru_hidden_channel = 64

optim_lr = 0.001
scheduler_patience = 15
device = torch.device("cpu")


model = GinGru(
    batch_size=batch_size,
    col_list=col_list,
    col_dims=col_dims,
    ad_col_index=ad_col_index,
    dis_col_index=dis_col_index,
    embedding_dim=embedding_dim,
    gin_hidden_channel=gin_hidden_channel,
    train_eps=True,
    gin_layers=gin_layers,
    gru_hidden_channel=gru_hidden_channel
)

model2 = GinGruForExplain2(
    batch_size=batch_size,
    col_list=col_list,
    col_dims=col_dims,
    ad_col_index=ad_col_index,
    dis_col_index=dis_col_index,
    embedding_dim=embedding_dim,
    gin_hidden_channel=gin_hidden_channel,
    train_eps=True,
    gin_layers=gin_layers,
    gru_hidden_channel=gru_hidden_channel
)

optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=scheduler_patience)

start_epoch, best_loss = load_checkpoint(model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        filename=checkpoint_path,
                                        device=device)

start_epoch, best_loss = load_checkpoint(model=model2,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        filename=checkpoint_path,
                                        device=device)
model.to(device)
model.eval()

model2.to(device)
model2.eval()

mi_dict_path = os.path.join(root, 'data', 'mi_dict_static.pickle')
edge_index = mi_edge_index_batched(batch_size=32, 
                                   num_nodes=len(ad_col_index), 
                                   mi_dict_path=mi_dict_path)
empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

_, _, test_dataloader = train_test_split_customed(dataset=dataset, batch_size=32)

prob_diff_sum = 0

'''print("model1")
for batch in tqdm(test_dataloader):
    x, y, los_batch = batch
    with torch.no_grad():
        logit = model(x, los_batch, edge_index, device)
        prob = torch.sigmoid(logit)
        
        logit_empty = model(x, los_batch, empty_edge_index, device)
        prob_empty = torch.sigmoid(logit_empty)
        
        prob_diff = abs(prob - prob_empty)
        prob_diff = prob_diff.sum()
        
        prob_diff_sum += prob_diff
    

prob_diff_mean = prob_diff_sum / len(test_dataset)        

print(prob_diff_mean)'''

'''print("model2")
for batch in tqdm(test_dataloader):
    x, y, los_batch = batch
    x = model2.entity_embedding_layer(x)
    with torch.no_grad():
        logit = model2(x, edge_index, LOS_batch=los_batch, device=device)
        prob = torch.sigmoid(logit)
        
        logit_empty = model2(x, empty_edge_index, LOS_batch=los_batch, device=device)
        prob_empty = torch.sigmoid(logit_empty)
        
        prob_diff = abs(prob - prob_empty)
        prob_diff = prob_diff.sum()

        prob_diff_sum += prob_diff

prob_diff_mean = prob_diff_sum / len(test_dataset)        

print(prob_diff_mean)'''


import torch
import numpy as np
from scipy import stats  # 통계 검정을 위해 추가
from tqdm import tqdm

# ... (이전 설정 코드 생략) ...

print("model2 Statistical Test")

# 결과를 모아둘 리스트 생성
all_probs_edge = []
all_probs_empty = []
all_y_true = []

for batch in tqdm(test_dataloader):
    x, y, los_batch = batch
    
    # 모델 2의 경우 entity embedding 처리
    x = model2.entity_embedding_layer(x)
    
    with torch.no_grad():
        # 1. Edge가 있는 경우
        logit = model2(x, edge_index, LOS_batch=los_batch, device=device)
        prob = torch.sigmoid(logit)
        
        # 2. Edge가 없는 경우 (Empty)
        logit_empty = model2(x, empty_edge_index, LOS_batch=los_batch, device=device)
        prob_empty = torch.sigmoid(logit_empty)
        
        # CPU로 내리고 Numpy 변환 후 리스트에 추가 (1차원 벡터로 펼침)
        all_probs_edge.append(prob.cpu().numpy().flatten())
        all_probs_empty.append(prob_empty.cpu().numpy().flatten())

        all_y_true.append(y.cpu().numpy().flatten())

# 배치별 결과를 하나의 큰 배열로 합침
probs_edge_concat = np.concatenate(all_probs_edge)
probs_empty_concat = np.concatenate(all_probs_empty)
y_true_concat = np.concatenate(all_y_true)

print(f"Total samples: {len(y_true_concat)}")

# --- 1. 기본 통계량 계산 ---
diff = np.abs(probs_edge_concat - probs_empty_concat)
mean_diff = np.mean(diff)
print(f"Mean Absolute Difference: {mean_diff:.6f}")

# --- 2. Paired T-test (대응표본 t-검정) ---
# 귀무가설: 두 모델(Edge 유/무)의 예측값 평균에는 차이가 없다.
t_stat, p_val_t = stats.ttest_rel(probs_edge_concat, probs_empty_concat)

print(f"Paired T-test | t-statistic: {t_stat:.4f}, p-value: {p_val_t:.4e}")

if p_val_t < 0.05:
    print(">> 결과: 통계적으로 유의미한 차이가 있습니다 (p < 0.05).")
else:
    print(">> 결과: 통계적으로 유의미한 차이가 발견되지 않았습니다.")

# --- 3. Wilcoxon Signed-Rank Test (윌콕슨 부호 순위 검정) ---
# 딥러닝 출력값(확률)은 정규분포가 아닐 가능성이 높으므로, 이 검정이 더 엄밀할 수 있습니다.
w_stat, p_val_w = stats.wilcoxon(probs_edge_concat, probs_empty_concat)

print(f"Wilcoxon Test | statistic: {w_stat:.4f}, p-value: {p_val_w:.4e}")

import matplotlib.pyplot as plt

# 차이값 분포 시각화
plt.figure(figsize=(10, 5))
plt.hist(probs_edge_concat - probs_empty_concat, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Prediction Differences (Edge - Empty)')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--') # 0점 기준선
plt.show()

# 1. 차이 계산 (Edge - Empty)
diff = probs_edge_concat - probs_empty_concat

# 2. 확률이 급격히 떨어진 데이터(-0.8 미만)의 인덱스 찾기
# (그래프에서 가장 왼쪽의 높은 기둥에 해당하는 데이터들입니다)
drop_indices = np.where(diff < -0.8)[0]

# 3. 해당 데이터들의 실제 정답(Label) 확인
y_in_drop_zone = y_true_concat[drop_indices]

# 4. 결과 출력
print(f"=== 검증 결과 ===")
print(f"확률이 급격히 떨어진 샘플 수: {len(drop_indices)}개")
print(f"전체 테스트 데이터 대비 비율: {len(drop_indices) / len(y_true_concat) * 100:.2f}%")

# 중요: 이들의 실제 정답 비율 (0: 실패/Negative, 1: 성공/Positive)
ratio_0 = np.mean(y_in_drop_zone == 0)
ratio_1 = np.mean(y_in_drop_zone == 1)

print(f"이들 중 실제 정답이 0인 비율: {ratio_0 * 100:.2f}%")
print(f"이들 중 실제 정답이 1인 비율: {ratio_1 * 100:.2f}%")

if ratio_0 > 0.8:
    print("\n>> [결론: 긍정적] GNN이 '가짜 성공(False Positive)'일 뻔한 데이터들을 올바르게 걸러냈습니다.")
    print(">> Node Feature만 보면 성공(1) 같지만, 관계를 보니 실패(0)인 케이스를 잘 잡아낸 것입니다.")
elif ratio_1 > 0.8:
    print("\n>> [결론: 부정적] GNN이 실제로 성공(1)한 케이스들을 잘못 억압(Suppress)하고 있습니다.")
    print(">> 관계 정보가 오히려 모델을 혼란스럽게 하여 성능을 떨어뜨릴 위험이 있습니다.")
else:
    print("\n>> [결론: 복합적] 0과 1이 섞여 있어 추가 분석이 필요합니다.")