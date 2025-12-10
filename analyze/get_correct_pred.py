import sys
import os
import random
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch_geometric.explain.config import ModelConfig
from tqdm import tqdm

cur_dir = os.path.dirname(__file__)
par_dir = os.path.join(cur_dir, '..')
sys.path.append(par_dir)

from teds_tensor_dataset import TEDSTensorDataset
from utils.processing_utils import mi_edge_index_batched
from models.gingru_for_explain import GinGruForExplain2

# 기존의 seed_everything 함수
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"SEED = {seed}")

seed_everything(2025)

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

batch_size = 32
embedding_dim = 32
gin_hidden_channel = 32
gin_layers = 2
gru_hidden_channel = 64

scheduler_patience = 15
# early_stopping_patience = 10
device = torch.device("cpu")
optim_lr = 0.001

epochs = 100
explainer_lr = 0.003
root = os.path.join(par_dir, 'data_tensor_cache')
model_path = os.path.join(par_dir, 'checkpoints', 'gingru', 'real')

checkpoint_path = os.path.join(par_dir, 'checkpoints', 'gingru', 'real', 'best_gingru_epoch_6_loss_0.3761.pth')
mi_dict_path = os.path.join(root, 'data', 'mi_dict_static.pickle')

dataset = TEDSTensorDataset(root)
col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info

model = GinGruForExplain2(
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
model.to(device)
model.eval()

def get_correct_stratified_subset(model, dataset, device, num_samples=30000):
    """
    1. 모델이 정답을 맞춘 데이터만 필터링
    2. 그 중에서 Class 0과 1을 균형 있게(Stratified) 샘플링하여 Subset 반환
    """
    print("Filtering correct predictions and stratifying...")
    model.eval()
    
    correct_indices_0 = [] # 맞춘 것 중 정답이 0인 인덱스
    correct_indices_1 = [] # 맞춘 것 중 정답이 1인 인덱스
    
    # 전체 데이터셋을 한번 훑어서 맞춘 인덱스만 수집
    # (DataLoader를 쓰지 않고 인덱스로 접근하거나, 큰 배치로 빠르게 훑습니다)
    temp_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(temp_loader, desc="Filtering Data"):
            x_batch, y_batch, los_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            los_batch = los_batch.to(device)
            
            # 모델 예측
            x_embedded = model.entity_embedding_layer(x_batch)
            # edge_index는 전체 데이터를 위한 것이므로 배치 처리가 필요하지만,
            # 여기서는 로직을 단순화하기 위해 "맞췄다/틀렸다" 판단만 빠르게 합니다.
            # (주의: 기존 get_y_pred 함수가 배치 처리를 잘 지원해야 함)
            
            # --- [중요] 기존 코드의 배치용 edge_index 로직 활용 ---
            # 배치 사이즈가 달라질 수 있으므로 유동적으로 edge_index 생성 필요
            curr_batch_size = x_batch.size(0)
            batch_edge_index = mi_edge_index_batched(batch_size=curr_batch_size,
                                                     num_nodes=len(ad_col_index),
                                                     mi_dict_path=mi_dict_path).to(device)
            
            logits = model(x_embedded=x_embedded, template_edge_index=batch_edge_index, LOS_batch=los_batch, device=device)
            preds = (logits > 0).float().view(-1)
            y_true = y_batch.view(-1)
            
            # 정답 여부 확인
            matches = (preds == y_true)
            
            # 인덱스 분류
            for i, is_correct in enumerate(matches):
                if is_correct:
                    current_idx = global_idx + i
                    true_label = y_true[i].item()
                    
                    if true_label == 0:
                        correct_indices_0.append(current_idx)
                    else:
                        correct_indices_1.append(current_idx)
            
            global_idx += curr_batch_size

    print(f"Total Correct -> Class 0: {len(correct_indices_0)}, Class 1: {len(correct_indices_1)}")

    # 2. Stratified Sampling (각 클래스당 절반씩)
    half_sample = num_samples // 2
    
    # 개수가 부족하면 있는 만큼만, 충분하면 랜덤 샘플링
    sample_0 = np.random.choice(correct_indices_0, min(len(correct_indices_0), half_sample), replace=False)
    sample_1 = np.random.choice(correct_indices_1, min(len(correct_indices_1), half_sample), replace=False)
    
    final_indices = np.concatenate([sample_0, sample_1])
    np.random.shuffle(final_indices) # 순서 섞기
    
    print(f"Final Dataset Size: {len(final_indices)} (Balanced & Correct)")
    
    return Subset(dataset, final_indices)


#####################################################################################
def filter_main():
    explainer_train_dataset = get_correct_stratified_subset(model, dataset, device, num_samples=30000)


    with open("correct_dataset.pickle", 'wb') as f:
        pickle.dump(explainer_train_dataset, f)