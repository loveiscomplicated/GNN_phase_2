
import os
import sys
cur_dir = os.path.dirname(__file__)
par_dir = os.path.join(cur_dir, '..')
sys.path.append(par_dir)
import pickle
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import ModelConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from teds_tensor_dataset import TEDSTensorDataset
from utils.processing_utils import train_test_split_customed_dataset, mi_edge_index_batched, train_test_split_customed
from models.gingru_for_explain import GinGruForExplain, GinGruForExplain2
from models.gin_gru import GinGru

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
early_stopping_patience = 10
device = torch.device("cpu")
optim_lr = 0.001

epochs = 100
explainer_lr = 0.003
edge_size = 0.0
edge_ent = 0.0
temp = [5.0, 0.5]

sample = False

save_dir = os.path.join(cur_dir, 'explainer_checkpoints', 'sample' if sample else 'real')
os.makedirs(save_dir, exist_ok=True)

if sample:
    root = os.path.join(par_dir, 'data_tensor_sampled')
    model_path = os.path.join(par_dir, 'checkpoints', 'gingru', 'sampled') #####
else:
    root = os.path.join(par_dir, 'data_tensor_cache')
    model_path = os.path.join(par_dir, 'checkpoints', 'gingru', 'real') #####

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
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.early_stopper import EarlyStopper
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=scheduler_patience)
early_stopper = EarlyStopper(patience=early_stopping_patience)

load_checkpoint(model, optimizer, scheduler, filename=checkpoint_path, device=device)
model.to(device)
model.eval()

model_config = ModelConfig(
    mode='binary_classification',
    task_level="graph",
    return_type="raw"
)

correct_dataset_path = os.path.join(cur_dir, "correct_dataset.pickle")
if os.path.exists(correct_dataset_path):
    with open(correct_dataset_path, 'rb') as f:
        explainer_train_dataset = pickle.load(f)
else:
    from get_correct_pred import filter_main
    filter_main()
    if os.path.exists(correct_dataset_path):
        with open(correct_dataset_path, 'rb') as f:
            explainer_train_dataset = pickle.load(f)
    else:
        raise Exception("error occured while filtering data(filter_main)")

class DebugPGExplainer(PGExplainer):
    def _add_mask_regularization(self, loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. Sigmoid 적용 (0~1 사이 값으로 변환)
        mask_sigmoid = mask.sigmoid()

        # 2. Size Regularization (평균 사용)
        # mean()을 쓰면 그래프 크기에 상관없이 계수(coeff)의 영향력이 일정해져서 튜닝하기 좋습니다.
        size_loss = mask_sigmoid.mean() * self.coeffs['edge_size']

        # 3. Entropy Regularization
        # log(0) 방지를 위한 클리핑
        masked = 0.99 * mask_sigmoid + 0.005
        mask_ent = -masked * masked.log() - (1 - masked) * (1 - masked).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']
        
        total_loss = loss + size_loss + mask_ent_loss

        if random.random() < 0.01:  # 빈도 조절
            mask_ratio = mask_sigmoid.mean().item() # 현재 마스크가 평균적으로 얼마나 켜져있는지
            print(f"--- Debug Step ---")
            print(f"Pred_Loss: {loss.item():.4f} | Size_Loss: {size_loss.item():.4f} | Ent_Loss: {mask_ent_loss.item():.4f}")
            print(f"Total_Loss: {total_loss.item():.4f}")
            print(f"Current Mask Ratio (Mean): {mask_ratio:.4f}") # 이 값이 0에 가까운지 1에 가까운지 보는 게 핵심
            print(f"Coeffs -> Size: {self.coeffs['edge_size']}, Ent: {self.coeffs['edge_ent']}")
            print("------------------")

        return loss + size_loss + mask_ent_loss

    
algorithm = DebugPGExplainer(
    epochs=epochs,
    lr=explainer_lr,
    edge_size=0.001,   # sum()일 땐 0.0001이었지만, mean()이므로 0.05 정도가 적당합니다.
    edge_ent=0.001,     # 마스크 값이 0 또는 1로 확실하게 갈리도록 강력하게 압박합니다.
    temp=temp
)

'''algorithm = PGExplainer(
    epochs=epochs,
    lr=explainer_lr,
    coeffs={
        'edge_size': edge_size,  
        'edge_ent': edge_ent,  
        'temp': temp,
    }
)'''

explainer = Explainer(
    model=model,
    algorithm=algorithm,
    # 원래 model의 예측을 기반으로 훈련하는 게 맞지만 이걸 model로 바꾸면 왜인지 모르게 오류가 남, 
    # 그래도 explanation_type과 무관하게, 학습 로직의 핵심은 훈련 시 target을 어떻게 정하는지에 달려 있음
    # 따라서 explanation_type이 무엇인지는 신경쓰지 않아도 됨
    explanation_type="phenomenon", 
    model_config=model_config,
    node_mask_type=None,
    edge_mask_type="object",
    threshold_config=None
)

num_nodes = len(ad_col_index)
train_dataset, val_dataset, test_dataset = train_test_split_customed_dataset(dataset=dataset)

dataset = val_dataset
train_loader, explainer_loader, test_loader = train_test_split_customed(dataset=dataset,
                                                                  batch_size=batch_size)


explainer_loader = DataLoader(dataset=explainer_train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              drop_last=True) 


edge_index = mi_edge_index_batched(batch_size=batch_size,
                                   num_nodes=num_nodes,
                                   mi_dict_path=mi_dict_path)
edge_index = edge_index.to(device)

def get_y_pred(x_batch, edge_index, los_batch, device):
    return model(x_embedded=x_batch,template_edge_index=edge_index, LOS_batch=los_batch, device=device)

def train_pgexplainer():
    best_model_path = ""
    print(f"Start Training PGExplainer on {len(explainer_loader.dataset)} samples...")
    for epoch in range(1, epochs):
        total_loss = 0
        for batch in tqdm(explainer_loader):
        #for batch in explainer_loader:
            x_batch, y_batch, los_batch = batch

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            los_batch = los_batch.to(device)

            with torch.no_grad():
                x_embedded = model.entity_embedding_layer(x_batch)

            original_logit = get_y_pred(x_embedded, edge_index, los_batch, device)
            # BCEWithLogitsLoss를 사용하여 학습했다면, 추론(Inference) 단계에서 굳이 sigmoid 함수를 계산하지 않고 
            # 로짓(Logit) 값의 부호(+, -)만 봐도 클래스를 완벽하게 판별할 수 있다.
            target_prediction = (original_logit > 0).float()

            
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=x_embedded,
                edge_index=edge_index,
                target=target_prediction,
                LOS_batch=los_batch,
                device=device
            )
            total_loss += loss
        avg_loss = round(total_loss / len(explainer_loader), 4)
        print(f"Epoch: {epoch} - loss: {avg_loss: .4f}")

        if sample:
            best_model_path = os.path.join(cur_dir, 'explainer_checkpoints', 'sample', f'PGExplainer_GINGRU_{avg_loss: .4f}.pth')
            
        else:
            best_model_path = os.path.join(cur_dir, 'explainer_checkpoints', 'real', f'PGExplainer_GINGRU_{avg_loss: .4f}.pth')

        torch.save(explainer.algorithm.state_dict(), best_model_path)
        print(f"  Loss updated to {avg_loss:.4f}. Model saved.")

def load_pgexplainer(best_model_path):
    print("Loading the best model state...")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        explainer.algorithm.load_state_dict(torch.load(best_model_path))
        explainer.algorithm.epochs = 0 # 이렇게 해야 로드해도 오류가 안 남. epoch를 
        return explainer
    else:
        print("Warning: No model saved properly.")
        return

def get_mean_edge_mask(best_model_path, loader, num_edges_single, edge_mask_mean_save_path):
    explainer = load_pgexplainer(best_model_path)
    edge_index = mi_edge_index_batched(batch_size=batch_size, num_nodes=num_nodes, mi_dict_path=mi_dict_path)
    model.eval() 
    total_graphs_count = 0
    
    edge_mask_sum = torch.zeros(num_edges_single, device=device)
    
    print("Calculating Mean Edge Mask...")
    
    for batch in tqdm(loader):
        x_batch, y_batch, los_batch = batch

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        los_batch = los_batch.to(device)
        
        # [안전장치] 로더에서 처리했더라도 혹시 모르니 체크 (비용 없음)
        if x_batch.size(0) != batch_size:
            continue

        with torch.no_grad():
            x_embedded = model.entity_embedding_layer(x_batch)

            explanation = explainer.algorithm.forward(
                model=model,
                x=x_embedded,
                edge_index=edge_index,
                target=y_batch,
                LOS_batch=los_batch,
                device=device
            )
        
        # 1. Reshape: [Batch_Size, Num_Edges]
        mask_reshaped = explanation.edge_mask.view(batch_size, num_edges_single)
        
        # 2. Sum over batch dimension
        batch_sum = mask_reshaped.sum(dim=0)

        # 3. Accumulate (둘 다 같은 device여야 함)
        edge_mask_sum += batch_sum
        total_graphs_count += batch_size
    
    # 평균 계산
    if total_graphs_count > 0:
        edge_mask_mean = edge_mask_sum / total_graphs_count
    else:
        print("Error: No graphs were processed.")
        return None

    # CPU로 내려서 저장
    edge_mask_mean_cpu = edge_mask_mean.cpu()
    
    with open(edge_mask_mean_save_path, 'wb') as f:
        pickle.dump(edge_mask_mean_cpu, f)
        
    print(f"Saved mean mask to {edge_mask_mean_save_path}")
    
    return edge_mask_mean_cpu

def train_all():
    # 전체 데이터에 대해 학습하기
    dataset = TEDSTensorDataset(root)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last=True)
        

    best_model_path = os.path.join(save_dir, 'PGExplainer_GINGRU_ 0.7275.pth')
    edge_mask_mean_save_path = os.path.join(save_dir, 'PGExplainer_GINGRU_edge_mask_mean_all.pickle')

    edge_index_single = mi_edge_index_batched(batch_size=1,
                                            num_nodes=num_nodes,
                                            mi_dict_path=mi_dict_path)
    num_edges_single = edge_index_single.shape[1]


    edge_mask_mean = get_mean_edge_mask(best_model_path=best_model_path, 
                                        loader=dataloader,
                                        num_edges_single=num_edges_single,
                                        edge_mask_mean_save_path = edge_mask_mean_save_path)

    with open(edge_mask_mean_save_path, 'rb') as f:
        edge_mask_mean = pickle.load(f)
    print(edge_mask_mean)

    return edge_mask_mean

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def get_diff_mask_pooled_auc(
    model,
    train_loader,
    device,
    save_dir,
    mi_dict_path,
    batch_size: int,
    edge_threshold: int = 10,
    is_topk: bool = True,
    max_batches: int = 100,
    model_output_is_logit: bool = True,  # True면 sigmoid 적용, False면 그대로 확률로 사용
):
    model.eval()

    # 1) PGExplainer 로드
    best_model_path = os.path.join(save_dir, 'PGExplainer_GINGRU_ 0.6189.pth')
    explainer = load_pgexplainer(best_model_path)

    # 2) batched edge_index 준비
    edge_index = mi_edge_index_batched(32, 60, mi_dict_path).to(device)

    # 누적용
    diff_sum = 0.0
    n_samples = 0

    y_true_all = []
    y_score_orig_all = []
    y_score_mask_all = []

    counter = 0
    for batch in train_loader:
        if counter >= max_batches:
            break

        x_batch, y_batch, los_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).view(-1)   # (B,)
        los_batch = los_batch.to(device)

        # 임베딩
        x_embedded = model.entity_embedding_layer(x_batch)

        # 설명(마스크) 계산
        explanation = explainer.algorithm.forward(
            model=model,
            x=x_embedded,
            edge_index=edge_index,
            target=y_batch,
            LOS_batch=los_batch,
            device=device
        )

        mask = explanation.edge_mask
        print(f"Mask Mean: {mask.mean().item(): .4f}")

        # --- masked_edge_index 생성 ---
        # IMPORTANT: 이 방식은 "배치 내 각 그래프가 동일한 edge template"이라는 가정이 있어야 안전함
        num_edges_single = edge_index.shape[1] // (batch_size * 2)
        mask_reshaped = mask.view(batch_size * 2, num_edges_single)

        masked_edge_index = None
        for i in range(mask_reshaped.shape[0]):
            local_idx = mask_reshaped[i].topk(edge_threshold, largest=is_topk).indices
            offset = i * num_edges_single
            global_idx = offset + local_idx
            masked_edge_index_single = edge_index[:, global_idx]

            if masked_edge_index is None:
                masked_edge_index = masked_edge_index_single
            else:
                masked_edge_index = torch.cat((masked_edge_index, masked_edge_index_single), dim=1)

        # --- 모델 출력 ---
        original_out = model(x_embedded, edge_index, LOS_batch=los_batch, device=device).view(-1)
        masked_out = model(x_embedded, masked_edge_index, LOS_batch=los_batch, device=device).view(-1)

        # 차이(샘플 단위 평균 diff)
        diff = torch.abs(original_out - masked_out)  # (B,)
        diff_sum += diff.sum().item()
        n_samples += diff.numel()

        print(f"Original out mean: {original_out.mean().item(): .4f} | Masked out mean: {masked_out.mean().item(): .4f} "
              f"| Mean |Δ|: {diff.mean().item(): .4f}")

        # pooled AUC용 score 저장
        if model_output_is_logit:
            orig_score = original_out.sigmoid()
            mask_score = masked_out.sigmoid()
        else:
            # 이미 확률이면 그대로 사용
            orig_score = original_out
            mask_score = masked_out

        y_true_all.append(y_batch.detach().cpu())
        y_score_orig_all.append(orig_score.detach().cpu())
        y_score_mask_all.append(mask_score.detach().cpu())

        counter += 1

    # --- pooled AUC 계산 ---
    y_true_all = torch.cat(y_true_all).numpy().reshape(-1)
    y_score_orig_all = torch.cat(y_score_orig_all).numpy().reshape(-1)
    y_score_mask_all = torch.cat(y_score_mask_all).numpy().reshape(-1)

    def safe_auc(y_true, y_score):
        try:
            return roc_auc_score(y_true, y_score)
        except ValueError:
            return np.nan  # 한 클래스만 있는 경우 등

    orig_auc = safe_auc(y_true_all, y_score_orig_all)
    mask_auc = safe_auc(y_true_all, y_score_mask_all)

    auc_diff = np.abs(orig_auc - mask_auc) if (np.isfinite(orig_auc) and np.isfinite(mask_auc)) else np.nan

    diff_mean = diff_sum / max(1, n_samples)

    print(f"\n=== Summary (pooled over {counter} batches, {n_samples} samples) ===")
    print(f"Mean |Δlogit(or prob)| per sample: {diff_mean: .6f}")
    print(f"Pooled Original AUC: {orig_auc: .6f}")
    print(f"Pooled Masked   AUC: {mask_auc: .6f}")
    print(f"Pooled AUC Diff     : {auc_diff: .6f}")

    return diff_mean, orig_auc, mask_auc, auc_diff

import torch
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def pooled_auc(y_true_all, logits_all):
    y_true = y_true_all.detach().cpu().numpy().reshape(-1)
    y_prob = torch.sigmoid(logits_all).detach().cpu().numpy().reshape(-1)
    return roc_auc_score(y_true, y_prob)

@torch.no_grad()
def eval_soft_mask_auc(
    model,
    loader,
    edge_index_sup,     # "B*2 graphs용" supersized edge_index여야 함
    device,
    batch_size,
    num_nodes,
    num_batches=100,
    mode="mask",        # "mask" or "remove"
):
    model.eval()
    best_model_path = os.path.join(save_dir, 'PGExplainer_GINGRU_ 0.6189.pth')
    explainer = load_pgexplainer(best_model_path)

    edge_index_sup = edge_index_sup.to(device)

    ys = []
    orig_logits = []
    masked_logits = []
    diffs = []

    counter = 0
    for batch in loader:
        if counter >= num_batches:
            break

        x_batch, y_batch, los_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        los_batch = los_batch.to(device)

        x_embedded = model.entity_embedding_layer(x_batch)

        # PGExplainer가 반환하는 edge_mask 길이는 edge_index_sup의 E와 같아야 함
        explanation = explainer.algorithm.forward(
            model=model,
            x=x_embedded,
            edge_index=edge_index_sup,
            target=y_batch,
            LOS_batch=los_batch,
            device=device
        )

        edge_mask = explanation.edge_mask  # shape [E_total]

        # remove 모드면 중요 edge를 약화시키는 방향
        if mode == "remove":
            edge_weight = 1.0 - edge_mask
        else:
            edge_weight = edge_mask

        # sanity: 길이 체크
        assert edge_weight.numel() == edge_index_sup.size(1), \
            f"edge_weight({edge_weight.numel()}) != num_edges({edge_index_sup.size(1)})"

        # 원본/소프트마스크 logits
        out_orig = model(x_embedded, edge_index_sup, LOS_batch=los_batch, device=device)
        out_mask = model(x_embedded, edge_index_sup, LOS_batch=los_batch, device=device, edge_weight=edge_weight)

        ys.append(y_batch.detach())
        orig_logits.append(out_orig.detach())
        masked_logits.append(out_mask.detach())
        diffs.append((out_orig - out_mask).abs().detach().cpu())

        counter += 1

    ys = torch.cat(ys, dim=0)
    orig_logits = torch.cat(orig_logits, dim=0)
    masked_logits = torch.cat(masked_logits, dim=0)

    mean_abs_logit_diff = torch.cat(diffs, dim=0).mean().item()
    auc_orig = pooled_auc(ys, orig_logits)
    auc_mask = pooled_auc(ys, masked_logits)
    auc_diff = abs(auc_orig - auc_mask)

    print(f"=== Soft-mask Summary (pooled over {counter} batches, {ys.shape[0]} samples) ===")
    print(f"Mode: {mode}")
    print(f"Mean |Δlogit| per sample:  {mean_abs_logit_diff:.6f}")
    print(f"Pooled Original AUC:  {auc_orig:.6f}")
    print(f"Pooled Masked   AUC:  {auc_mask:.6f}")
    print(f"Pooled AUC Diff     :  {auc_diff:.6f}")

    return {
        "mean_abs_logit_diff": mean_abs_logit_diff,
        "auc_orig": auc_orig,
        "auc_masked": auc_mask,
        "auc_diff": auc_diff,
        "n_samples": int(ys.shape[0]),
        "n_batches": counter,
    }



# train_pgexplainer()
'''get_diff_mask_pooled_auc(model=model,
                         train_loader=test_loader,
                         device=device,
                         save_dir=save_dir,
                         mi_dict_path=mi_dict_path,
                         batch_size=batch_size,
                         edge_threshold=10,
                         is_topk=False,
                         max_batches=100,
                         model_output_is_logit=True)
'''
eval_soft_mask_auc(model=model,
                   loader=test_loader,
                   edge_index_sup=edge_index,
                   device=device,
                   batch_size=batch_size,
                   num_nodes=num_nodes,
                   num_batches=100,
                   mode='mask')

eval_soft_mask_auc(model=model,
                   loader=test_loader,
                   edge_index_sup=edge_index,
                   device=device,
                   batch_size=batch_size,
                   num_nodes=num_nodes,
                   num_batches=100,
                   mode='remove')


