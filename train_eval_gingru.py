import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from utils.write_log import enable_dual_output
from utils.early_stopper import EarlyStopper
from utils.processing_utils import mi_edge_index_batched, train_test_split_customed
from utils.device_set import device_set
from teds_tensor_dataset import TEDSTensorDataset
from models.gin_gru import GinGru

cur_dir = os.path.dirname(__file__)
enable_dual_output(f'gingru_1124.txt')

def train(model, dataloader, criterion, optimizer, edge_index, device):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch, los_batch in tqdm(dataloader, desc="train_process", leave=True):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        los_batch = los_batch.to(device)

        optimizer.zero_grad()

        logits = model(
            x_batch,
            los_batch,
            edge_index,
            device
        )

        logits = logits.squeeze(1)
        loss = criterion(logits, y_batch.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, val_dataloader, criterion, decision_threshold, device, edge_index):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_targets = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for x_batch, y_batch, los_batch in tqdm(val_dataloader, desc="eval_process", leave=True):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            los_batch = los_batch.to(device)

            logits = model(
                x_batch,
                los_batch,
                edge_index,
                device
            )

            logits = logits.squeeze(1)
            loss = criterion(logits, y_batch.float())

            with torch.no_grad():
                scores = torch.sigmoid(logits)            # [B]
                predicted = (scores >= decision_threshold).long()

            running_loss += loss.item() * x_batch.size(0)

            all_targets.append(y_batch.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_scores.append(scores.cpu().numpy()) # AUC 계산을 위해 확률(Scores) 저장
            
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_scores = np.concatenate(all_scores)

    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_accuracy = total_correct / total_samples

    try:
        # 이진 분류이므로 all_scores는 (N, 1)이 아닌 (N,) 형태여야 합니다.
        # squeeze(1)을 통해 이를 보장했습니다.
        epoch_auc = roc_auc_score(all_targets, all_scores) 
    except ValueError:
        print("Warning: AUC score could not be calculated.")
        epoch_auc = 0.0

    epoch_precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    epoch_recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    print("Valid preds label counts:", torch.bincount(preds))
    print("Valid true label counts:", torch.bincount(y_batch))


    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_auc


def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, filename, map_location=None):
    """
    저장된 체크포인트(.pth)를 불러와서 
    model, optimizer, scheduler 상태를 복구합니다.

    Parameters:
        model (nn.Module): 모델 객체
        optimizer (torch.optim.Optimizer): 옵티마이저 객체
        scheduler: 스케줄러 객체
        filename (str): 저장된 체크포인트 경로
        map_location: CPU로 로드하고 싶으면 'cpu' 또는 torch.device('cpu')

    Returns:
        start_epoch (int): 다음 훈련을 시작할 epoch 번호
        best_loss (float): 저장된 최소 validation loss
    """
    checkpoint = torch.load(filename, map_location=map_location)

    # --- Load states ---
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']

    return start_epoch, best_loss


if __name__ == "__main__":
    # device = device_set()
    device = torch.device('cpu')
    BATCH_SIZE = 32
    embedding_dim = 64
    gin_hidden_channel = 64
    train_eps = True
    gin_layers = 2
    gru_hidden_channel = 128
    decision_threshold = 0.5

    EPOCH = 100
    scheduler_patience = 10
    early_stopping_patience = 15
    learning_rate = 0.001

    sample = False

    if sample:
        root = os.path.join(cur_dir, 'data_tensor_sampled')
        model_path = os.path.join(cur_dir, 'checkpoints', 'gingru', 'sampled') #####
    else:
        root = os.path.join(cur_dir, 'data_tensor_cache')
        model_path = os.path.join(cur_dir, 'checkpoints', 'gingru', 'real') #####

    mi_dict_path = os.path.join(root, 'data', 'mi_dict_static.pickle')

    dataset = TEDSTensorDataset(root)

    col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info

    num_nodes = len(ad_col_index)
    
    edge_index = mi_edge_index_batched(batch_size=BATCH_SIZE,
                                            mi_dict_path=mi_dict_path,
                                            num_nodes=num_nodes,
                                            top_k=6,
                                            return_edge_attr=False)
    
    edge_index = edge_index.to(device) # type: ignore
    
    model = GinGru(batch_size=BATCH_SIZE,
                   col_dims=col_dims,
                   col_list=col_list,
                   ad_col_index=ad_col_index, 
                   dis_col_index=dis_col_index,
                   embedding_dim=embedding_dim,
                   gin_hidden_channel=gin_hidden_channel,
                   train_eps=train_eps,
                   gin_layers=gin_layers,
                   gru_hidden_channel=gru_hidden_channel)
    model = model.to(device=device)
    '''
    print(model)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")
    '''
    train_dataloader, val_dataloader, test_dataloader = train_test_split_customed(dataset=dataset,
                                                 batch_size=BATCH_SIZE)
    '''
    all_labels = []
    print("데이터로더를 순회하며 레이블 추출 중...")
    # train_dataloader를 사용합니다. 데이터셋 튜플의 두 번째 요소(y)가 라벨입니다.
    for x_batch, y_batch, los_batch in train_dataloader:
        # 텐서를 CPU로 이동하고 NumPy 배열로 변환
        all_labels.append(y_batch.cpu().numpy())
        
    # 추출된 모든 레이블 배열을 하나로 합칩니다.
    all_labels = np.concatenate(all_labels)

    # 2. 클래스별 개수 계산
    n_neg = np.sum(all_labels == 0) # 음성 샘플 수 (레이블 0)
    n_pos = np.sum(all_labels == 1) # 양성 샘플 수 (레이블 1)
    n_total = len(all_labels)

    # 3. pos_weight 계산
    if n_pos > 0:
        pos_weight_value = n_neg / n_pos
    else:
        pos_weight_value = 1.0 # 양성 샘플이 없는 경우 1.0으로 설정

    # 4. 결과 출력 및 pos_weight 텐서 생성 (바로 사용 가능)
    print("\n--- 클래스 비율 분석 결과 ---")
    print(f"전체 학습 샘플 수: {n_total:,}")
    print(f"음성 샘플 (0): {n_neg:,} ({n_neg/n_total*100:.2f}%)")
    print(f"양성 샘플 (1): {n_pos:,} ({n_pos/n_total*100:.2f}%)")
    print(f"계산된 pos_weight: {pos_weight_value:.4f}")
    '''
    # pos_weight_value = 1.2504 # 음성 / 양성 
    # pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=scheduler_patience)
    early_stopper = EarlyStopper(patience=early_stopping_patience)

    with torch.no_grad():
        batch = next(iter(val_dataloader))
        x_batch, los_batch, edge_index, y = batch[0], batch[2], edge_index, batch[1]

        logits = model(x_batch, los_batch, edge_index, device)
        probs = torch.sigmoid(logits)

        print("logits mean/std:", logits.mean().item(), logits.std().item())
        print("probs min/max/mean:", probs.min().item(), probs.max().item(), probs.mean().item())
        print("y label mean:", y.float().mean().item())

        pos_mask = (y == 1)
        neg_mask = (y == 0)

        print("pos logits mean:", logits[pos_mask].mean().item())
        print("neg logits mean:", logits[neg_mask].mean().item())

    for epoch in tqdm(range(EPOCH)):
        train_loss = train(model, train_dataloader, criterion, optimizer, edge_index, device)

        result = evaluate(model, val_dataloader, criterion, decision_threshold, device, edge_index)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = result

        scheduler.step(val_loss)

        if val_loss < early_stopper.best_validation_loss:
            print(f"🎉 New best validation loss: {val_loss:.4f}. Saving model...")
            
            best_val_loss = val_loss
            
            file_name = f"best_gingru_epoch_{epoch+1}_loss_{best_val_loss:.4f}.pth"
            full_save_path = os.path.join(model_path, file_name)
            save_checkpoint(epoch + 1, 
                            model, 
                            optimizer, 
                            scheduler, 
                            best_val_loss, 
                            full_save_path)

        should_stop = early_stopper(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch+1}/{EPOCH}]")
        print(f"  [Train] LR: {current_lr:.6f} | Loss: {train_loss:.4f}")
        print(f"  [Valid] Loss: {val_loss:.4f} | Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        if should_stop:
            print("\n--- Early Stopping activated. Learning terminated. ---")
            break

    print("\n--- 학습 완료 ---")
    
    with torch.no_grad():
        result = evaluate(model, test_dataloader, criterion, decision_threshold, device, edge_index)
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = result
    
        print(f"\n[Test] Loss: {test_loss:.4f} | Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
