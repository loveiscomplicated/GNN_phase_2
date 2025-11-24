import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.a3tgcn import A3TGCNCat2
from utils.device_set import device_set
from utils.metrics import compute_metrics
from utils.write_log import enable_dual_output
from utils.early_stopper import EarlyStopper
# train_eval_a3tgcn.py 파일의 상단 import 부분 아래에 추가

def create_dataloader(dataset, batch_size):
    """
    데이터셋을 지정된 배치 크기로 나누어 리스트로 반환합니다.
    """
    dataloader = []
    for i in range(0, len(dataset), batch_size):
        dataloader.append(dataset[i:i + batch_size])
    return dataloader

# BATCH_SIZE 정의
BATCH_SIZE = 32 # 🚨 이 값은 GPU 메모리에 따라 조정되어야 합니다.

CUR_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(CUR_DIR, 'data', 'Sampled_temporal_graph_data_fully_connected.pickle')
enable_dual_output('a3tgcn_1110.txt')

def save_checkpoint(epoch, model, optimizer, lr_scheduler, best_val_loss, save_dir, filename="best_model.pth"):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to: {filepath}")


def load_checkpoint(model, optimizer, lr_scheduler, filepath):
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return None, None, None

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"Checkpoint loaded from {filepath}")
    return start_epoch, best_val_loss


def eval_A3TGCNCat2(model, device, template_edge_index, test_dataset, criterion):
    
    model.eval()

    total_loss = 0.0
    total_eval_samples = 0
    y_true_list = []
    y_pred_list = []
    y_scores_list = []
    
    total_samples = len(test_dataset)
    
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for signal in tqdm(test_dataset, desc="Evaluating"):
            
            batch_list = []
            for batch in signal:
                if hasattr(batch, 'to'):
                    batch_list.append(batch.to(device))
                else:
                    batch_list.append(batch)
            
            # 🚨 [수정 1] .ptr 대신 .batch 속성 사용
            labels = batch_list[0].y
            logits = model.forward(batch_list, template_edge_index)
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * batch_list[0].batch_size 

            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1]

            y_true_list.extend(labels.cpu().tolist())
            y_pred_list.extend(logits.argmax(dim=1).cpu().tolist())
            y_scores_list.extend(score_class_1.cpu().tolist())

            current_batch_size = batch_list[0].batch_size
            total_loss += loss.item() * current_batch_size
            total_eval_samples += current_batch_size # ⬅️ 실제 배치 크기 누적
            
    # 평균 손실 계산 수정
    # 🚨 참고: test_dataset[0][0]이 존재하지 않을 수 있으므로, 
    #    평균 배치 크기를 가정하거나, eval 루프 내에서 총 샘플 수를 세는 것이 더 안전합니다.
    #    여기서는 간단히 첫 번째 배치의 크기를 사용합니다.
    avg_loss = total_loss / total_eval_samples
    
    metrics = compute_metrics(
        y_true_list, 
        y_pred_list, 
        y_scores_list,
        num_classes=2
    )
    return avg_loss, *metrics

def train_A3TGCNCat2(model, device, template_edge_index, train_dataset, val_dataset, criterion, optimizer, lr_scheduler, MODEL_SAVE_PATH, num_epochs=10):
    
    model.train()

    softmax = nn.Softmax(dim=1)
    early_stopper = EarlyStopper(patience=15, min_delta=0.001)

    print(f"Starting training on {len(train_dataset)} batches...")
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss_sum = 0.0
        total_train_samples = 0

        y_true_list, y_pred_list, y_scores_list = [], [], []

        # 루프 원상 복구: train_dataset을 직접 순회합니다.
        for signal in tqdm(train_dataset, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            
            # 'signal'은 이미 T_max개의 PyG Batch 객체 리스트 (배치 상태)
            batch_list = []
            for batch in signal:
                if hasattr(batch, 'to'):
                    batch_list.append(batch.to(device))
                else:
                    batch_list.append(batch)
            
            # 그래프의 시작 노드 인덱스를 사용하여 y 텐서에서 레이블 추출
            labels = batch_list[0].y # (B,)

            logits = model.forward(batch_list, template_edge_index)   
            loss = criterion(logits, labels) # (B, 2) vs (B,)
            
            loss.backward()
            optimizer.step()

            # train_loss_sum 누적 방식 수정
            train_loss_sum += loss.item() * batch_list[0].batch_size

            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1]
            predicted_class = logits.argmax(dim=1)

            y_true_list.extend(labels.cpu().tolist())
            y_pred_list.extend(predicted_class.cpu().tolist())
            y_scores_list.extend(score_class_1.cpu().tolist())

            current_batch_size = batch_list[0].batch_size
            train_loss_sum += loss.item() * current_batch_size
            total_train_samples += current_batch_size # ⬅️ 실제 배치 크기 누적

        # 평균 손실 계산 수정
        avg_train_loss = train_loss_sum / total_train_samples

        train_metrics = compute_metrics(y_true_list, y_pred_list, y_scores_list, num_classes=2)
        train_acc, train_prec, train_rec, train_f1, train_auc = train_metrics

        # 검증 호출 (eval 함수도 수정되었으므로 BATCH_SIZE 인자 불필요)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = eval_A3TGCNCat2(
            model, device, template_edge_index, val_dataset, criterion
        )

        # best_validation_loss는 EarlyStopper 객체에서
        if val_loss < early_stopper.best_validation_loss - early_stopper.min_delta:
            print(f"🎉 New best validation loss: {val_loss:.4f}. Saving model...")
            
            # best_val_loss 변수 업데이트 (EarlyStopper 내부 값과 동일해짐)
            best_val_loss = val_loss 
            
            # 모델 저장
            filename = f"best_model_epoch_{epoch+1}_loss_{best_val_loss:.4f}.pth"
            
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                best_val_loss=best_val_loss,
                save_dir=MODEL_SAVE_PATH, # 폴더 경로 사용
                filename=filename
            )

        # 🚨 [수정] 2. 검증 및 저장 로직이 끝난 후,
        #    EarlyStopper의 상태를 업데이트합니다.
        should_stop = early_stopper(val_loss)
        
        # 학습률 로깅
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        print(f"  [Train] LR: {current_lr:.6f} | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"  [Valid] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        if should_stop:
            print("\n--- Early Stopping activated. Learning terminated. ---")
            break

    print("\n--- 학습 완료 ---")


if __name__ == "__main__":
    # gpu 세팅
    DEVICE = device_set()

    # 데이터 로드
    CURDIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(CURDIR, 'data', 'temporal_graph_data_mi_sampled.pickle')
    with open(DATA_PATH, 'rb') as f:
        pickle_dataset = pickle.load(f)
        print("pickle_dataset loaded!!")
        
    train_dataset, val_dataset, test_dataset, col_info = pickle_dataset
    col_list, col_dim = col_info
    
    # Edge Index 준비 및 DEVICE로 이동
    template_edge_index = torch.as_tensor(train_dataset[0][0].edge_index, dtype=torch.long)
    template_edge_index = template_edge_index.to(DEVICE)

    # model 준비
    model = A3TGCNCat2(col_dims=col_dim, 
                       col_list=col_list, 
                       num_layers=2, 
                       hidden_channel=64, 
                       out_channel=2)
    model.to(DEVICE)

    # 🚨 MODEL_SAVE_PATH를 폴더 경로로 변경하고 생성합니다.
    MODEL_SAVE_PATH = os.path.join(CUR_DIR, 'a3tgcn_sample_checkpoints') 
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True) # 저장 폴더 생성

    # 손실 함수 정의
    criterion = nn.CrossEntropyLoss()

    # 최적화 알고리즘 정의
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=3)
    
    # train 함수 실행
    train_A3TGCNCat2(model, DEVICE, template_edge_index, train_dataset, val_dataset, criterion, optimizer, lr_scheduler, MODEL_SAVE_PATH, num_epochs=100)