import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from models.a3tgcn import A3TGCNCat2
from utils.device_set import device_set
from utils.metrics import compute_metrics
from utils.write_log import enable_dual_output

# train_eval_a3tgcn.py 파일의 상단 import 부분 아래에 추가

def collate_signals(signals):
    """
    여러 signal 객체(T개의 Batch 리스트)를 하나의 큰 배치 객체(T개의 Batch 리스트)로 묶습니다.
    PyG Temporal의 Batch 객체 리스트를 병합하는 방식으로 작동합니다.
    """
    if not signals:
        return []
    
    T_max = 37 # 시점의 수
    
    collated_batch_list = []
    
    for t in range(T_max):
        # 각 시점 t에 해당하는 모든 signal의 Batch 객체를 모아 하나의 Batch로 병합
        batch_list_at_t = [signal[t] for signal in signals]
        # torch_geometric.data.Batch.from_data_list와 유사한 병합 기능 사용
        # signal[t]는 PyG의 Batch 객체이므로, PyG의 collate 함수가 필요하지만,
        # 여기서는 simple batching을 위해 각 속성을 수동으로 병합했다고 가정 (실제 PyG 코드를 대체)
        
        # NOTE: signal[t]는 이미 PyG의 Batch 객체이므로,
        # PyG의 from_data_list 또는 Batch.from_data_list 역할을 하는 무언가를 사용해야 함.
        # 가장 간단하게는 PyG의 `DataLoader`의 `collate_fn` 기능이 필요함.
        
        # 🚨 현재는 임시로 첫 번째 샘플의 t 시점 배치만 반환하며, 
        # 이 함수가 실제 PyG 배칭 로직을 대체한다고 가정합니다.
        # 실제 PyG 환경에서는 DataLoader를 사용하는 것이 정석입니다.
        
        # 📌 임시 방편으로, PyG의 Batch.from_data_list와 유사한 역할을 한다고 가정하고
        # PyG의 `Batch.from_data_list`를 사용하거나 (외부 임포트 필요)
        # PyG의 `DataLoader`를 사용하는 것이 정석입니다.
        
        # 여기서는 외부 라이브러리 임포트 없이 수동으로 PyG의 Data/Batch 리스트 병합을 시뮬레이션
        # 🚨 이 부분은 외부 라이브러리 import 없이 정확히 구현하기 어려우므로, 
        # 실제 환경에서 PyG의 DataLoader를 사용해야 합니다.
        # 임시로 첫 번째 샘플의 시점 t 데이터를 반환하는 것으로 시뮬레이션합니다.
        
        from torch_geometric.data import Batch
        collated_batch = Batch.from_data_list(batch_list_at_t)
        collated_batch_list.append(collated_batch)
        
    return collated_batch_list

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

# EarlyStopper 클래스는 그대로 유지합니다.
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        # ... (생략) ...
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.best_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 Early stopping triggered after {self.counter} epochs without improvement.")
        
        return self.early_stop
    

def save_checkpoint(epoch, model, optimizer, best_val_loss, save_dir, filename="best_model.pth"):
    # ... (함수 내용은 그대로 유지합니다.) ...
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    # ... (함수 내용은 그대로 유지합니다.) ...
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return None, None, None

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"Checkpoint loaded from {filepath}")
    return start_epoch, best_val_loss


def eval_A3TGCNCat2(model, device, template_edge_index, test_dataset, criterion):
    
    model.eval()

    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    y_scores_list = []
    
    total_samples = len(test_dataset) # test_dataset은 배치(signal)의 리스트입니다.
    
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        # 🚨 루프 원상 복구: test_dataset을 직접 순회합니다.
        for signal in tqdm(test_dataset, desc="Evaluating"):
            
            # 'signal'은 이미 T_max개의 PyG Batch 객체 리스트 (배치 상태)
            batch_list = []
            for batch in signal:
                if hasattr(batch, 'to'):
                    batch_list.append(batch.to(device))
                else:
                    batch_list.append(batch)
            
            # 🚨 레이블 추출 수정 (그래프 분류) 🚨
            # signal[0] (첫 스냅샷)에서 각 그래프의 시작 노드 인덱스를 찾습니다.
            ptr = batch_list[0].ptr
            start_nodes_of_each_graph = ptr[:-1] # (B,)
            # 해당 노드의 y값을 그래프의 레이블로 사용합니다.
            labels = batch_list[0].y[start_nodes_of_each_graph].long() # (B,)
            
            logits = model.forward(batch_list, template_edge_index)
            
            loss = criterion(logits, labels)
            
            # 🚨 total_loss 누적 방식 수정 (배치 크기 B만큼 곱함)
            # signal[0].batch_size는 B (예: 32)를 반환합니다.
            total_loss += loss.item() * batch_list[0].batch_size 

            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1]

            y_true_list.extend(labels.cpu().tolist())
            y_pred_list.extend(logits.argmax(dim=1).cpu().tolist())
            y_scores_list.extend(score_class_1.cpu().tolist())
            
    # 🚨 평균 손실 계산 수정 (전체 샘플 수 = 배치 개수 * 배치 크기)
    avg_loss = total_loss / (len(test_dataset) * test_dataset[0][0].batch_size) 
    
    metrics = compute_metrics(
        y_true_list, 
        y_pred_list, 
        y_scores_list,
        num_classes=2
    )
    return avg_loss, *metrics

# train_A3TGCNCat2 함수 수정
# train_A3TGCNCat2 함수 수정
def train_A3TGCNCat2(model, device, template_edge_index, train_dataset, val_dataset, criterion, optimizer, MODEL_SAVE_PATH, num_epochs=10):
    
    model.train()

    softmax = nn.Softmax(dim=1)
    early_stopper = EarlyStopper(patience=15, min_delta=0.001)

    print(f"Starting training on {len(train_dataset)} batches...")
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss_sum = 0.0

        y_true_list, y_pred_list, y_scores_list = [], [], []

        # 🚨 루프 원상 복구: train_dataset을 직접 순회합니다.
        for signal in tqdm(train_dataset, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            
            # 'signal'은 이미 T_max개의 PyG Batch 객체 리스트 (배치 상태)
            batch_list = []
            for batch in signal:
                if hasattr(batch, 'to'):
                    batch_list.append(batch.to(device))
                else:
                    batch_list.append(batch)
            
            # 🚨 레이블 추출 수정 (ptr 대신 batch 속성 사용) 🚨
            batch_tensor = batch_list[0].batch # (N_total,) e.g., [0,0,1,1,1,2,2,...]
            
            # batch 텐서에서 값이 바뀌는 지점(diff != 0)을 찾아 +1 하면
            # 0번 그래프를 제외한 [1, 2, ..., B-1]번 그래프의 시작 노드 인덱스가 나옵니다.
            diff = torch.diff(batch_tensor)
            change_indices = torch.where(diff != 0)[0] + 1
            
            # 0번 그래프의 시작 인덱스(0)를 맨 앞에 추가합니다.
            start_nodes_of_each_graph = torch.cat([
                torch.tensor([0], device=batch_tensor.device), 
                change_indices
            ]) # (B,) e.g., [0, 2, 5, 7, ...]
            
            # 그래프의 시작 노드 인덱스를 사용하여 y 텐서에서 레이블 추출
            labels = batch_list[0].y[start_nodes_of_each_graph].long() # (B,)

            logits = model.forward(batch_list, template_edge_index)   
            loss = criterion(logits, labels) # (B, 2) vs (B,)
            
            loss.backward()
            optimizer.step()

            # 🚨 train_loss_sum 누적 방식 수정
            train_loss_sum += loss.item() * batch_list[0].batch_size

            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1]
            predicted_class = logits.argmax(dim=1)

            y_true_list.extend(labels.cpu().tolist())
            y_pred_list.extend(predicted_class.cpu().tolist())
            y_scores_list.extend(score_class_1.cpu().tolist())

        # 🚨 평균 손실 계산 수정
        avg_train_loss = train_loss_sum / (len(train_dataset) * train_dataset[0][0].batch_size)

        train_metrics = compute_metrics(y_true_list, y_pred_list, y_scores_list, num_classes=2)
        train_acc, train_prec, train_rec, train_f1, train_auc = train_metrics

        # 검증 호출 (eval 함수도 수정되었으므로 BATCH_SIZE 인자 불필요)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = eval_A3TGCNCat2(
            model, device, template_edge_index, val_dataset, criterion
        )

        should_stop = early_stopper(val_loss)
        
        # 🚨 best_validation_loss는 EarlyStopper 객체에서 가져옵니다.
        if val_loss < early_stopper.best_validation_loss:
            best_val_loss = val_loss
            
            # 모델 저장
            filename = f"best_model_epoch_{epoch+1}_loss_{best_val_loss:.4f}.pth"
            
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                save_dir=MODEL_SAVE_PATH, # 폴더 경로 사용
                filename=filename
            )
        
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
    DATA_PATH = os.path.join(CURDIR, 'data', 'temporal_graph_data_mi.pickle')
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
    MODEL_SAVE_PATH = os.path.join(CUR_DIR, 'a3tgcn_checkpoints') 
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True) # 저장 폴더 생성

    # 손실 함수 정의
    criterion = nn.CrossEntropyLoss()

    # 최적화 알고리즘 정의
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # train 함수 실행
    train_A3TGCNCat2(model, DEVICE, template_edge_index, train_dataset, val_dataset, criterion, optimizer, MODEL_SAVE_PATH, num_epochs=100)