import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from models.a3tgcn import A3TGCNCat2
from utils.device_set import device_set
from utils.metrics import compute_metrics
from utils.write_log import enable_dual_output

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


# eval_A3TGCNCat2 함수 수정
def eval_A3TGCNCat2(model, device, template_edge_index, test_dataset, criterion):
    
    model.eval()

    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    y_scores_list = []
    
    # 🚨 수정: total_samples는 test_dataset의 길이여야 함
    total_samples = len(test_dataset)
    
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        # 🚨 tqdm 적용
        for signal in tqdm(test_dataset, desc="Evaluating"):
            # signal 객체 내부의 Batch 객체만 to()를 호출
            batch_list = []
            for batch in signal:
                if hasattr(batch, 'to'):
                    batch_list.append(batch.to(device))
                else:
                    batch_list.append(batch)
                    
            logits = model.forward(batch_list, template_edge_index)
            
            y_true = batch_list[0].y.to(device)
            loss = criterion(logits, y_true)
            total_loss += loss.item()

            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1]

            # 🚨 수정: .item() 대신 .extend()와 .tolist() 사용
            y_true_list.extend(y_true.cpu().tolist())
            y_pred_list.extend(logits.argmax(dim=1).cpu().tolist())
            y_scores_list.extend(score_class_1.cpu().tolist())
            

    avg_loss = total_loss / total_samples
    metrics = compute_metrics(
        y_true_list, 
        y_pred_list, 
        y_scores_list,
        num_classes=2
    )
    return avg_loss, *metrics
    

# train_A3TGCNCat2 함수 수정
def train_A3TGCNCat2(model, device, template_edge_index, train_dataset, val_dataset, criterion, optimizer, MODEL_SAVE_PATH, num_epochs=10):
    
    model.train()

    softmax = nn.Softmax(dim=1)
    early_stopper = EarlyStopper(patience=15, min_delta=0.001)

    print(f"Starting training on {len(train_dataset)} samples...")
    # 🚨 tqdm 적용
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss_sum = 0.0

        y_true_list, y_pred_list, y_scores_list = [], [], []

        # 🚨 tqdm 적용
        for signal in tqdm(train_dataset, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            
            # signal 객체 내부의 Batch 객체만 to()를 호출
            batch_list = []
            for batch in signal:
                if hasattr(batch, 'to'):
                    batch_list.append(batch.to(device))
                else:
                    batch_list.append(batch)
            
            label = signal[0].y.to(device) # 레이블 이동 (이전 수정)

            logits = model.forward(batch_list, template_edge_index)   
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            # AUC 점수 수집
            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1]
            predicted_class = logits.argmax(dim=1)

            # 🚨 .extend()와 .tolist()는 이미 올바르게 적용됨
            y_true_list.extend(label.cpu().tolist())
            y_pred_list.extend(predicted_class.cpu().tolist())
            y_scores_list.extend(score_class_1.cpu().tolist())

        avg_train_loss = train_loss_sum / len(train_dataset)
        # 🚨 AUC 계산 시 y_scores_list를 전달하도록 수정 (이전 코드에서는 빈 리스트를 전달하고 있었음)
        train_metrics = compute_metrics(y_true_list, y_pred_list, y_scores_list, num_classes=2)
        train_acc, train_prec, train_rec, train_f1, train_auc = train_metrics

        # 검증 호출
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
    DATA_PATH = os.path.join(CURDIR, 'data', 'temporal_graph_data.pickle')
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