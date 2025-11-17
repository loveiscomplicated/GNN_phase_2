import os
import pickle
import torch
import torch.nn as nn
# train_eval_a3tgcn.py에서 정의된 함수/클래스들을 import 합니다.
# 실제 환경에서는 이들이 모듈로 정의되어 있어야 import 가능합니다.
# 여기서는 편의상 동일 디렉토리에 있다고 가정하고 필요한 import를 유지합니다.
from models.a3tgcn import A3TGCNCat2
from utils.device_set import device_set
from utils.metrics import compute_metrics
# from utils.write_log import enable_dual_output # 테스트 시에는 불필요
from torch.optim.lr_scheduler import ReduceLROnPlateau # load_checkpoint에 필요
from tqdm import tqdm # 평가 함수에 필요

# --- train_eval_a3tgcn.py 에서 필요한 함수들을 복사 또는 import ---

# load_checkpoint 함수 (train_eval_a3tgcn.py에서 복사 또는 import)
def load_checkpoint(model, optimizer, lr_scheduler, filepath):
    """체크포인트를 로드하여 모델, 옵티마이저, 스케줄러 상태를 복원합니다."""
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return None, None, None

    # CPU로 로드 후 모델을 DEVICE로 이동시킵니다.
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저와 스케줄러는 학습에는 필요하지만 평가만 할 경우 상태 로드가 필수는 아닙니다.
    # 그러나 함수 정의에 포함되어 있으므로 상태만 로드합니다.
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"Checkpoint loaded from {filepath}")
    return start_epoch, best_val_loss


# eval_A3TGCNCat2 함수 (train_eval_a3tgcn.py에서 복사 또는 import)
def eval_A3TGCNCat2(model, device, template_edge_index, test_dataset, criterion):
    """모델을 테스트 데이터셋으로 평가합니다."""
    
    model.eval()

    total_loss = 0.0
    total_eval_samples = 0
    y_true_list = []
    y_pred_list = []
    y_scores_list = []
    
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for signal in tqdm(test_dataset, desc="Evaluating Test Set"):
            
            batch_list = []
            for batch in signal:
                if hasattr(batch, 'to'):
                    batch_list.append(batch.to(device))
                else:
                    batch_list.append(batch)
            
            # 레이블 및 예측
            labels = batch_list[0].y # (B,)
            logits = model.forward(batch_list, template_edge_index) # (B, 2)
            
            loss = criterion(logits, labels)
            
            current_batch_size = batch_list[0].batch_size
            total_loss += loss.item() * current_batch_size
            total_eval_samples += current_batch_size
            
            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1] # 1번 클래스 (양성) 확률

            y_true_list.extend(labels.cpu().tolist())
            y_pred_list.extend(logits.argmax(dim=1).cpu().tolist())
            y_scores_list.extend(score_class_1.cpu().tolist())
            
    # 평균 손실 계산
    avg_loss = total_loss / total_eval_samples
    
    # 지표 계산
    metrics = compute_metrics(
        y_true_list, 
        y_pred_list, 
        y_scores_list,
        num_classes=2
    )
    return avg_loss, *metrics
    
# --- 메인 실행 로직 ---

if __name__ == "__main__":
    
    # 1. GPU 및 경로 설정
    DEVICE = device_set()
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(CUR_DIR, 'data', 'temporal_graph_data_mi.pickle')
    # 🚨 로드할 체크포인트 파일 경로 설정
    # 학습 스크립트에서 저장한 실제 파일 이름으로 변경해야 합니다.
    MODEL_SAVE_PATH = os.path.join(CUR_DIR, 'a3tgcn_checkpoints')
    CHECKPOINT_FILENAME = "best_model_epoch_4_loss_0.9801.pth" # ⬅️ 실제 파일명으로 수정
    CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILENAME)
    
    print(f"Device: {DEVICE}")
    print(f"Loading data from: {DATA_PATH}")
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

    # 2. 데이터 로드
    try:
        with open(DATA_PATH, 'rb') as f:
            pickle_dataset = pickle.load(f)
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        exit()
        
    train_dataset, val_dataset, test_dataset, col_info = pickle_dataset
    col_list, col_dim = col_info

    # Edge Index 준비 및 DEVICE로 이동
    # edge_index는 모든 샘플에서 동일하다고 가정합니다.
    if not test_dataset:
        print("Error: Test dataset is empty.")
        exit()
    if not test_dataset[0]:
        print("Error: Test dataset first element is empty.")
        exit()

    template_edge_index = torch.as_tensor(test_dataset[0][0].edge_index, dtype=torch.long)
    template_edge_index = template_edge_index.to(DEVICE)
    
    # 3. 모델 및 최적화 객체 준비 (체크포인트 로드에 필요)
    model = A3TGCNCat2(col_dims=col_dim, 
                       col_list=col_list, 
                       num_layers=2, 
                       hidden_channel=64, 
                       out_channel=2)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=3)
    
    # 4. 체크포인트 로드
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, lr_scheduler, CHECKPOINT_PATH)
    
    if start_epoch is None:
        print("Model loading failed. Exiting.")
        exit()
        
    print(f"Model checkpoint restored (Epoch: {start_epoch}, Best Val Loss: {best_val_loss:.4f})")
    
    # 5. 모델 평가
    print("\n--- Starting Test Set Evaluation ---")
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = eval_A3TGCNCat2(
        model, 
        DEVICE, 
        template_edge_index, 
        test_dataset, 
        criterion
    )

    # 6. 결과 출력
    print("\n--- Test Evaluation Results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print("-------------------------------")