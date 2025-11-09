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

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience (int): 검증 손실이 개선되지 않아도 기다릴 에폭 수.
            min_delta (float): 새로운 손실이 개선으로 간주될 최소 변화량 (tolerance).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0              # 개선되지 않은 에폭 수 카운터
        self.best_validation_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        """
        새로운 검증 손실을 받아 조기 종료 여부를 결정합니다.
        Args:
            validation_loss (float): 현재 에폭의 검증 손실.
        Returns:
            bool: True이면 학습 중단, False이면 학습 계속.
        """
        
        # 1. 개선 여부 판단: 현재 손실이 이전 최적 손실보다 'min_delta'만큼 좋아졌는지 확인
        if validation_loss < self.best_validation_loss - self.min_delta:
            # 개선됨: 최적 손실 업데이트, 카운터 초기화
            self.best_validation_loss = validation_loss
            self.counter = 0
            
            # (선택 사항: 여기서 'best model'을 저장하는 로직을 호출하거나 포함할 수 있습니다.)
            # save_best_model(model, validation_loss) 
            
        elif validation_loss > self.best_validation_loss + self.min_delta:
            # 악화됨: 카운터 증가
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 Early stopping triggered after {self.counter} epochs without improvement.")
        
        return self.early_stop
    

def save_checkpoint(epoch, model, optimizer, best_val_loss, save_dir, filename="best_model.pth"):
    """
    현재 학습 상태를 체크포인트 파일로 저장
    """
    # 저장할 딕셔너리 생성
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    
    # 저장 경로 생성
    filepath = os.path.join(save_dir, filename)
    
    # 저장 실행
    torch.save(state, filepath)
    print(f"Checkpoint saved to: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """
    체크포인트 파일에서 모델과 옵티마이저 상태를 로드합니다.
    """
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return None, None, None

    # 로드 실행
    checkpoint = torch.load(filepath)
    
    # 모델의 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저 상태 로드
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 메타데이터 추출
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"Checkpoint loaded from {filepath}")
    
    return start_epoch, best_val_loss


def eval_A3TGCNCat2(model, device, template_edge_index, test_dataset, criterion):
    
    model.eval()

    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    y_scores_list = [] # AUC 계산을 위한 점수 리스트 추가
    total_samples = len(train_dataset)
    
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for signal in test_dataset:
            batch_list = [batch.to(device) for batch in signal]
            logits = model.forward(batch_list, template_edge_index)
            
            y_true = batch_list[0].y.to(device)
            loss = criterion(logits, y_true)
            total_loss += loss.item()

            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1] # 클래스 1에 대한 확률 (AUC 점수)


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
    

def train_A3TGCNCat2(model, device, template_edge_index, train_dataset, val_dataset, criterion, optimizer, MODEL_SAVE_PATH, num_epochs=10):
    
    model.train()

    softmax = nn.Softmax(dim=1)
    early_stopper = EarlyStopper(patience=15, min_delta=0.001)

    print(f"Starting training on {len(train_dataset)} samples...")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss_sum = 0.0

        y_true_list, y_pred_list, y_scores_list = [], [], []

        for signal in tqdm(train_dataset):
            optimizer.zero_grad()
            batch_list = [batch.to(device) for batch in signal]
            label = signal[0].y.to(device)

            logits = model.forward(batch_list, template_edge_index)   
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            # AUC 점수 수집
            probabilities = softmax(logits)
            score_class_1 = probabilities[:, 1]
            predicted_class = logits.argmax(dim=1)

            y_true_list.extend(label.cpu().tolist())
            y_pred_list.extend(logits.argmax(dim=1).cpu().tolist())
            y_scores_list.extend(score_class_1.cpu().tolist())

        avg_train_loss = train_loss_sum / len(train_dataset)
        train_metrics = compute_metrics(y_true_list, y_pred_list, [], num_classes=2)
        train_acc, train_prec, train_rec, train_f1, train_auc = train_metrics

        # 검증 호출 인자 순서/의미 수정
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = eval_A3TGCNCat2(
            model, device, template_edge_index, val_dataset, criterion
        )

        should_stop = early_stopper(val_loss)
        
        if val_loss < early_stopper.best_validation_loss:
            # 1. 최적 손실 값 업데이트
            best_val_loss = val_loss
            
            # 2. 모델 저장 함수 호출
            # 저장 파일명에 epoch 정보를 포함하여 관리 용이성 높임
            filename = f"best_model_epoch_{epoch+1}_loss_{best_val_loss:.4f}.pth"
            
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                save_dir=MODEL_SAVE_PATH,
                filename=filename
            )
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        print(f"  [Train] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"  [Valid] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

    print("\n--- 학습 완료 ---")


if __name__ == "__main__":
    # gpu 세팅
    DEVICE = device_set()

    # 데이터 로드
    #with open(DATA_PATH, 'rb') as f:
    #    pickle_dataset = pickle.load(f)
    from data.processing_temporal import processing_temporal_main
    pickle_dataset = processing_temporal_main()
    train_dataset, val_dataset, test_dataset, col_info = pickle_dataset
    col_list, col_dim = col_info
    template_edge_index = torch.as_tensor(train_dataset[0][0].edge_index, dtype=torch.long)
    template_edge_index = template_edge_index.to(DEVICE)

    # model 준비
    model = A3TGCNCat2(col_dims=col_dim, 
                       col_list=col_list, 
                       num_layers=2, 
                       hidden_channel=64, 
                       out_channel=2)
    model.to(DEVICE)

    # MODEL_SAVE_PATH
    MODEL_SAVE_PATH = os.path.join(CUR_DIR, 'a3tgcn1110.pt')

    # 손실 함수 정의
    criterion = nn.CrossEntropyLoss()

    # 최적화 알고리즘 정의
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # train 함수 실행
    train_A3TGCNCat2(model, DEVICE, template_edge_index, train_dataset, val_dataset, criterion, optimizer, MODEL_SAVE_PATH, num_epochs=3)
    
    


    
    
