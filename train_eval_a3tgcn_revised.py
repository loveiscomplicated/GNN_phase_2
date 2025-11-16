import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


from models.a3tgcn_revised import A3TGCNCat1
from teds_tensor_dataset import TEDSTensorDataset
from utils.device_set import device_set
from utils.metrics import compute_metrics
from utils.write_log import enable_dual_output
from utils.early_stopper import EarlyStopper

BATCH_SIZE = 32
CUR_DIR = os.path.dirname(__file__)
enable_dual_output(f'a3tgcn_1116.txt')

def train_test_split_customed(dataset, ratio=[0.7, 0.15, 0.15], seed=42):

    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=ratio,
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Test Set Size: {len(test_dataset)}")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader
    
    

def train():
    pass

def eval():
    pass

if __name__ == "__main__":
    # 디바이스 세팅(cpu or gpu)
    device = device_set()
    
    # 데이터 로드, train_test_split, Dataloader 생성
    root = os.path.join(CUR_DIR, 'data_cache')

    dataset = TEDSTensorDataset(root)

    train_dataloader, val_dataloader, test_dataloader = train_test_split_customed(dataset, ratio=(0.7, 0.15, 0.15))
    
    # 추가 정보 로드
    col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info
    LOS = dataset.LOS
    
    # 모델 정의
    model = A3TGCNCat1(batch_size=BATCH_SIZE, col_list=col_list,
                       col_dims=col_dims, hidden_channel=64)
    model.to(device)

    # 각종 기법들 정의 (EarlyStopping, LRScheduler, model save)
    

    # 에포크 반복문
    

