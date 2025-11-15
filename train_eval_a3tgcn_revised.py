import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from models.a3tgcn import A3TGCNCat2
from teds_temporal_dataset import TedsTemporalDataset
from utils.device_set import device_set
from utils.metrics import compute_metrics
from utils.write_log import enable_dual_output
from utils.early_stopper import EarlyStopper

BATCH_SIZE = 32
CUR_DIR = os.path.dirname(__file__)
enable_dual_output(f'a3tgcn_1114.txt')


def train():
    pass

def eval():
    pass

if __name__ == "__main__":
    # 디바이스 세팅(cpu or gpu)
    device = device_set()
    
    # 데이터 로드, train_test_split, Dataloader 생성
    root = os.path.join(CUR_DIR, 'data_cache')

    dataset = TedsTemporalDataset(root)
    print(f"총 데이터셋 크기: {dataset.NUM_GRAPH}")

    train_ratio, val_ratio, test_ratio = (0.7, 0.15, 0.15)

    total_size = len(dataset)
    train_size = total_size * train_ratio
    val_size = total_size * val_ratio
    test_size = total_size * test_ratio

    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Test Set Size: {len(test_dataset)}")

    train_dataloader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_dataloader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=False)
    test_dataloader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE, shuffle=False)

    # 모델 정의


    # 각종 기법들 정의 (EarlyStopping, LRScheduler, model save)


    # 에포크 반복문
    
