import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from utils.write_log import enable_dual_output
from utils.processing_utils import train_test_split_customed, mi_edge_index_batched_cor
from teds_tensor_dataset import TEDSTensorDataset
from models.ctmp_gin import CtmpGIN

cur_dir = os.path.dirname(__file__)


def evaluate(model, dataloader, criterion, decision_threshold, device, edge_index):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_targets = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for x_batch, y_batch, los_batch in tqdm(dataloader, desc="test_process", leave=True):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            los_batch = los_batch.to(device)

            logits = model(x_batch, los_batch, edge_index).squeeze(1)
            loss = criterion(logits, y_batch.float())

            scores = torch.sigmoid(logits)  # [B]
            predicted = (scores >= decision_threshold).long()

            running_loss += loss.item() * x_batch.size(0)

            all_targets.append(y_batch.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_scores = np.concatenate(all_scores)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = total_correct / total_samples

    try:
        epoch_auc = roc_auc_score(all_targets, all_scores)
    except ValueError:
        print("Warning: AUC score could not be calculated.")
        epoch_auc = 0.0

    epoch_precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    epoch_recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_auc


def load_checkpoint(model, optimizer, scheduler, filename, map_location=None):
    checkpoint = torch.load(filename, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])

    # optimizer/scheduler는 "학습 재개"가 목적일 때만 의미 있지만,
    # 네가 저장한 포맷이 있으니 일단 복구 가능하도록 유지
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('best_loss', None)
    return start_epoch, best_loss


def build_edge_index(dataset, batch_size, device):
    root = os.path.join(cur_dir, 'data_tensor_cache')
    mi_dict_ad_path = os.path.join(root, 'data', 'mi_dict_ad.pickle')
    mi_dict_dis_path = os.path.join(root, 'data', 'mi_dict_dis.pickle')

    col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info
    num_nodes = len(ad_col_index)

    edge_index = mi_edge_index_batched_cor(
        batch_size=batch_size,
        num_nodes=num_nodes,
        mi_dict_ad_path=mi_dict_ad_path,
        mi_dict_dis_path=mi_dict_dis_path,
        top_k=6,
        return_edge_attr=False
    )
    return edge_index.to(device)


def build_model(dataset, device,
                embedding_dim=32,
                gin_hidden_channel=32,
                gin_1_layers=2,
                gin_hidden_channel_2=32,
                gin_2_layers=2,
                dropout_p=0.2,
                los_embedding_dim=8):
    model = CtmpGIN(
        col_info=dataset.col_info,
        embedding_dim=embedding_dim,
        gin_hidden_channel=gin_hidden_channel,
        gin_1_layers=gin_1_layers,
        gin_hidden_channel_2=gin_hidden_channel_2,
        gin_2_layers=gin_2_layers,
        device=device,
        dropout_p=dropout_p,
        los_embedding_dim=los_embedding_dim,
        max_los=37,
        train_eps=True,
        gate_hidden_ch=None
    )
    return model.to(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="체크포인트(.pth) 경로")
    parser.add_argument("--device", type=str, default="mps", help="mps/cuda/cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--log", type=str, default="ctmp_gin_test.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    enable_dual_output(args.log)

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        # 기본 mps
        device = torch.device("mps")

    # dataset & dataloaders
    root = os.path.join(cur_dir, 'data_tensor_cache')
    dataset = TEDSTensorDataset(root)

    train_dataloader, val_dataloader, test_dataloader = train_test_split_customed(
        dataset=dataset,
        batch_size=args.batch_size
    )

    # edge_index (학습과 동일하게 생성)
    edge_index = build_edge_index(dataset, args.batch_size, device)

    import pickle

    with open('edge_index.pickle', 'wb') as f:
        pickle.dump(edge_index, f)
    
    # model
    model = build_model(dataset, device)

    # criterion
    criterion = nn.BCEWithLogitsLoss()

    # optimizer/scheduler (로드 포맷 맞추기용)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5)

    # load checkpoint
    ckpt_path = args.ckpt
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    start_epoch, best_loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        filename=ckpt_path,
        map_location=device
    )

    print(f"[Loaded checkpoint] {ckpt_path}")
    print(f"  start_epoch: {start_epoch}")
    print(f"  best_loss: {best_loss}")

    # test eval
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model=model,
        dataloader=test_dataloader,
        criterion=criterion,
        decision_threshold=args.threshold,
        device=device,
        edge_index=edge_index
    )

    print("\n--- Test Result ---")
    print(f"[Test] Loss: {test_loss:.4f} | "
          f"Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, "
          f"Rec: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
