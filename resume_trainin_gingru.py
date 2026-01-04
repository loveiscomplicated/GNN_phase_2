# train_gingru_resume.py
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
from teds_tensor_dataset import TEDSTensorDataset

# 너가 쓰는 모델
from models import gin_gru_2_point


# -------------------------
# Train / Eval (원본과 동일)
# -------------------------
def train(model, dataloader, criterion, optimizer, edge_index, device):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch, los_batch in tqdm(dataloader, desc="train_process", leave=True):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        los_batch = los_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch, los_batch, edge_index, device)
        logits = logits.squeeze(1)

        loss = criterion(logits, y_batch.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

    return running_loss / len(dataloader.dataset)


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

            logits = model(x_batch, los_batch, edge_index, device)
            logits = logits.squeeze(1)

            loss = criterion(logits, y_batch.float())

            scores = torch.sigmoid(logits)
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

    epoch_loss = running_loss / len(val_dataloader.dataset)
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


# -------------------------
# Checkpoint save / load
# -------------------------
def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, filename):
    checkpoint = {
        "epoch": epoch,  # 1-based epoch
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, scheduler, filename, map_location=None):
    ckpt = torch.load(filename, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1  # 다음 epoch부터
    best_loss = ckpt["best_loss"]
    return start_epoch, best_loss


# -------------------------
# Main
# -------------------------
def main():
    cur_dir = os.path.dirname(__file__)
    enable_dual_output("gingru_resume.txt")

    # ==============
    # Config
    # ==============
    device = torch.device("cpu")  # 필요하면 cuda/mps로 변경
    BATCH_SIZE = 32
    embedding_dim = 32
    gin_hidden_channel = 32
    train_eps = True
    gin_layers = 2
    gru_hidden_channel = 64
    decision_threshold = 0.5

    EPOCH = 100
    scheduler_patience = 5
    early_stopping_patience = 10
    learning_rate = 0.001

    sample = False
    if sample:
        root = os.path.join(cur_dir, "data_tensor_sampled")
        model_path = os.path.join(cur_dir, "checkpoints", "gingru", "sampled")
    else:
        root = os.path.join(cur_dir, "data_tensor_cache")
        model_path = os.path.join(cur_dir, "checkpoints", "gingru", "real")

    # ----------------
    # RESUME 설정 (여기만 바꾸면 됨)
    # ----------------
    RESUME = True
    RESUME_CKPT = os.path.join(
        model_path,
        "1218_gingru_epoch_37_loss_0.3287.pth"   # <- 파일명 바꿔
    )

    # ==============
    # Data / Graph
    # ==============
    mi_dict_path = os.path.join(root, "data", "mi_dict_static.pickle")

    dataset = TEDSTensorDataset(root)
    col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info
    num_nodes = len(ad_col_index)

    edge_index = mi_edge_index_batched(
        batch_size=BATCH_SIZE,
        mi_dict_path=mi_dict_path,
        num_nodes=num_nodes,
        top_k=6,
        return_edge_attr=False
    ).to(device)

    train_dataloader, val_dataloader, test_dataloader = train_test_split_customed(
        dataset=dataset,
        batch_size=BATCH_SIZE
    )

    # ==============
    # Model
    # ==============
    model = gin_gru_2_point.GinGru(
        batch_size=BATCH_SIZE,
        col_dims=col_dims,
        col_list=col_list,
        ad_col_index=ad_col_index,
        dis_col_index=dis_col_index,
        embedding_dim=embedding_dim,
        gin_hidden_channel=gin_hidden_channel,
        train_eps=train_eps,
        gin_layers=gin_layers,
        gru_hidden_channel=gru_hidden_channel
    ).to(device)

    print(model)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")

    # ==============
    # Optim / Sched / EarlyStop
    # ==============
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=scheduler_patience)
    early_stopper = EarlyStopper(patience=early_stopping_patience)

    # ==============
    # Resume
    # ==============
    start_epoch = 1
    best_val_loss = float("inf")

    if RESUME:
        if not os.path.exists(RESUME_CKPT):
            raise FileNotFoundError(f"Checkpoint not found: {RESUME_CKPT}")

        start_epoch, best_val_loss = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            filename=RESUME_CKPT,
            map_location=device
        )

        # EarlyStopper best 기준도 맞춰줘야 함
        early_stopper.best_validation_loss = best_val_loss

        print(f"\n[Resume] Loaded: {RESUME_CKPT}")
        print(f"[Resume] start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}\n")

    # ==============
    # Train loop
    # ==============
    for epoch in tqdm(range(start_epoch, EPOCH + 1)):
        train_loss = train(model, train_dataloader, criterion, optimizer, edge_index, device)

        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = evaluate(
            model, val_dataloader, criterion, decision_threshold, device, edge_index
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            print(f"🎉 New best validation loss: {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss

            file_name = f"resume_gingru_epoch_{epoch}_loss_{best_val_loss:.4f}.pth"
            full_save_path = os.path.join(model_path, file_name)

            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_val_loss,
                filename=full_save_path
            )

        should_stop = early_stopper(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n[Epoch {epoch}/{EPOCH}]")
        print(f"  [Train] LR: {current_lr:.6f} | Loss: {train_loss:.4f}")
        print(f"  [Valid] Loss: {val_loss:.4f} | Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        if should_stop:
            print("\n--- Early Stopping activated. Learning terminated. ---")
            break

    print("\n--- 학습 완료 ---")

    with torch.no_grad():
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate(
            model, test_dataloader, criterion, decision_threshold, device, edge_index
        )

        print(f"\n[Test] Loss: {test_loss:.4f} | Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
