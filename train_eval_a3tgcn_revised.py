import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from models.a3tgcn_revised import A3TGCNCat1
from teds_tensor_dataset import TEDSTensorDataset
from utils.device_set import device_set
from utils.write_log import enable_dual_output
from utils.early_stopper import EarlyStopper
from utils.processing_utils import mi_edge_index_batched, train_test_split_customed

CUR_DIR = os.path.dirname(__file__)
enable_dual_output(f'a3tgcn_1116.txt')

def train(model, dataloader, criterion, optimizer, edge_index, device):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch, los_batch in tqdm(dataloader, desc="train_process", leave=True):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        los_batch = los_batch.to(device)

        optimizer.zero_grad()

        result = model(
            ad_col_index,
            dis_col_index,
            x_batch,
            los_batch,
            edge_index,
            device
        )
        loss = criterion(result, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, val_dataloader, criterion, device, ad_col_index, dis_col_index, edge_index):
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

            result = model(
                ad_col_index,
                dis_col_index,
                x_batch,
                los_batch,
                edge_index,
                device
            )

            loss = criterion(result, y_batch)
            running_loss += loss.item() * x_batch.size(0)

            _, predicted = torch.max(result, 1)

            all_targets.append(y_batch.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_scores.append(result.cpu().numpy())

            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_scores = np.concatenate(all_scores)

    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_accuracy = total_correct / total_samples

    epoch_precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    epoch_recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    try:
        epoch_auc = roc_auc_score(all_targets, all_scores, multi_class='ovr')
    except ValueError:
        print("Warning: AUC score could not be calculated (requires multiple classes or specific binary format).")
        epoch_auc = 0.0

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


if __name__ == "__main__":
    root = './data_tensor_cache'
    # root = './data_tensor_sampled'
    EPOCH = 100
    scheduler_patience = 10
    early_stopping_patience = 15
    model_path = os.path.join(root, 'model')
    embedding_dim=32
    hidden_channel=64 ###이게 좀 걸림 30분씩 늘어남

    device = device_set()

    BATCH_SIZE = 16 
    num_workers= 0


    from torch.optim.lr_scheduler import ReduceLROnPlateau

    dataset = TEDSTensorDataset(root)

    train_dataloader, val_dataloader, test_dataloader = train_test_split_customed(dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info

    model = A3TGCNCat1(batch_size=BATCH_SIZE, col_list=col_list,
                        col_dims=col_dims, embedding_dim=embedding_dim, hidden_channel=hidden_channel)
    model.to(device)
    print("compiling model...")
    model = torch.compile(model=model, mode="max-autotune", dynamic=False)
    print("compile finished")

    mi_dict_path = os.path.join(root, 'data', 'mi_dict_static.pickle')
    edge_index = mi_edge_index_batched(batch_size=BATCH_SIZE,
                                            mi_dict_path=mi_dict_path,
                                            top_k=6,
                                            return_edge_attr=False)
    edge_index = edge_index.to(device)

    counter = 0

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=10)

    early_stopper = EarlyStopper(patience=early_stopping_patience)

    for epoch in tqdm(range(EPOCH)):
        train_loss = train(model, train_dataloader, criterion, optimizer, edge_index, device)

        result = evaluate(model, val_dataloader, criterion, device, ad_col_index, dis_col_index, edge_index)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = result

        scheduler.step(val_loss)

        if val_loss < early_stopper.best_validation_loss:
            print(f"🎉 New best validation loss: {val_loss:.4f}. Saving model...")
            
            best_val_loss = val_loss
            
            file_name = f"best_model_epoch_{epoch+1}_loss_{best_val_loss:.4f}.pth"
            full_save_path = os.path.join(model_path, file_name)
            save_checkpoint(epoch + 1, 
                            model, 
                            optimizer, 
                            scheduler, 
                            best_val_loss, 
                            full_save_path)

        should_stop = early_stopper(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch+1}/{epoch}]")
        print(f"  [Train] LR: {current_lr:.6f} | Loss: {train_loss:.4f}")
        print(f"  [Valid] Loss: {val_loss:.4f} | Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        if should_stop:
            print("\n--- Early Stopping activated. Learning terminated. ---")
            break

    print("\n--- 학습 완료 ---")