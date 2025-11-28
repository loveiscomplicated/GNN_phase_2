import os
import sys
import torch
from torch.utils.data import DataLoader 
from torch_geometric.explain.explainer import Explainer, ModelConfig, ExplainerConfig, ThresholdConfig 
from torch_geometric.explain.algorithm import DummyExplainer, GNNExplainer

CURDIR = os.path.dirname(__file__)
parent_dir = os.path.join(CURDIR, '..')
sys.path.append(parent_dir)

from models.gingru_for_explain import ExplainerCompatibleGinGru
from utils.processing_utils import train_test_split_customed, mi_edge_index_batched, mi_edge_index_improved
from teds_tensor_dataset import TEDSTensorDataset


def explain_all(dataloader: DataLoader, 
                explainer: Explainer, 
                model: ExplainerCompatibleGinGru,
                device):
    for batch in dataloader:
        x_label, y_batch, los_batch = batch
        
        with torch.no_grad():
            x_emb = model.entity_embedding_layer(x_label) # [32, 72] --> [32, 72, 32]
        
        for graph_idx in range(batch_size):
            input_ = x_emb[graph_idx].detach().to(device) # [72, 32]
            los_ = los_batch[graph_idx].detach().to(device)
            target_ = y_batch[graph_idx].detach().to(device)

    pass

root = os.path.join(parent_dir, 'data_tensor_cache')
mi_dict_path = os.path.join(root, 'data', 'mi_dict_static.pickle')
dataset = TEDSTensorDataset(root)

batch_size = 32
epochs = 100
lr = 0.01
num_nodes = 60

col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info

edge_index = mi_edge_index_improved(mi_dict_path=mi_dict_path)

_, _, test_dataloader = train_test_split_customed(dataset=dataset,
                                                  batch_size=batch_size)


model = ExplainerCompatibleGinGru(
    batch_size=batch_size,
    col_list=col_list,
    col_dims=col_dims,
    ad_col_index=ad_col_index,
    dis_col_index=dis_col_index,
    train_eps=True,
    gin_hidden_channel=32,
    embedding_dim=32,
    gin_layers=2,
    gru_hidden_channel=64
)

model.eval()

model_config = ModelConfig(
    mode="binary_classification",
    task_level="graph",
    return_type="raw")

algorithm = GNNExplainer(
    epochs=epochs,
    lr=lr
)

explainer = Explainer(
    model=model,
    algorithm=algorithm,
    explanation_type="phenomenon",
    model_config=model_config,
    node_mask_type=None,
    edge_mask_type="object",
    threshold_config=None
)

def get_explain(explainer: Explainer, dataloader: DataLoader, model: ExplainerCompatibleGinGru):
    for batch in dataloader:
        x_label, y_batch, los_batch = batch
        x_emb = model.entity_embedding_layer(x_label)
        