import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

def generate_non_local_graph(args, feat_trans, H, A, num_edge, num_nodes):
    K = args.K
    # if not args.knn:    
    # pdb.set_trace()
    x = F.relu(feat_trans(H))
    # D_ = torch.sigmoid(x@x.t())
    D_ = x@x.t()
    _, D_topk_indices = D_.t().sort(dim=1, descending=True)
    D_topk_indices = D_topk_indices[:,:K]
    D_topk_value = D_.t()[torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K), D_topk_indices]
    edge_j = D_topk_indices.reshape(-1)
    edge_i = torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K).reshape(-1).to(H.device)
    edge_index = torch.stack([edge_i, edge_j])
    edge_value = (D_topk_value).reshape(-1)
    edge_value = D_topk_value.reshape(-1)
    return [edge_index, edge_value]

def _norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ),
                                dtype=dtype,
                                device=edge_index.device)
    edge_weight = edge_weight.view(-1)
    assert edge_weight.size(0) == edge_index.size(1)
    row, col = edge_index.detach()
    deg = scatter_add(edge_weight.clone(), row.clone(), dim=0, dim_size=num_nodes)                                                          
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
    return deg_inv_sqrt, row, col
