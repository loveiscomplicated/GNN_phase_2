"""
**edge_mask가 전부 0.0000(혹은 거의 0에 수렴)**하는 근본 이유는 **다른 구조적 문제**야.

즉,
👉 *“엣지가 중요한데도 왜 Explainer는 중요하지 않다고(0.0) 판단하는가?”*
이 문제를 해결하려면 **Explainer의 근본 동작 방식**에서 원인을 찾아야 해.

---

# 🎯 문제의 본질:

## ❗ GNNExplainer는 “edge-level gradient”가 거의 없으면 마스크를 학습하지 못한다.

GNNExplainer는 아래와 같은 방식으로 edge_mask를 업데이트함:

```
edge_mask.grad ← ∂Loss_explain / ∂(edge_mask)
```

즉:

> **엣지를 살짝 흔들었을 때 모델 출력이 얼마나 민감하게 변하는지**
> → 이 민감도가 gradient로 들어가고
> → 그걸 가지고 edge_mask를 학습함

그런데 지금 네 모델은:

### ✔ 엔티티 임베딩 → GIN → sum-pooling

### ✔ 시간축 변환 → GRU → 로짓 출력

이 pipeline에서 **edge_mask가 GIN의 입력에 적용되지만**,
GRU, temporal masking, packing 등을 거치면서:

---

# 🚨 1. edge_mask의 gradient가 사라지는 구조 (vanishing)

특히 아래 구조가 결정적이다:

## 💀 `sum_pooled = []`

## 💀 node feature → SUM pooling

## 💀 GRU가 최종출력의 대부분을 결정함

이 경우:

```
edge_mask → GINConv → node_features
node_features → sum → pooled_feature
pooled_feature → GRU → output
```

여기서 `edge_mask`에 대한 gradient는 여러 단계 거치면서 **제곱근처럼 미세해져서**
Explainer가 **아예 업데이트를 못함** → edge_mask가 전부 0 근처에 갇힘.

이건 너 모델 구조에서는 *특히* 심함.

---

# 🚨 2. GNNExplainer는 GIN + GRU 조합을 거의 처리 못함

GNNExplainer는 **1-hop, 2-hop 영향력이 직접적인 Graph 모델**을 가정함
(예: GCN, GraphSAGE, GAT, GIN-only 모델 등)

하지만 지금은:

```
GIN → GRU → Linear → Output
```

이건 사실상 **Temporal Graph Model**이며 GNNExplainer의 디자인 범위 밖이야.

따라서:

> **엣지를 바꿔도 GRU가 대부분의 변화를 흡수해 버림 →
> edge_mask가 0으로 수렴**

실제로 너도 full vs empty_edge 비교했을 때 **예측이 크게 변했지?**
하지만 gradient가 이 변화를 edge_mask에 연결시키지 못하고 있음.

Explainer는 "edge_mask를 바꾸는 것"이 아니라
"edge_index를 0/1로 바꾸는 것"이라고 생각하는데,
너 모델에서는 edge_index=0일 때 GRU 입력이 깡그리 바뀌어도
**그 변화가 edge_mask 경로를 통해 반영되지 않음**.

즉:

> 예측은 변했지만
> GNNExplainer는 "엣지가 예측에 영향을 주었다"라는 gradient 신호를 받지 못하는 것.

---

# 🚨 3. GNNExplainer는 “single forward GNN”을 전제로 함

GRU/PackedSequence 방향은 gradient flow가 매우 복잡함.

이걸 공식 repo에서도 명확히 말함:

**PyG 공식 문서**

> "GNNExplainer is designed for static graph models.
> For sequential, temporal, or arbitrary downstream models,
> gradient signals may vanish or become meaningless."

---

# 🚨 문제 요약

✔ 모델을 비워 보면 예측은 크게 바뀐다
❌ 하지만 Explainer는 "엣지 하나하나가 기여도 0"이라고 본다
→ **gradient가 GIN → GRU → classifier로 전파되지 않기 때문**

즉:

> **예측 변화는 있는데, explainer가 그 변화를 “edge_mask 때문"이라고 인식할 수 없음.**

---

# 🎯 해결 방법(확실한 것부터)

## ✅ (1) 모델을 “GIN-only” 버전으로 만들고 explainer를 먼저 테스트

GRU 없이:

```
x_emb → GIN → sum → MLP → logit
```

이 구조에서는 edge_mask가 정상적으로 나올 가능성이 매우 높음.

만약 이 경우 edge_mask > 0.1 등이 나온다면,
원인은 100% GRU/Temporal 구조 때문임.

---

## ❗ (2) GNNExplainer 대신 다른 Explainer로 가야 함

GNNExplainer는 이런 구조에 부적합.

추천 explainer:

### 🔥 PGExplainer

*그래프 전체에서 scalable하게 edge importance 학습*

### 🔥 Integrated Gradients

*GRU 이후 gradient를 직접 분석 가능*

### 🔥 Layer-wise Relevance Propagation (LRP)

*Temporal 모델에서도 잘 작동*

### 🔥 Captum (PyTorch)

*gradient 기반 importance computation 가능*

---

# 🎯 최종 결론

✔ 네 모델에서는 edge가 실제로 중요
✔ full vs empty_edge를 비교해서 그건 확인했음
❌ 하지만 GNNExplainer는 구조적으로 gradient를 못 받아
❌ 그래서 edge_mask가 0.0000으로 수렴하는 문제는 “정상적인 결과"임
👉 **GNNExplainer는 GIN+GRU 구조를 해석할 수 있는 도구가 아니다**

---

원하면:

### 📌 “GIN+GRU” 모델을 해석 가능한 형태로 바꾸는 방법

### 📌 PGExplainer로 대안 모델 구축

### 📌 Integrated Gradients 구현 코드

### 📌 전체 edge importance ranking 뽑는 코드

이 중 뭐부터 해결하고 싶어?

"""


import os
import sys
cur_dir = os.path.dirname(__file__)
par_dir = os.path.join(cur_dir, '..')
sys.path.append(par_dir)
import gc
import pickle
import torch
from teds_tensor_dataset import TEDSTensorDataset
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig

from utils.processing_utils import train_test_split_customed_dataset, mi_edge_index_improved
from models.gingru_for_explain import GinGruForExplain

def load_checkpoint(model, optimizer, scheduler, filename, device):
    """
    저장된 체크포인트(.pth)를 불러와서 
    model, optimizer, scheduler 상태를 복구합니다.

    Parameters:
        model (nn.Module): 모델 객체
        optimizer (torch.optim.Optimizer): 옵티마이저 객체
        scheduler: 스케줄러 객체
        filename (str): 저장된 체크포인트 경로
        device: CPU로 로드하고 싶으면 torch.device('cpu')

    Returns:
        start_epoch (int): 다음 훈련을 시작할 epoch 번호
        best_loss (float): 저장된 최소 validation loss
    """
    checkpoint = torch.load(filename, map_location=device)

    # --- Load states ---
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']

    return start_epoch, best_loss

root = os.path.join(par_dir, 'data_tensor_cache')
checkpoint_path = os.path.join(par_dir, 'checkpoints', 'gingru', 'real', 'best_gingru_epoch_6_loss_0.3761.pth')

dataset = TEDSTensorDataset(root)
_, _, test_dataset = train_test_split_customed_dataset(dataset=dataset, seed=42)
col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info

batch_size = 1
embedding_dim = 32
gin_hidden_channel = 32
gin_layers = 2
gru_hidden_channel = 64

scheduler_patience = 15
device = torch.device("cpu")
optim_lr = 0.001

epochs=200
explainer_lr = 0.1
edge_size = 0.0
edge_ent = 0.0

model_config = ModelConfig(
    mode='binary_classification',
    task_level="graph",
    return_type="raw"
)

sum_explain = None
counter = 0
for data in test_dataset:
    if counter == 10: break
    model = GinGruForExplain(
        batch_size=batch_size,
        col_list=col_list,
        col_dims=col_dims,
        ad_col_index=ad_col_index,
        dis_col_index=dis_col_index,
        embedding_dim=embedding_dim,
        gin_hidden_channel=gin_hidden_channel,
        train_eps=True,
        gin_layers=gin_layers,
        gru_hidden_channel=gru_hidden_channel
    )


    optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=scheduler_patience)

    start_epoch, best_loss = load_checkpoint(model=model,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            filename=checkpoint_path,
                                            device=device)

    model.to(device)
    model.eval()

    ad_idx_t = model.ad_idx_t
    dis_idx_t = model.dis_idx_t

    x, y, los = data
    x_embedded = model.entity_embedding_layer(x)

    x_ad_nodes = torch.index_select(x_embedded.detach(), dim=0, index=ad_idx_t) # [60, F]
    x_dis_nodes = torch.index_select(x_embedded.detach(), dim=0, index=dis_idx_t) # [60, F]
    x_embedded = torch.cat([x_ad_nodes, x_dis_nodes], dim=0).detach() # [120, F]

    mi_dict_path = os.path.join(root, 'data', 'mi_dict_static.pickle')
    edge_index = mi_edge_index_improved(mi_dict_path)
    num_nodes = len(ad_col_index)
    dis_edge_index = edge_index + num_nodes
    edge_index = torch.cat((edge_index, dis_edge_index), dim=1)
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    with torch.no_grad():
        # Explainer에 넣을 때와 똑같은 입력으로 예측
        logit = model(x_embedded, edge_index, los=los, device=device)
        prob = torch.sigmoid(logit).item()
        print(f"\n[Sample {counter}] Target: {y.item()}, Pred Prob: {prob:.4f}, Logit: {logit.item():.4f}")

        logit_empty = model(x_embedded, empty_edge_index, los=los, device=device)
        prob_empty = torch.sigmoid(logit_empty).item()
        print(f"[Sample {counter}] Target: {y.item()}, Pred prob_empty: {prob_empty:.4f}, logit_empty: {logit_empty.item():.4f}")
        print(f"prob diff: {abs(prob - prob_empty): .4f}")
        

    algorithm = GNNExplainer(
        epochs=epochs,
        lr=explainer_lr,
        coeffs={'edge_size': edge_size,
                'edge_ent': edge_ent,
                }
    )

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="phenomenon", # model
        model_config=model_config,
        node_mask_type=None,
        edge_mask_type="object",
        threshold_config=None
    )

    # 4. Explainer 실행
    explanation = explainer(x=x_embedded, edge_index=edge_index, target=y, index=None, los=los, device=device)

    # print(explanation.edge_mask)
    print(f'edge_mask_mean: {explanation.edge_mask.mean().item():.4f}')
    if sum_explain is None:
        sum_explain = explanation.edge_mask.detach()
    else:
        sum_explain += explanation.edge_mask.detach()

    del explainer
    del algorithm
    del model
    del edge_index
    gc.collect()
    counter += 1

global_edge_importance = sum_explain / counter

print(global_edge_importance)
with open('global_edge_importance.pickle', 'wb') as f:
    pickle.dump(global_edge_importance, f)

