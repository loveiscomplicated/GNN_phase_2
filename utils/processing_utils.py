import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

def get_initial_data(random_state=42):
    '''
    missing_corrected.csv를 불러오고,
    데이터 스플릿까지 하는 함수
    Args:
        random_state(int, optional): 데이터 스플릿할 때 필요한 랜덤 시드
    '''
    CURDIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(CURDIR, 'missing_corrected.csv')
    DATA = pd.read_csv(DATA_PATH)
    DATA.head()

    # 범주형 변수 더미화 시 train/test 간 불일치를 방지하기 위해
    # 스플릿 전에 전체 데이터 기준으로 가능한 범주를 정의하고 CategoricalDtype으로 고정
    for col in DATA.columns:
        cats = list(DATA[col].unique())
        DATA[col] = DATA[col].astype(pd.api.types.CategoricalDtype(categories=cats))

    # validation set, test set이 imputation 설계하는 데 들어가면 안 됨, 일종의 사후판단 정보가 들어갈 수 있기 때문
    y = DATA['REASONb']
    X = DATA.drop('REASONb', axis=1)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_initial_data_sampled(size=100, random_state=42):
    '''
    missing_corrected.csv를 불러오고,
    @@전체는 너무 많으니까 잘 돌아가는지 보기 위해 조금만 추출해서@@
    데이터 스플릿까지 하는 함수
    Args:
        random_state(int, optional): 데이터 스플릿할 때 필요한 랜덤 시드
    '''
    CURDIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(CURDIR, 'missing_corrected.csv')
    DATA = pd.read_csv(DATA_PATH)
    DATA.head()

    # 범주형 변수 더미화 시 train/test 간 불일치를 방지하기 위해
    # 스플릿 전에 전체 데이터 기준으로 가능한 범주를 정의하고 CategoricalDtype으로 고정
    for col in DATA.columns:
        cats = list(DATA[col].unique())
        DATA[col] = DATA[col].astype(pd.api.types.CategoricalDtype(categories=cats))

    DATA = DATA.iloc[:size]

    # validation set, test set이 imputation 설계하는 데 들어가면 안 됨, 일종의 사후판단 정보가 들어갈 수 있기 때문
    y = DATA['REASONb']
    X = DATA.drop('REASONb', axis=1)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_initial_data_sampled_stratified(size=100, random_state=42):
    '''
    missing_corrected.csv를 불러오고,
    @@타겟 변수 'REASONb'를 기준으로 계층적 샘플링을 통해 지정된 크기(size)만큼 추출 후@@
    학습, 검증, 테스트 세트로 분할하는 함수
    Args:
        size(int, optional): 추출할 최종 데이터셋의 크기
        random_state(int, optional): 데이터 스플릿할 때 필요한 랜덤 시드
    '''
    CURDIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(CURDIR, 'missing_corrected.csv')
    DATA = pd.read_csv(DATA_PATH)
    DATA.head()

    # 범주형 변수 더미화 시 train/test 간 불일치를 방지하기 위해
    # 스플릿 전에 전체 데이터 기준으로 가능한 범주를 정의하고 CategoricalDtype으로 고정
    for col in DATA.columns:
        cats = list(DATA[col].unique())
        DATA[col] = DATA[col].astype(pd.api.types.CategoricalDtype(categories=cats))
    
    # y와 X 분리
    y_full = DATA['REASONb']
    X_full = DATA.drop('REASONb', axis=1)

    
    # 2. 🎯 계층적 샘플링을 통한 데이터 추출 (iloc 대체)
    # 전체 데이터셋이 size보다 작을 경우, 전체 데이터를 사용합니다.
    if len(X_full) <= size:
        print(f"Warning: Dataset size ({len(X_full)}) is smaller than requested size ({size}). Using full dataset.")
        X_sampled = X_full
        y_sampled = y_full
    else:
        # train_test_split을 활용하여 'y_full'을 기준으로 계층적 추출
        # test_size를 이용하여 size만큼의 비율을 계산
        sample_ratio = size / len(X_full)
        
        # NOTE: 추출할 데이터셋을 X_temp로, 버릴 데이터를 X_discard로 간주합니다.
        # test_size = 1 - sample_ratio 를 해야 sample_ratio 만큼의 데이터가 남게 됨
        X_discard, X_sampled, y_discard, y_sampled = train_test_split(
            X_full, y_full, test_size=sample_ratio, random_state=random_state, stratify=y_full
        )
        print(f"Successfully sampled {len(X_sampled)} records using stratification.")

    
    # 3. 추출된 데이터셋을 학습(70%), 임시(30%)로 계층 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sampled, y_sampled, test_size=0.30, random_state=random_state, stratify=y_sampled
    )
    
    # 4. 임시 데이터셋을 검증(50%), 테스트(50%)로 계층 분할
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    # 🚨 중요: 범주형 변수의 범주 정보가 drop되지 않도록 처리
    # 만약 범주형 변수의 범주가 추출된 데이터에 모두 없다면 오류가 날 수 있으므로
    # 이후 데이터 전처리 단계에서 CategoricalDtype을 다시 설정하는 것이 안전합니다.
    # 현재는 원본의 cats를 유지한 채 추출합니다.

    # 추출된 데이터의 클래스 분포 확인 (선택 사항)
    print("\n--- Final Split Distribution (Target: REASONb) ---")
    print(f"Train Set: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Validation Set: {y_val.value_counts(normalize=True).to_dict()}")
    print(f"Test Set: {y_test.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def fully_connected_edge_index(num_nodes, self_loops=False):
    '''
    이름 그대로, num_nodes 즉 변수 개수만 알면 됨
    '''
    nodes = torch.arange(num_nodes)
    row, col = torch.meshgrid(nodes, nodes, indexing="ij")
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)
    if not self_loops:
        mask = row != col
        edge_index = edge_index[:, mask.reshape(-1)]
    return edge_index

def fully_connected_edge_index_batched(num_nodes, batch_size, self_loops=False):
    '''
    batched edge_index는 결국 옆으로 이어붙인 것일 뿐, 즉 shape: [2, sum(num_edges{i})]
    '''
    single = fully_connected_edge_index(num_nodes=num_nodes)
    batch_list = [single for i in range(batch_size)]
    return torch.concatenate(batch_list, dim=1)


def mi_edge_index_improved(mi_dict_path, top_k=6, threshold=0.01, pruning_ratio=0.5, return_edge_attr=False):
    """
    개선된 MI 기반 그래프 생성 함수 (Strategies 1, 2, 3 적용)
    
    Args:
        mi_dict_path (str): 피클 파일 경로
        top_k (int): 상위 k개 선택
        threshold (float): [Strategy 2] MI 값이 이보다 작으면 연결하지 않음 (기본값 0.01)
        pruning_ratio (float): [Strategy 3] In-Degree가 전체 노드 수의 이 비율을 넘으면 하위 엣지 삭제 (기본값 0.5 = 50%)
        return_edge_attr (bool): 가중치 반환 여부
    """

    with open(mi_dict_path, 'rb') as f:
        mi_dict = pickle.load(f)

    cols = list(mi_dict.keys())
    num_nodes = len(cols)
    col_to_idx = {c: i for i, c in enumerate(cols)}

    # 임시 저장을 위한 리스트 (source, target, weight)
    raw_edges = []

    # 1. 초기 유향 엣지 생성 (Threshold & Top-k 적용)
    for src in cols:
        series = mi_dict[src]

        # 유효 변수 필터링 & 자기 자신 제외
        series = series[series.index.isin(cols)]
        if src in series.index:
            series = series.drop(index=src)

        # [Strategy 2] Threshold 적용 (너무 약한 관계 끊기)
        series = series[series >= threshold]

        # Top-k 선택
        top_neighbors = series.head(top_k)

        src_idx = col_to_idx[src]
        for dst, w in top_neighbors.items():
            dst_idx = col_to_idx[dst]
            raw_edges.append((src_idx, dst_idx, float(w)))

    # 2. [Strategy 3] 구조적 Pruning (Hub 노드 견제)
    # Target 노드별로 엣지를 모아서 In-Degree가 너무 높으면 약한 것부터 잘라냅니다.
    
    # Target별로 그룹화: {dst_idx: [(src, dst, w), ...]}
    edges_by_target = {}
    for edge in raw_edges:
        dst = edge[1]
        if dst not in edges_by_target:
            edges_by_target[dst] = []
        edges_by_target[dst].append(edge)
    
    final_edges = []
    max_in_degree = int(num_nodes * pruning_ratio) # 허용 가능한 최대 In-Degree (예: 60개 중 30개)

    for dst, edges in edges_by_target.items():
        # 만약 특정 노드(예: STFIPS)로 들어오는 엣지가 너무 많다면?
        if len(edges) > max_in_degree:
            # 가중치(MI) 기준 내림차순 정렬 후 상위 N개만 남김
            edges.sort(key=lambda x: x[2], reverse=True)
            kept_edges = edges[:max_in_degree]
            final_edges.extend(kept_edges)
        else:
            final_edges.extend(edges)

    # 텐서 변환 준비
    if not final_edges:
        print("⚠️ 주의: 조건에 맞는 엣지가 하나도 없습니다. Threshold를 낮추세요.")
        return torch.empty((2, 0), dtype=torch.long)

    src_list, dst_list, weight_list = zip(*final_edges)
    
    # Directed Edge Index 생성
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weight_list, dtype=torch.float) if return_edge_attr else None

    # 3. [Strategy 1] 무방향(Undirected) 그래프로 변환
    # A->B가 있으면 B->A도 생성 (정보 흐름 개선)
    # to_undirected는 중복된 엣지는 제거하고, 양방향을 보장해줍니다.
    if return_edge_attr:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=num_nodes)
        return edge_index, edge_attr
    else:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        return edge_index

def mi_edge_index_batched(batch_size, num_nodes, mi_dict_path, top_k=6, threshold=0.01, pruning_ratio=0.5, return_edge_attr=False, edge_attr_single=None):
    """
    배치 처리를 위한 Wrapper 함수
    """
    # 곱하기 2를 하는 이유는 ad, dis를 concat하는 것의 영향으로 x tensor shape: [B * 2 * 60, F]가 될 것
    batch_size_d = batch_size * 2

    if return_edge_attr:
        single, edge_attr= mi_edge_index_improved(
            mi_dict_path=mi_dict_path, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    else:
        single = mi_edge_index_improved(
            mi_dict_path=mi_dict_path, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )

    edge_list = []
    attr_list = []

    for g in range(batch_size_d):
        offset = num_nodes * g
        edge_i = single + offset
        edge_list.append(edge_i)

        if return_edge_attr:
            attr_list.append(edge_attr)
    
    batched_edge_index = torch.cat(edge_list, dim=1)

    if return_edge_attr:
        batched_attr_list = torch.cat(attr_list, dim=0)
        return batched_edge_index, batched_attr_list
    
    return batched_edge_index
    
def get_col_dims(df: pd.DataFrame):
    '''
    변수별 범주의 개수 파악
    '''
    col_dims = [len(df[col].unique()) for col in df.columns]
    return col_dims

def get_ad_dis_col(df:pd.DataFrame):
    '''
    admission 시의 컬럼, discharge 시의 컬럼을 나누어 리턴
    Args:
        df(pd.DataFrame): 원본 데이터프레임, REASONb는 자동으로 제외됨
    Returns: 
        (admission 시의 컬럼 list, discharge 시의 컬럼 list)
    '''
    cols = list(df.columns)

    if 'LOS' in cols:
        cols.remove('LOS')

    if 'REASONb' in cols:
        cols.remove('REASONb')

    change = []
    change_D = []

    for i in cols:
        if i.endswith('_D'):
            change_D.append(i)
            change.append(i[:-2])
    
    ad = [i for i in cols if i not in change_D]
    dis = ad.copy()
    for i in range(len(ad)):
        if dis[i] in change:
            dis[i] = dis[i] + '_D'

    return ad, dis

def find_indices(lst, targets):
    return [lst.index(t) if t in lst else None for t in targets]

def get_ad_dis_index(df: pd.DataFrame):
    col_list = list(df.columns)
    ad, dis = get_ad_dis_col(df)
    ad_col_index = find_indices(col_list, ad)
    dis_col_index = find_indices(col_list, dis)
    return ad_col_index, dis_col_index

def get_col_info(df: pd.DataFrame):
    '''
    Returns: (tuple)
        col_list, col_dims, ad_col_index, dis_col_index

        col_list: 보관용, 데이터에 등장하는 열 이름의 순서
        col_dims: col_list 순서대로 변수별 범주의 개수
        ad_col_index: admission에 해당하는 변수의 integer position
        dis_col_index: discharge에 해당하는 변수의 integer position
    '''
    col_list = list(df.columns)
    col_dims = get_col_dims(df)
    ad_col_index, dis_col_index = get_ad_dis_index(df)
    return col_list, col_dims, ad_col_index, dis_col_index

def organize_labels(df: pd.DataFrame):
    '''
    -9가 있는 변수를 그대로 엔티티 임베딩에 넣으면 이상해짐
    왜냐하면 엔티티 임베딩 모델은 레이블들이 연속된 정수들의 범위로 있다고 가정하기 때문
    -9, 1, 2, 3 이렇게 있었다면
    -9, -8, -7, -6, -5, ~~~ 이런 것으로 가정함

    -9, 1, 2, 3를
    0, 1, 2, 3으로 바꿈 (-9 -> 4)
    
    + CBSA2020
    이것도 문제가 됨
    10000 24242 32646 75577 이런 식이라 연속된 정수들의 레이블이 아님
    10000 24242 32646 75577 -> 1, 2, 3, 4
    '''

    for col in df.columns:
        labels = sorted(df[col].unique())
        replace_dict = {labels[i]: i for i in range(len(labels))}
        df[col] = df[col].replace(replace_dict)

    return df

def df_to_tensor(df: pd.DataFrame | pd.Series, dtype=torch.long):
    df_np = df.to_numpy()
    return torch.tensor(df_np, dtype=dtype)

def get_total_dim(df: pd.DataFrame):
    total_dim = 0
    for col in df.columns:
        col_dim = len(df[col].unique())
        total_dim += col_dim
    return total_dim


'''def train_test_split_customed(dataset, batch_size, ratio=[0.7, 0.15, 0.15], seed=42, num_workers=0):

    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=ratio,
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Test Set Size: {len(test_dataset)}")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader'''




def train_test_split_customed(dataset, batch_size,
                              ratio=[0.7, 0.15, 0.15],
                              seed=42,
                              num_workers=0):

    assert abs(sum(ratio) - 1.0) < 1e-6, "ratio must sum to 1.0"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 전체 인덱스 & 라벨 추출
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    indices = np.arange(len(dataset))

    unique_labels = np.unique(labels)

    train_idx = []
    val_idx = []
    test_idx = []

    # --- Stratified Split ---
    for ul in unique_labels:
        cls_idx = indices[labels == ul]
        np.random.shuffle(cls_idx)

        n_total = len(cls_idx)
        n_train = int(n_total * ratio[0])
        n_val = int(n_total * ratio[1])
        # 남은 건 test
        n_test = n_total - n_train - n_val

        # 분할
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    # 셔플 (선택)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    # Subset dataset 생성
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Valid Set Size: {len(val_dataset)}")
    print(f"Test Set Size: {len(test_dataset)}")

    # DataLoader 생성
    # drop_last=True를 해야 마지막 자투리 배치를 위해 따로 배치 엣지 인덱스를 만들 필요가 없음
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader

def train_test_split_customed_dataset(dataset,
                                      ratio=[0.7, 0.15, 0.15],
                                      seed=42
                                      ):

    assert abs(sum(ratio) - 1.0) < 1e-6, "ratio must sum to 1.0"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 전체 인덱스 & 라벨 추출
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    indices = np.arange(len(dataset))

    unique_labels = np.unique(labels)

    train_idx = []
    val_idx = []
    test_idx = []

    # --- Stratified Split ---
    for ul in unique_labels:
        cls_idx = indices[labels == ul]
        np.random.shuffle(cls_idx)

        n_total = len(cls_idx)
        n_train = int(n_total * ratio[0])
        n_val = int(n_total * ratio[1])
        # 남은 건 test
        n_test = n_total - n_train - n_val

        # 분할
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    # 셔플 (선택)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    # Subset dataset 생성
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Valid Set Size: {len(val_dataset)}")
    print(f"Test Set Size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

import torch
from torch_geometric.utils import is_undirected, degree
import numpy as np

def verify_graph_structure(edge_index, num_nodes, pruning_ratio):
    print("\n🔍 [Graph Structure Verification Report]")
    print("=" * 50)

    # 1. 무방향성 체크
    is_sym = is_undirected(edge_index)
    print(f"1. 무방향(Undirected) 그래프인가? : {'✅ Yes' if is_sym else '❌ No'}")
    
    if not is_sym:
        print("   -> 경고: to_undirected가 제대로 적용되지 않았습니다.")

    # 2. 허브 노드(Max Degree) 체크
    # 무방향이므로 in-degree = out-degree
    deg = degree(edge_index[0], num_nodes=num_nodes)
    max_deg = deg.max().item()
    max_deg_node = deg.argmax().item()
    
    limit = int(num_nodes * pruning_ratio)
    
    print(f"2. 가장 연결이 많은 노드 (Max Degree)")
    print(f"   - Node Index: {max_deg_node}")
    print(f"   - Degree: {int(max_deg)} (Limit: {limit})")
    
    if max_deg <= limit:
        print(f"   ✅ Pass: 허브 노드 프루닝이 잘 되었습니다. (Max {int(max_deg)} <= Limit {limit})")
    else:
        print(f"   ❌ Fail: 여전히 너무 강력한 허브가 존재합니다.")

    # 3. 고립 노드 체크 (Degree가 0인 노드)
    # Threshold가 너무 높으면 고립 노드가 생길 수 있음 (괜찮을 수도 있지만 확인 필요)
    isolated_nodes = (deg == 0).sum().item()
    print(f"3. 고립된(연결 없는) 노드 개수: {isolated_nodes}개 / 전체 {num_nodes}개")
    
    # 4. 전체 밀도 (Density)
    num_edges = edge_index.shape[1]
    possible_edges = num_nodes * (num_nodes - 1) # Self-loop 제외 시
    density = num_edges / possible_edges
    print(f"4. 그래프 밀도: {density:.4f} (총 엣지 수: {num_edges})")
    print("=" * 50)

# --- 테스트 실행 예시 ---
if __name__ == "__main__":
    # 임시 테스트용 경로 및 설정
    import os
    cur_dir = os.path.dirname(__file__)
    
    MI_PATH = os.path.join(cur_dir, 'mi_dict.pickle') # 경로 확인 필요
    TOP_K = 6
    THRESH = 0.01
    PRUNING = 0.5
    
    # 함수 실행
    try:
        # 개선된 함수 호출
        edge_index = mi_edge_index_improved(
            mi_dict_path=MI_PATH, 
            top_k=TOP_K, 
            threshold=THRESH, 
            pruning_ratio=PRUNING
        )
        
        # 전체 노드 수는 mi_dict 열어서 확인하거나 대략 72로 가정
        # (정확히 하려면 mi_dict 로드해서 len(keys) 해야 함)
        import pickle
        with open(MI_PATH, 'rb') as f:
            cols = list(pickle.load(f).keys())
            NUM_NODES = len(cols)

        # 검증
        verify_graph_structure(edge_index, NUM_NODES, PRUNING)

    except Exception as e:
        print(f"테스트 실패: {e}")



