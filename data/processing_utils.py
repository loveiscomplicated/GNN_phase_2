import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

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

def get_col_dims(df: pd.DataFrame):
    '''
    변수별 범주의 개수 파악
    '''
    col_dims = [len(df[col].cat.categories) for col in df.columns]
    return col_dims