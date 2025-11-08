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