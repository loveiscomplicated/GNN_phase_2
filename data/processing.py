import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch
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

def get_ad_dis_col(df:pd.DataFrame):
    '''
    admission 시의 컬럼, discharge 시의 컬럼을 나누어 리턴
    Args:
        df(pd.DataFrame): 원본 데이터프레임

    Returns: 
        (admission 시의 컬럼 list, discharge 시의 컬럼 list)
    '''
    cols = list(df.columns)

    if 'LOS' in cols:
        cols.remove('LOS')

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


def los_to_time(vec:pd.Series, ad:list, dis:list):
    '''
    원본 케이스(데이터프레임의 한 행)를 시계열 데이터로 변환

    Args:
        vec(pd.Series): 원본 데이터프레임의 한 행
        ad(list): admission 시의 컬럼 list, LOS 제외
        dis(list): discharge 시의 컬럼 list, LOS 제외
    Returns:
        
    '''
    if 'LOS' not in vec.index:
        raise ValueError("no LOS available")
    
    los = vec['LOS']
    admission = vec[ad]
    discharge = vec[dis]


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

def get_graph(x:pd.Series, y=None, edge_index=None):
    '''
    make numpy data to graph data (torch_geometric.data.Data)
    Args:
        x(pd.Series): row of dataframe
        y(Optional): y-label, Default: None
        edge_index(): COO matix, Default: None
    Returns: instance of torch_geometric.data.Data
    '''
    x_tensor = torch.as_tensor(x.to_numpy()).unsqueeze(-1)

    if y is not None:
        y_tensor = torch.as_tensor(y.to_numpy(), dtype=torch.long)
    else:
        y_tensor = None

    return Data(x=x_tensor, y=y_tensor, edge_index=edge_index)


class DataBundle:
    def __init__(self, xdf:pd.DataFrame, ysr:pd.Series):
        '''
        Args:
            x: X_train, X_val, X_test 중 하나
            y: y_train, y_val, y_test 중 하나
        '''
        self.xdf = xdf
        self.ysr = ysr

        # not temporal data
        self.edge_index = fully_connected_edge_index(self.xdf.shape[1])
        self.col_order = list(self.xdf.columns)
        self.col_dims = get_col_dims(self.xdf) # 나중에 엔티티 임베딩할 때 쓸 거
        self.graph_list = []

        # temporal data
        self.ad, self.dis = get_ad_dis_col(self.xdf)
        self.edge_index_tem = fully_connected_edge_index(len(self.ad))
        self.col_dims_tem = get_col_dims(self.xdf[self.ad])
        self.signal_list = []

    def get_graph_lists(self):
        '''
        pytorch_geometric.data.Data 객체들의 리스트 반환
        '''
        for idx in tqdm(list(self.xdf.index)):
            vec = self.xdf.loc[idx].to_numpy()
            X = torch.as_tensor(vec).unsqueeze(-1) # 명시적으로 [num_nodes, 1] 형태로 만들기
            y = torch.as_tensor(self.ysr.loc[idx], dtype=torch.long) # 손실 함수가 원하는 타입으로 맞추기 -> torch.long으로 해야 함
            self.graph_list.append(Data(X, self.edge_index, y=y))
        return self
    
    def get_temporal_graph_batches(self, batch_size=16):
        '''
        매 시점마다 여러 개의 그래프를 하나로 묶어서 배치로 만드는 것
        즉 한 시점 안에서
            여러 개의 그래프의 정보가 들어 있는 노드 피처 매트릭스, 엣지 인덱스, (y 라벨-이건 종료시에만)
        이와 동시에 종료 시점이 그래프마다 상이하므로 모두 가장 긴 경우(37)에 맞춰서 패딩 & 마스크를 해야 함
        이걸 37개 시점까지 만들어서 리스트로 묶어서 StaticGraphTemporalSignalBatch에 넣으면 됨

        이걸 130만 / 배치 사이즈 만큼 해야 함

        1. 개별 시계열 그래프 데이터 구성 및 패딩
        2. 마스크 생성 및 저장 - 패딩된 데이터가 실제 데이터인지, 패딩된 가짜 데이터인지를 구분할 수 있는 마스크 시퀀스를 별도로 생성
            예를 들어 1이면 유효한 값, 0이면 패딩되어 무시해야 하는 값
            이건 StaticGraphTemporalSignalBatch이 기본적으로 받는 인수가 아니므로 추가 feature로 집어넣거나 따로 가지고 있어야 함
            마스크는 (batch_size, 37), 배치 범위 안에 있던 행 인덱스, 위에서 아래로 추가됨
        3. 2단계: StaticGraphTemporalSignalBatch 입력 데이터 준비
            총 130만 개의 그래프를 배치 사이즈(B) 단위로 묶어 처리 (볼록 대각선 행렬 방식)
        4. StaticGraphTemporalSignalBatch 객체 생성
        '''
        # 공통으로 들어가는 변수들 먼저 정의
        edge_index_np = self.edge_index_tem.detach().cpu().numpy()

        ad, dis = self.ad, self.dis
        num_nodes = len(ad)
        T = 37 # max LOS
        zero_vec = np.zeros((num_nodes,), dtype=np.float32)

        # 1. 개별 시계열 그래프 데이터 구성 및 패딩
        for i in tqdm(range(batch_size, self.xdf.shape[0] + 1, batch_size)): # 데이터프레임의 행 인덱스들을 위에서부터 배치 단위로 가져 옴
            idx_list = list(self.xdf.iloc[i - batch_size : i].index)
            idx_los = {i: int(self.xdf.loc[i]['LOS']) for i in idx_list}

            def vec_to_x(v):
                v = np.asarray(v, dtype=np.float32).reshape(-1, 1)
                return torch.as_tensor(v) # [60, 1]
            
            idx_ad_x = {
                idx: vec_to_x(self.xdf.loc[idx, ad].to_numpy())
                for idx in idx_list
            }

            idx_dis_x = {
                idx: vec_to_x(self.xdf.loc[idx, dis].to_numpy())
                for idx in idx_list
            }

        features_seq = []
        targets_seq  = []
        mask_seq     = []
        batch_vec_np = None

        for t in range(1, T + 1):
            data_list_t = []
            y_t = []
            for idx in idx_list:
                los = idx_los[idx]
                y_i = int(self.ysr.loc[idx])
                if t > los:
                    x = torch.tensor(zero_vec, dtype=torch.float32).unsqueeze(-1)
                elif t == los:
                    x = idx_dis_x[idx]
                else:
                    x = idx_ad_x[idx]
                data_list_t.append(Data(x=x, y=torch.as_tensor(y_i)))
                y_t.append(y_i)
            
            batch_t = Batch.from_data_list(data_list_t)
            features_seq.append(batch_t.x.detach().cpu().numpy())
            targets_seq.append(np.asarray(y_t, dtype=np.int64))
            if batch_vec_np is None:
                batch_vec_np = batch_t.batch.detach().cpu().numpy()
            
            # valid = 1 if any non-zero node exists (los >= t) else 0
            # Here, define valid by LOS: valid=1 if t <= los else 0
            mask_seq.append(np.asarray([1 if t <= idx_los[idx] else 0 for idx in idx_list], dtype=np.int64))

        signal = StaticGraphTemporalSignalBatch(
            edge_index=edge_index_np,
            edge_weight=None,
            features=features_seq,
            targets=targets_seq,          # graph-level labels per timestep
            batches=batch_vec_np,         # node→graph mapping (constant across t)
            mask=mask_seq                 # optional additional temporal feature
        )

        self.signal_list.append(signal)
        return self

            
            
def processing_static_main():
    print("loading initial data...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_initial_data(random_state=42)
    print("loading initial data done !!!")

    print("converting into graph: train dataset")
    train_data_bundle = DataBundle(X_train, y_train).get_graph_lists() # 메서드 체이닝
    print("train dataset done !!", '\n')

    print("converting into graph: validation dataset")
    val_data_bundle = DataBundle(X_val, y_val).get_graph_lists()
    print("validation dataset done !!", '\n')

    print("converting into graph: test dataset")
    test_data_bundle = DataBundle(X_test, y_test).get_graph_lists()
    print("test dataset done !!", '\n')

    return train_data_bundle.graph_list, val_data_bundle.graph_list, test_data_bundle.graph_list

def processing_temporal_main():
    print("loading initial data...(SAMPLED)")
    X_train, X_val, X_test, y_train, y_val, y_test = get_initial_data(random_state=42)
    print("loading initial data done !!!")

    print("converting into graph: train dataset")
    train_data_bundle = DataBundle(X_train, y_train).get_temporal_graph_batches() # 메서드 체이닝
    print("train dataset done !!", '\n')

    print("converting into graph: validation dataset")
    val_data_bundle = DataBundle(X_val, y_val).get_temporal_graph_batches()
    print("validation dataset done !!", '\n')

    print("converting into graph: test dataset")
    test_data_bundle = DataBundle(X_test, y_test).get_temporal_graph_batches()
    print("test dataset done !!", '\n')

    return train_data_bundle.signal_list, val_data_bundle.signal_list, test_data_bundle.signal_list
    

if __name__ == "__main__":
    result = processing_temporal_main()
    import pickle
    save_path = 'temporal_grpah_data.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    print("data saved !!")

    
