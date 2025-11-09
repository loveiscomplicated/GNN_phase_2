
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch

from processing_utils import get_initial_data, get_initial_data_sampled, fully_connected_edge_index, get_col_dims

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

def get_batch_index(current_batch_size: int, num_nodes=60) -> np.ndarray:
    '''
    현재 배치의 사이즈에 맞춰서 batch index (torch_geometric.data.Batch.batch)를 수동으로 생성합니다.
    (더 빠름, 기존에는 Data 객체를 일일이 생성한 뒤, 이를 Batch 객체에 init해야 함,
    이건 넘파이로만 조작하기 때문에 더 빠름)
    
    Args:
        current_batch_size (int): "현재" batch_size. 마지막 배치는 사이즈가 달라질 수 있음
        num_nodes (int, Optional): 각 그래프에 포함된 노드의 수. default=60
        
    Returns:
        np.ndarray: PyTorch Geometric의 batch index 배열. size: (current_batch_size * num_nodes,)
    '''
    
    batch_indices = []
    for i in range(current_batch_size):
        # i 번째 그래프에 해당하는 노드 인덱스 배열 (길이: num_nodes, 값: i)
        index_array = np.full((num_nodes,), i, dtype=np.int64)
        batch_indices.append(index_array)
        
    # 2. 모든 인덱스 배열을 하나로 합칩니다.
    # np.concatenate는 리스트/튜플 형태의 배열 시퀀스를 첫 번째 인자로 받습니다.
    if batch_indices:
        batch_index = np.concatenate(batch_indices, axis=0)
    else:
        # 엣지 케이스 처리 (배치 사이즈가 0인 경우)
        batch_index = np.array([], dtype=np.int64)

    return batch_index

class DataBundle:
    def __init__(self, xdf:pd.DataFrame, ysr:pd.Series):
        '''
        Args:
            x: X_train, X_val, X_test 중 하나
            y: y_train, y_val, y_test 중 하나
        '''
        self.xdf = xdf
        self.ysr = ysr

        # temporal data
        self.ad, self.dis = get_ad_dis_col(self.xdf)
        self.edge_index_tem = fully_connected_edge_index(len(self.ad))
        self.col_dims_tem = get_col_dims(self.xdf[self.ad])
        self.signal_list = []
    
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
            zero_vec = np.zeros((num_nodes,), dtype=np.int64)

            # 1. 개별 시계열 그래프 데이터 구성 및 패딩
            for i in tqdm(range(0, self.xdf.shape[0], batch_size)): # 데이터프레임의 행 인덱스들을 위에서부터 배치 단위로 가져 옴
                end = min(i + batch_size, self.xdf.shape[0])
                idx_list = list(self.xdf.iloc[i : end].index)
                idx_los = {i: int(self.xdf.loc[i]['LOS']) for i in idx_list}

                def vec_to_x(v):
                    v = np.asarray(v, dtype=np.int64).reshape(-1, 1)
                    return torch.as_tensor(v, dtype=torch.long) # [60, 1]
                
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
                batch_vec_np = get_batch_index(len(idx_list), num_nodes=num_nodes)

                for t in range(1, T + 1):
                    data_list_t = []
                    y_t = []
                    for idx in idx_list:
                        los = idx_los[idx]
                        y_i = int(self.ysr.loc[idx])
                        if t > los:
                            x = torch.tensor(zero_vec, dtype=torch.float32).unsqueeze(-1)
                        elif t == los: # discharge
                            x = idx_dis_x[idx]
                        else: # admission
                            x = idx_ad_x[idx]
                        data_list_t.append(Data(x=x, y=torch.as_tensor(y_i)))
                        y_t.append(y_i)
                    
                    batch_t = Batch.from_data_list(data_list_t)
                    features_seq.append(batch_t.x.detach().cpu().numpy()) # type: ignore
                    targets_seq.append(np.asarray(y_t, dtype=np.int64))
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

def processing_temporal_main():
    print("loading initial data...(SAMPLED)")
    X_train, X_val, X_test, y_train, y_val, y_test = get_initial_data_sampled(size=1000, random_state=42)
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
    save_path = 'Sampled_temporal_graph_data_fully_connected.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    print("data saved !!")

    
