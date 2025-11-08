import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data

from processing_utils import get_initial_data, get_initial_data_sampled, fully_connected_edge_index, get_col_dims


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



if __name__ == "__main__":
    result = processing_static_main()
    import pickle
    save_path = 'temporal_grpah_data.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    print("data saved !!")

    
