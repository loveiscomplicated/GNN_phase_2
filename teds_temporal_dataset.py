import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Batch, Dataset

def split_ad_dis_df(df: pd.DataFrame):
    
    pass

def _process(raw_data_path):
    '''
    TedsTemporalDataset.process 내부에서 전처리 과정을 수행하는 함수
    Args:
        raw_data_path: 원본 데이터가 들어 있는 폴더 경로
    '''
    df = pd.read_csv(raw_data_path)
    df_np = df[:10].to_numpy()
    df_tensor_X = torch.tensor(df_np[:, :-1], dtype=torch.long)
    df_tensor_y = torch.tensor(df_np[:, -1], dtype=torch.long)
    
    return df_tensor_X, df_tensor_y

class TedsTemporalDataset(Dataset):
    NUM_GRAPH = 1_394_138
    def __init__(self, root):
        '''
        Args:
            root(str): 캐시된 데이터가 저장되어 있는 디렉토리 (폴더 경로)
            나머지는 안 만져도 됨 
        '''
        super().__init__(root)
        
    
    @property
    def processed_file_names(self):
        '''
        130만 개 가량의 파일 이름을 생성
        로직이 복잡하면서 속성으로 접근하면 좋으니까 @property 사용 
        (메소드를 속성처럼 사용할 수 있게 해줌)
        '''
        return tuple(f'data_{i}.pt' for i in range(self.NUM_GRAPH))
    
    def process(self):
        '''
        self.root가 비어 있는지 보고 processed_file_names와 파일 이름들을 비교해서 
        파일이 하나라도 없거나 폴더가 비어 있으면: process() 메서드를 자동으로 호출
        '''
        print("전처리된 파일이 하나라도 없거나 폴더가 비어 있기 때문에 전처리를 시작합니다...")

        raw_data_path = os.path.join(self.root, 'raw', 'missing_corrected.csv')
        X, Y = _process(raw_data_path)
        # X는 

        for i in tqdm(range(self.NUM_GRAPH), desc='Processing Graphs'):
            x = X[i, ]
            y = Y[i]
            graph = Data(x=x, edge_index=None, y=y)
            torch.save(graph, os.path.join(self.processed_dir, f'data_{i}.pt')) 
            # self.processed_dir를 수동으로 할당할 필요 없이 
            # root에 processed라는 폴더가 있다면 알아서 그 경로를 self.processed_dir로 저장함
            # 만약 root에 processed라는 폴더가 있다면 자동으로 폴더 생성
        
    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        
    def len(self):
        return self.NUM_GRAPH
    


if __name__ == "__main__":
    CURDIR = os.path.dirname(__file__)
    root = os.path.join(CURDIR, 'data_cache')
    dataset = TedsTemporalDataset(root)
    