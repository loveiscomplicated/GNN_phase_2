# TODO 배치 단위로 데이터셋 저장하기, 즉 process랑 get 부분을 수정해야 함, 
# 이거 하기 전에 먼저 shape 어떻게 할 건지 생각하기
# processed_file_names도 달라져야 할 것
# 나머지 달라져야 하는 게 뭐가 있는지도 알아보기
# process, processed_file_name, get



import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Dataset

from utils.processing_utils import get_ad_dis_col
from utils.device_set import device_set

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

def get_graph_Data(ad_vector: torch.Tensor, dis_vector: torch.Tensor, y: torch.Tensor, los: int, device):
    '''
    Args:
        ad_vector (torch.Tensor): vector by integer location based on ad_tensor, shape of ad_tensor - [num_cases, num_variables]
        dis_vector (torch.Tensor): vector by integer location based on dis_vector, shape of dis_vector - [num_cases, num_variables]
        y (torch.Tensor): 1-element vector by integer location based on y_vector, shape of y_vector - [num_cases,]
        los (int): the los of given case
        device: gpu or cpu
    '''
    # padding을 위한 텐서 생성, shape: [max_los(=37) - los, num_nodes(=60)]
    zero_tensor = torch.zeros((37 - los, 60), dtype=torch.long, device=device)

    # admission을 가지고 los-1 번째 타임스탬프까지 채움
    timestamps = [ad_vector for _ in range(los - 1)]

    # discharge를 가지고 마지막 타임스템흐를 채움
    timestamps.append(dis_vector)

    # 리스트로 묶인 타임스탬프들을 텐서로 변환, shape: [los, num_nodes(=60)]
    timestamps_tensor = torch.stack(timestamps, dim=0)

    # 실제 데이터와 패딩을 합침, shape: [37(=max_los), 60=(num_nodes)] 
    # 이 shape으로 해야 엔티티 임베딩할 때 37개를 배치 단위로 인식하게 하여 한꺼번에 처리 가능
    time_padded = torch.concatenate((timestamps_tensor, zero_tensor), dim=0)

    # edge_index=None인 이유는 어차피 모든 데이터에 대해 동일하므로 나중에 따로 들고 있으면 됨
    # 굳이 모든 객체에 똑같은 걸 넣을 필요는 없음 - 공간 낭비
    return Data(x=time_padded, edge_index=None, y=y)


class TedsTemporalDataset(Dataset):
    NUM_GRAPH = 1_394_138
    BATCH_SIZE = 10_000
    device = device_set()
    

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
        num_files = (self.NUM_GRAPH + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        return tuple(f'data_{i}.pt' for i in range(num_files))
    
    def process(self):
        '''
        self.root가 비어 있는지 보고 processed_file_names와 파일 이름들을 비교해서 
        파일이 하나라도 없거나 폴더가 비어 있으면: process() 메서드를 자동으로 호출
        '''
        num_files = (self.NUM_GRAPH + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        print("전처리된 파일이 하나라도 없거나 폴더가 비어 있기 때문에 전처리를 시작합니다...")
        device = self.device
        df = pd.read_csv(os.path.join(self.raw_dir, 'missing_corrected.csv'))
        df = organize_labels(df)
        ad, dis = get_ad_dis_col(df)

        LOS_tensor = df_to_tensor(df['LOS']).to(device)
        y_tensor = df_to_tensor(df['REASONb']).to(device)
        ad_tensor = df_to_tensor(df[ad]).to(device)
        dis_tensor = df_to_tensor(df[dis]).to(device)

        data_list = []
        counter = 0
        for i in tqdm(range(self.NUM_GRAPH), desc='Processing Graphs'):
            graph_Data = get_graph_Data(ad_tensor[i, :], dis_tensor[i, :], y_tensor[i], LOS_tensor[i].item(), device=device)  # type: ignore
            data_list.append(graph_Data)

            if (i + 1) % self.BATCH_SIZE == 0 or i == self.NUM_GRAPH - 1:
                save_path = os.path.join(self.processed_dir, f'data_{counter}.pt')
                torch.save(data_list, save_path)
                print(f"> data_{counter}.pt saved / total {num_files}s has to be saved...")
                data_list = []
                counter += 1
            # self.processed_dir를 수동으로 할당할 필요 없이 
            # root에 processed라는 폴더가 있다면 알아서 그 경로를 self.processed_dir로 저장함
            # 만약 root에 processed라는 폴더가 있다면 자동으로 폴더 생성
        
    def get(self, idx):
        file_num = idx // self.BATCH_SIZE
        data_in_file_idx = idx % self.BATCH_SIZE

        file_path = os.path.join(self.processed_dir, f'data_{file_num}.pt')
        data_list = torch.load(file_path, weights_only=False)

        data = data_list[data_in_file_idx]

        return data
            
    def len(self):
        return self.NUM_GRAPH
    

################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
'''
결국 pyg Data / Batch를 생성해서 저장 또는 로드할 필요가 없었고, 오히려 불편함, 저장공간 낭비였음
37개의 타임스탬프를 미리 만들어 저장하는 것은 원본 데이터의 37배에 달하는 용량으로 뻥튀기가 되는 것
우리는 어차피 필요한 게 1. 입소 시, 2. 퇴소 시, 3. 제로 패딩 이기 때문에
1,2만 들고 엔티티 임베딩까지 한 뒤에
모델 입력하기 직전에 시계열 데이터로 변환하면 됨

이 모든 걸 텐서로만 진행할 수 있음 + pyg Data / Batch를 쓰면 기본 알고리즘 때문에 쉐입이 이상해지던가 객체를 생성해서 저장하기 때문에 용량이 더 커짐
'''

class TEDSTensorDataset()







if __name__ == "__main__":
    CURDIR = os.path.dirname(__file__)
    root = os.path.join(CURDIR, 'data_cache')
    dataset = TedsTemporalDataset(root)
    
    