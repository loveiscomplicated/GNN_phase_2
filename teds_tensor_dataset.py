'''
결국 pyg Data / Batch를 생성해서 저장 또는 로드할 필요가 없었고, 오히려 불편함, 저장공간 낭비였음
37개의 타임스탬프를 미리 만들어 저장하는 것은 원본 데이터의 37배에 달하는 용량으로 뻥튀기가 되는 것
우리는 어차피 필요한 게 1. 입소 시, 2. 퇴소 시, 3. 제로 패딩 이기 때문에
1,2만 들고 엔티티 임베딩까지 한 뒤에
모델 입력하기 직전에 시계열 데이터로 변환하면 됨

이 모든 걸 텐서로만 진행할 수 있음 + pyg Data / Batch를 쓰면 기본 알고리즘 때문에 쉐입이 이상해지던가 객체를 생성해서 저장하기 때문에 용량이 더 커짐
'''
import torch
from torch.utils.data import Dataset

class TEDSTensorDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)()
    
    def __len__(self):
        pass

if __name__ == "__main__":
    pass