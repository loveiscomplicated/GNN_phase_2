import os
import pandas as pd
import numpy as np
import torch
import torch_geometric
import torch_geometric_temporal


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
        3. 2단계: StaticGraphTemporalSignalBatch 입력 데이터 준비
            총 130만 개의 그래프를 배치 사이즈(B) 단위로 묶어 처리 (볼록 대각선 행렬 방식)
        4. StaticGraphTemporalSignalBatch 객체 생성
'''


