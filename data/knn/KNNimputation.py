import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sympy import N
from tqdm import tqdm
import itertools
from tqdm import tqdm
import itertools

CURDIR = os.path.dirname(__file__)
print(CURDIR)
DATA_PATH = os.path.join(CURDIR, 'missing_corrected.csv')
DATA = pd.read_csv(DATA_PATH)

# 범주형 변수 더미화 시 train/test 간 불일치를 방지하기 위해
# 스플릿 전에 전체 데이터 기준으로 가능한 범주를 정의하고 CategoricalDtype으로 고정
for col in DATA.columns:
    cats = list(DATA[col].unique())
    DATA[col] = DATA[col].astype(pd.api.types.CategoricalDtype(categories=cats))

def train_val_test_split(DATA:pd.DataFrame):
    # validation set, test set이 imputation 설계하는 데 들어가면 안 됨, 일종의 사후판단 정보가 들어갈 수 있기 때문
    y = DATA['REASONb']
    X = DATA.drop('REASONb', axis=1)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    # X_train만 가지고 imputation 모델 설계, 이후 X_val, X_test에 해당 모델 적용
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_data_dum(X_train):
    df = X_train.copy()
    df = df.where(df != -9, np.nan)
    for col in df.columns:
        try:
            df[col] = df[col].cat.remove_categories(-9)
        except ValueError:
            continue
    return df

def get_mask(df):
    mask = ~df.isna()
    data_dum = pd.get_dummies(df, dtype=int)
    dum_temp = data_dum.copy()

    for target_col in mask.columns:
        mask_specified = mask[target_col]
        target_col = target_col + '_'
        object_cols = [i for i in data_dum.columns if target_col in i]
        dum_temp.loc[mask_specified, object_cols] = np.nan

    mask_dum = ~dum_temp.isna() # 결측일 때 False가 되게
    mask_dum = mask_dum.astype(int)

    return mask, mask_dum

def get_pearson_dum(data_dum):
    '''pearson_dum = data_dum.corr()
    pearson_dum.to_csv("pearson_dum.csv")
    pearson_dum'''

    if not os.path.exists('pearson_dum.csv'):
        pearson_dum = data_dum.corr()
        pearson_dum.to_csv("pearson_dum.csv")
    pearson_dum = pd.read_csv("pearson_dum.csv", index_col=0)
    pearson_dum = pearson_dum.fillna(0)
    # 5시간 걸려서 만든 데이터프레임에 결측치가 있었음 -> 일단 0으로 채우고 진행
    return pearson_dum

def get_prefix(col):
    """
    더미 변수 컬럼명(예: 'STFIPS_1')에서 
    원본 변수명(예: 'STFIPS')을 접두사로 추출합니다.
    """
    
    # .rsplit('_', 1)은 문자열을 *오른쪽(뒤)*에서부터 *첫 번째* '_'를 기준으로 1번만 분리합니다.
    # 예: 'FREQ_ATND_SELF_HELP_D_1' -> ['FREQ_ATND_SELF_HELP_D', '1']
    # 예: 'STFIPS_51' -> ['STFIPS', '51']
    # 예: 'METRO_D_1' -> ['METRO_D', '1']
    parts = col.rsplit('_', 1)
    
    # 만약 '_'가 없는 컬럼명(예: 'REASONb')이라 분리가 안 된 경우,
    # parts는 [col]이 됩니다. 이 경우 컬럼명 전체를 반환합니다.
    if len(parts) == 1:
        return col
    
    # 분리된 경우, 첫 번째 조각(접두사)을 반환합니다.
    prefix = parts[0]
    return prefix

def get_prefix_dict(data_dum):
    cols = data_dum.columns
    if not cols.any():
        return {}
        
    prefix_dict = {}
    dict_elem = []
    cur_prefix = get_prefix(cols[0])
    
    for count, col in enumerate(cols):
        new_prefix = get_prefix(col)
        
        if new_prefix == cur_prefix:
            dict_elem.append(count)
        else:
            prefix_dict[cur_prefix] = dict_elem
            cur_prefix = new_prefix
            dict_elem = [count] 
    if dict_elem:
        prefix_dict[cur_prefix] = dict_elem
        
    return prefix_dict

def gaussian(distance, tau):
    if tau == 0:
        u = np.divide(distance, tau, out=np.zeros_like(distance), where=tau!=0)
    else:
        u = distance / tau
    constant = 1.0 / np.sqrt(2.0 * np.pi)
    K_u = constant * np.exp(-0.5 * u**2)
    return K_u

# (get_prefix, get_prefix_dict, gaussian, get_distance 함수는 셀 88, 89, 90에 정의되어 있다고 가정)
# [수정] get_distance는 prefix_dict를 인자로 받도록 변경
def get_distance(w, a_v_prefix, a_i, a_j, mask_dum, pearson_dum, data_dum, prefix_dict):
    '''
    a_v_prefix: 원래 데이터에서 결측치가 속한 변수명 (예: 'STFIPS')
    a_i: 결측치가 속한 행 인덱스
    a_j: 거리를 재고자 하는 행 인덱스
    prefix_dict: 미리 계산된 접두사 맵
    '''
    mask_i = mask_dum.loc[a_i]
    mask_j = mask_dum.loc[a_j]
    mask_common = mask_i * mask_j

    a_ij = np.sum(mask_common)
    
    if a_ij == 0:
        return np.inf # 공동 관측치가 없으면 무한대 거리

    vec_i = data_dum.loc[a_i]
    vec_j = data_dum.loc[a_j]
    vec_diff = (vec_i - vec_j) ** 2

    # [수정] a_v_prefix (예: 'STFIPS')를 키로 사용
    r_vec = np.mean(pearson_dum.iloc[prefix_dict[a_v_prefix]], axis=0)
    r_vec = np.abs(r_vec) ** w

    # 0으로 나누기 방지
    if a_ij == 0:
        return np.inf
        
    return np.sqrt((1 / a_ij) * np.sum(vec_diff * mask_common * r_vec))

# [수정] weighting_scheme은 prefix_dict를 인자로 받도록 변경
def weighting_scheme(w, tau, a_v_prefix, target_idx, mask_dum, pearson_dum, data_dum, prefix_dict):
    dist_list = []
    
    # [수정] target_idx가 아닌 행들만 가져옴
    other_indices = data_dum.index.drop(target_idx)
    
    for i in other_indices:
        dist = get_distance(w, a_v_prefix, target_idx, i, mask_dum, pearson_dum, data_dum, prefix_dict)
        dist_list.append(dist)

    dist_np = np.array(dist_list)
    
    # 0으로 나누기 방지
    if np.sum(dist_np) == 0:
         # 모든 거리가 0이면 균등 가중치
        return np.full_like(dist_np, 1.0 / len(dist_np))

    weights = gaussian(dist_np, tau)
    
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        # 가우시안 결과가 0 (거리가 너무 멈)이면, 거리의 역수에 비례하는 가중치 사용
        inv_dist = 1.0 / (dist_np + 1e-9) # 1e-9는 0으로 나누기 방지
        weights = inv_dist / np.sum(inv_dist)
    else:
        weights = weights / sum_weights
        
    return weights, other_indices

# [수정] impute_cell (오류 수정)
def impute_cell(w, tau, a_v_prefix, target_idx, mask_dum, pearson_dum, data_dum, prefix_dict):
    '''
    a_v_prefix: 결측 변수의 접두사 (예: 'STFIPS')
    '''
    
    # 1. [수정] prefix_dict에서 해당 접두사의 더미 컬럼 인덱스를 가져옴
    target_dummy_indices = prefix_dict.get(a_v_prefix)
    
    if target_dummy_indices is None:
        raise KeyError(f"접두사 '{a_v_prefix}'를 prefix_dict에서 찾을 수 없습니다.")
        
    # 2. [수정] 컬럼 이름 리스트 생성
    col_list = data_dum.columns[target_dummy_indices]
    
    # 3. [수정] target_idx를 제외한 가중치와 인덱스 계산
    weights, other_indices = weighting_scheme(w, tau, a_v_prefix, target_idx, mask_dum, pearson_dum, data_dum, prefix_dict)

    # 4. [수정] other_indices (target_idx가 제외된)를 사용하여 데이터 슬라이싱
    sliced_df = data_dum.loc[other_indices, col_list]
    sliced_m = sliced_df.to_numpy() # (N-1, k) 배열

    # 5. [수정] (N-1, k) * (N-1, 1) -> (N-1, k)
    weighted_sliced_m = sliced_m * weights[:, np.newaxis]
    
    vec = np.sum(weighted_sliced_m, axis=0) # (k,) 벡터
    
    # 6. [수정] 확률 벡터 표준화
    sum_vec = np.sum(vec)
    if sum_vec > 0:
        vec = vec / sum_vec
    else:
        # 모든 가중 합이 0이면 (예: 모든 이웃이 0), 첫 번째 범주 선택
        vec[0] = 1 

    # 7. [수정] 컬럼 이름(col_list)에서 argmax를 찾고 접두사 제거
    imputed_col_name = col_list[np.argmax(vec)]
    
    # 8. [수정] 접두사 제거 로직 (get_prefix 논리 활용)
    prefix_len = len(a_v_prefix)
    if a_v_prefix.endswith('_D'): # 'METRO_D' 같은 경우
        result = imputed_col_name[prefix_len+1:] # 'METRO_D_1' -> '1'
    else: # 'STFIPS' 같은 경우
        result = imputed_col_name[prefix_len+1:] # 'STFIPS_51' -> '51'
        
    return int(result)

# [수정] impute (최종)
def impute(w, tau, mask_dum, pearson_dum, data_dum, mask, raw_df):
    df = raw_df.copy()
    
    # 1. [최적화] prefix_dict를 한 번만 계산
    prefix_dict = get_prefix_dict(data_dum)
    
    # 2. [최적화] 결측치가 있는 행만 순회
    missing_rows = mask[mask.eq(0).any(axis=1)]
    
    for i in tqdm(missing_rows.index, desc=f"Imputing {len(missing_rows)} rows", leave=False): # i = 결측이 있는 행 인덱스
        target_vec = mask.loc[i]
        target_cols = target_vec[target_vec==0].index # 결측인 변수명 리스트 (예: ['STFIPS', 'METRO_D'])
        
        for col_prefix in target_cols: # col_prefix = 결측인 변수명 (예: 'STFIPS')
            
            # 3. [수정] prefix_dict를 인자로 전달
            value_impute = impute_cell(w, tau, col_prefix, i, mask_dum, pearson_dum, data_dum, prefix_dict)
            
            # 4. [수정] 정확한 셀에 대치
            df.loc[i, col_prefix] = value_impute 
        
    return df

def create_cv_sample(X_train_original, sample_size, num_to_hide):
    """
    X_train에서 'sample_size'만큼의 행을 샘플링한 후,
    'num_to_hide' 개수만큼의 인공 결측값을 생성합니다.

    Args:
        X_train_original (pd.DataFrame): -9가 포함된 원본 학습 데이터 (예: temp_df)
        sample_size (int): CV에 사용할 총 행의 수 (예: 5000)
        num_to_hide (int): 성능 평가에 사용할 인공 결측값의 수 (예: 1000)

    Returns:
        X_cv_sample (pd.DataFrame): (sample_size, P) 크기의 DF, 원본 NaN + 인공 NaN 포함
        true_values (dict): 인공 결측값의 정답. {(row_label, col_name): true_value}
    """
    
    print(f"원본 {len(X_train_original)}개 행에서 {sample_size}개 행을 샘플링합니다...")
    # 1. 원본 데이터에서 sample_size만큼 행을 샘플링
    X_train_sample = X_train_original.sample(n=sample_size, random_state=42)
    
    # 2. -9를 np.nan으로 변환
    X_train_nan = X_train_sample.where(X_train_sample != -9, np.nan)
    
    # 3. 샘플 내에서 관측된(결측이 아닌) 값들의 위치를 찾음
    mask_observed = X_train_nan.notna()

    # (row_idx, col_idx) 튜플의 리스트로 변환, 즉 관측된 값의 위치 좌표
    observed_indices_tuples = list(zip(*np.where(mask_observed.values)))

    num_observed = len(observed_indices_tuples)
    
    if num_observed < num_to_hide:
        raise ValueError(
            f"샘플 내 관측값({num_observed})이 숨기려는 값({num_to_hide})보다 적습니다."
        )

    print(f"샘플 내 {num_observed}개의 관측값 중 {num_to_hide}개를 인공 결측값으로 만듭니다...")
    
    # 4. 숨길 인덱스를 무작위로 샘플링 (튜플 리스트의 인덱스를 샘플링)
    np.random.seed(42)
    indices_to_hide_flat = np.random.choice(len(observed_indices_tuples), num_to_hide, replace=False)
    # (r_idx, c_idx) 튜플 리스트
    indices_to_hide = [observed_indices_tuples[i] for i in indices_to_hide_flat]

    true_values = {}
    X_cv_sample = X_train_nan.copy()
    
    # 5. 정답(true_values)을 저장하고, 해당 위치를 NaN으로 변경
    for r_idx, c_idx in indices_to_hide:
        # iloc[r_idx]를 통해 실제 DataFrame 라벨(인덱스명)을 가져옴
        r_label = X_cv_sample.index[r_idx]
        c_name = X_cv_sample.columns[c_idx]
        
        # 정답 저장
        true_values[(r_label, c_name)] = X_cv_sample.iloc[r_idx, c_idx]
        
        # 인공 결측값 생성
        X_cv_sample.iloc[r_idx, c_idx] = np.nan
        
    return X_cv_sample, true_values

def prepare_data_for_imputation(X_cv_nan):
    """
    CV용 데이터(NaN 포함)를 받아 impute 함수에 필요한 모든 구성요소를 반환합니다.
    (주피터 노트북의 셀 86, 87의 로직을 따릅니다.)
    """
    
    # 1. 원본 mask 생성 (인공+원본 NaN 기준)
    # mask.eq(0)인 위치가 대치 대상이 됨
    mask_cv = X_cv_nan.notna().astype(int) # 1=관측, 0=결측
    
    # 2. 더미 변수화 (NaN은 무시됨, 즉 모든 더미가 0이 됨)
    data_dum_cv = pd.get_dummies(X_cv_nan, dtype=int)
    
    # 3. 더미 마스크(mask_dum) 생성
    dum_temp = data_dum_cv.copy()
    
    # 원본 변수가 NaN이었던 행의 모든 관련 더미 변수를 NaN으로 설정
    for target_col in mask_cv.columns:
        # mask_cv[target_col] == 0 인 행 (즉, 결측인 행)
        missing_rows_mask = (mask_cv[target_col] == 0)
        
        # 해당 변수의 더미 컬럼들 (예: 'STFIPS_1', 'STFIPS_2'...)
        object_cols = [c for c in data_dum_cv.columns if c.startswith(target_col + '_')]
        
        if object_cols:
            # 결측인 행의 해당 더미 변수들을 np.nan으로 설정
            dum_temp.loc[missing_rows_mask, object_cols] = np.nan
            
    # 최종 더미 마스크 (True = 관측됨(0 또는 1), False = 결측(NaN))
    mask_dum_cv = dum_temp.notna().astype(int)
    
    # 4. Pearson 상관관계 계산 (NaN은 0으로 채움)
    # 논문 [cite: 1286]에서는 상관관계 계산을 위해 1차 대치(예: 5-NN)를 제안했으나,
    # 여기서는 계산 편의상 NaN을 0으로 채운 후 상관관계를 계산합니다.
    pearson_dum_cv = data_dum_cv.corr().fillna(0)
    
    return data_dum_cv, mask_dum_cv, pearson_dum_cv, mask_cv

def run_cross_validation_sampled(
    X_train_original, 
    w_grid, 
    tau_grid, 
    sample_size, 
    num_to_hide
):
    """
    w와 tau 그리드 서치를 '샘플링된' 데이터로 수행합니다.
    """
    
    # 1. CV용 샘플 데이터 생성 (인공 결측 + 정답셋)
    # (셀 69의 temp_df를 X_train_original로 전달)
    X_cv_sample, true_values = create_cv_sample(
        X_train_original, 
        sample_size=sample_size, 
        num_to_hide=num_to_hide
    )
    
    # 2. CV용 샘플 데이터 전처리 (이 과정은 샘플 크기(N)만큼만 소요됨)
    print("CV 샘플 데이터 전처리를 시작합니다...")
    # data_dum_cv는 (sample_size, 1133) 크기
    data_dum_cv, mask_dum_cv, pearson_dum_cv, mask_cv = prepare_data_for_imputation(X_cv_sample)
    print("CV 샘플 데이터 전처리 완료.")
    
    cv_results = {} # (w, tau) : pfc

    # 3. 그리드 서치
    param_grid = list(itertools.product(w_grid, tau_grid))
    for w, tau in tqdm(param_grid, desc="Cross-validation Grid Search"):
        
        # 4. 대치 실행 (sample_size 크기의 데이터로 실행됨)
        df_imputed = impute(w, tau, mask_dum_cv, pearson_dum_cv, data_dum_cv, mask_cv, X_cv_sample)
        
        # 5. PFC (오분류 비율) 계산 (num_to_hide 개수만큼만 비교)
        num_false = 0
        for (r_label, c_name), true_val in true_values.items():
            imputed_val = df_imputed.loc[r_label, c_name]
            
            # 타입 변환 후 비교
            if str(imputed_val) != str(true_val):
                num_false += 1
        
        pfc = num_false / len(true_values)
        cv_results[(w, tau)] = pfc

    # 6. 최적 파라미터 찾기
    best_params = min(cv_results, key=cv_results.get)
    best_pfc = cv_results[best_params]
    
    print("\n--- 교차 검증 완료 ---")
    print(f"최적 파라미터 (w, tau): {best_params}")
    print(f"최소 PFC: {best_pfc}")
    
    return best_params, cv_results

# --- 실행 예시 ---
# (셀 69의 'temp_df'를 사용)
w_grid = [1]
tau_grid = [0.5, 1.0, 2.0]

# (전체 97만 행 대신) 5000개 행을 샘플링하고, 
# (전체 관측치의 10% 대신) 1000개의 셀만 숨겨서 테스트
# best_params, cv_results = run_cross_validation_sampled(
#     temp_df, 
#     w_grid=w_grid, 
#     tau_grid=tau_grid,
#     sample_size=5000,
#     num_to_hide=1000
# )
X_train = train_val_test_split(DATA)[0]
temp_df = get_data_dum(X_train)


best_params, cv_results = run_cross_validation_sampled(
    temp_df, 
    w_grid=w_grid, 
    tau_grid=tau_grid,
    sample_size=5000,
    num_to_hide=1000
)

