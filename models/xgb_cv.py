import os
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from scipy.stats import uniform, randint




CURDIR = os.path.dirname(__file__)
PARDIR = os.path.join(CURDIR, '..')
root = os.path.join(PARDIR, 'data_tensor_cache')
data_path = os.path.join(root, 'raw', 'missing_corrected.csv')

# 1. 데이터 불러오기
data = pd.read_csv(data_path)
X = data.drop(columns=['REASONb'])
y = data['REASONb']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. XGBoost 모델 인스턴스
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

def grid():
    # 3. 탐색할 하이퍼파라미터의 범위(Grid) 정의
    # 이 범위는 데이터와 상황에 따라 적절하게 설정해야 합니다.
    param_grid = {
        'n_estimators': [100, 200, 500],             # 부스팅 트리의 개수
        'max_depth': [3, 5, 7],                      # 트리의 최대 깊이
        'learning_rate': [0.01, 0.1, 0.3],           # 학습률
        'subsample': [0.8, 1.0]                      # 데이터 샘플링 비율
    }

    # 4. 평가 지표 설정 (GridSearchCV는 하나의 지표를 사용해 최적화를 수행)
    # 이진 분류에서는 ROC-AUC를 최적화 지표로 사용하는 것이 일반적입니다.
    scorer = make_scorer(roc_auc_score)

    # 5. StratifiedKFold 교차 검증 설정
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 6. GridSearchCV 객체 초기화 및 학습
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=scorer,       # 최적화할 평가 지표 (가장 높은 ROC-AUC를 찾음)
        cv=cv,                # 교차 검증 폴드 설정
        n_jobs=1,            # 모든 코어 사용 x
        verbose=3             # 진행 상황 출력
    )

    print("Grid Search 시작...")
    grid_search.fit(X_train, y_train)
    print("Grid Search 완료.")

    # 7. 최적의 결과 확인
    print("\n--- 최적의 하이퍼파라미터 조합 ---")
    print(grid_search.best_params_)

    print("\n--- 최고 ROC-AUC 점수 ---")
    print(f"{grid_search.best_score_:.4f}")

    # 8. 최적의 모델 추출
    best_xgb_model = grid_search.best_estimator_

def random():

    # 4. 평가 지표 설정 (GridSearchCV는 하나의 지표를 사용해 최적화를 수행)
    # 이진 분류에서는 ROC-AUC를 최적화 지표로 사용하는 것이 일반적입니다.
    scorer = make_scorer(roc_auc_score)


    # 탐색할 파라미터의 분포 정의 (Random Search용)
    param_distributions = {
        'n_estimators': randint(50, 500),         # 50부터 500 사이의 정수
        'max_depth': randint(3, 10),              # 3부터 10 사이의 정수
        'learning_rate': uniform(0.01, 0.3)       # 0.01부터 0.3 사이의 실수
    }

    # 5. StratifiedKFold 교차 검증 설정
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=20, # 시도할 조합의 개수 (예: 20개 조합만 테스트)
        scoring=scorer,
        cv=cv,
        n_jobs=1,
        random_state=42,
        verbose=3
    )

    print("Random Search 시작...")
    random_search.fit(X_train, y_train) 
    print("Random Search 완료.")

    # 7. 최적의 결과 확인
    print("\n--- 최적의 하이퍼파라미터 조합 ---")
    print(random_search.best_params_)

    print("\n--- 최고 ROC-AUC 점수 ---")
    print(f"{random_search.best_score_:.4f}")

    # 8. 최적의 모델 추출
    best_xgb_model = random_search.best_estimator_


print("What Method? Type 0 for grid, 1 for random")
method = input(": ")
if method == "0":
    grid()
elif method == "1":
    random()
else:
    raise Exception("Invalid Input")