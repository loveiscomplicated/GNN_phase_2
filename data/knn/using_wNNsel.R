library(wNNSel)
setwd('/Users/jeong-yunseong/Documents/programming/practiceLabForVAMwithGNN/Phase_2/data')

# 1. 원본 데이터 로드
df <- read.csv('data_dum_nan.csv')
print("원본 데이터 로드 완료.")

# 2. 데이터의 일부 (10,000개 행)를 무작위로 샘플링
print("샘플링 시작...")
n_rows <- nrow(df)
sample_indices <- sample(1:n_rows, 5000) # 100 -> 10000으로 수정
data_sample <- df[sample_indices, ]

# 3. *샘플 데이터*로 cv.wNNsel 실행 (메모리 문제 해결)
print("샘플 데이터로 교차 검증 시작...")
# *** 중요 ***: 샘플 데이터를 matrix로 변환해야 합니다.
data_sample_matrix <- as.matrix(data_sample) 

# 샘플 matrix를 함수에 전달
cv_results <- cv.wNNSel(data_sample_matrix)
print("교차 검증 완료.")

# 4. 샘플에서 찾은 최적의 파라미터 확인
# *** 중요 ***: 'cv_results_sample'이 아니라 'cv_results'를 사용해야 합니다.
lambda_opt <- cv_results$lambda.opt
m_opt <- cv_results$m.opt

print(paste("찾아낸 최적 Lambda:", lambda_opt))
print(paste("찾아낸 최적 m:", m_opt))

# 5. 이 파라미터를 *전체* 데이터에 적용하여 결측치 대체
print("전체 데이터에 결측치 대체 시작...")

# *** 중요 ***: wNNSel.impute 함수도 data.frame이 아닌 matrix가 필요합니다.
# 원본 df를 matrix로 변환 (이미 3단계에서 하려던 것)
data_matrix <- as.matrix(df)

final_imputed_data <- wNNSel.impute(data_matrix, lambda = lambda_opt, m = m_opt)
print("결측치 대체 완료.")

# 6. (선택 사항) 결과를 다시 data.frame으로 변환
final_imputed_df <- as.data.frame(final_imputed_data)
