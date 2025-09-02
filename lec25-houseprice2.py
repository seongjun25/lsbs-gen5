import pandas as pd

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/houseprice/train.csv')
test_df = pd.read_csv('./data/houseprice/test.csv')

train_df = train_df.select_dtypes(include=['number'])

# 결측치 제거 (간단히 처리)
train_df = train_df.dropna()

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df.drop(columns='SalePrice')
y_train = train_df['SalePrice']


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 모델 학습
lr.fit(X_train, y_train)
lr.coef_
lr.intercept_

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
test_df = test_df.select_dtypes(include=['number'])
test_df = test_df.fillna(train_df.mean())

# 예측
y_pred = lr.predict(test_df)
submit = pd.read_csv('./data/houseprice/sample_submission.csv')
submit["SalePrice"]=y_pred

# CSV로 저장
submit.to_csv('./data/houseprice/baseline.csv', index=False)


# 데이터 분할
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X_train,  
                   y_train, 
                   test_size = 0.2,                     
                   random_state = 0, 
                   shuffle = True, 
                   stratify = None 
                   )

# random_state=2025
# shuffle = True
# test size = 0.3
# 정보를 사용해서 
# train_df 를 train_X, test_X, train_y, test_y로
# 나눠보는 코드 작성. 넘파이 판다스 사용.

# 1) tr_df 총 행수 확인
# 2) 벡터가 주어졌을때 랜덤으로 순서를 섞는 함수 알아낼것
#    1:n 벡터를 섞어주세요!
# 3) 앞의 80% 인덱스는 train으로 뒤 20% 인덱스는 test로 설정
import numpy as np
# train_df.iloc[np.array([2, 3]), :]
n=train_df.shape[0]
idx_vec=np.arange(n)
np.random.shuffle(idx_vec)
k1=int(n*0.33)
k2=int(n*0.66)

df1_Xy=train_df.iloc[idx_vec[:k1], :]
df2_Xy=train_df.iloc[idx_vec[k1:k2], :]
df3_Xy=train_df.iloc[idx_vec[k2:], :]


# 모의고사 세트(validation set) 3개를 활용해서 모델
# 성능 점수를 계산

from sklearn.linear_model import LinearRegression
lr = LinearRegression()


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)

# RMSE 기준으로 판단해보자!



# valid_set
from sklearn.model_selection import train_test_split
train_X_sub, valid_X, train_y_sub, valid_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=1
)

