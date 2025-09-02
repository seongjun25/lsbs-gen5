import pandas as pd

# CSV 파일 불러오기
train_df = pd.read_csv("./data/houseprice/train.csv")
test_df = pd.read_csv("./data/houseprice/test.csv")
submit_csv = pd.read_csv("./data/houseprice/sample_submission.csv")

train_df.info()
test_df.info()
train_df["SalePrice"]
# test_df["SalePrice"]

train_df.head()
test_df.head()

# train_df에서 수치형 변수 선택
train_df = train_df.select_dtypes(
    include=["int64", "float64"]
    )

# NA 처리
# drop na말고 다른 값으로 채울 수 있는 칼럼 없어?
train_df=train_df.dropna()

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# 의미없는 변수들을 빼면 어떨까?
# 
# x와 y 설정
train_X = train_df.drop(["Id", "SalePrice"], axis=1)
train_y = train_df["SalePrice"]

# 모델 학습
model.fit(train_X, train_y)
model.coef_
model.intercept_

# test_df에서 수치형 변수 선택
test_df = test_df.select_dtypes(
    include=["int64", "float64"]
    )

# 모든 NA 값을 0으로 채움
# NA말고 다른 값으로 채울 수 있는 칼럼없어?
test_df=test_df.fillna(0)

# import numpy as np
# model.intercept_ + np.sum(model.coef_ * test_df.iloc[0,:])
result=model.predict(test_df)
result

# train_df 의 수치형변수를 선택해서 train_df 업데이트
# 모든 변수 사용해서 선형회귀분석 fit(계수찾기)
# test_df 의 정보를 사용해서 집값 예측
# 예측한 값을 사용해서 submit_csv의 집값 채우기

submit_csv["SalePrice"] = result
submit_csv

submit_csv.to_csv("./data/houseprice/output.csv",
                  index=False,
                  encoding="utf-8")