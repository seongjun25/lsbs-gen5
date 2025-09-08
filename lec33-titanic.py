import pandas as pd

# 타이타닉 데이터 불러오세요!
train_df = pd.read_csv('./data/titanic/train.csv')
test_df = pd.read_csv('./data/titanic/test.csv')
train_df.columns
test_df.columns
# train_df = train_df.select_dtypes(include=['number'])

# 각 칼럼별 결측치 몇개가 있을까?
train_df.isna().sum(axis=0)

num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop("Survived")
cat_columns = train_df.select_dtypes(include=['object']).columns
cat_columns = cat_columns.drop(["Name", "Ticket", "Cabin"])

# 결측치 채우기 (간단히 처리)
from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')

train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
# freq_impute.statistics_
# mean_impute.statistics_

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
onehot = OneHotEncoder(handle_unknown='ignore',                        
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")

train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])
# train_df[num_columns].mean(axis=0)
# train_df[num_columns].std(axis=0, ddof=1)
# train_df_num.mean(axis=0)
# std_scaler.mean_

train_df_all = pd.concat([train_df_cat,
                          train_df_num], axis = 1)

train_df_all

# 독립변수(X)와 종속변수(y) 분리
# np.log(y_train).hist()
X_train = train_df_all
y_train = train_df["Survived"]

from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier(criterion='gini')
dct.get_params()

import numpy as np
dct_params = {'max_depth' : np.arange(1, 8),
              'ccp_alpha': np.linspace(0, 1, 5)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, 
           shuffle=True, 
           random_state=2025)

# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                          param_grid=dct_params, 
                          cv = cv, 
                          scoring='accuracy')
dct_search.fit(X_train, y_train)
