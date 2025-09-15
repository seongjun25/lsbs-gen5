import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/USArrests.csv')
print(df.head())

from sklearn.preprocessing import StandardScaler
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), 
                        columns = numeric_data.columns)
print(df_trans.head(2))

from sklearn.cluster import KMeans # K-평균 군집분석 불러오기
kmeans = KMeans(n_clusters = 4, 
                random_state = 1)
labels = kmeans.fit_predict(df_trans)
print(labels)


# 팔머 펭귄 데이터를 bill_length, bill_depth 변수사용
# kmeans 알고리즘으로 3개 그룹으로 분류해보세요.

from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins
