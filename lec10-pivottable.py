import pandas as pd
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}
df = pd.DataFrame(data)
print(df)

df_melted = pd.melt(
    df, 
    id_vars='Date',
    value_vars=['Temperature', 'Humidity'],
    var_name='측정요소', 
    value_name='측정값')
df_melted

# 원래 형식으로 변환
# Date, Temperature, Humidity
df_pivot=pd.pivot_table(
    df_melted,
    index="Date",
    columns="측정요소",
    values="측정값",
    aggfunc="count"
).reset_index()
df_pivot.columns.name = None
df_pivot


# 학생성적 데이터
df = pd.read_csv('./data/dat.csv')
print(df.head())

df.columns

# 칼럼명 변경하기
df = df.rename(columns = {'Dalc' : 'dalc', 'Walc' : 'walc'})

df.info()

df.loc[:, ['famrel', 'dalc']].astype({'famrel' : 'object', 'dalc' : 'float64'})
df.info()


def classify_famrel(famrel):
    if famrel <= 2:
        return 'Low'
    elif famrel <= 4:
        return 'Medium'
    else:
        return 'High'
    
classify_famrel(1)    
classify_famrel(4)    
classify_famrel(5)
df["famrel"]

df=df.assign(famrel = df['famrel'].apply(classify_famrel))
df


df.select_dtypes('number')
df.select_dtypes('object')

import numpy as np
def standardize(x):
    return ((x - np.nanmean(x))/np.std(x))

vec_a=np.arange(5)
vec_a

standardize(vec_a)

df.select_dtypes('number').apply(standardize)

df_std=df.select_dtypes('number').apply(standardize)
df_std.mean(axis=0)
df_std.std(axis=0)


index_f=df.columns.str.startswith('f')
df.loc[:, index_f].head()


df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [90, 85, 88]
})

df.to_csv("mydata.csv",
          index=False)


import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins()

penguins

# 문제 1: 펭귄 종별 평균 부리 길이 구하기
penguins.groupby("species")["bill_length_mm"].mean()
penguins.pivot_table(
    index="species",
    values="bill_length_mm",
    aggfunc="mean"
).reset_index()

# 문제 2: 섬별 몸무게 중앙값 구하기
penguins.pivot_table(
    index="island",
    values="body_mass_g",
    aggfunc="median"
).reset_index()

# 문제 3: 성별에 따른 부리 길이와 몸무게 평균 구하기
penguins.columns
penguins.pivot_table(
    index=["sex", "species"],
    values=["bill_length_mm", "body_mass_g"],
    aggfunc="mean"
).reset_index()

# 문제 4: 종과 섬에 따른 평균 지느러미 길이 구하기
penguins.pivot_table(
    index=["species", "island"],
    values="flipper_length_mm",
    aggfunc="mean",
    dropna=False
).reset_index()

penguins.pivot_table(
    index="species",
    columns="island",
    values="flipper_length_mm",
    aggfunc="count",
    dropna=False,
    fill_value="개체수 없음"
).reset_index()

# 문제 6
# 종별 몸무게의 변동 범위(Range) 구하기
def my_range(vec_x):
    return np.max(vec_x) - np.min(vec_x)
# my_range(penguins["body_mass_g"])

penguins.pivot_table(
    index="species",
    values="body_mass_g",
    aggfunc=my_range
).reset_index()
