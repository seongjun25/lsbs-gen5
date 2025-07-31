import pandas as pd

# 데이터 프레임 생성
df = pd.DataFrame({
    'col1': ['one', 'two', 'three', 'four', 'five'],
    'col2': [6, 7, 8, 9, 10]
})
print(df)
df
df.shape

df["col1"]

my_df = pd.DataFrame({
    'name': ['issac', 'bomi'],
    'birthmonth': [5, 4]
})
my_df
my_df.info()



import pandas as pd
url = "https://bit.ly/examscore-csv"
mydata = pd.read_csv(url)
print(mydata.head())

mydata.shape


mydata["gender"].head()
mydata["gender"].tail()

mydata[["midterm", "final"]].head()
mydata[mydata["midterm"] > 15].head()

# iloc - 숫자사용 indexing
mydata.iloc[:,1]
mydata.head()
mydata.iloc[1:5, 2]
mydata.iloc[1:4,:3]

mydata.iloc[:, 1].head()
mydata.iloc[:, [1]].head()
mydata.iloc[:, [1]].squeeze()

mydata.iloc[:, [1, 0, 1]].head()

# 라벨 인덱싱 loc[]
# T & F 필터링 가능
mydata.loc[:, "midterm"]
mydata.loc[:, 2]
mydata.loc[1:4, "midterm"]

mydata[mydata['midterm'] <= 15].head()
mydata.loc[mydata['midterm'] <= 15, "gender"]
mydata.loc[mydata['midterm'] <= 15, ["gender", "student_id"]]


mydata['midterm'].isin([28, 38, 52])
# 중간고사 점수 28, 38, 52인 애들의
# 기말고사 점수와 성별 정보 가져오세요!

mydata.loc[mydata['midterm'].isin([28, 38, 52]), ["final", "gender"]]

import numpy as np
check_index=np.where(mydata['midterm'].isin([28, 38, 52]))[0]
mydata.iloc[check_index, [3, 1]]

# 일부 데이터를 NA로 설정
mydata.iloc[0, 1] = np.nan
mydata.iloc[4, 0] = np.nan

mydata.head()

mydata["gender"].isna().sum()

mydata.dropna()

# 1번
mydata["student_id"].isna().head()

# 2번
vec_2=~mydata["student_id"].isna()

# 3번
vec_3=~mydata["gender"].isna()

mydata[vec_2 & vec_3]

mydata['total'] = mydata['midterm'] + mydata['final']
mydata.head()

mydata["average"] = (mydata['total'] / 2).rename("average")
mydata.head()

mydata["average^2"] = mydata["average"]**2
mydata.head()


mydata.head()

del mydata["average^2"]

mydata.head()
mydata.rename(columns={"student_id": "std-id"},
              inplace=True)

mydata.head()

# concat
df1 = pd.DataFrame({
'A': ['A0', 'A1', 'A2'],
'B': ['B0', 'B1', 'B2']
})

df2 = pd.DataFrame({
'A': ['A3', 'A4', 'A5'],
'B': ['B3', 'B4', 'B5']
})
result = pd.concat([df1, df2],
                   ignore_index=True)
result

df4 = pd.DataFrame({
'A': ['A2', 'A3', 'A4'],
'B': ['B2', 'B3', 'B4'],
'C': ['C2', 'C3', 'C4']
})

pd.concat([df1, df4],
           join='outer')
import pandas as pd
df = pd.read_csv('./data/penguins.csv')
df.info()

# Q1. bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g 중에서 
# 결측치가 하나라도 있는 행은 몇 개인가요?
n=df.iloc[:, 2:6].shape[0]
filled_n=df.iloc[:, 2:6].dropna().shape[0]
n-filled_n

# Q2. 몸무게(body_mass_g)가 4000g 이상 5000g 
# 이하인 펭귄은 몇 마리인가요?
cond1=df["body_mass_g"] >= 4000
cond2=df["body_mass_g"] <= 5000
sum(cond1 & cond2)

df["body_mass_g"].between(4000, 5000).sum()

# Q3. 펭귄 종(species)별로 최대 부리 길이(bill_length_mm)는
# 어떻게 되나요?
df["species"].unique()
df.loc[df["species"] == 'Adelie', "bill_length_mm"].mean() 
df.loc[df["species"] == 'Chinstrap', "bill_length_mm"].mean() 
df.loc[df["species"] == 'Gentoo', "bill_length_mm"].mean() 
# Q4. 성별(sex)이 결측치가 아닌 데이터 중, 
# 성별 비율은 각각 몇 퍼센트인가요?


# Q5. 섬(island)별로 평균 날개 길이(flipper_length_mm)가 
# 가장 긴 섬은 어디인가요?
mean_vec=df.groupby('island')["flipper_length_mm"].mean()
mean_vec.index[mean_vec.argmax()]


# 1~12조 별 발표
# 답안 코드는 배운 내용으로 구성할 것


df.describe()


df.sort_values(by='bill_length_mm')
df.sort_values(by=['bill_length_mm', 'flipper_length_mm'])
df.sort_values(by='bill_length_mm', ascending=False)
df.sort_values(by=['bill_length_mm', 'flipper_length_mm'],
               ascending=[True, False])


