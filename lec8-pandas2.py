import pandas as pd
df = pd.read_csv('./data/penguins.csv')
df.info()

df.describe()
df.sort_values("bill_length_mm")

# groupby()
result=df.groupby("species")["bill_length_mm"].mean()
# idxmax()
# result.index[result.values.argmax()]
result.idxmax()

result.values.argmax()

# 예제 데이터 프레임 생성
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})

# 두 데이터 프레임 병합
merged_df = pd.merge(df1, df2, 
                     on='key',
                     how='inner')
print(merged_df)

merged_df = pd.merge(df1, df2, 
                     on='key',
                     how='outer')
print(merged_df)


mid_df1 = pd.DataFrame({'std_id': [1, 2, 4, 5],
                        'midterm': [10, 20, 40, 30]})
fin_df2 = pd.DataFrame({'std_id': [1, 2, 3, 6],
                        'final': [4, 5, 7, 2]})

# 중간 & 기말 둘다 본 학생들의 데이터를 만들어보세요.
midfin_df = pd.merge(mid_df1, fin_df2, 
                    #  on= 'std_id',
                     left_on='std_id',
                     right_on='student_id',
                     how='inner')
del midfin_df["student_id"]
midfin_df

# 학생들 전체의 중간 & 기말 데이터를 만들어보세요.
total_df = pd.merge(mid_df1, fin_df2, 
                     on='std_id',
                     how='outer')
total_df

# merge how = "left"
# 가장 빈번하게 사용됨
left_df = pd.merge(mid_df1, fin_df2,
                   on='std_id',
                   how='left')
left_df

# df = pd.DataFrame({'name': ["홍길동", "김민정", "이삭", "홍길동"],
#                    'loc': ["서울", "춘천", "안성", "대구"],
#                    'score': [10, 20, 40, 30]})

# 판다스 melt()
wide_df = pd.DataFrame({
    '학생': ['철수', '영희', '민수'],
    '수학': [90, 80, 70],
    '영어': [85, 95, 75]
})
wide_df


long_df=pd.melt(wide_df, 
        id_vars='학생', # 기준칼럼
        var_name='과목',
        value_name='점수')
long_df


# wide 형식의 출석 데이터
w_df = pd.DataFrame({
    '반': ['A', 'B', 'C'],
    '1월': [20, 18, 22],
    '2월': [19, 20, 21],
    '3월': [21, 17, 23]
})

pd.melt(w_df,
        id_vars="반",
        var_name="월",
        value_name="출석일수")

#    반   월  출석일수
# 0  A  1월     20
# 1  B  1월     18
# 2  C  1월     22
# 3  A  2월     19
# 4  B  2월     20
# 5  C  2월     21
# 6  A  3월     21
# 7  B  3월     17
# 8  C  3월     23

# id_vars 두개
w_df = pd.DataFrame({
    '학년': [1, 1, 2],
    '반': ['A', 'B', 'C'],
    '1월': [20, 18, 22],
    '2월': [19, 20, 21],
    '3월': [21, 17, 23]
})

pd.melt(w_df,
        id_vars=["학년", "반"],
        var_name="월",
        value_name="출석일수")

# submelt()
df3 = pd.DataFrame({
    '학생': ['철수', '영희', '민수'],
    '국어': [90, 80, 85],
    '수학': [70, 90, 75],
    '영어': [88, 92, 79],
    '학급': ['1반', '1반', '2반']
})
pd.melt(df3,
        id_vars=["학급", "학생"],
        var_name="언어과목",
        value_vars=["국어", "영어"],
        value_name="성적")

df = pd.read_csv('./data/grade.csv')
df.info()
df.head()

# average열 생성
df["average"] = (df["midterm"] + df["final"] + df["assignment"])/3
result=df.iloc[:,2:].groupby("gender", as_index=False).mean()
result=pd.melt(result,
        id_vars="gender",
        var_name="variable",
        value_name="score").sort_values("gender")
result


# 판다스 pivot_table()
long_df=pd.melt(wide_df, 
        id_vars='학생', # 기준칼럼
        var_name='과목',
        value_name='점수')
long_df

pd.pivot_table(long_df,
        index="학생",
        columns="과목",
        values="점수")

result_df=long_df.pivot_table(
    index="학생",
    columns="과목",
    values="점수"
).reset_index()
result_df.columns.name = None
result_df



import pandas as pd
import nycflights13 as flights

# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

# 예시: 항공편 데이터 확인
print(df_flights.head())