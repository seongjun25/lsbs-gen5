import pandas as pd
df = pd.read_csv('./data/bike_data.csv')
df = df.astype({'datetime' : 'datetime64[ns]', 'weather' : 'int64', 
                'season' : 'object', 'workingday' : 'object', 
                'holiday' : 'object'})

df_sub = df.loc[df.season == 1, ]
# 시간 정보 추출
df_sub.loc[:, 'hour'] = df_sub['datetime'].dt.hour

# count가 가장 큰 hour 찾기
max_count_hour = df_sub.loc[df_sub['count'].idxmax(), 'hour']
max_count = df['count'].max()
print(f"count가 가장 큰 hour는 {max_count_hour}시이며, 대여량은 {max_count}입니다.")