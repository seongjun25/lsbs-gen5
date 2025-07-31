import matplotlib.pyplot as plt
plt.plot([4, 1, 3, 2],
         marker='o', linestyle='None')
plt.ylabel('Some Numbers')
plt.show()

# x-y plot (산점도)
plt.plot([1, 2, 3, 4], [1, 4, 9, 16],
         marker='o',
         linestyle='None')
plt.show()

import numpy as np
plt.plot(np.arange(10),
         np.arange(10),
         marker='o',
         linestyle='None')
plt.show()


# 펭귄데이터 불러오자
import pandas as pd
df = pd.read_csv('./data/penguins.csv')
df.info()

# 부리 길이(x축) vs. 부리 깊이(y축)
# bill_length_mm vs. bill_depth_mm
plt.plot(df["bill_length_mm"],
         df["bill_depth_mm"],
         marker='o',
         linestyle='None',
         color = "red")
plt.xlabel('bill_length_mm')
plt.ylabel('bill_depth_mm')
plt.show()

plt.scatter(df["bill_length_mm"],
            df["bill_depth_mm"],
            # c = "red",
            c = np.repeat("red", 344))
plt.xlabel('bill_length_mm')
plt.ylabel('bill_depth_mm')
plt.show()

# 100 - 녹색
# 244 - 빨간색
x=np.repeat("green", 100)
y=np.repeat("red", 244)
my_color=np.concatenate([x, y])

# df["species"]의 값에 따라서
# 아델리면 "red"
# 친스트랩 "blue"
# 겐터면 "green"
# 색깔 벡터를 만들어 보세요.
# 종에 따라 색상 매핑
color_map = {
    "Adelie": "red",
    "Chinstrap": "blue",
    "Gentoo": "green"
}
# 색깔 벡터 생성
color_vector = df["species"].map(color_map)
color_vector

plt.scatter(df["bill_length_mm"],
            df["bill_depth_mm"],
            c = color_vector)
plt.xlabel('bill_length_mm')
plt.ylabel('bill_depth_mm')
plt.show()

# 숫자 벡터
x=np.repeat(0, 100)
y=np.repeat(1, 100)
z=np.repeat(2, 144)
my_color=np.concatenate([x, y, z])

plt.scatter(df["bill_length_mm"],
            df["bill_depth_mm"],
            c = my_color)
plt.xlabel('bill_length_mm')
plt.ylabel('bill_depth_mm')
plt.show()

# 카데고리 변수에 대하여
df['species'] = df['species'].astype('category')
df.info()

df['species'].cat.categories
df['species'].cat.codes

plt.scatter(df["bill_length_mm"],
            df["bill_depth_mm"],
            c = df['species'].cat.codes)
plt.xlabel('bill_length_mm')
plt.ylabel('bill_depth_mm')
plt.show()


# 서브 플랏
names = ['A', 'B', 'C']
values = [1, 10, 100]
plt.figure(figsize=(9, 3))
plt.subplot(231)
plt.bar(names, values)  # 막대 그래프
plt.subplot(235)
plt.scatter(names, values)  # 산점도
plt.subplot(233)
plt.plot(names, values)  # 선 그래프
plt.suptitle('Categorical Plotting')
plt.show()

# 한글 표현
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.plot([1, 2, 3, 4], [10, 20, 30, 40])
plt.text(2, 25, # 특정 위치에 텍스트 추가
        '한글 테스트 입니다!', 
        fontsize=18, 
        color='red')
plt.show()


plt.plot([1, 2, 3, 4], 
         [1, 4, 9, 16], 
        label="y = x^2")  
plt.title("Example Plot") # 재목 설정 
plt.xlabel("X Axis") # 축 라벨 
plt.ylabel("Y Axis") # 축 라벨  
plt.legend(loc="upper left") # 범례 표시
plt.show()


# 펭귄데이터 불러오자
import pandas as pd
df = pd.read_csv('./data/penguins.csv')
df.info()
df["island"].unique()
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# 종별 평균 부리 길이 계산 및 내림차순 정렬
mean_bill_length = df.groupby('species')['bill_length_mm'].mean().sort_values(ascending=False)

# 색상 지정: Chinstrap은 빨간색, 나머지는 회색
colors = ['red' if species == 'Chinstrap' else 'gray' for species in mean_bill_length.index]

# 막대그래프 그리기
plt.bar(mean_bill_length.index, mean_bill_length.values, color=colors)
plt.xlabel('펭귄 종류')
plt.ylabel('평균 부리 길이 (mm)')
plt.title('펭귄 종류별 평균 부리 길이 (내림차순)')

# 막대 위에 값 표시
for idx, value in enumerate(mean_bill_length.values):
    plt.text(idx, value + 0.1, f'{value:.1f} mm', ha='center', va='bottom', fontsize=10)

plt.show()

# 섬별 평균 몸무게 막대그래프
# 가장 평균 몸무게 작은 섬 (파란색)
# 나머지 회색처리
# 그려보세요~!

# 펭귄 종별 부리 길이(x) vs 깊이(y) 산점도
# 한글 제목, x, y축 제목 설정
# 아델리 - 빨간색
# 친스트랩 - 회색
# 겐투 - 회색
# 범례: 오른쪽 하단 위치
# 아델리 평균 중심점 표시
# 점 찍고 텍스트로 아래와 같이 출력
# (평균 부리길이: xx.xx mm,
#  평균 부리깊이: xx.xx mm)
# 종별 색상 지정: 아델리 - 빨강, 나머지 회색
# 색상 맵핑: 아델리만 빨강, 나머지는 회색
color_map = {'Adelie': 'red', 'Chinstrap': 'gray', 'Gentoo': 'gray'}
df['color'] = df['species'].map(color_map)

# 산점도 그리기
plt.figure(figsize=(8, 6))

# 빨간색(아델리) 점
adelie = df[df['species'] == 'Adelie']
plt.scatter(adelie['bill_length_mm'], adelie['bill_depth_mm'], c='red', label='아델리')

# 회색(기타종) 점
others = df[df['species'] != 'Adelie']
plt.scatter(others['bill_length_mm'], others['bill_depth_mm'], c='gray', label='기타종')

# 아델리 평균점
adelie_mean = adelie[['bill_length_mm', 'bill_depth_mm']].mean()

# 평균 중심점 X 마커 (범례 미포함)
plt.scatter(adelie_mean['bill_length_mm'], 
            adelie_mean['bill_depth_mm'], 
            color='black', s=100, marker='x')

# 텍스트 위치 설정 (왼쪽하단), 화살표 연결
text_x = adelie_mean['bill_length_mm'] - 8
text_y = adelie_mean['bill_depth_mm'] - 4
plt.annotate(
    f'(평균 부리길이: {adelie_mean["bill_length_mm"]:.2f} mm,\n 평균 부리깊이: {adelie_mean["bill_depth_mm"]:.2f} mm)',
    xy=(adelie_mean['bill_length_mm'], adelie_mean['bill_depth_mm']),  # 시작점 (X표 위치)
    xytext=(text_x, text_y),  # 텍스트 위치
    textcoords='data',
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=14,
    ha='left'
)

# 축 제목, 제목, 범례
plt.xlabel('부리 길이 (mm)')
plt.ylabel('부리 깊이 (mm)')
plt.title('펭귄 종별 부리 길이 vs 부리 깊이 산점도')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# 연속적인 값을 가지는 변수들을 쪼개서
# 범주형 변수로 변환시키는 테크닉
# 몸무게

min_weight = df['body_mass_g'].min()
max_weight = df['body_mass_g'].max()

print(f"몸무게 범위: {min_weight}g ~ {max_weight}g")

# 몸무게 범위를 3등분한 구간 경계 설정
bins = [2700.0, 3900.0, 5100.0, 6300.0]  # 2700~3900, 3900~5100, 5100~6300

# 범주형 변수 생성
df['body_mass_cat'] = pd.cut(df['body_mass_g'],
                             bins=bins, 
                             labels=['low', 'middle', 'high'])
df['body_mass_cat']

df.info()

# 각 범주별 펭귄 마리 수 계산
count_by_category = df['body_mass_cat'].value_counts().sort_index()

count_by_category.sum()

# 막대그래프 그리기
plt.figure(figsize=(6, 4))
plt.bar(count_by_category.index,
        count_by_category.values, 
        color='skyblue',
        alpha=0.5)
plt.xlabel('몸무게 범주')
plt.ylabel('펭귄 마리 수')
plt.title('몸무게 범주별 펭귄 수')
plt.grid(axis='y',
         linestyle='--',
         alpha=1)
plt.show()


# 파이차트

# 섬 목록과 한글 이름 매핑
island_labels = {
    'Torgersen': '토거센(Torgersen)',
    'Biscoe': '비스코(Biscoe)',
    'Dream': '드림(Dream)'
}

# 종별 색상 지정 (아델리: 빨강, 친스트랩: 초록, 겐투: 파랑)
species_colors = {
    'Adelie': '#e74c3c',     # 빨간 계열
    'Chinstrap': '#2ecc71',  # 초록 계열
    'Gentoo': '#3498db'      # 파란 계열
}

# 서브플롯 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('각 섬별 펭귄 종 서식 현황', fontsize=24)

# 각 섬에 대해 파이차트 그리기
for i, island in enumerate(island_labels.keys()):
    data = df[df['island'] == island]['species'].value_counts()
    labels = data.index
    sizes = data.values
    colors = [species_colors[sp] for sp in labels]
    
    wedges, texts, autotexts = axes[i].pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white'},
        textprops={'fontsize': 20}
    )
    
    axes[i].set_title(island_labels[island], fontsize=16)
    axes[i].axis('equal')

# 범례 추가 (오른쪽 하단)
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Adelie', markerfacecolor=species_colors['Adelie'], markersize=15),
    plt.Line2D([0], [0], marker='o', color='w', label='Chinstrap', markerfacecolor=species_colors['Chinstrap'], markersize=15),
    plt.Line2D([0], [0], marker='o', color='w', label='Gentoo', markerfacecolor=species_colors['Gentoo'], markersize=15)
]
fig.legend(handles=handles, title='펭귄 종', loc='lower right', fontsize=14, title_fontsize=16)

# 레이아웃 조정
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 전체 타이틀과 범례 공간 확보
plt.show()


# 펭귄 종별 성비를 나타내는 파이차트
# 그려보세요!




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 불러오기
df = pd.read_csv('./data/penguins.csv')

# 변수 및 라벨
features = ['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'flipper_length_mm']
feature_labels = ['부리길이', '부리깊이', '몸무게', '날개 길이']

# 결측치 제거
df_clean = df.dropna(subset=features + ['species'])

# 종별 평균 계산
grouped = df_clean.groupby('species')[features].mean()

# 등수 계산 (값이 클수록 1등)
ranked = grouped.rank(method='min', ascending=False)

# 1등 → 3, 2등 → 2, 3등 → 1로 점수화
score_transformed = 4 - ranked

# -------------------------------
# 레이더 차트 설정
# -------------------------------
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 도형 닫기용

colors = {'Adelie': '#e74c3c', 'Chinstrap': '#2ecc71', 'Gentoo': '#3498db'}

# 차트 생성
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.suptitle('펭귄 종별 특성 등수 점수 비교 (1등 → 3점)', fontsize=16, y=1.1)

for species in score_transformed.index:
    values = score_transformed.loc[species].tolist()
    values += values[:1]
    
    ax.plot(angles, values, label=species, color=colors[species])
    ax.fill(angles, values, alpha=0.25, color=colors[species])

# 각 변수 이름을 축에 배치
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), feature_labels, fontsize=13)

# r축 설정
ax.set_rlabel_position(0)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['1점 (3등)', '2점 (2등)', '3점 (1등)'], fontsize=10)
ax.tick_params(colors='#777777')

# 범례 추가
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=12)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
df = pd.read_csv('./data/penguins.csv')

# 변수 및 라벨 설정
features = ['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'flipper_length_mm']
feature_labels = ['부리길이', '부리깊이', '몸무게', '날개 길이']

# 결측치 제거
df_clean = df.dropna(subset=features + ['species'])

# 종별 평균 계산
grouped = df_clean.groupby('species')[features].mean()

# -----------------------------
# 등수 계산 (높을수록 1등, 낮을수록 3등)
# -----------------------------
# rank(method='min', ascending=False): 값이 클수록 높은 등수
ranked = grouped.rank(method='min', ascending=False)

# -----------------------------
# 정규화: 등수 1~3 → 1등은 가장 높게 보이도록 뒤집어서 시각화
# 예: 1등 → 1.0, 3등 → 0.0
# -----------------------------
ranked_norm = 1 - (ranked - 1) / (ranked.max() - 1)

# -----------------------------
# 레이더 차트용 설정
# -----------------------------
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 닫기용 첫 점 다시 추가

# 색상 지정
colors = {'Adelie': '#e74c3c', 'Chinstrap': '#2ecc71', 'Gentoo': '#3498db'}

# 레이더 차트 그리기
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.suptitle('펭귄 종별 특성 등수 기반 비교 (1등 = 더 큼)', fontsize=16, y=1.1)

for species in ranked_norm.index:
    values = ranked_norm.loc[species].tolist()
    values += values[:1]
    
    ax.plot(angles, values, label=species, color=colors[species])
    ax.fill(angles, values, alpha=0.25, color=colors[species])

# 축 라벨 설정
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), feature_labels, fontsize=13)

# r축 설정
ax.set_rlabel_position(0)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(['3등', '2등', '1등'], fontsize=10)
ax.tick_params(colors='#777777')

# 범례
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=12)

plt.tight_layout()
plt.show()
