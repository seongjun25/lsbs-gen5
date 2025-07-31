import numpy as np
import pandas as pd

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5])  # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"])  # 문자형 벡터 생성
c = np.array([True, False, True, True])  # 논리형 벡터 생성

type(a)
# d = np.array(["q", 2, [1, 2]])
d = np.array(["q", 2])
d

a + 3
a * 20

# b = np.array([6, 7, 8, 9, 10])
b = a + 5
b

a + b

a**2

a
a.cumsum()

np.arange(4, 10, step=0.5)
a = np.arange(4, 10, step=0.5)
len(a)

# 1000이하의 7의 배수를
# 발생시켜보세요!
vec_a = np.arange(7, 1001, step=7)
vec_a

sum(vec_a)
vec_a.sum()

np.cumsum(vec_a)
vec_a.cumsum()


# pip install palmerpenguins
# 데이터로드
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins


vec_m=np.array(penguins["body_mass_g"])
vec_m.shape

np.max(vec_m)
np.min(vec_m)

vec_m.argmax()

vec_m.mean()

# 평균 4.2kg라고하면
# 4.2kg 몸무게보다 작은 펭귄들은 몇마리 인가?
sum(vec_m < 4200)

# 3.0kg 이상인 펭귄들 몇마리 인가요?
sum(vec_m >= 3000)
sum((vec_m >= 3000) & (vec_m < 4200))

# 인생 띵곡 & 애창곡 list
# 가수: 제목
{
    "샤샤슬롯": "Older",
    "김광석": "잊어야 한다는 마음으로",
    "다섯": "YA,YA",
    "SS501": "내 머리가 나빠서",
    "Lany": "It even rains in LA",
    "빅뱅": "거짓말",
    "IZI": "응급실",
    "윤수일": "아파트",
    "김현성": "헤븐",
    "fiftyfifty": "Pookie",
    "아일릿": "아몬드쵸콜렛(한국어버전)",
    "뽀로로": "바나나차차"
}

