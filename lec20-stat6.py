# 팔머펭귄 데이터의
# 각 종별 부리길이의 사분위수를 계산하세요.

# pip install palmerpenguins
# 데이터로드
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins

adeli_bill=penguins.loc[penguins["species"] == "Adelie"]["bill_length_mm"]
gentoo_bill=penguins.loc[penguins["species"] == "Gentoo"]["bill_length_mm"]
chinstrap_bill=penguins.loc[penguins["species"] == "Chinstrap"]["bill_length_mm"]

def cal_quartile(data):
    import numpy as np
    sorted_data = np.sort(data) # 데이터 정렬
    median = np.median(sorted_data) # 중앙값
    lower_half = sorted_data[sorted_data < median] # 중앙값보다 크거나, 작은 데이터들 필터
    upper_half = sorted_data[sorted_data > median]
    q1 = np.median(lower_half) # 1사분위수와 3사분위수
    q3 = np.median(upper_half)
    print("Q1:", q1,
          "Q2:", np.round(median, 1),
          "Q3:", q3)

cal_quartile(adeli_bill)
cal_quartile(gentoo_bill)
cal_quartile(chinstrap_bill)

import numpy as np
np.quantile(adeli_bill, 0.25)
np.quantile(adeli_bill, 0.5)
np.quantile(adeli_bill, 0.75)

# 이상치 판별하는 방법
# 1Q에서 -1.5*IQR
# 3Q에서 +1.5*IQR
# 구간에서 벗어나는 데이터는 이상치로 판단

import numpy as np
import matplotlib.pyplot as plt

# 데이터 정의
data = np.array([155, 126, 10, 82, 115, 140,
                 73, 92, 110, 134])

# 박스플랏 그리기
plt.figure(figsize=(6, 4))
plt.boxplot(data, vert=True)
plt.title("Boxplot of Data")
plt.ylabel("Values")
plt.grid(True)

# 그래프 출력
plt.show()


# 데이터를 사용해서 박스 플랏을 그려보세요!
scores = np.array([88, 92, 95, 91, 87, 89,
                    94, 90, 92, 100, 43])

q1=np.quantile(scores, 0.25)
q3=np.quantile(scores, 0.75)
np.percentile(scores, 25)
iqr = q3 - q1

q3 + 1.5 * iqr
# 박스플랏 그리기
plt.figure(figsize=(6, 4))
plt.boxplot(scores, vert=True)
plt.title("Boxplot of Data")
plt.ylabel("Values")
plt.grid(True)

# 그래프 출력
plt.show()

data = np.array([155, 126, 27, 82, 115, 140, 73, 92, 110, 134])
sorted_data = np.sort(data)
n = len(data)

np.quantile(data, [0.25, 0.5, 0.75])
# np.percentile(data, [25, 50, 75])
# 넘파이 0.01 ~ 0.99
np.arange(0.01, 1, step=0.01)
data_q=np.quantile(data, np.arange(0.01, 1, step=0.01))

from scipy.stats import norm

data.mean()
data.std(ddof=1)
X = norm(loc = data.mean(),
         scale = data.std(ddof=1))
norm_q=X.ppf(np.arange(0.01, 1, step=0.01))

data_q
norm_q

import matplotlib.pyplot as plt
plt.scatter(data_q, norm_q)
plt.plot([25, 200], [25, 200],
          color='red', 
          label='y = x')
plt.xlabel('data')
plt.ylabel('theory')
plt.show()

import scipy.stats as sp
sp.probplot(data,
            dist="norm", plot = plt)
plt.show()


import numpy as np
import scipy.stats as sp
sorted_data = np.array([155, 126, 27, 82, 115, 140, 73, 92, 110, 134])
percentiles = np.array([sp.percentileofscore(sorted_data, value, kind='rank') for value in sorted_data])
percentiles


import scipy.stats as sp
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55,
3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9])
w, p_value = sp.shapiro(data_x)
print("W:", w, "p-value:", p_value)


from statsmodels.distributions.empirical_distribution import ECDF
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
11.97, 2.16, 3.24, 10.91, 11.36, 0.87])
ecdf = ECDF(data_x)

x = np.linspace(min(data_x), max(data_x))
y = ecdf(x)
plt.plot(x,y,marker='o', linestyle='none')
plt.title("Estimated CDF")
plt.xlabel("X-axis")
plt.ylabel("ECDF")

k = np.arange(min(data_x), max(data_x), 0.1)
plt.plot(k, norm.cdf(k, 
                     loc=np.mean(data_x),
                     scale=np.std(data_x, ddof=1)), 
                     color='red')
plt.show()


# K-S test
from scipy.stats import kstest, norm
import numpy as np

sample_data = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
11.97, 2.16, 3.24, 10.91, 11.36, 0.87])

# 표본 평균과 표준편차로 정규분포 생성
loc = np.mean(sample_data)
scale = np.std(sample_data, ddof=1)

# 정규분포를 기준으로 K-S 검정 수행
result = kstest(sample_data, 'norm', args=(loc, scale))
print("검정통계량:", result.statistic)

result.statistic
result.pvalue



from scipy.stats import anderson, norm
sample_data = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
11.97, 2.16, 3.24, 10.91, 11.36, 0.87])

result = sp.anderson(data, dist='norm') # Anderson-Darling 검정 수행
print('검정통계량',result[0], "\n",
      '임계값:',result[1], "\n",
      '유의수준:',result[2])
