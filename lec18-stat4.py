from scipy.stats import norm

X = norm(loc=5, scale=3)

sample = X.rvs(size=17)
sample.round(0)
sample.mean()


# 동전을 던져서 앞면이 나온 수

X = norm(loc=14, scale=2.227)
X.ppf(0.025)
X.ppf(0.975)

# 데이터가 오른쪽과
# 같을때 모평균에 
# 대한 90% 신뢰구간
# 을 구하세요,
# (단, 모분산은 9) 
# array([14., 17., 12., 14., 13., 14., 16.,
#        10., 14., 15., 13., 12., 16.,
#        17., 12., 12., 16.])
import numpy as np
x=np.array([14., 17., 12., 14., 13., 14., 16.,
       10., 14., 15., 13., 12., 16.,
       17., 12., 12., 16.])
n=len(x)

# 방법 1
x.mean() # 파란벽돌
sd= 3 / np.sqrt(n)

X = norm(loc=x.mean(),
         scale=sd)
X.ppf(0.05)
X.ppf(0.95)

# 방법 2
# X_bar +- z_0.05 * sigma / sqrt(n)
z_05 = norm.ppf(0.025, loc=0, scale=1)

x.mean() + z_05 * 3 / np.sqrt(n)
x.mean() - z_05 * 3 / np.sqrt(n)





# 95% 신뢰구간의 의미 
# 1. X_i ~ 모분포는 균일분포 (3, 7)로 설정
# 2. i = 1, 2, ..., 20
# 3. Xi들에서 표본을 하나씩 뽑은 후 표본평균을 계산하고,
#    95% 신뢰구간을 계산 (모분산값 1번 정보로 사용)
# 4. 3번의 과정을 1000번 실행해서(즉, 1000개의 신뢰구간 발생)
# 각 신뢰구간이 모평균을 포함하고 있는지 체크해보세요.

# # 시각화-심화
# 5. 이론적인 표본평균의 분포의 pdf를 그리고, 모평균을 
#    빨간색 막대기를 사용해서 표현
# 6. 3번에서 뽑은 표본들을 x축에 녹색점들로 표현 한 후,
#    95% 신뢰구간을 녹색 막대기 2개 표현
# 7. 표본이 바뀔때마다 녹색 막대기 안에 빨간 막대기가 
#    존재하는지 확인


# Z_0.05
X = norm(loc=0,
         scale=1)
X.ppf(0.95) # 약 1.65



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# 자유도 설정
df = 1  # Change to other values like 1, 10, 30 for comparison

# x 범위 설정
x = np.linspace(-5, 5, 1000)

# 확률밀도 계산
t_pdf = t.pdf(x, df)
normal_pdf = norm.pdf(x)

# 그래프 그리기
plt.plot(x, t_pdf, label=f"t-distribution (df={df})", color='blue', linestyle='--')
plt.plot(x, normal_pdf, label="Standard Normal Distribution", color='red')

# 그래프 설정
plt.title("Comparison of t-distribution and Standard Normal Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()


# 표본 데이터
data = [4.3, 4.1, 5.2, 4.9, 5.0, 4.5, 4.7, 4.8, 5.2, 4.6]

from scipy.stats import t

# 표본 평균
mean = np.mean(data)
# 표본 크기
n = len(data)
# 표준 오차
se = np.std(data, ddof=1) / np.sqrt(n)


mean - t.ppf(0.975, n-1) * se
mean + t.ppf(0.975, n-1) * se

ci = t.interval(0.95, loc=mean, scale=se, df=n-1)
ci

# H0: mu = 7 vs. HA: mu != 7
x = [4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
11.97, 2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9]
t_value = (np.mean(x)-7) / (np.std(x, ddof=1) / np.sqrt(len(x)))
round(t_value, 3)

# p-value
t.cdf(t_value, df=14) * 2

# 유의수준: p-value가 큰지, 작은지 판단하는 기준
# 따라서 0.2216은 유의수준보다 크므로, 귀무가설을
# 기각 하지 못한다.


# 문제 1
norm.sf(510, loc = 500, scale = 5)

# 문제 2
from scipy.stats import binom
Y = binom(n = 20, p = 0.05)
Y.pmf(2)
Y.cdf(2)
1 - Y.cdf(2) # P(X >= 3)
Y.sf(2) # P(X > 2)

t.cdf(-1.087, df = 9) * 2

(53 - 50) / (8 / np.sqrt(40))

2 * norm.sf(2.3717, loc = 0, scale = 1)