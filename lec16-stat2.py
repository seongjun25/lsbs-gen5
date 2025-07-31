import numpy as np

# 확률 변수 X의 가능한 값:
# 0 - 지지하지 않는다.
# 1 - 지지한다.
values = np.array([0, 1, 2])
probs = np.array([0.36, 0.48, 0.16])

# 조건에 따라 확률변수 X 생성
X = np.random.choice(values, size=333, p=probs)
X.mean()

exp_X = np.sum(values * probs)
exp_X


values = np.array([1, 2, 3, 4])
probs = np.array([0.1, 0.3, 0.2, 0.4])

exp_X = np.sum(values * probs)
exp_X

# 조건에 따라 확률변수 X 생성
X = np.random.choice(values, size=300, p=probs)
X.mean()

import matplotlib.pyplot as plt
# 히스토그램 그리기
plt.hist(X, bins=np.arange(1, 6)-0.5,
         density=True,
         edgecolor='black',
         rwidth=0.8, align='mid')
plt.xticks(values)
plt.xlabel("value")
plt.ylabel("count")
plt.title("hist")
# 빨간선 추가: 실제 확률 분포
plt.vlines(values, ymin=0, ymax=probs,
           colors='red', linewidth=2, label='True probabilities')




values = np.array([1, 2, 3, 4])
probs = np.array([0.1, 0.3, 0.2, 0.4])
E_X = np.sum(values * probs)
X = np.random.choice(values, size=30,
                     p=probs)
# np.sum((X - X.mean())**2) / (300-1)
# np.sum((X - X.mean())**2) / (300)
var_n1=X.var(ddof=1)
var_n=X.var()

# 위의 var_n1과 var_n을 1000번씩 발생 시킨 후, 
# 각각의 히스토그램을 그려보세요.
# 각 히스토그램의 중심점(표본평균)을 초록색 막대기로 표시
# 이론 분산값(1.09)을 빨간 막대기로 표시


import numpy as np
import matplotlib.pyplot as plt

# Given parameters
values = np.array([1, 2, 3, 4])
probs = np.array([0.1, 0.3, 0.2, 0.4])
E_X = np.sum(values * probs)
theoretical_var = np.sum((values - E_X)**2 * probs)

# Simulation parameters
n_samples = 30
n_trials = 1000

# Arrays to store variance estimates
var_n1_samples = np.empty(n_trials)
var_n0_samples = np.empty(n_trials)

# Run simulations
for i in range(n_trials):
    X = np.random.choice(values, size=n_samples, p=probs)
    var_n1_samples[i] = X.var(ddof=1)  # unbiased sample variance
    var_n0_samples[i] = X.var(ddof=0)  # population variance

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Histogram for unbiased sample variance (ddof=1)
axs[0].hist(var_n1_samples, bins=30, edgecolor='black')
axs[0].axvline(var_n1_samples.mean(), color='green', linestyle='--', linewidth=2, label=f"Sample mean = {var_n1_samples.mean():.2f}")
axs[0].axvline(theoretical_var, color='red', linestyle='-', linewidth=2, label=f"Theoretical var = {theoretical_var:.2f}")
axs[0].set_title("Histogram of Unbiased Sample Variance (ddof=1)")
axs[0].set_xlabel("Variance")
axs[0].set_ylabel("Frequency")
axs[0].legend()
axs[0].grid(True)

# Histogram for population variance (ddof=0)
axs[1].hist(var_n0_samples, bins=30, edgecolor='black')
axs[1].axvline(var_n0_samples.mean(), color='green', linestyle='--', linewidth=2, label=f"Sample mean = {var_n0_samples.mean():.2f}")
axs[1].axvline(theoretical_var, color='red', linestyle='-', linewidth=2, label=f"Theoretical var = {theoretical_var:.2f}")
axs[1].set_title("Histogram of Population Variance (ddof=0)")
axs[1].set_xlabel("Variance")
axs[1].set_ylabel("Frequency")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# (X - E_X)**2  # 확률변수

# 확률변수가 갖는 값과 확률은?
np.sum((values - E_X)**2 * probs)

# 균일분포
X = np.random.uniform(0, 1)
X



import numpy as np
from scipy.stats import uniform

# X~균일분포 U(2, 4)
a = 2     # 하한
b = 4     # 상한
X = uniform(loc=a, scale=b - a)
X.mean() # (a + b) / 2
X.var()  # (b - a)^2 / 12
x=X.rvs(size=100)
x

# P(X <= 3.5) 의 값
X.cdf(3.5)

# (3.2 - 2.1) * 0.5
# P(2.1 < X <= 3.2)
X.cdf(3.2) - X.cdf(2.1)


# PDF 시각화를 위한 x축 값
x_vals = np.linspace(1.5, 4.5, 500)  # 분포의 밖까지 포함해서 표현
pdf_vals = X.pdf(x_vals)

# PDF 그래프 그리기
plt.figure(figsize=(8, 4))
plt.plot(x_vals, pdf_vals, color='blue', label='PDF of U(2, 4)')
plt.fill_between(x_vals, pdf_vals, where=(x_vals >= a) & (x_vals <= b), alpha=0.3)

# 수직선: 평균 위치
plt.axvline(X.mean(), color='green', linestyle='--', label=f'Expected value (mean) = {X.mean():.2f}')

# 축 정보 및 제목
plt.title("균일분포 U(2, 4)의 확률밀도함수 (PDF)")
plt.xlabel("x")
plt.ylabel("밀도 (PDF)")
plt.legend()
plt.grid(True)
plt.show()


# 문제. 지수분포를 따르는 확률변수 X를 만들어
# 보세요. scale = 0.5
# X ~ exp(theta = 0.5)

import numpy as np
from scipy.stats import expon

# 지수분포 정의
X1 = expon(scale=0.5)
X2 = expon(scale=3)    

X1.cdf(6) - X1.cdf(2)
X2.cdf(6) - X2.cdf(2)

X1.mean()
X2.mean()

X1.var()
X2.var()

x1=X1.rvs(size=100)
x2=X2.rvs(size=100)
sum(x1 <= 2)
sum(x2 <= 2)

X1.ppf(0.2)

# x 값 범위
x_vals = np.linspace(0, 10, 500)

# 각각의 PDF 계산
pdf1 = X1.pdf(x_vals)
pdf2 = X2.pdf(x_vals)

# 그래프 그리기
plt.figure(figsize=(8, 4))
plt.plot(x_vals, pdf1, label='Exponential(scale=0.5)', color='blue')
plt.plot(x_vals, pdf2, label='Exponential(scale=3)', color='orange')
plt.xlabel("x")
plt.ylabel("밀도 (PDF)")
plt.legend()
plt.grid(True)
plt.show()


# 정규분포(loc=0, scale=1) 확률변수를
# X1 만들어보세요.
import numpy as np
from scipy.stats import norm

# 지수분포 정의
X1 = norm(loc=0, scale=1)
X1
# 1. 평균 계산
X1.mean() # loc
# 2. 분산 계산 Var(X)
X1.var()
# 표준편차
X1.std() # scale

# 3. pdf 그려보기 - 특징 유추
# x 값의 범위 생성
x = np.linspace(-4, 4, 1000)

# PDF 계산
pdf_values = X1.pdf(x)

# 그래프 그리기
plt.plot(x, pdf_values, label='정규분포 N(0,1)')
plt.title('표준 정규분포의 확률밀도함수(PDF)')
plt.xlabel('x')
plt.ylabel('확률밀도')
plt.grid(True)
plt.legend()
plt.show()

# 정규분포(loc=2, scale=3) 확률변수를
# X2 만들어보세요.
X2 = norm(loc=2, scale=3)
X2
# 4. loc, scale 패러미터의 효과 알아내기
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 정규분포 정의
X1 = norm(loc=0, scale=1)   # 평균 0, 표준편차 1
X2 = norm(loc=2, scale=3)   # 평균 2, 표준편차 3

# x 값의 범위 생성
x = np.linspace(-10, 10, 1000)

# PDF 계산
pdf_X1 = X1.pdf(x)
pdf_X2 = X2.pdf(x)

# 그래프 그리기
plt.plot(x, pdf_X1, label='X1: N(0, 1)', color='blue')
plt.plot(x, pdf_X2, label='X2: N(2, 3)', color='orange')
plt.title('정규분포 PDF 비교')
plt.xlabel('x')
plt.ylabel('확률밀도')
plt.grid(True)
plt.legend()
plt.show()

# 5. P(X <= 3) 값은?
X2.cdf(3)

# 6. P(1.5 < X < 5) 값은?
X2.cdf(5) - X2.cdf(1.5)
# 7. 표본 300개를 뽑아서 히스토그램을 그린 후,
# pdf와 겹쳐서 그려보기
x2=X2.rvs(size=300)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 정규분포 X2 정의
X2 = norm(loc=2, scale=3)

# 7. 표본 300개 추출
x2 = X2.rvs(size=300)

# x 값 범위 정의 (히스토그램과 PDF 겹치기 위한 범위)
x = np.linspace(-10, 15, 1000)

# PDF 계산
pdf_X2 = X2.pdf(x)

# 히스토그램 + PDF 겹쳐서 그리기
plt.hist(x2, bins=30, density=True, alpha=0.6, color='skyblue', label='표본 히스토그램')
plt.plot(x, pdf_X2, color='red', linewidth=2, label='이론적 PDF: N(2, 3)')
plt.title('정규분포 N(2, 3): 표본 히스토그램과 PDF 비교')
plt.xlabel('x')
plt.ylabel('밀도')
plt.legend()
plt.grid(True)
plt.show()
# 8. X2에서 나올 수 있는 값들 중, 상위 10%에
# 해당하는 값은?
X2.ppf(0.9)

# 표준 정규분포
# -1 에서 1사이 값이 나올 확률
X1.cdf(2) - X1.cdf(-2)
# X ~ N(2, 3^2)에서 
# (2 - 3*2, 2 + 3*2) 사이에 값이 나올 확률
X2.cdf(8) - X2.cdf(-4)