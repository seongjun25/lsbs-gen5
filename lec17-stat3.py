import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

X = bernoulli(p=0.3)
X.mean()
X.var()

X = bernoulli(p=0.4)
X.pmf(0)
X.pmf(1)
# 이항분포 확률변수
from scipy.stats import binom

X = binom(n=7, p=0.5)
X.mean()
X.pmf(0)
X.pmf(1)
X.pmf(2)
X.pmf(3)
X.pmf(4)
X.pmf(5)+ X.pmf(6) + X.pmf(7)
# 1 - P(X <= 4)
1 - X.cdf(4)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# 이항분포 정의
n = 5
p = 0.3
X = binom(n=n, p=p)

# x 값과 PMF 계산
x_vals = np.arange(0, n + 1)
pmf_vals = X.pmf(x_vals)

# PMF 시각화
plt.figure(figsize=(8, 4))
markerline, stemlines, baseline = plt.stem(x_vals, pmf_vals, basefmt=" ")
plt.setp(markerline, color='blue', marker='o')
plt.setp(stemlines, color='blue')
plt.title("이항분포 PMF: Binomial(n=5, p=0.3)")
plt.xlabel("x")
plt.ylabel("P(X = x)")
plt.xticks(x_vals)
plt.grid(True)
plt.show()


from scipy.special import comb

# nCr 계산: 예) 5C2
comb(5, 3)
# P(X=3)
comb(5, 3) * (0.3**3) * (0.7**2)

# 앞면이 나올 확률이 0.6인 동전을
# 10번 던져서 앞면이 5번 나올 확률은?
X ~ B(n=10, p=0.6)
P(X = 5)=?

comb(10, 5) * (0.6**5) * (0.4**5)


# 이항분포 확률변수
from scipy.stats import binom

X = binom(n=5, p=0.3)
X.mean()
X.var()

Y = bernoulli(p=0.3)
sum(Y.rvs(size=5)) # B(5, 0.3)

30 * 29 * 28 / 6


from scipy.stats import poisson
# λ = 2, x = 3
# prob = poisson.pmf(3, mu=2)
# prob
# X 0, 1, 2, 3, ...  정수 형태 값을 갖는
# 이산형 확률변수
X = poisson(mu = 4)
1 - (X.pmf(0) + X.pmf(1))
1 - X.cdf(1)

X = poisson(mu = 3.5)
X.cdf(2)
X.mean()
X.var()

import matplotlib.pyplot as plt
from scipy.stats import bernoulli
x = [0, 1]
p = 0.7
pmf = bernoulli.pmf(x, p)
plt.bar(x, pmf)
plt.title("Bernoulli PMF (p=0.7)")
plt.xlabel("x")
plt.ylabel("P(X=x)")
plt.show()


X = binom(n=4, p=0.5)
X.pmf([0, 1, 2, 3, 4])
# x=np.arange(5)
# x
# X.pmf(x)


X = poisson(mu = 1.5)
X.pmf([1, 2, 3]).sum()


# binom.cdf(1, 5, 0.2)
X = binom(n=5, p=0.2)
X.cdf(1)


visits = [0, 1, 2, 0, 3, 1, 4, 2, 2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 4, 2]
lambda_hat = np.mean(visits)
lambda_hat
X = poisson(mu = lambda_hat)
X.pmf(0)

# 하루동안 5명 이상 고객이 방문할 확률은?
1 - X.cdf(4)

np.var(visits, ddof=1)

import matplotlib.pyplot as plt
from scipy.stats import poisson
# 관측값
values, counts = np.unique(visits, return_counts=True)
prob_obs = counts / len(visits)
# 추정된 파라미터
lambda_hat = np.mean(visits)
x = np.arange(0, max(values)+1)
pmf_theory = poisson.pmf(x, mu=lambda_hat)
# 시각화
plt.bar(x -   0.2, prob_obs, width=0.4, label="Observed", color="skyblue")
plt.bar(x + 0.2, pmf_theory, width=0.4, label="Poisson Fit", color="orange")
plt.xlabel("Visits")
plt.ylabel("Probability")
plt.title("Observed vs. Fitted Poisson PMF")
plt.legend()
plt.grid(True)
plt.show()