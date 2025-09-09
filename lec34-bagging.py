import numpy as np

# 원 데이터
x = np.array([2, 4, 5, 7, 9])

n, B = len(x), 1000

# 부트스트랩 샘플링 (B x n 행렬)
samples = np.random.choice(x, size=(B, n), replace=True)

# 각 샘플의 평균
means = samples.mean(axis=1)
means

print("원 평균:", x.mean())
print("부트스트랩 평균의 평균:", means.mean())
print("부트스트랩 표준오차(SE):", means.std(ddof=1))
print("95% 신뢰구간:", np.percentile(means, [2.5, 97.5]))

# 시각화 코드
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
plt.hist(means, bins=20, color="skyblue", edgecolor="k", alpha=0.7)
plt.axvline(x.mean(), color="red", linestyle="--", label="원 평균")
plt.xlabel("부트스트랩 평균")
plt.ylabel("빈도")
plt.title("Bootstrap distribution of the mean")
plt.legend()
plt.show()
