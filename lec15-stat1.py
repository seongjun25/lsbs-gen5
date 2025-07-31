import numpy as np

samples = np.random.choice(np.arange(1, 6), 
                           size=3,
                           replace=True)
print(samples)

set([1, 2])


import numpy as np

# 사전 확률: [수니, 젤리, 뭉이]
prior = np.array([0.5, 0.3, 0.2])

# 조건부 확률 (접시 깨질 확률): [수니, 젤리, 뭉이]
likelihood = np.array([0.01, 0.02, 0.03])

# 분모: 전체 접시가 깨질 확률 (정규화 상수)
p_break = np.sum(prior * likelihood)

# 사후 확률: Bayes 정리 적용
posterior = (prior * likelihood) / p_break
posterior

# ===================
prior = posterior
p_break = np.sum(prior * likelihood)
p_break

# 사후 확률: Bayes 정리 적용
posterior = (prior * likelihood) / p_break
posterior


np.array([0.16, 0.18, 0.20]) / 0.54