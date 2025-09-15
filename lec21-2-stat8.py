import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 비교할 자유도 목록
dfs = [3]

# x 구간: 가장 큰 df 기준으로 99.9% 분위까지
x_max = chi2.ppf(0.999, max(dfs))
x = np.linspace(0, x_max, 1000)

plt.figure(figsize=(7, 5))
for df in dfs:
    pdf = chi2.pdf(x, df=df)
    plt.plot(x, pdf, label=f"df={df}")

plt.title("Chi-square PDF (scipy.stats.chi2)")
plt.xlabel("x")
plt.ylabel("density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

from scipy.stats import chi2
from scipy.stats import norm

X = chi2(df=3)
1-X.cdf(8)

Y = norm(loc=3, scale=2)
data_set = Y.rvs(500 * 15).reshape(500, -1)
s_2=data_set.var(ddof=1, axis=1)
statistics = s_2 * (15 - 1) / 2**2
statistics

# 5. 히스토그램 + 이론적 카이제곱 PDF
x = np.linspace(0, max(statistics), 500)
pdf = chi2.pdf(x, df=14)  # 자유도 = n-1 = 14

plt.figure(figsize=(7,5))
plt.hist(statistics, bins=30, density=True, alpha=0.6, label="Simulated")
plt.plot(x, pdf, "r-", lw=2, label="Chi2 PDF (df=14)")
plt.title("Sample variance → Chi-square distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


x = np.array([10.67, 9.92, 9.62, 9.53, 9.14, 9.74, 8.45, 12.65, 11.47, 8.62])
stat_t=(10-1) * x.var(ddof=1) / 1.3

1-chi2.cdf(stat_t, df = 9)
# 유의확률인 0.235가 유의수준 5%보다 크므로
# 귀무가설을 기각하지 못한다.
# 현재 생산 적격이라 판단함.

# 1표본 t 검정 => 모평균이 특정값인지를 검정
# 1표본 카이제곱 검정 => 모분산이 특정값인지를 검정
1-chi2.cdf(15.55, df=1)



from scipy.stats import chi2_contingency
table = np.array([[14, 4],
                  [0, 10]])
chi2, p, df, expected = chi2_contingency(table, correction=False)
print('X-squared:', chi2.round(3), 
      'df:', df, 
      'p-value:', p.round(3))

expected



1-chi2.cdf(0.6478, df=2)

o_i = np.array([13, 23, 24, 20, 27, 18, 15])
e_i = np.repeat(20, 7)
sum((o_i - e_i)**2 / e_i)

1-chi2.cdf(7.6, df=6)

from scipy.stats import chisquare
import numpy as np

observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)

statistic, p_value = chisquare(observed, 
                               f_exp=expected)

X = norm(loc=3, scale = 2)

from palmerpenguins import load_penguins
import pandas as pd
df = load_penguins()
penguins=df.dropna()

x = penguins["bill_length_mm"]
x.mean()
x.std(ddof=1)

# 정규분포 평균 43.993, 표준편차 5.468
X = norm(loc=43.993, scale = 5.468)
X.cdf(52.725) - X.cdf(45.85)
X.cdf(45.85) - X.cdf(38.975)
333 * 0.3119
333 * 0.4535
len(x)
pd.cut(x, bins=4).value_counts()


# =============== 연습문제
import pandas as pd
from scipy.stats import chi2_contingency
# 데이터 불러오기
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
# 임신 유무 파생변수 생성
dat['Pregnancy_status'] = (dat['Pregnancies'] > 0).astype(int)

from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
ct = pd.crosstab(dat['Pregnancy_status'], dat['Outcome'], normalize='index')
ct.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
plt.tight_layout()
plt.show()


from statsmodels.graphics.mosaicplot import mosaic
dat['Pregnancy_status_label'] = dat['Pregnancy_status'].map({0: 'Not Pregnant', 1: 'Pregnant'})
dat['Outcome_label'] = dat['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})
plt.figure(figsize=(8, 5))
mosaic(dat, ['Pregnancy_status_label', 'Outcome_label'], 
       title='Mosaic Plot: Pregnancy vs Diabetes')
plt.tight_layout()
plt.show()

ct = pd.crosstab(
    dat['Pregnancy_status'],
    dat['Outcome']
    )
ct


chi2, p, dof, expected = chi2_contingency(ct, correction=False)
expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
expected_df



import pandas as pd
from scipy.stats import chi2_contingency
# 데이터 불러오기
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
# 연령대 그룹 생성
dat['Age_group'] = pd.cut(dat['Age'], bins=[0, 39, 100], labels=['Under40', 'Over40'])

mosaic(pd.crosstab(dat['Age_group'], dat['Outcome']).stack())
plt.show()

ct = pd.crosstab(dat['Age_group'], dat['Outcome'])
ct

chi2, p, dof, expected = chi2_contingency(ct, correction=False)
pd.DataFrame(expected, index=ct.index, columns=ct.columns)


import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
table = np.array([[1, 4],
                  [2, 8],
                  [4, 6],
                  [15, 20]])
df = pd.DataFrame(table, 
                  index=["자주 거름", "불규칙", "하루 2끼", "규칙적 3끼"], 
                  columns=["건강함", "건강하지 않음"])
print(df)

chi2, p, dof, expected = chi2_contingency(table, correction=False)
pd.DataFrame(expected, index=df.index, columns=df.columns)

