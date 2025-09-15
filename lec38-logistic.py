import pandas as pd
import numpy as np
admission_data = pd.read_csv("./data/admission.csv")
print(admission_data.head())
print(admission_data.shape)

p_hat = admission_data['admit'].mean()
print(np.round(p_hat / (1 - p_hat), 3))

unique_ranks = sorted(admission_data['rank'].unique())

grouped_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean'))
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])

2.48 / (2.48 + 1)



odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)


import matplotlib.pyplot as plt

# 산점도
plt.figure(figsize=(6,4))
plt.scatter(odds_data['rank'], odds_data['log_odds'], color='blue', s=80)

# 보조: 각 점에 값 표시
for i, row in odds_data.iterrows():
    plt.text(row['rank']+0.05, row['log_odds'], f"{row['log_odds']:.2f}", va='center')

# 라벨과 제목
plt.title("Rank vs Log Odds")
plt.xlabel("Rank")
plt.ylabel("Log Odds")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary())

# y_hat = 0.6327 + -0.5675 * x
# 1.5등급 학생의 입학확률을 구하세요.
odds_15=np.exp(0.6327 -0.5675 * 1.5) # 오즈
odds_15 / (odds_15 + 1)

log_odds = 0.6327 + -0.5675 * x

x 랭크
odds_x = exp(0.6327 -0.5675 * x)
odds_xp1 = exp(0.6327 -0.5675 * (x+1))

1등급의 오즈는? 1.06737
np.exp(0.6327 -0.5675 * 1)

2등급의 오즈는? 0.60513
np.exp(0.6327 -0.5675 * 2)

0.6051372422396532 / 1.067372477533673

3등급의 오즈는? 0.3430771259828181
np.exp(0.6327 -0.5675 * 3)

0.3430771259828181 / 0.6051372422396532

np.exp(-0.5675)

model = smf.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()
model
print(model.summary())

# gpa 계수가 0.7753
# 이것을 해석하지?
# np.exp(0.7753)
# 2.17 ->
# x가 1증가 -> 오즈가 2.17배가 된다.
# gre 0.0023
# 해석?
# np.exp(0.0023)
# x가 1증가 -> 오즈가 1.0023배 증가


(model.predict(admission_data) > 0.5).astype(int)







import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
train = pd.read_csv("./data/problem19.csv")
train_y = train.absences
train_X = train.drop(["absences"], axis=1)
# -----------------------------
# 4) 모델 &amp; 그리드서치 (MAE 기준)
# -----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=0)
# (A) 의사결정나무
tree = DecisionTreeRegressor()
tree_param = {"ccp_alpha": np.arange(0.1, 1.0, 0.1)}
tree_search = GridSearchCV(
    estimator=tree,
    param_grid=tree_param,
    cv=cv,
    scoring="neg_mean_absolute_error",
)
tree_search.fit(train_X, train_y)
print("[DecisionTree]")
print("Best ccp_alpha:", tree_search.best_params_["ccp_alpha"])
print("Best CV MAE:", -tree_search.best_score_)
# (B) Lasso
lasso = Lasso()
lasso_param = {"alpha": np.arange(0.1, 1.0, 0.1)}
lasso_search = GridSearchCV(
    estimator=lasso,
    param_grid=lasso_param,
    cv=cv,
    scoring="neg_mean_absolute_error",
)
lasso_search.fit(train_X, train_y)
print("\n[Lasso]")
print("Best alpha:", lasso_search.best_params_["alpha"])
print("Best CV MAE:", -lasso_search.best_score_)


import pandas as pd
import numpy as np
train = pd.read_csv("./data/problem19.csv")
test = pd.read_csv("./data/problem19_test.csv")
import joblib
lasso_search = joblib.load("./data/lasso_model.pkl")
print("모델이 불러와졌습니다.")

all_dat = pd.concat([train, test], axis=0)
feat_cols = train.columns.drop('absences')
N = 1000
cor_result = []
for i in range(1, N):
    sub_dat = all_dat.sample(frac=0.6, random_state=i)
    X_sub = sub_dat[feat_cols]  # 피처만
    y_sub = sub_dat["absences"].to_numpy()  # 타깃
    pred = lasso_search.predict(X_sub)  # 파이프라인이므로 원본 X_sub 그대로 입력
    cor_result.append(np.corrcoef(pred, y_sub)[0, 1])
result = pd.DataFrame(cor_result, columns=["cor"])
result.plot(kind="box")
# plt.show()
q1 = result["cor"].quantile(0.25)
q3 = result["cor"].quantile(0.75)
iqr = (q3 - q1).round(2)
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)