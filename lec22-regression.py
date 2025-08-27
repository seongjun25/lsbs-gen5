import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. Iris 데이터 로드
df_iris = load_iris()

# 2. pandas DataFrame으로 변환
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] #컬럼명 변경시

# 3. 타겟(클래스) 추가
iris["Species"] = df_iris.target

# 4. 클래스 라벨을 실제 이름으로 변환 (0: setosa, 1: versicolor, 2: virginica)
iris["Species"] = iris["Species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

iris.info()

# iris["Species"].value_counts()

import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Length",
                data = iris).fit()
print(model.summary())

coefficients=model.params[1:]
coefficients.idxmax()

model.tvalues
model.pvalues
np.abs(model.tvalues).idxmax()

conf_intervals = model.conf_int(alpha=0.10)
conf_intervals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 3. 회귀 계수 가져오기
intercept, slope = model.params
print(f"절편: {intercept:.3f}, 기울기: {slope:.3f}")

# 4. 시각화
plt.figure(figsize=(8,6))

# 산점도
sns.scatterplot(x="Petal_Width", y="Petal_Length", hue="Species", data=iris, alpha=0.7)

# 회귀 직선 (예측값 계산)
x_vals = np.linspace(iris["Petal_Width"].min(), iris["Petal_Width"].max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="regression line")

plt.title("Iris data: Petal Width vs Petal Length")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# ==============
model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Length + C(Species)", data=iris).fit()
print(model.summary())



# =====================
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Iris 데이터 로드
df_iris = load_iris()
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
iris["Species"] = df_iris.target
iris["Species"] = iris["Species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# 2. 회귀 모델 적합
model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Length", data=iris).fit()
print(model.summary())

# 3. 회귀계수
b0, b1, b2 = model.params
print(f"절편={b0:.3f}, Petal_Width 계수={b1:.3f}, Sepal_Length 계수={b2:.3f}")

# 4. 산점도 + 회귀평면 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 산점도 (색상은 종별 구분)
species_colors = {"setosa": "red", "versicolor": "green", "virginica": "blue"}
for sp, color in species_colors.items():
    df_sp = iris[iris["Species"] == sp]
    ax.scatter(df_sp["Petal_Width"], df_sp["Sepal_Length"], df_sp["Petal_Length"],
               label=sp, color=color, alpha=0.6)

# 회귀평면 생성
x_surf, y_surf = np.meshgrid(
    np.linspace(iris["Petal_Width"].min(), iris["Petal_Width"].max(), 30),
    np.linspace(iris["Sepal_Length"].min(), iris["Sepal_Length"].max(), 30)
)
z_surf = b0 + b1 * x_surf + b2 * y_surf

ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color="yellow")

# 축 레이블
ax.set_xlabel("Petal Width")
ax.set_ylabel("Sepal Length")
ax.set_zlabel("Petal Length")
ax.set_title("3D Regression Plane: Petal_Length ~ Petal_Width + Sepal_Length")
ax.legend()

plt.show()





# ===================================
import pandas as pd
import numpy as np
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)

np.random.seed(2022)
train_index=np.random.choice(penguins.shape[0],200)
train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
train_data.head()

train_data

# # 1
# 팔머펭귄 데이터의 부리길이를 종속변수
# 부리 깊이를 독립변수로 설정하여
# 회귀직선을 구하시오.

# 1) 산점도를 그린 후, 구해진 직선을 시각화
# 해보세요.
import matplotlib.pyplot as plt
import seaborn as sns
# Scatter plot using seaborn
plt.figure(figsize=(8,4))
sns.scatterplot(data=train_data,
                x='bill_depth_mm',
                y='bill_length_mm',
                edgecolor='w', s=50)
plt.title('Bill Length vs Bill Depth by Species')
plt.grid(True)
plt.show()

# 2) 독립변수와 종속변수의 관계를 직선
# 계수를 사용해서 해석해보세요.

from statsmodels.formula.api import ols
model1 = ols("bill_length_mm ~ bill_depth_mm", data=train_data).fit()
model1.params

sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                edgecolor='w', s=50)
x_values = train_data['bill_depth_mm']
y_values = 55.4110 - 0.7062 * x_values
# 부리깊이(독립변수)가 1 단위만큼 증가하면
# 부리길이가 0.7062 만큼 감소한다.
plt.plot(x_values, y_values, 
         color='red', label='Regression Line')
plt.grid(True)
plt.legend()
plt.show()

# 3) 계수 유의성을 통해 해석가능성을
# 이야기해보세요.
print(model1.summary())
# 부리깊이에 대응하는 계수의 유의확률이 0.05보다
# 작으므로, 부리깊이 계수가 0이 아니라는 통계적
# 근거가 충분하다. 따라서, 위 해석은 타당하다.

# 넘파이를 사용해서, 직선과 주어진 점들의
# 수직거리를 한변으로하는 사각형들의 넓이
# 합을 계산하세요.

y = train_data["bill_length_mm"]

x_values = train_data['bill_depth_mm']
y_line = 55.4110 - 0.7062 * x_values

# 오차 지표
# 잔차제곱합(Sum of Squared Residual)
np.sum((y - y_line)**2)


x=np.array([0, 0, 0, 1, 1, 1])
y=np.array([2, 4, 6, 2, 4, 6]) 
(x - 0.5) * (y - 4.2)
px=np.array([0.2, 0, 0.3, 0.1, 0.3, 0.1])
CovXY = np.sum((x - 0.5) * (y - 4.2) * px)
CovXY / (np.sqrt(0.25) * np.sqrt(2.76))


### 상관계수 0인 두 확률변수에서 표본 뽑기
import numpy as np

# 1. 가능한 (X, Y) 조합
x_vals = [0, 0, 0, 1, 1, 1]
y_vals = [2, 4, 6, 2, 4, 6]

# 2. 각 조합에 대한 확률
probs = [0.15, 0.15, 0.20,
         0.15, 0.15, 0.20]

N = 300
idx = np.random.choice(len(x_vals), size=N, p=probs)
x = np.array([x_vals[i] for i in idx])
y = np.array([y_vals[i] for i in idx])

upper=np.sum((x - x.mean()) * (y - y.mean()))
lower_l=np.sqrt(np.sum((x - x.mean())**2))
lower_r=np.sqrt(np.sum((y - y.mean())**2))
upper / (lower_l * lower_r)

import scipy.stats as stats
corr_coeff, p_value = stats.pearsonr(x, y)




x = np.array([2, 5, 3, 7])
noise = np.random.normal(loc=0, scale=2, size=4)
y = 0*x + 0 + noise

# 산점도
plt.scatter(x, y, color='black', label='points')

# y = 2x + 3 직선 추가
x_line = np.linspace(min(x)-1, max(x)+1, 100)  # 직선을 위한 x 범위
y_line = 0*x_line + 0
plt.plot(x_line, y_line, color='blue', 
         label='y = 2x + 3')

plt.grid(True)
plt.legend()
plt.show()



import numpy as np
x = np.array([10, 20, 30, 40, 50])
y = np.array([5, 15, 25, 35, 48]).reshape(-1, 1)

x = x.reshape(-1, 1)
X = np.hstack([np.ones((x.shape[0], 1)), x])

beta = np.array([2.0, 1.0]).reshape(-1, 1)
beta

def ssr(beta_vec):
    return (y - X @ beta_vec).transpose() @ (y - X @ beta_vec)

ssr(beta)

from scipy.optimize import minimize

# 예시 함수: SSR
def ssr(beta):
    beta = beta.reshape(-1, 1)
    r = y - X @ beta
    return float((r.T @ r))

# 초기값 아무거나 줌
res = minimize(ssr, x0=np.zeros(2))
print("최적해 beta =", res.x)
print("최소 SSR =", res.fun)


a=np.linalg.inv(X.transpose() @ X)
b=X.transpose() @ y
a @ b
