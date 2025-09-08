from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins_a=penguins.loc[penguins["species"] != "Adelie",]

x = penguins_a[["bill_length_mm", "bill_depth_mm"]]
y = penguins_a["species"]

# x,y 평면상에
# bill_length, bill_depth에 점을
# 표시, 단, 겐투는 사각형, 친스트랩은
# 동그라미로 표시
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=penguins_a,
    x="bill_length_mm", 
    y="bill_depth_mm",
    style="species",   # 종에 따라 모양 다르게
    hue="species",     # 종에 따라 색도 다르게
    s=80,              # 점 크기
    edgecolor="k"
)
plt.title("Penguins (Gentoo vs Chinstrap)")
plt.grid(True)
plt.show()


x = penguins_a[["bill_length_mm", "bill_depth_mm"]]
y = penguins_a["species"]

left_x = x.loc[x["bill_length_mm"] <= 50,:] 
right_x = x.loc[x["bill_length_mm"] > 50,:] 
left_y = y.loc[x["bill_length_mm"] <= 50]
right_y = y.loc[x["bill_length_mm"] > 50]

# 지니 인덱스
# 특정 노드에 Gentoo만 있는 경우!
# 특정 노드에 Chinstrap만 있는 경우!
# 1 - (p_g^2 + p_c^2)
# 1 - (1^2 + 0^2) = 1 - 1 = 0
# 1 - (0^2 + 1^2) = 1 - 1 = 0

# 특정 노드에 100 데이터 -> Gentoo 50개, Chinstrap 50개
# 1 - (p_g^2 + p_c^2)
# 1 - (0.5**2 + 0.5**2)

# bill_depth 16.5 기준으로
# 지니 인덱스 구해보세요!

import numpy as np
np.sum(y == "Gentoo")
len(y)
# 나누기 전 GI
1 - ((119/187)**2 + (68/187)**2)

# 윗 상자
upper_y=y[x["bill_depth_mm"] >= 16.5]
len(upper_y) # 75
np.sum(upper_y == "Gentoo") # 8
# GI: 0.1906
1 - ((8/75)**2 + (67/75)**2)

# 아래 상자
lower_y=y[x["bill_depth_mm"] < 16.5]
len(lower_y) # 112
np.sum(lower_y == "Gentoo") # 111
# GI: 0.0177
1 - ((111/112)**2 + (1/112)**2)

# mean GI: 0.087
# 0.4628 - 0.087

from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier(criterion='gini')
dct.get_params()

import numpy as np
dct_params = {'max_depth' : np.arange(1, 8),
              'ccp_alpha': np.linspace(0, 1, 5)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, 
           shuffle=True, 
           random_state=2025)

# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                          param_grid=dct_params, 
                          cv = cv, 
                          scoring='accuracy')
dct_search.fit(x, y)
dct_search.best_params_

dct_search.predict(x)
dct_search.predict_proba(x)

from sklearn import tree
import matplotlib.pyplot as plt

# 최적 모델 꺼내오기
best_tree = dct_search.best_estimator_

plt.figure(figsize=(14,7))
tree.plot_tree(
    best_tree,
    feature_names=["bill_length_mm", "bill_depth_mm"],  # 입력 변수 이름
    class_names=best_tree.classes_,                     # 클래스 이름
    filled=True, rounded=True
)
plt.show()


