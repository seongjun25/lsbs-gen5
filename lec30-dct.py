from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins_a=penguins.loc[penguins["species"] == "Adelie",]


x = penguins_a[["bill_length_mm"]]
y = penguins_a["bill_depth_mm"]

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(x, y, alpha=0.6, color="blue", edgecolor="k")
plt.grid(True)
plt.show()

from sklearn.tree import DecisionTreeRegressor

dct = DecisionTreeRegressor(max_depth=1)
dct.get_params()

dct.fit(x, y)
pred_y=dct.predict(x)

plt.figure(figsize=(6,4))
plt.scatter(x, y, alpha=0.6, color="blue", edgecolor="k")
plt.scatter(x, pred_y, alpha=0.6, color="red", edgecolor="k")
plt.grid(True)
plt.show()