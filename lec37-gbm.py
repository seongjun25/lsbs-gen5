from matplotlib import pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins
import pandas as pd
df = load_penguins()
penguins=df.dropna()

# x와 y 설정
x = penguins[["bill_length_mm"]]
y = penguins["bill_depth_mm"]

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()

# 모델 학습
model1.fit(x, y)
model1.coef_
model1.intercept_

sns.scatterplot(data=penguins,
                x='bill_length_mm', 
                y='bill_depth_mm',
                palette='deep',
                edgecolor='w', s=50)
y_pred1=model1.predict(x)
y - y_pred1

sns.scatterplot(x=penguins["bill_length_mm"], 
                y=y - y_pred1,
                palette='deep',
                edgecolor='w', s=50)
