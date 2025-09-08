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