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
특정 노드에 Gentoo만 있는 경우!
특정 노드에 Chinstrap만 있는 경우!
1 - (p_g^2 + p_c^2)
1 - (1^2 + 0^2) = 1 - 1 = 0
1 - (0^2 + 1^2) = 1 - 1 = 0

특정 노드에 100 데이터 -> Gentoo 50개, Chinstrap 50개
1 - (p_g^2 + p_c^2)
1 - (0.5**2 + 0.5**2)


