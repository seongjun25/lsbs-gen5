import pandas as pd
from palmerpenguins import load_penguins

# 1. 데이터 불러오기 및 전처리
penguins = load_penguins()
penguins = penguins.dropna()
penguins = penguins[penguins["species"] == "Adelie"]
df = penguins[["bill_length_mm", 
               "bill_depth_mm", 
               "body_mass_g"]]
df

import seaborn as sns
sns.pairplot(df, kind="scatter",
             diag_kind="hist");

df.corr()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_data = pd.DataFrame(scaled_data, 
                           columns=df.columns)

scaled_data


from sklearn.decomposition import PCA
pca = PCA(n_components=3)

pca_array = pca.fit_transform(scaled_data)
my_pca = pd.DataFrame(pca_array,
                      index = scaled_data.index,
                      columns=["pc1", "pc2", "pc3"])
my_pca.shape
my_pca.corr()
my_pca.cov()

my_pca["pc1"].var(ddof=1)
my_pca["pc2"].var(ddof=1)
my_pca["pc3"].var(ddof=1)

pca.explained_variance_.round(3) # 아이겐베류
x_to_pc = pd.DataFrame(
    pca.components_,
    columns=scaled_data.columns,
    index=['pca1','pca2','pca3']).round(3)
x_to_pc


# 어떻게 PC들을 만들었나?
# 1) 스케일된 데이터의 공분산행렬 계산
# 2) 행렬 분해(아이겐벨류 디컴포지션) 적용
#  => 아이겐벨류, 아이겐벡터 두개 결과값이 나옴
scaled_data.cov(ddof=1)

from numpy import linalg
import numpy as np
eig_values, eig_vectors = linalg.eig(scaled_data.cov(ddof=1))
np.sqrt(eig_values[0] / eig_values[2])
eig_vectors

# PC1 = 0.548 * 부리길이 + 0.564 * 부리깊이 + 0.618 * 몸무게

my_pca
scaled_data.iloc[0,:]

import matplotlib.pyplot as plt
plt.bar(range(1, 4), pca.explained_variance_ratio_);
plt.show()
pca.explained_variance_ratio_[:2].sum()


def biplot(score, coeff, pcax, pcay, labels=None):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley)
    
    for i in range(n):
        plt.arrow(0, 0, coeff[pca1, i], coeff[pca2, i],color='r',alpha=0.5)
        if labels is None:
            plt.text(coeff[pca1, i]* 1.15, coeff[pca2, i] * 1.15,
            "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[pca1, i]* 1.15, coeff[pca2, i] * 1.15,
            labels[i], color='g', ha='center', va='center')

    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid()

biplot(pca_array, pca.components_, 1, 2, labels=scaled_data.columns)
plt.show()


x_to_pc