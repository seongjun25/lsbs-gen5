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
                      columns=["pca1", "pca2", "pca3"])