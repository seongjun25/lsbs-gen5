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
