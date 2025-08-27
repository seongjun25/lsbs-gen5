sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 
          10.52, 14.83, 13.03, 16.46, 10.84, 12.45]

from scipy.stats import ttest_1samp

t_statistic, p_value = ttest_1samp(sample, popmean=10, 
                                   alternative='greater')
t_statistic
p_value

# 유의수준 5% 하에서 통계적 판단은?


# 2표본
import pandas as pd

sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]
gender = ["Female"]*7 + ["Male"]*5
my_tab2 = pd.DataFrame({"score": sample, "gender": gender})
print(my_tab2)

from scipy.stats import ttest_ind
male_score = my_tab2[my_tab2['gender'] == 'Male']["score"]
female_score = my_tab2[my_tab2['gender'] == 'Female']["score"]
ttest_ind(female_score, male_score, 
          equal_var=True, alternative='less')


import numpy as np

before = np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15])
after = np.array([10.52, 14.83, 13.03, 16.46, 10.84, 12.45])

x = after - before
x
ttest_1samp(x, popmean=0, alternative="greater")
