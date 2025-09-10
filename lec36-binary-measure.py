import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
df.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='target'), 
    df['target'], 
    test_size=0.3, 
    random_state=42
    )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

np.set_printoptions(suppress=True)
y_prob_org = model.predict_proba(X_test)
len(y_prob_org[:,1])
y_prob_org[:,1] > 0.5 # 암환자라고 확신정도
(y_prob_org[:,1] > 0.5).astype(int)

model.predict(X_test)

# ==============================
y_prob_org.shape
y_prob_org.sum(axis=1)

print(pd.DataFrame(y_prob_org[:4].round(3)))

y_pred = model.predict(X_test)

p = 0.98989898989899
p=np.linspace(0, 1, 100)
y_pred = (y_prob_org[:,1] > p).astype(int)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred);
recall =  cm[1, 1] / sum(cm[1,:])
specificity = cm[0, 0] / sum(cm[0,:])
fpr = 1 - specificity

isp = ConfusionMatrixDisplay(confusion_matrix=cm);
isp.plot(cmap=plt.cm.Blues);
plt.show()




# 여러 threshold 정의
thresholds = np.linspace(0, 1, 100)

tprs = []  # recall
fprs = []  # false positive rate

for p in thresholds:
    y_pred = (y_prob_org[:,1] > p).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    # recall (TPR)
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # specificity
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    # fpr
    fpr = 1 - specificity

    tprs.append(recall)
    fprs.append(fpr)

# ROC curve 그리기
plt.figure(figsize=(6, 6))
plt.plot(fprs, tprs, marker='.', label="ROC curve (custom)")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random guess")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (Recall/TPR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

