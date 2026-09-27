import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X_full = loadmat('data/KSC.mat')['KSC']
y_full = loadmat('data/KSC_gt.mat')['KSC_gt']


X = X_full.reshape(-1, X_full.shape[-1])
y = y_full.ravel()


mask = y > 0
X = X[mask]
y = y[mask]


np.random.seed(42)
train_indices = []
for cls in range(1, 14):
    cls_indices = np.where(y == cls)[0]
    selected = np.random.choice(cls_indices, size=10, replace=False)
    train_indices.extend(selected)
train_indices = np.array(train_indices)


X_train1 = X[train_indices]
y_train1 = y[train_indices]
X_test = np.delete(X, train_indices, axis=0)
y_test = np.delete(y, train_indices, axis=0)

rf1 = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42)
rf1.fit(X_train1, y_train1)
y_pred1 = rf1.predict(X_test)
oa1 = accuracy_score(y_test, y_pred1) * 100


remaining_indices = np.setdiff1d(np.arange(len(y)), train_indices)
extra_indices = np.random.choice(remaining_indices, size=500, replace=False)
train_indices2 = np.concatenate([train_indices, extra_indices])

X_train2 = X[train_indices2]
y_train2 = y[train_indices2]
X_test2 = np.delete(X, train_indices2, axis=0)
y_test2 = np.delete(y, train_indices2, axis=0)

rf2 = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42)
rf2.fit(X_train2, y_train2)
y_pred2 = rf2.predict(X_test2)
oa2 = accuracy_score(y_test2, y_pred2) * 100


print(f"RF₁ (10 пікселів на клас): Overall Accuracy = {oa1:.2f}%")
print(f"RF₂ (10 пікселів на клас + 500 випадкових): Overall Accuracy = {oa2:.2f}%")