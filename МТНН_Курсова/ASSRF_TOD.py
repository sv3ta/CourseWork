import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib
# matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt

X_full = loadmat('data/KSC.mat')['KSC']
y_full = loadmat('data/KSC_gt.mat')['KSC_gt']

X = X_full.reshape(-1, X_full.shape[-1])
y = y_full.ravel()
mask = y > 0
X = X[mask]
y = y[mask] - 1


n_classes = len(np.unique(y))
np.random.seed(42)
labeled_idx = []
for c in range(n_classes):
    idx = np.where(y == c)[0]
    labeled_idx += list(np.random.choice(idx, 10, replace=False))
unlabeled_idx = list(set(range(len(X))) - set(labeled_idx))


test_idx = np.setdiff1d(np.arange(len(y)), labeled_idx)
np.random.seed(42)
test_idx = np.random.choice(test_idx, size=int(0.4 * len(y)), replace=False)
X_test = X[test_idx]
y_test = y[test_idx]


n_iters = 50
active_batch = 10
pseudo_batch = 10
ema_decay = 0.999
lambda_unsup = 0.05
confidence_threshold = 0.95
temp = 1.0
temp_min = 0.5
temp_decay = 0.95


rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', n_jobs=-1, random_state=42)
prev_proba = np.zeros((len(X), n_classes))


kmeans = KMeans(n_clusters=12, random_state=42)
clusters = kmeans.fit_predict(X[unlabeled_idx])


ema_proba = None
for epoch in range(n_iters):
    print(f"\nEpoch {epoch + 1}/{n_iters}")


    rf.fit(X[labeled_idx], y[labeled_idx])
    proba = rf.predict_proba(X)


    if epoch == 0:
        ema_proba = proba.copy()
    else:
        ema_proba = ema_decay * ema_proba + (1 - ema_decay) * proba


    unsup_loss = np.mean(np.sum((proba[unlabeled_idx] - ema_proba[unlabeled_idx]) ** 2, axis=1))
    print(f"Unsupervised TOD loss: {unsup_loss:.4f}")


    entropy = -np.sum(proba[unlabeled_idx] * np.log(proba[unlabeled_idx] + 1e-10), axis=1)
    cod_score = np.linalg.norm(proba[unlabeled_idx] - prev_proba[unlabeled_idx], axis=1) if epoch > 0 else entropy
    tod_score = np.sum((proba[unlabeled_idx] - ema_proba[unlabeled_idx]) ** 2, axis=1)

    if len(unlabeled_idx) < len(clusters):
        cluster_idx_map = {old_idx: i for i, old_idx in enumerate(unlabeled_idx)}
        valid_clusters = np.array([clusters[cluster_idx_map[idx]] for idx in unlabeled_idx if idx in cluster_idx_map])
        clusters = valid_clusters
    elif len(unlabeled_idx) != len(clusters):
        raise ValueError(f"Mismatch: len(unlabeled_idx)={len(unlabeled_idx)}, len(clusters)={len(clusters)}")

    cluster_scores = np.zeros(len(unlabeled_idx))
    for c in range(100):
        cluster_mask = clusters == c
        if np.sum(cluster_mask) > 0:
            cluster_entropy = entropy[cluster_mask]
            cluster_scores[cluster_mask] = np.mean(cluster_entropy)


    if epoch== 0:
        combined_score = 0.7 * entropy + 0.2 * cluster_scores+0.1 * cod_score
    else:
        combined_score = 0.7 * entropy + 0.1 * cluster_scores + 0.1 * cod_score + 0.1 * tod_score


    active_idx = np.array(unlabeled_idx)[np.argsort(-combined_score)[:active_batch]]


    confidences = np.max(proba[unlabeled_idx] / temp, axis=1)
    pseudo_idx = np.array(unlabeled_idx)[np.argsort(-confidences)[:pseudo_batch]]
    pseudo_labels = np.argmax(proba[pseudo_idx], axis=1)

    labeled_idx += list(active_idx) + list(pseudo_idx)
    y[pseudo_idx] = pseudo_labels
    unlabeled_idx = list(set(unlabeled_idx) - set(active_idx) - set(pseudo_idx))

    temp = max(temp_min, temp * temp_decay)
    confidence_threshold = max(0.7, confidence_threshold * 0.95)

    prev_proba = proba.copy()
    print(f"Labeled samples: {len(labeled_idx)} | Remaining unlabeled: {len(unlabeled_idx)}")

    y_pred_test = rf.predict(X_test)
    oa = accuracy_score(y_test, y_pred_test) * 100
    print(f"Test Overall Accuracy: {oa:.2f}%")

rf.fit(X[labeled_idx], y[labeled_idx])
final_proba = rf.predict_proba(X)
final_preds = np.argmax((final_proba ), axis=1)  # EMA fusion

acc = accuracy_score(y_test, final_preds[test_idx])
print(f"\n✅ Final Test Accuracy (ASSRF+TOD): {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, final_preds[test_idx], digits=3))


def show_rgb_composite(X_raw, title):
    rgb = X_raw[:, :, [30, 20, 10]]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    plt.figure(figsize=(6, 5))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_label_map(pred_flat, title):
    pred_img = np.zeros_like(y_full)
    pred_img[mask.reshape(y_full.shape)] = pred_flat + 1
    plt.figure(figsize=(6, 5))
    plt.imshow(pred_img, cmap='tab20')
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


show_rgb_composite(X_full, "KSC RGB Composite")
show_label_map(y, "Ground Truth Labels")
show_label_map(final_preds, "Predicted Labels (ASSRF+TOD)")
plot_conf_matrix(y_test, final_preds[test_idx])