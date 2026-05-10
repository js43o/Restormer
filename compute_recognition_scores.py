import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_curve

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
HQ_PATH = "results/BlindFFHQ_6.6M/visualization/lfw_custom-aligned_validation/gt"
PRED_PATH = "results/BlindFFHQ_6.6M/visualization/lfw_custom-aligned_validation/pred"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64
NUM_POS = 10_000
NUM_NEG = 100_000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────
# 모델 & 전처리
# facenet-pytorch의 VGGFace2 모델은 160×160, [-1, 1] 정규화를 기대함
# ──────────────────────────────────────────
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

transform = transforms.Compose(
    [
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


# ──────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────
def get_identity(filename: str) -> str:
    """'Jiseung_Maeng_0001.jpg' → 'Jiseung_Maeng'"""
    return filename.rsplit("_", 1)[0]


def extract_features(folder: str, filenames: list[str]) -> np.ndarray:
    feats = []
    for i in tqdm(
        range(0, len(filenames), BATCH), desc=f"  {os.path.basename(folder)}"
    ):
        batch = filenames[i : i + BATCH]
        imgs = torch.stack(
            [
                transform(Image.open(os.path.join(folder, f)).convert("RGB"))
                for f in batch
            ]
        ).to(DEVICE)
        with torch.no_grad():
            feat = F.normalize(model(imgs), dim=1)
        feats.append(feat.cpu().numpy())
    return np.vstack(feats)


def tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    idx = np.where(fpr <= target)[0]
    return float(tpr[idx[-1]]) if len(idx) else 0.0


# ──────────────────────────────────────────
# 피처 추출
# 두 폴더의 파일명이 동일하게 쌍을 이룬다고 가정
# ──────────────────────────────────────────
filenames = sorted(os.listdir(HQ_PATH))
labels = np.array([get_identity(f) for f in filenames])
N = len(filenames)

print("Extracting features...")
hq_feats = extract_features(HQ_PATH, filenames)  # (N, 512)
pred_feats = extract_features(PRED_PATH, filenames)  # (N, 512)

# ──────────────────────────────────────────
# Verification
# probe: pred, gallery: hq
# positive pair: 같은 identity, negative: 다른 identity
# ──────────────────────────────────────────
label_to_idx = {}
for i, l in enumerate(labels):
    label_to_idx.setdefault(l, []).append(i)

# Positive pairs — identity별로 고르게 샘플링
pos_pairs = []
id_list = [idxs for idxs in label_to_idx.values() if len(idxs) >= 2]
while len(pos_pairs) < NUM_POS:
    idxs = random.choice(id_list)
    i, j = random.sample(idxs, 2)
    pos_pairs.append((i, j))

# Negative pairs
all_idx = list(range(N))
neg_pairs = []
while len(neg_pairs) < NUM_NEG:
    i, j = random.sample(all_idx, 2)
    if labels[i] != labels[j]:
        neg_pairs.append((i, j))


# 유사도 계산 (이미 L2-정규화 완료 → 내적 = cosine similarity)
def compute_scores(pairs, a_feats, b_feats):
    idx_a = np.array([p[0] for p in pairs])
    idx_b = np.array([p[1] for p in pairs])
    return (a_feats[idx_a] * b_feats[idx_b]).sum(axis=1)


scores = np.concatenate(
    [
        compute_scores(pos_pairs, pred_feats, hq_feats),
        compute_scores(neg_pairs, pred_feats, hq_feats),
    ]
)
gt = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))

# Best accuracy over thresholds
thresholds = np.linspace(-1, 1, 1000)
best_acc = max(
    ((scores[None] >= thresholds[:, None]).astype(int) == gt[None]).mean(axis=1)
)

fpr, tpr, _ = roc_curve(gt, scores)

# ──────────────────────────────────────────
# Identification  (Rank-1 / Rank-5)
# probe: pred, gallery: hq — 배치 처리로 메모리 절약
# ──────────────────────────────────────────
rank1 = rank5 = 0
K = min(5, N)

for start in tqdm(range(0, N, BATCH), desc="Identification"):
    end = min(start + BATCH, N)
    sims = np.dot(pred_feats[start:end], hq_feats.T)  # (B, N)
    top5 = np.argpartition(-sims, K, axis=1)[:, :K]  # (B, K) unsorted

    for b, g in enumerate(range(start, end)):
        top5_b = top5[b]
        best = top5_b[np.argmax(sims[b, top5_b])]
        rank1 += int(labels[g] == labels[best])
        rank5 += int(labels[g] in labels[top5_b])

rank1 /= N
rank5 /= N

# ──────────────────────────────────────────
# 출력
# ──────────────────────────────────────────
print("\n===== Results =====")
print(f"Verification Accuracy : {best_acc:.4f}")
print(f"TAR @ FAR=0.01        : {tar_at_far(fpr, tpr, 0.01):.4f}")
print(f"TAR @ FAR=0.001       : {tar_at_far(fpr, tpr, 0.001):.4f}")
print(f"Rank-1                : {rank1:.4f}")
print(f"Rank-5                : {rank5:.4f}")
