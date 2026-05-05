import os
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from pyiqa import create_metric
import argparse
import face_alignment
import numpy as np

from helpers.arcface.models import resnet_face18
from helpers.utils import process_arcface_input

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pred_path",
    type=str,
    default="results/BlindFFHQ_8.8M/visualization/celeba-hq_custom-aligned_validation/pred",
)
parser.add_argument(
    "--gt_path",
    type=str,
    default="results/BlindFFHQ_8.8M/visualization/celeba-hq_custom-aligned_validation/gt",
)
parser.add_argument(
    "--match", action="store_true", help="Match the number of pred and GT samples"
)
args = parser.parse_args()

gt_filenames = sorted(os.listdir(os.path.join(args.gt_path)))
pred_filenames = sorted(os.listdir(os.path.join(args.pred_path)))

device = "cuda"

if args.match:
    gt_filenames = [f for f in gt_filenames if f in pred_filenames]

gt_filenames.sort()
pred_filenames.sort()

assert len(gt_filenames) == len(
    pred_filenames
), "The number of GT and pred images must be the same. %s vs. %s" % (
    len(gt_filenames),
    len(pred_filenames),
)

get_psnr = create_metric("psnr", device=device)
get_ssim = create_metric("ssim", device=device)
get_lpips = create_metric("lpips", device=device)
get_niqe = create_metric("niqe", device=device)
get_fid = create_metric("fid", device=device)
get_musiq = create_metric("musiq", device=device)
get_vif = create_metric("vif", device=device)

# Identity Model
id_model = resnet_face18(use_se=False).to(device)
id_model.load_state_dict(
    torch.load("helpers/arcface/weights/resnet18_110_wo_dist.pth", weights_only=False)
)
id_model.requires_grad_(False)  # freeze the identity model
id_model.eval()

# Facial Landmarks Detector
landmarks_detector = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, flip_input=False
)


psnr = 0.0
ssim = 0.0
lpips = 0.0
niqe = 0.0
musiq = 0.0
ids = 0.0
lmd = 0.0
lmd_count = 0
vif = 0.0

for idx, filename in enumerate(gt_filenames):
    print("🍊 %s/%s" % (idx + 1, len(gt_filenames)))

    gt_filepath = os.path.join(args.gt_path, filename)
    gt_image = (
        to_tensor(
            Image.open(gt_filepath)
            .convert("RGB")
            .resize((128, 128), resample=Image.Resampling.BICUBIC)
        )
        .unsqueeze(0)
        .to(device)
    )

    pred_filepath = os.path.join(args.pred_path, filename)
    pred_image = (
        to_tensor(
            Image.open(pred_filepath)
            .convert("RGB")
            .resize((128, 128), resample=Image.Resampling.BICUBIC)
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        # Arcface Feature 추출
        pred_feature = id_model(process_arcface_input(pred_image))
        gt_feature = id_model(process_arcface_input(gt_image))

        pred_feature = F.normalize(pred_feature, dim=1)
        gt_feature = F.normalize(gt_feature, dim=1)

        ids_score = F.cosine_similarity(pred_feature, gt_feature, dim=1).item()

        # 얼굴 랜드마크 추출
        pred_lds = landmarks_detector.get_landmarks(pred_filepath)
        gt_lds = landmarks_detector.get_landmarks(gt_filepath)

        if pred_lds is None or gt_lds is None:
            print("No face detected. Skip calculating lmd.")
            lmd_score = None
        else:
            pred_landmarks, gt_landmarks = np.array(pred_lds[0]), np.array(gt_lds[0])
            lmd_score = np.linalg.norm(pred_landmarks - gt_landmarks, axis=1).mean()
            lmd_count += 1

    psnr += get_psnr(gt_image, pred_image).item()
    ssim += get_ssim(gt_image, pred_image).item()
    lpips += get_lpips(gt_image, pred_image).item()
    niqe += get_niqe(pred_image).item()
    musiq += get_musiq(pred_image).item()
    vif += get_vif(gt_image, pred_image).item()

    ids += ids_score
    if lmd_score is not None:
        lmd += lmd_score

fid = get_fid(args.gt_path, args.pred_path)

scores = {
    "PSNR": psnr / len(gt_filenames),
    "SSIM": ssim / len(gt_filenames),
    "LPIPS": lpips / len(gt_filenames),
    "NIQE": niqe / len(gt_filenames),
    "IDS": ids / len(gt_filenames),
    "FID": fid.item(),
    "MUSIQ": musiq / len(gt_filenames),
    "VIF": vif / len(gt_filenames),
    "LMD": lmd / lmd_count,
}
results = ["%s=%s" % (k, v) for k, v in scores.items()]

print("✅ Done!")
print("\n".join(results))
