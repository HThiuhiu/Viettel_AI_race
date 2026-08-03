"""
Tinh diem tong hop THAT theo dung cong thuc BTC cham:
    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
    PSNR_norm = clamp(PSNR / PSNR_MAX, 0, 1)   (PSNR_MAX mac dinh 50dB)

Doc truc tiep tu outputs/metrics_summary.csv (do kaggle_train_t4x2.sh tu ghi sau
moi lan train+ns-eval). Muc dich: khi A/B 2 cau hinh hyperparameter (vd
CULL_ALPHA_THRESH=0.003 vs 0.005), quyet dinh cau hinh nao tot hon phai dua tren
DIEM TONG HOP THAT, khong phai nhin rieng le PSNR/SSIM/LPIPS roi doan mo ho -
vi 1 cau hinh co the tang PSNR nhung giam LPIPS (hoac nguoc lai), va chi cong
thuc tong hop moi noi len no co thuc su tot hon cho DIEM SO cuoi cung hay khong.

Cach dung:
    python pipeline/compute_score.py --metrics_csv /kaggle/working/outputs/metrics_summary.csv
    python pipeline/compute_score.py --metrics_csv outputs/metrics_summary.csv --psnr_max 50

Neu ban biet gia tri PSNR_MAX khac (hoac cong thuc PSNR_norm khac) so voi mac
dinh 50dB dung o day, sua --psnr_max hoac ham compute_score() ben duoi cho khop.
"""

import argparse
import csv
import sys
from collections import defaultdict


def compute_score(psnr: float, ssim: float, lpips: float, psnr_max: float) -> float:
    psnr_norm = max(0.0, min(psnr / psnr_max, 1.0))
    return 0.4 * (1.0 - lpips) + 0.3 * ssim + 0.3 * psnr_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, required=True,
                         help="Duong dan metrics_summary.csv (do kaggle_train_t4x2.sh tu tao)")
    parser.add_argument("--psnr_max", type=float, default=50.0,
                         help="Gia tri PSNR dung de chuan hoa PSNR_norm=clamp(PSNR/psnr_max,0,1). "
                              "Mac dinh 50.0 - SUA LAI neu BTC cong bo gia tri khac.")
    args = parser.parse_args()

    rows = []
    with open(args.metrics_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                psnr = float(r["psnr"])
                ssim = float(r["ssim"])
                lpips = float(r["lpips"])
            except (KeyError, ValueError, TypeError):
                continue
            score = compute_score(psnr, ssim, lpips, args.psnr_max)
            r["score"] = score
            rows.append(r)

    if not rows:
        print(f"[LOI] Khong doc duoc dong nao co du psnr/ssim/lpips tu {args.metrics_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"{'scene':<12}{'strategy':<10}{'cull_alpha':<12}{'ssim_lambda':<12}"
          f"{'psnr':>8}{'ssim':>8}{'lpips':>8}{'SCORE':>10}")
    for r in rows:
        print(f"{r.get('scene',''):<12}{r.get('strategy',''):<10}{r.get('cull_alpha_thresh',''):<12}"
              f"{r.get('ssim_lambda',''):<12}{float(r['psnr']):>8.3f}{float(r['ssim']):>8.4f}"
              f"{float(r['lpips']):>8.4f}{r['score']:>10.4f}")

    # Neu 1 scene co nhieu dong (do chay lai/A-B nhieu lan), gom nhom lay dong
    # co score cao nhat lam "best config" cho scene do - day chinh la cau hinh
    # nen dung khi render submission cuoi cung cho scene nay.
    best_per_scene = {}
    for r in rows:
        scene = r.get("scene", "")
        if scene not in best_per_scene or r["score"] > best_per_scene[scene]["score"]:
            best_per_scene[scene] = r

    print(f"\n=== Cau hinh tot nhat (theo SCORE) cho tung scene ===")
    total = 0.0
    for scene, r in sorted(best_per_scene.items()):
        print(f"[{scene}] score={r['score']:.4f}  (psnr={float(r['psnr']):.2f}, ssim={float(r['ssim']):.4f}, "
              f"lpips={float(r['lpips']):.4f}, config={r.get('config_path','')})")
        total += r["score"]

    avg = total / len(best_per_scene)
    print(f"\n=== DIEM TRUNG BINH UOC LUONG (proxy cho leaderboard, tinh tren "
          f"{len(best_per_scene)} scene, dung eval-split noi bo KHONG phai test that) = {avg:.4f} ===")
    print("Luu y: day la uoc luong tren anh eval giu lai TU TAP TRAIN (Nerfstudio tu tach "
          "~10% anh train lam eval), KHONG phai tap test that cua BTC - nhung la proxy tot "
          "nhat hien co vi khong co ground-truth cho test_poses.csv.")


if __name__ == "__main__":
    main()
