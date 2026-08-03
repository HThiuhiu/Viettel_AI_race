"""
ENSEMBLE NHIEU CHECKPOINT: render CUNG 1 pose tu nhieu checkpoint khac nhau cua
CUNG 1 lan train, roi TRUNG BINH PIXEL, va do Score that tren duong render nop bai.

=== VI SAO ===
Ky thuat chuan trong cac cuoc thi: nhieu checkpoint gan nhau co cung cau truc
dung nhung nhieu (noise) NGAU NHIEN DOC LAP nhau (floater nho, dao dong opacity
con lai luc train). Trung binh lam trieu tieu phan nhieu doc lap do -> thuong
tang PSNR/SSIM ma khong can train them 1 giay GPU nao.

=== CANH BAO PHAI DO, KHONG DUOC GIA DINH ===
Trung binh cung GAY MO NHE. LPIPS chiem 40% diem so va dang o muc TOT (~0.10),
LPIPS phat rat nang su mo. Nen ensemble co the LOI PSNR/SSIM nhung HAI LPIPS,
tong lai co the AM. Script nay vi vay luon in ca:
  - Score cua TUNG checkpoint rieng le
  - Score cua ban ensemble
de so sanh truc tiep. CHI dung ensemble cho submission that neu no THANG ro rang.

=== THU TU XU LY (khop DUNG eval_submission_path.py / render_and_submit.py) ===
Voi TUNG checkpoint:  render -> appearance transfer -> resize ve target size
Sau khi trung binh:   post_process (neu bat) -> tinh metric
post_process de o CUOI cung vi trong duong nop bai that no cung la buoc cuoi
(sau resize), va ap dung no 1 lan len anh da trung binh moi dung y nghia.

=== CACH DUNG ===
    python tools/ensemble_checkpoints.py --config <...>/config.yml \
        --steps 15000,20000,25000 \
        --supersample_factor 2.5 --appearance_transfer \
        --appearance_mode rgb_gain --appearance_k_neighbors 3 \
        --distortion_mode un_redistort

Bo nho: chi giu 1 bo accumulator float32 (~380MB cho 24 anh 1320x989) + 1
pipeline tai 1 thoi diem - pipeline cu duoc giai phong truoc khi nap cai moi.
"""

import argparse
import csv
import gc
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Script nay nam trong tools/ nhung dung chung ham voi pipeline/ - them
# pipeline/ vao sys.path de import duoc bat ke goi tu dau (root hay trong tools/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

# Tai su dung TRUC TIEP cac ham cua eval_submission_path.py thay vi chep lai
# logic - da co tien le hong: render_scale cua reference tung bi lech giua 2
# file va lam sai lech Score do duoc. Import bao dam 2 duong luon giong nhau.
from eval_submission_path import (
    REF_RENDER_SCALE,
    apply_gain,
    combine_gains,
    combine_lab_stats,
    compute_score,
    estimate_gain,
    find_k_nearest_train_idx,
    gain_weights_from_scores,
    pil_to_tensor_bchw,
    render_camera,
    render_camera_un_redistort,
    tensor_to_pil,
)
from image_postprocess import (
    apply_lab_color_transfer,
    enhance_image,
    lab_color_transfer_stats,
)


def load_pipeline_at_step(config_path: Path, step):
    """Nap pipeline tai DUNG checkpoint step (None = checkpoint moi nhat)."""
    from nerfstudio.utils.eval_utils import eval_setup

    def _force(cfg):
        if step is not None:
            cfg.load_step = step
        return cfg

    _, pipeline, _, _ = eval_setup(config_path, test_mode="inference", update_config_callback=_force)
    pipeline.eval()
    return pipeline


def render_one_image(model, eval_dataset, train_dataset, idx, args, gain_cache, device):
    """Render 1 anh eval qua DUNG duong nop bai (render -> appearance -> resize).
    Tra ve PIL Image o dung target size. KHONG ap post_process o day - buoc do
    duoc lam 1 lan sau khi trung binh (xem docstring dau file)."""
    gt_hwc = eval_dataset[idx]["image"]
    if gt_hwc.dtype != torch.float32:
        gt_hwc = gt_hwc.float() / 255.0
    target_size = tensor_to_pil(gt_hwc).size

    if args.distortion_mode == "un_redistort":
        rgb = render_camera_un_redistort(model, eval_dataset.cameras, idx, args.supersample_factor, device,
                                          prefilter=args.resample_prefilter)
    else:
        rgb = render_camera(model, eval_dataset.cameras, idx, args.supersample_factor, device)
    pred_pil = tensor_to_pil(rgb)

    if args.appearance_transfer:
        eval_c2w = eval_dataset.cameras.camera_to_worlds[idx]
        top_k = find_k_nearest_train_idx(eval_c2w, train_dataset.cameras, k=args.appearance_k_neighbors)
        per_ref = []
        for ref_idx, _ in top_k:
            if ref_idx not in gain_cache:
                if args.distortion_mode == "un_redistort":
                    ref_rgb = render_camera_un_redistort(model, train_dataset.cameras, ref_idx,
                                                          REF_RENDER_SCALE, device,
                                                          prefilter=args.resample_prefilter)
                else:
                    ref_rgb = render_camera(model, train_dataset.cameras, ref_idx,
                                             REF_RENDER_SCALE, device)
                ref_gt_hwc = train_dataset[ref_idx]["image"]
                if ref_gt_hwc.dtype != torch.float32:
                    ref_gt_hwc = ref_gt_hwc.float() / 255.0
                ref_render_pil, ref_gt_pil = tensor_to_pil(ref_rgb), tensor_to_pil(ref_gt_hwc)
                if args.appearance_mode == "lab_transfer":
                    gain_cache[ref_idx] = (lab_color_transfer_stats(ref_render_pil),
                                            lab_color_transfer_stats(ref_gt_pil))
                else:
                    gain_cache[ref_idx] = estimate_gain(ref_render_pil, ref_gt_pil)
            per_ref.append(gain_cache[ref_idx])

        if len(top_k) == 1:
            correction = per_ref[0]
        else:
            weights = gain_weights_from_scores([s for _, s in top_k])
            if args.appearance_mode == "lab_transfer":
                correction = (combine_lab_stats([c[0] for c in per_ref], weights),
                              combine_lab_stats([c[1] for c in per_ref], weights))
            else:
                correction = combine_gains(per_ref, weights)

        if args.appearance_mode == "lab_transfer":
            src_stats, dst_stats = correction
            pred_pil = apply_lab_color_transfer(pred_pil, src_stats, dst_stats)
        else:
            pred_pil = apply_gain(pred_pil, correction)

    if pred_pil.size != target_size:
        pred_pil = pred_pil.resize(target_size, Image.LANCZOS)
    return pred_pil


def score_images(pred_pils, gt_pils, metrics, args, device):
    """Tinh PSNR/SSIM/LPIPS/Score trung binh cho 1 danh sach anh da render."""
    psnr_fn, ssim_fn, lpips_fn = metrics
    ps, ss, ls = [], [], []
    for pred_pil, gt_pil in zip(pred_pils, gt_pils):
        if args.post_process != "none":
            pred_pil = enhance_image(pred_pil, mode=args.post_process,
                                      denoise_strength=args.denoise_strength,
                                      sharpen_amount=args.sharpen_amount,
                                      lowpass_sigma=args.lowpass_sigma,
                                      despeckle_threshold=args.despeckle_threshold)
        if args.jpeg_quality > 0:
            buf = io.BytesIO()
            pred_pil.save(buf, format="JPEG", quality=args.jpeg_quality, subsampling=0)
            buf.seek(0)
            pred_pil = Image.open(buf).convert("RGB")
        pred_t = pil_to_tensor_bchw(pred_pil, device)
        gt_t = pil_to_tensor_bchw(gt_pil, device)
        with torch.no_grad():
            ps.append(float(psnr_fn(gt_t, pred_t)))
            ss.append(float(ssim_fn(gt_t, pred_t)))
            ls.append(float(lpips_fn(gt_t, pred_t)))
    p, s, l = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
    return p, s, l, compute_score(p, s, l, args.psnr_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--steps", type=str, required=True,
                     help="Danh sach checkpoint step, cach nhau dau phay (vd '15000,20000,25000')")
    ap.add_argument("--supersample_factor", type=float, default=2.5)
    ap.add_argument("--appearance_transfer", action="store_true")
    ap.add_argument("--appearance_mode", type=str, default="rgb_gain",
                     choices=["rgb_gain", "lab_transfer"])
    ap.add_argument("--appearance_k_neighbors", type=int, default=3)
    ap.add_argument("--distortion_mode", type=str, default="un_redistort",
                     choices=["rasterizer", "un_redistort"])
    ap.add_argument("--resample_prefilter", action="store_true",
                     help="Loc trung binh vung ve target size truoc khi grid_sample ap meo "
                          "(xem distortion_redistort.py). Dat GIONG voi cau hinh da do thang o "
                          "eval_submission_path.py, neu khong con so ensemble se khong so sanh duoc.")
    ap.add_argument("--post_process", type=str, default="none",
                     choices=["none", "denoise", "sharpen", "denoise_sharpen", "lowpass",
                              "despeckle", "despeckle_lowpass"])
    ap.add_argument("--despeckle_threshold", type=float, default=0.03)
    ap.add_argument("--lowpass_sigma", type=float, default=0.3)
    ap.add_argument("--jpeg_quality", type=int, default=0,
                     help="Mo phong nen JPEG cuoi cung (0 = tat). Dat GIONG voi cau hinh da do o "
                          "eval_submission_path.py, neu khong con so ensemble khong so sanh duoc.")
    ap.add_argument("--denoise_strength", type=float, default=0.3)
    ap.add_argument("--sharpen_amount", type=float, default=0.5)
    ap.add_argument("--psnr_max", type=float, default=50.0)
    ap.add_argument("--max_images", type=int, default=0, help="0 = tat ca anh eval")
    ap.add_argument("--out_csv", type=str, default="ensemble_scores.csv")
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    if len(steps) < 2:
        raise SystemExit("[LOI] Can it nhat 2 checkpoint de ensemble.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = Path(args.config)
    print(f"Ensemble {len(steps)} checkpoint: {steps}")
    print(f"Cau hinh render: supersample_factor={args.supersample_factor} "
          f"appearance_transfer={args.appearance_transfer} (mode={args.appearance_mode}, "
          f"k={args.appearance_k_neighbors}) distortion_mode={args.distortion_mode}")

    accum = None          # tong tich luy (float32) cua anh cuoi tung checkpoint
    gt_pils = None        # anh ground-truth (giong nhau moi checkpoint)
    metrics = None        # (psnr, ssim, lpips) - lay 1 lan, doc lap voi Gaussian
    per_step_rows = []

    for si, step in enumerate(steps):
        print(f"\n=== [{si + 1}/{len(steps)}] Nap checkpoint step={step} ===")
        pipeline = load_pipeline_at_step(config_path, step)
        model = pipeline.model
        dm = pipeline.datamanager
        eval_dataset, train_dataset = dm.eval_dataset, dm.train_dataset
        n_eval = len(eval_dataset)
        if args.max_images > 0:
            n_eval = min(n_eval, args.max_images)

        if metrics is None:
            # Cac module metric la torchmetrics thuan, khong phu thuoc Gaussian
            # cua checkpoint nao - giu lai de dung sau khi da giai phong pipeline.
            metrics = (model.psnr, model.ssim, model.lpips)
            gt_pils = []
            for i in range(n_eval):
                g = eval_dataset[i]["image"]
                if g.dtype != torch.float32:
                    g = g.float() / 255.0
                gt_pils.append(tensor_to_pil(g))

        gain_cache = {}
        preds = []
        for i in range(n_eval):
            preds.append(render_one_image(model, eval_dataset, train_dataset, i, args, gain_cache, device))
        print(f"  Da render {n_eval} anh o step={step}")

        p, s, l, sc = score_images(preds, gt_pils, metrics, args, device)
        print(f"  RIENG LE step={step}: psnr={p:.3f} ssim={s:.4f} lpips={l:.4f} score={sc:.4f}")
        per_step_rows.append({"step": step, "psnr": p, "ssim": s, "lpips": l, "score": sc})

        arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.float32) for im in preds], axis=0)
        accum = arr if accum is None else accum + arr
        del preds, arr, pipeline, model, dm
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    mean_arr = accum / float(len(steps))
    ens_pils = [Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)) for a in mean_arr]
    p, s, l, sc = score_images(ens_pils, gt_pils, metrics, args, device)

    print("\n=== KET QUA ===")
    for r in per_step_rows:
        print(f"  step={r['step']:>7} rieng le : psnr={r['psnr']:.3f} ssim={r['ssim']:.4f} "
              f"lpips={r['lpips']:.4f} score={r['score']:.4f}")
    print(f"  ENSEMBLE ({len(steps)} ckpt): psnr={p:.3f} ssim={s:.4f} lpips={l:.4f} score={sc:.4f}")

    best_single = max(per_step_rows, key=lambda r: r["score"])
    delta = sc - best_single["score"]
    print(f"\n  Checkpoint don TOT NHAT: step={best_single['step']} (score={best_single['score']:.4f})")
    print(f"  Ensemble - tot nhat don  = {delta:+.4f}")
    if delta > 0.005:
        print("  => Ensemble THANG ro rang - nen dung cho submission that.")
    elif delta < -0.005:
        print("  => Ensemble THUA (nhieu kha nang do trung binh lam mo, LPIPS bi phat) "
              "- dung checkpoint don tot nhat.")
    else:
        print("  => Chenh lech nam trong nhieu do (<0.005) - KHONG du bang chung, "
              "uu tien checkpoint don cho don gian.")

    out_csv = Path(args.out_csv)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["config_path", "kind", "steps", "supersample_factor", "appearance_mode",
                        "appearance_k_neighbors", "distortion_mode", "post_process",
                        "psnr", "ssim", "lpips", "score"])
        for r in per_step_rows:
            w.writerow([str(config_path), "single", r["step"], args.supersample_factor,
                        args.appearance_mode, args.appearance_k_neighbors, args.distortion_mode,
                        args.post_process, f"{r['psnr']:.4f}", f"{r['ssim']:.5f}",
                        f"{r['lpips']:.5f}", f"{r['score']:.5f}"])
        w.writerow([str(config_path), "ensemble", "|".join(str(x) for x in steps),
                    args.supersample_factor, args.appearance_mode, args.appearance_k_neighbors,
                    args.distortion_mode, args.post_process,
                    f"{p:.4f}", f"{s:.5f}", f"{l:.5f}", f"{sc:.5f}"])
    print(f"\nDa ghi ket qua vao {out_csv}")


if __name__ == "__main__":
    main()
