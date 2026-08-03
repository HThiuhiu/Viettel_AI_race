"""
Do DIEM SO THAT (Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm) tren anh
eval giu lai, nhung render QUA DUNG DUONG NOP BAI - tuc la co ca supersampling
va appearance transfer, roi resize LANCZOS ve dung kich thuoc muc tieu.

VI SAO CAN SCRIPT NAY: ns-eval (nguon so lieu cua metrics_summary.csv) render
THANG tu model, KHONG di qua render_and_submit.py. Nghia la --supersample_factor
va --appearance_transfer - 2 thu tac dong truc tiep len anh nop bai - chua bao gio
duoc do bang so. Script nay lam dung dieu do: mo phong chinh xac duong render nop
bai tren tap anh CO ground-truth, de biet bat/tat chung co that su tang diem khong,
thay vi doan.

Metric dung LAI DUNG doi tuong psnr/ssim/lpips cua chinh model (pipeline.model.psnr,
.ssim, .lpips) - cung thu ns-eval dung - nen con so so sanh duoc truc tiep voi
metrics_summary.csv, khong bi lech do khac thu vien/khac cach chuan hoa.

Cach dung (A/B tung cau hinh render, moi lan ghi 1 dong vao CSV de so sanh):
    python pipeline/eval_submission_path.py --config <duong_dan>/config.yml
    python pipeline/eval_submission_path.py --config <...>/config.yml --supersample_factor 1.5
    python pipeline/eval_submission_path.py --config <...>/config.yml --supersample_factor 1.5 --appearance_transfer
"""

import argparse
import csv
import io
from pathlib import Path

import copy

import numpy as np
import torch
from PIL import Image

from image_postprocess import enhance_image, lab_color_transfer_stats, apply_lab_color_transfer
from distortion_redistort import redistort_to_target

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# Phai khop DUNG voi render_scale mac dinh cua render_pose_lowres() trong
# render_and_submit.py - do la do phan giai THAT su duoc dung de uoc luong
# gain/LAB-stats khi tao submission that. Neu 2 gia tri lech nhau, Score do
# duoc o day khong con mo phong dung duong nop bai (xem audit note ben duoi).
REF_RENDER_SCALE = 0.35


def best_shift_psnr(gt_arr: np.ndarray, pred_arr: np.ndarray, max_shift: int = 10):
    """CHAN DOAN misalignment: do PSNR khi cho phep dich pred toi da +-max_shift
    pixel (nguyen, theo ca 2 truc) truoc khi so voi gt, roi lay muc PSNR TOT NHAT
    trong toan bo luoi dich chuyen.

    LY DO: PSNR/SSIM so tung pixel tuyet doi, cuc nhay voi lech vi tri du chi vai
    pixel (vd do camera-optimizer chi sua pose LUC TRAIN, render lai dung pose
    COLMAP tho, hoac sai so con lai trong dataparser_transform). LPIPS (so dac
    trung sau, dung sai khong gian tot hon) it bi anh huong boi kieu lech nay.
    Neu PSNR "tot nhat sau khi dich" cao hon han PSNR goc (vai dB), xac nhan phan
    lon diem bi mat la do LECH VI TRI he thong, khong phai sai noi dung/mau sac -
    huong sua se khac han (kiem tra pose/camera-optimizer) so voi huong "train
    them" hay "sua mau".

    CHI DUNG DE CHAN DOAN - khong dung shift nay de sua anh nop bai thuc te."""
    gt_gray = gt_arr.mean(axis=2)
    pred_gray = pred_arr.mean(axis=2)
    h, w = gt_gray.shape
    best_mse, best_dy, best_dx = float("inf"), 0, 0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0g, y1g = max(0, dy), h + min(0, dy)
            x0g, x1g = max(0, dx), w + min(0, dx)
            y0p, y1p = max(0, -dy), h + min(0, -dy)
            x0p, x1p = max(0, -dx), w + min(0, -dx)
            g = gt_gray[y0g:y1g, x0g:x1g]
            p = pred_gray[y0p:y1p, x0p:x1p]
            if g.size == 0:
                continue
            mse = float(np.mean((g - p) ** 2))
            if mse < best_mse:
                best_mse, best_dy, best_dx = mse, dy, dx
    psnr_best = 10 * np.log10(1.0 / max(best_mse, 1e-10))
    return psnr_best, (best_dy, best_dx)


def compute_score(psnr: float, ssim: float, lpips: float, psnr_max: float) -> float:
    psnr_norm = max(0.0, min(psnr / psnr_max, 1.0))
    return 0.4 * (1.0 - lpips) + 0.3 * ssim + 0.3 * psnr_norm


def tensor_to_pil(rgb_hwc: torch.Tensor) -> Image.Image:
    arr = (rgb_hwc.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor_bchw(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def render_camera(model, cameras, idx: int, supersample_factor: float, device: str):
    """Render 1 camera tu dataset. Neu supersample_factor != 1.0, phong to
    intrinsics+resolution truoc khi render (Gaussian doc lap voi do phan giai
    render nen day la supersampling that su, khong phai upscale anh)."""
    camera = cameras[idx: idx + 1].to(device)
    if supersample_factor != 1.0:
        try:
            camera.rescale_output_resolution(supersample_factor)
        except Exception as e:
            print(f"[CANH BAO] Khong rescale duoc camera ({e}) - render o do phan giai goc.")
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera)
    return outputs["rgb"]


def render_camera_un_redistort(model, cameras, idx: int, supersample_factor: float, device: str,
                                prefilter: bool = False):
    """(Moi) Bien the un_redistort cua render_camera(): render camera PINHOLE
    (distortion_params ep ve 0, GIU nguyen fx/fy/cx/cy/pose) roi warp nguoc ve
    dung luoi meo cua camera GOC bang redistort_to_target() (xem
    distortion_redistort.py). CHUA VERIFY tren GPU that viec deepcopy() +
    zero-hoa distortion_params khong lam vo cache noi bo cua Cameras - kiem
    tra ky output truoc khi tin dung."""
    camera_orig = cameras[idx: idx + 1].to(device)
    render_w = int(camera_orig.width[0, 0])
    render_h = int(camera_orig.height[0, 0])
    render_fx = float(camera_orig.fx[0, 0])
    render_fy = float(camera_orig.fy[0, 0])
    render_cx = float(camera_orig.cx[0, 0])
    render_cy = float(camera_orig.cy[0, 0])
    k1 = float(camera_orig.distortion_params[0, 0]) if camera_orig.distortion_params is not None else 0.0
    k2 = float(camera_orig.distortion_params[0, 1]) if camera_orig.distortion_params is not None else 0.0

    camera_zero = copy.deepcopy(camera_orig)
    if camera_zero.distortion_params is not None:
        camera_zero.distortion_params = torch.zeros_like(camera_zero.distortion_params)
    if supersample_factor != 1.0:
        try:
            camera_zero.rescale_output_resolution(supersample_factor)
        except Exception as e:
            print(f"[CANH BAO] Khong rescale duoc camera ({e}) - render o do phan giai goc.")

    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera_zero)
    rgb_pinhole = outputs["rgb"].clamp(0, 1).detach()

    render_scale = supersample_factor if supersample_factor != 1.0 else 1.0
    out_chw = redistort_to_target(
        rgb_pinhole.permute(2, 0, 1),
        render_fx=render_fx * render_scale, render_fy=render_fy * render_scale,
        render_cx=render_cx * render_scale, render_cy=render_cy * render_scale,
        target_w=render_w, target_h=render_h,
        target_fx=render_fx, target_fy=render_fy, target_cx=render_cx, target_cy=render_cy,
        k1=k1, k2=k2, prefilter=prefilter,
    )
    return out_chw.permute(1, 2, 0)


def find_k_nearest_train_idx(eval_c2w: torch.Tensor, train_cameras, k: int = 1,
                              dir_weight: float = 0.5) -> list:
    """Tim K view train gan nhat voi 1 camera eval (ca 2 deu DA o training space
    nen so sanh truc tiep duoc, khong can dataparser transform). Dung khoang
    cach tam camera + phat huong nhin, giong logic trong render_and_submit.py.
    Tra ve danh sach [(idx, score), ...] sap theo score TANG DAN."""
    tc = eval_c2w[:3, 3]
    tf = eval_c2w[:3, 2]
    scored = []
    for i in range(len(train_cameras)):
        rc = train_cameras.camera_to_worlds[i][:3, 3]
        rf = train_cameras.camera_to_worlds[i][:3, 2]
        dist = float(torch.norm(rc - tc))
        denom = float(torch.norm(tf) * torch.norm(rf)) + 1e-9
        dir_dot = float(torch.dot(tf, rf)) / denom
        score = dist + dir_weight * (1.0 - dir_dot)
        scored.append((i, score))
    scored.sort(key=lambda x: x[1])
    return scored[:max(1, k)]


def estimate_gain(ref_render: Image.Image, ref_gt: Image.Image, lo=0.6, hi=1.7,
                   saturation_thresh: float = 0.92) -> np.ndarray:
    """ROBUST: loai pixel co gia tri kenh toi da (o CA render lan GT) vuot
    saturation_thresh TRUOC khi tinh mean - tranh vung loa/bao hoa (kim loai
    phan chieu) keo lech ty le gain. Xem estimate_gain_from_reference() trong
    render_and_submit.py (cung logic, duy tri song song vi 2 file dang lap
    logic render-time tu truoc)."""
    r = np.asarray(ref_render.convert("RGB")).astype(np.float32) / 255.0
    g = np.asarray(ref_gt.convert("RGB").resize(ref_render.size, Image.LANCZOS)).astype(np.float32) / 255.0
    mask = (r.max(axis=-1) <= saturation_thresh) & (g.max(axis=-1) <= saturation_thresh)
    if mask.sum() < 0.05 * mask.size:
        print("[CANH BAO][estimate_gain] Mask loai outlier con lai <5% pixel - dung lai toan bo pixel.")
        mask = np.ones_like(mask)
    gains = []
    for c in range(3):
        val = float(g[..., c][mask].mean()) / max(float(r[..., c][mask].mean()), 1e-4)
        gains.append(min(max(val, lo), hi))
    return np.array(gains, dtype=np.float32)


def gain_weights_from_scores(scores: list, eps: float = 1e-6) -> np.ndarray:
    """w_i = 1/(score_i + eps), chuan hoa tong = 1 - xem ham cung ten trong
    render_and_submit.py (giu song song, khong import cheo giua 2 script
    render doc lap)."""
    inv = np.array([1.0 / (s + eps) for s in scores], dtype=np.float64)
    return inv / inv.sum()


def combine_gains(gains: list, weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(gains, axis=0)
    return (stacked * weights[:, None]).sum(axis=0).astype(np.float32)


def combine_lab_stats(stats_list: list, weights: np.ndarray):
    """XAP XI don gian hoa (trung binh co trong so cua std, khong phai
    pooled-variance chinh xac) - xem ham cung ten trong render_and_submit.py."""
    means = np.stack([s[0] for s in stats_list], axis=0)
    stds = np.stack([s[1] for s in stats_list], axis=0)
    mean = (means * weights[:, None]).sum(axis=0)
    std = (stds * weights[:, None]).sum(axis=0).clip(min=1e-6)
    return mean, std


def apply_gain(img: Image.Image, gain: np.ndarray) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    arr = np.clip(arr * gain[None, None, :], 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Duong dan config.yml cua scene da train")
    parser.add_argument("--supersample_factor", type=float, default=1.0,
                         help="Render o do phan giai cao hon roi downsample LANCZOS (vd 1.5, 2.0)")
    parser.add_argument("--appearance_transfer", action="store_true",
                         help="Bu phoi sang bang view train gan nhat (mo phong dung duong nop bai)")
    parser.add_argument("--appearance_mode", type=str, default="rgb_gain",
                         choices=["rgb_gain", "lab_transfer"],
                         help="(Moi) 'rgb_gain' (mac dinh, khong doi hanh vi cu) hay 'lab_transfer' "
                              "(Reinhard 2001, sua ca contrast/phan tan mau trong khong gian LAB - xem "
                              "image_postprocess.py). Dung de do Score THAT truoc khi doi mac dinh o "
                              "render_and_submit.py.")
    parser.add_argument("--appearance_k_neighbors", type=int, default=1,
                         help="(Moi) So view train GAN NHAT dung lam reference, ket hop qua trung binh "
                              "co trong so. Mac dinh 1 (khong doi hanh vi cu). THU 2-3 de on dinh hon.")
    parser.add_argument("--psnr_max", type=float, default=50.0)
    parser.add_argument("--max_images", type=int, default=0, help="Chi do N anh dau (0 = tat ca)")
    parser.add_argument("--diagnose_misalignment", action="store_true",
                         help="Do them PSNR 'tot nhat sau khi dich +-N pixel' de biet PSNR/SSIM "
                              "thap co phai do LECH VI TRI he thong (vd camera-optimizer chi sua "
                              "pose luc train) thay vi sai noi dung/mau sac. Cham hon (~1-3s/anh).")
    parser.add_argument("--max_shift_px", type=int, default=10,
                         help="Bien do dich pixel toi da khi --diagnose_misalignment")
    parser.add_argument("--out_csv", type=str, default="submission_path_scores.csv")
    parser.add_argument("--jpeg_quality", type=int, default=0,
                         help="(Moi) >0 = mo phong buoc ma hoa JPEG cuoi cung cua render_and_submit.py "
                              "truoc khi tinh metric. Mac dinh 0 = TAT (giu nguyen hanh vi cu de con so "
                              "cu van so sanh duoc). Anh nop bai THUC TE luon bi nen JPEG, nen muon biet "
                              "diem THAT phai bat cai nay - dat 75 de tai hien bai nop cu (PIL mac dinh), "
                              "95 de do muc sau khi da sua.")
    parser.add_argument("--lowpass_sigma", type=float, default=0.3,
                         help="Sigma cua --post_process lowpass. 0.3 la muc khop pho tan so voi anh chup "
                              "that nhat (do tren 60 anh nop bai HCM0421 vs 240 anh that).")
    parser.add_argument("--despeckle_threshold", type=float, default=0.03,
                         help="Nguong cua --post_process despeckle: va diem VUA toi hon trung vi cuc bo "
                              "qua nguong nay VUA toi tuyet doi. Do tren du lieu that: render co 0.94% "
                              "diem gan-den vs 0.21% o anh chup that (thua 4.5x, phan bo deu khap anh = "
                              "lo thung tai tao). 0.03 la muc can bang tot nhat do duoc.")
    parser.add_argument("--post_process", type=str, default="none",
                         choices=["none", "denoise", "sharpen", "denoise_sharpen", "lowpass",
                                  "despeckle", "despeckle_lowpass"],
                         help="(Moi) Hau xu ly 2D anh render TRUOC KHI tinh PSNR/SSIM/LPIPS - mo phong "
                              "DUNG anh se duoc cham diem neu bat cung flag nay o render_and_submit.py. "
                              "Mac dinh 'none' (an toan). Dung script nay de do bang so truoc khi quyet dinh bat.")
    parser.add_argument("--denoise_strength", type=float, default=0.3)
    parser.add_argument("--sharpen_amount", type=float, default=0.5)
    parser.add_argument("--distortion_mode", type=str, default="rasterizer",
                         choices=["rasterizer", "un_redistort"],
                         help="(Moi) 'rasterizer' (mac dinh, khong doi hanh vi cu) hay 'un_redistort' "
                              "(render pinhole roi warp nguoc - xem distortion_redistort.py). Dung de "
                              "do Score THAT truoc khi doi mac dinh o render_and_submit.py.")
    parser.add_argument("--resample_prefilter", action="store_true",
                         help="(Moi) Chi co tac dung voi --distortion_mode un_redistort. Loc trung binh "
                              "vung (area/box filter) ve target size TRUOC khi grid_sample ap meo, thay vi "
                              "de bilinear 2x2 tu giam do phan giai - bilinear chi doc 4 diem anh nen voi "
                              "supersample > 2 se vut bo phan lon thong tin va gay rang cua. Nghi ngo day la "
                              "nguyen nhan Score CHUNG LAI o supersample 2.5-3.0. Bat co thu, do bang so.")
    parser.add_argument("--load_step", "--load-step", type=int, default=None,
                         help="(Moi) Neu dat, ep eval_setup load DUNG checkpoint step nay "
                              "(vd 20000) thay vi mac dinh luon lay checkpoint MOI NHAT trong "
                              "load_dir. Dung de so sanh Score giua cac checkpoint (vd 10000 vs "
                              "20000 vs checkpoint cuoi) va chon checkpoint tot nhat, vi da co "
                              "bang chung thuc te checkpoint giua chung co the tot hon checkpoint "
                              "cuoi (overfitting/thoai hoa cuoi qua trinh train).")
    args = parser.parse_args()

    from nerfstudio.utils.eval_utils import eval_setup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = Path(args.config)
    print(f"Load checkpoint: {config_path}" +
          (f" (step={args.load_step})" if args.load_step is not None else " (step=moi nhat)"))

    def _force_load_step(cfg):
        if args.load_step is not None:
            cfg.load_step = args.load_step
        return cfg

    config, pipeline, _, _ = eval_setup(config_path, test_mode="inference",
                                         update_config_callback=_force_load_step)
    pipeline.eval()
    model = pipeline.model

    dm = pipeline.datamanager
    eval_dataset = dm.eval_dataset
    train_dataset = dm.train_dataset
    n_eval = len(eval_dataset)
    if args.max_images > 0:
        n_eval = min(n_eval, args.max_images)
    print(f"So anh eval (co ground-truth) se do: {n_eval} / {len(eval_dataset)}")
    print(f"Cau hinh render: supersample_factor={args.supersample_factor} "
          f"appearance_transfer={args.appearance_transfer}")

    gain_cache = {}
    psnrs, ssims, lpipss = [], [], []
    shift_gains, control_gains = [], []
    gt_cache = {}  # luu anh GT da giai ma de dung lam doi chung "khong lien quan"

    for i in range(n_eval):
        gt_hwc = eval_dataset[i]["image"]
        if gt_hwc.dtype != torch.float32:
            gt_hwc = gt_hwc.float() / 255.0
        gt_pil = tensor_to_pil(gt_hwc)
        target_size = gt_pil.size

        if args.distortion_mode == "un_redistort":
            rgb = render_camera_un_redistort(model, eval_dataset.cameras, i, args.supersample_factor, device,
                                              prefilter=args.resample_prefilter)
        else:
            rgb = render_camera(model, eval_dataset.cameras, i, args.supersample_factor, device)
        pred_pil = tensor_to_pil(rgb)

        if args.appearance_transfer:
            eval_c2w = eval_dataset.cameras.camera_to_worlds[i]
            top_k = find_k_nearest_train_idx(eval_c2w, train_dataset.cameras, k=args.appearance_k_neighbors)
            per_ref_corrections = []
            for ref_idx, _ in top_k:
                if ref_idx not in gain_cache:
                    # (SUA LOI NHAT QUAN) Reference PHAI render bang DUNG che do
                    # voi anh eval chinh (dong 295-298) - truoc day luon dung
                    # render_camera() rasterizer bat ke --distortion_mode.
                    #
                    # (SUA LOI NHAT QUAN #2, phat hien qua audit) Truoc day dung
                    # supersample_factor=1.0 (full-res) o day, trong khi
                    # render_and_submit.py::render_pose_lowres() dung render_scale=0.35
                    # cho DUNG viec nay (chi can thong ke mau, khong can chi tiet) khi
                    # tao submission that. 2 do phan giai khac nhau cho gain/LAB-stats
                    # khac nhau (du nho, vi la thong ke trung binh kenh mau) -> Score do
                    # o day khong con mo phong CHINH XAC duong nop bai that. Dong bo lai
                    # ve REF_RENDER_SCALE = 0.35 cho khop.
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
                        gain_cache[ref_idx] = (lab_color_transfer_stats(ref_render_pil), lab_color_transfer_stats(ref_gt_pil))
                    else:
                        gain_cache[ref_idx] = estimate_gain(ref_render_pil, ref_gt_pil)
                per_ref_corrections.append(gain_cache[ref_idx])

            if len(top_k) == 1:
                correction = per_ref_corrections[0]
            else:
                weights = gain_weights_from_scores([s for _, s in top_k])
                if args.appearance_mode == "lab_transfer":
                    src_stats = combine_lab_stats([c[0] for c in per_ref_corrections], weights)
                    dst_stats = combine_lab_stats([c[1] for c in per_ref_corrections], weights)
                    correction = (src_stats, dst_stats)
                else:
                    correction = combine_gains(per_ref_corrections, weights)

            if args.appearance_mode == "lab_transfer":
                src_stats, dst_stats = correction
                pred_pil = apply_lab_color_transfer(pred_pil, src_stats, dst_stats)
            else:
                pred_pil = apply_gain(pred_pil, correction)

        if pred_pil.size != target_size:
            pred_pil = pred_pil.resize(target_size, Image.LANCZOS)

        # (Moi) Ap dung DUNG luc voi render_and_submit.py (sau resize, truoc
        # khi tinh metric/luu) - de con so o day phan anh CHINH XAC Score se
        # nhan duoc neu bat cung --post_process cho submission that.
        if args.post_process != "none":
            pred_pil = enhance_image(pred_pil, mode=args.post_process,
                                      denoise_strength=args.denoise_strength,
                                      sharpen_amount=args.sharpen_amount,
                                      lowpass_sigma=args.lowpass_sigma,
                                      despeckle_threshold=args.despeckle_threshold)

        # (Moi - VA LAP LO HONG DO LUONG) Mo phong DUNG buoc ma hoa JPEG cuoi
        # cung cua render_and_submit.py. Truoc day script nay tinh metric tren
        # anh PIL trong BO NHO, bo qua hoan toan viec anh nop bai bi nen JPEG -
        # nghia la moi con so A/B tu truoc toi nay (0.7151, 0.7236...) deu LAC
        # QUAN hon bai nop that. Dat o day (sau post_process, ngay truoc khi
        # tinh metric) de khop chinh xac thu tu cua duong nop bai.
        if args.jpeg_quality > 0:
            buf = io.BytesIO()
            pred_pil.save(buf, format="JPEG", quality=args.jpeg_quality, subsampling=0)
            buf.seek(0)
            pred_pil = Image.open(buf).convert("RGB")

        pred_t = pil_to_tensor_bchw(pred_pil, device)
        gt_t = pil_to_tensor_bchw(gt_pil, device)

        with torch.no_grad():
            p = float(model.psnr(gt_t, pred_t))
            s = float(model.ssim(gt_t, pred_t))
            l = float(model.lpips(gt_t, pred_t))
        psnrs.append(p); ssims.append(s); lpipss.append(l)
        img_name = Path(eval_dataset.image_filenames[i]).name
        line = (f"  [{i + 1}/{n_eval}] {img_name} psnr={p:.3f} ssim={s:.4f} lpips={l:.4f} "
                f"score={compute_score(p, s, l, args.psnr_max):.4f}")

        if args.diagnose_misalignment:
            gt_np = np.asarray(gt_pil).astype(np.float32) / 255.0
            pred_np = np.asarray(pred_pil).astype(np.float32) / 255.0
            gt_cache[i] = gt_np
            p_best, (dy, dx) = best_shift_psnr(gt_np, pred_np, args.max_shift_px)
            gain = p_best - p
            shift_gains.append((gain, dy, dx))
            line += f"  | PSNR sau dich toi uu={p_best:.3f} (dich=({dy},{dx})px, gain={gain:+.2f}dB)"

            # DOI CHUNG: so cung anh render voi GT cua 1 anh KHAC (khong lien quan)
            # de do "muc lac quan gia tao" cua chinh phep do nay tren du lieu that -
            # da xac nhan qua test rieng: dich chuyen tim duoc TOT NHAT trong nhieu
            # ung vien luon lac quan hon thuc te (thien lech chon-max thong ke), can
            # tru phan nay di truoc khi ket luan co misalignment that hay khong.
            if i > 0:
                other_gt = gt_cache[i - 1]
                if other_gt.shape == pred_np.shape:
                    p_ctrl_raw = 10 * np.log10(1.0 / max(float(np.mean((other_gt - pred_np) ** 2)), 1e-10))
                    p_ctrl_best, _ = best_shift_psnr(other_gt, pred_np, args.max_shift_px)
                    control_gains.append(p_ctrl_best - p_ctrl_raw)

        print(line)

    avg_psnr = sum(psnrs) / len(psnrs)
    avg_ssim = sum(ssims) / len(ssims)
    avg_lpips = sum(lpipss) / len(lpipss)
    score = compute_score(avg_psnr, avg_ssim, avg_lpips, args.psnr_max)

    print("\n=== KET QUA TREN DUNG DUONG RENDER NOP BAI ===")
    print(f"  supersample_factor = {args.supersample_factor}")
    print(f"  appearance_transfer = {args.appearance_transfer} (mode={args.appearance_mode}, "
          f"k_neighbors={args.appearance_k_neighbors})")
    print(f"  post_process = {args.post_process} (denoise_strength={args.denoise_strength}, sharpen_amount={args.sharpen_amount})")
    print(f"  distortion_mode = {args.distortion_mode} (resample_prefilter={args.resample_prefilter})")
    print(f"  jpeg_quality = {args.jpeg_quality if args.jpeg_quality > 0 else 'TAT (khong mo phong nen JPEG)'}")
    print(f"  PSNR  = {avg_psnr:.3f}")
    print(f"  SSIM  = {avg_ssim:.4f}")
    print(f"  LPIPS = {avg_lpips:.4f}")
    print(f"  SCORE = {score:.4f}   (0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR/{args.psnr_max:g})")

    if args.diagnose_misalignment and shift_gains:
        avg_gain = sum(g for g, _, _ in shift_gains) / len(shift_gains)
        avg_dy = sum(abs(dy) for _, dy, _ in shift_gains) / len(shift_gains)
        avg_dx = sum(abs(dx) for _, _, dx in shift_gains) / len(shift_gains)
        print(f"\n=== CHAN DOAN MISALIGNMENT ===")
        print(f"  PSNR tang trung binh neu cho phep dich toi uu +-{args.max_shift_px}px: {avg_gain:+.2f} dB (RAW - chua tru thien lech)")
        print(f"  Do dich trung binh (tri tuyet doi): dy={avg_dy:.1f}px  dx={avg_dx:.1f}px")

        if control_gains:
            # QUAN TRONG: da tu kiem chung (xem docstring best_shift_psnr) rang tim
            # "shift tot nhat trong N ung vien" luon lac quan hon thuc te, KE CA khi
            # 2 anh khong lien quan gi nhau (thien lech chon-max thong ke thuan tuy).
            # Doi chung nay do dung muc lac quan gia tao DO CHINH BO DU LIEU NAY gay
            # ra, roi tru di - cong bang hon han 1 nguong co dinh dat truoc.
            avg_ctrl = sum(control_gains) / len(control_gains)
            net_gain = avg_gain - avg_ctrl
            print(f"  Doi chung (so voi anh GT KHONG lien quan): thien lech gia tao = {avg_ctrl:+.2f} dB")
            print(f"  ==> GAIN THUC (da tru thien lech) = {net_gain:+.2f} dB")
            if net_gain > 1.5:
                print(f"  [KET LUAN] Gain thuc > 1.5dB sau khi tru doi chung -> XAC NHAN misalignment "
                      f"he thong la nguyen nhan dang ke. Uu tien kiem tra camera-optimizer.mode (SO3xR3 "
                      f"chi sua pose LUC TRAIN, khong sua luc render) va dataparser_transform/distortion.")
            elif net_gain > 0.5:
                print(f"  [KET LUAN] Gain thuc vua phai (0.5-1.5dB) - co mot phan do lech vi tri nhung "
                      f"khong phai nguyen nhan duy nhat. Ket hop voi huong sua noi dung/artifact.")
            else:
                print(f"  [KET LUAN] Gain thuc nho (<0.5dB sau khi tru doi chung) -> lech vi tri KHONG "
                      f"phai van de chinh. PSNR/SSIM thap chu yeu do sai noi dung that su (floater, mo, "
                      f"sai mau) - tap trung huong do thay vi nghi ngo camera-optimizer.")
        else:
            print(f"  [CANH BAO] Khong tinh duoc doi chung (can >=2 anh) - GAIN RAW o tren CHUA DANG TIN, "
                  f"da xac nhan qua kiem thu rang no lac quan hon thuc te. Chay lai voi --max_images >= 3.")

    out_csv = Path(args.out_csv)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["config_path", "supersample_factor", "appearance_transfer", "appearance_mode",
                        "appearance_k_neighbors", "post_process", "denoise_strength", "sharpen_amount",
                        "distortion_mode", "n_images", "psnr", "ssim", "lpips", "score"])
        # Ghi them load_step va resample_prefilter NGAY TRONG 2 cot san (dang chuoi)
        # thay vi them cot moi: header CSV chi duoc ghi 1 lan, nen doi so cot se lam
        # cac dong moi lech han so voi header cu cua file da ton tai - loi am tham
        # kho phat hien hon nhieu so voi ten dai mot chut.
        cfg_field = f"{config_path}#step{args.load_step}" if args.load_step is not None else str(config_path)
        dist_field = args.distortion_mode + ("+prefilter" if args.resample_prefilter else "")
        if args.jpeg_quality > 0:
            dist_field += f"+jpeg{args.jpeg_quality}"
        w.writerow([cfg_field, args.supersample_factor, args.appearance_transfer, args.appearance_mode,
                    args.appearance_k_neighbors, args.post_process, args.denoise_strength, args.sharpen_amount,
                    dist_field, n_eval, f"{avg_psnr:.4f}", f"{avg_ssim:.5f}", f"{avg_lpips:.5f}", f"{score:.5f}"])
    print(f"\nDa ghi them 1 dong vao {out_csv} - chay lai voi cau hinh khac de so sanh truc tiep.")


if __name__ == "__main__":
    main()
