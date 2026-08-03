

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from image_postprocess import enhance_image, lab_color_transfer_stats, apply_lab_color_transfer
from distortion_redistort import redistort_to_target

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def row_to_c2w_ns_raw(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """world-to-camera (qvec,tvec) -> camera-to-world 4x4, dung convention
    'Nerfstudio raw' (sau khi flip truc COLMAP->OpenGL-style, GIONG HET
    nhung TRUOC KHI ap dataparser_transform/scale). Day chinh la dang pose
    duoc ghi trong transforms.json['frames'][i]['transform_matrix']."""
    w2c = np.eye(4)
    w2c[:3, :3] = qvec_to_rotmat(qvec)
    w2c[:3, 3] = tvec
    c2w = np.linalg.inv(w2c)
    c2w[0:3, 1:3] *= -1
    return c2w


def load_dataparser_transform(config_path: Path):
    """Doc dataparser_transforms.json (luu cung thu muc voi config.yml sau
    khi train xong). Tra ve (T 4x4, scale float)."""
    transform_path = config_path.parent / "dataparser_transforms.json"
    if not transform_path.exists():
        print(f"[CANH BAO] Khong tim thay {transform_path}. Dung transform=identity, scale=1.0 "
              f"- CHI dung neu ban chac chan train voi auto_scale_poses=False va "
              f"orientation_method=none (KHONG phai mac dinh Nerfstudio). Neu khong, render se sai.")
        return np.eye(4), 1.0
    with open(transform_path) as f:
        d = json.load(f)
    T = np.eye(4)
    T[:3, :4] = np.array(d["transform"])
    scale = float(d["scale"])
    return T, scale


def apply_dataparser_transform(c2w_ns_raw: np.ndarray, T: np.ndarray, scale: float) -> np.ndarray:
    """Chuyen pose tu 'Nerfstudio raw space' (dung quy uoc trong
    transforms.json, dau ra cua row_to_c2w_ns_raw) sang 'training space' -
    khong gian thuc su model da hoc trong do.

    Cong thuc suy nguoc tu transform_poses_to_original_space() (chieu
    nguoc lai) trong nerfstudio/data/dataparsers/base_dataparser.py:
        goc:  P_raw_opencv = flip( inv(T) @ scale_div(pad(P_train), s) )
        =>    M = T @ pad(P_ns_raw)          # P_ns_raw = flip(P_raw_opencv)
              P_train.rotation    = M.rotation
              P_train.translation = M.translation * s
    """
    M = T @ np.vstack([c2w_ns_raw[:3, :4], [0.0, 0.0, 0.0, 1.0]])
    c2w_train = M[:3, :4].copy()
    c2w_train[:3, 3] *= scale
    return c2w_train


def load_pipeline(config_path: Path, device: str):
    from nerfstudio.utils.eval_utils import eval_setup

    config, pipeline, _, _ = eval_setup(config_path, test_mode="inference")
    pipeline.eval()
    return pipeline


_warned_fisheye_scenes = set()


def _normalize_intrinsics_entry(entry):
    """Ho tro ca dinh dang moi (dict co distortion, tu preprocess v2) va
    dinh dang cu / thieu (None) de khong crash."""
    if entry is None:
        return {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0, "camera_model": "OPENCV"}
    if isinstance(entry, dict):
        return {
            "k1": entry.get("k1", 0.0), "k2": entry.get("k2", 0.0),
            "k3": entry.get("k3", 0.0), "k4": entry.get("k4", 0.0),
            "p1": entry.get("p1", 0.0), "p2": entry.get("p2", 0.0),
            "camera_model": entry.get("camera_model", "OPENCV"),
        }
    return {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0, "camera_model": "OPENCV"}


def build_camera(row: pd.Series, c2w_train: np.ndarray, distortion_entry: dict, device: str, scene_id: str = ""):
    from nerfstudio.cameras import camera_utils
    from nerfstudio.cameras.cameras import Cameras, CameraType

    dist = _normalize_intrinsics_entry(distortion_entry)
    c2w_t = torch.tensor(c2w_train[:3, :4], dtype=torch.float32).unsqueeze(0)

    if dist["camera_model"] == "OPENCV_FISHEYE" and scene_id not in _warned_fisheye_scenes:
        _warned_fisheye_scenes.add(scene_id)
        print(f"[CANH BAO][{scene_id}] Camera model la OPENCV_FISHEYE nhung Splatfacto/gsplat CHI "
              f"ho tro rasterization voi camera perspective. Dung PERSPECTIVE + k1,k2 xap xi radial, "
              f"co the khong chinh xac cho goc rong that su.")

    is_fisheye = dist["camera_model"] == "OPENCV_FISHEYE"
    distortion_params = camera_utils.get_distortion_params(
        k1=dist["k1"], k2=dist["k2"],
        k3=0.0 if is_fisheye else dist.get("k3", 0.0),
        k4=0.0 if is_fisheye else dist.get("k4", 0.0),
        p1=dist["p1"], p2=dist["p2"],
    )

    camera = Cameras(
        camera_to_worlds=c2w_t,
        fx=torch.tensor([float(row.fx)]),
        fy=torch.tensor([float(row.fy)]),
        cx=torch.tensor([float(row.cx)]),
        cy=torch.tensor([float(row.cy)]),
        width=torch.tensor([[int(row.width)]]),
        height=torch.tensor([[int(row.height)]]),
        distortion_params=distortion_params.unsqueeze(0),
        camera_type=CameraType.PERSPECTIVE,
    )
    return camera.to(device)


def load_train_reference_frames(processed_data_root: Path, scene_id: str):
    """Doc transforms.json cua scene -> danh sach view train (pose ns_raw +
    duong dan anh GT + intrinsics). Dung lam 'reference' de uoc luong hieu
    chinh phoi sang/mau cho tung view test lan can (xem estimate_gain_from_reference)."""
    transforms_path = processed_data_root / scene_id / "transforms.json"
    images_dir = processed_data_root / scene_id / "images"
    if not transforms_path.exists():
        return [], None
    with open(transforms_path) as f:
        transforms = json.load(f)
    intr = (transforms["fl_x"], transforms["fl_y"], transforms["cx"], transforms["cy"],
            int(transforms["w"]), int(transforms["h"]))
    refs = []
    for fr in transforms.get("frames", []):
        gt_path = images_dir / Path(fr["file_path"]).name
        if not gt_path.exists():
            continue
        c2w = np.array(fr["transform_matrix"])
        refs.append({"c2w_ns_raw": c2w, "gt_path": gt_path})
    return refs, intr


def find_k_nearest_references(test_c2w_ns_raw: np.ndarray, refs: list, k: int = 1, exclude_path=None,
                               dir_weight: float = 0.5):
    """Tim K view train gan nhat voi 1 pose test, theo khoang cach tam camera +
    1 phan phat huong nhin (tranh chon view gan ve vi tri nhung quay nguoc
    huong). Ca 2 deu o khong gian ns_raw nen so sanh truc tiep duoc. Tra ve
    danh sach [(ref, score), ...] sap theo score TANG DAN (gan nhat truoc).

    K>1 (mac dinh 1 = hanh vi cu, chi 1 view gan nhat): dung trung binh co
    trong so cua nhieu view thay vi chi 1 view - on dinh hon neu view gan
    nhat NHAT tinh co anh sang/goc chup hoi bat thuong (xem gain_weights_from_scores)."""
    tc = test_c2w_ns_raw[:3, 3]
    tf = test_c2w_ns_raw[:3, 2]
    scored = []
    for r in refs:
        if exclude_path is not None and r["gt_path"] == exclude_path:
            continue
        rc = r["c2w_ns_raw"][:3, 3]
        rf = r["c2w_ns_raw"][:3, 2]
        dist = float(np.linalg.norm(rc - tc))
        dir_dot = float(np.dot(tf, rf) / (np.linalg.norm(tf) * np.linalg.norm(rf) + 1e-9))
        score = dist + dir_weight * (1.0 - dir_dot)
        scored.append((r, score))
    scored.sort(key=lambda x: x[1])
    return scored[:max(1, k)]


def gain_weights_from_scores(scores: list, eps: float = 1e-6) -> np.ndarray:
    """w_i = 1/(score_i + eps), chuan hoa tong = 1. Khong dung softmax/nhiet do
    de tranh them 1 hyperparameter can tune - ty le nghich don gian voi score
    (thap hon = gan hon = tin cay hon) la du dung cho muc dich hieu chinh gan dung."""
    inv = np.array([1.0 / (s + eps) for s in scores], dtype=np.float64)
    return inv / inv.sum()


def render_pose_lowres(pipeline, c2w_ns_raw, T, scale, intr, distortion_entry, device,
                        scene_id, render_scale=0.35, distortion_mode="rasterizer"):
    """Render 1 pose (voi intrinsics train) o do phan giai thap (chi de uoc
    luong thong ke mau, khong can chi tiet) -> tra ve PIL Image.

    distortion_mode="un_redistort" (SUA LOI NHAT QUAN, phat hien khi ra soat
    lai theo yeu cau nguoi dung): TRUOC DAY ham nay LUON dung rasterizer
    (distortion_entry day du) bat ke --distortion_mode dang chon cho anh
    chinh la gi - nghia la anh REFERENCE dung de uoc luong gain/LAB-stats bi
    render KHAC CACH voi anh dang duoc hieu chinh, khong nhat quan. Gio ho
    tro dung ca 2 che do, khop voi anh chinh dang render bang gi."""
    c2w_train = apply_dataparser_transform(c2w_ns_raw, T, scale)
    fx, fy, cx, cy, w, h = intr
    s = render_scale
    row = pd.Series({
        "fx": fx * s, "fy": fy * s, "cx": cx * s, "cy": cy * s,
        "width": max(1, int(round(w * s))), "height": max(1, int(round(h * s))),
    })
    dist_norm = _normalize_intrinsics_entry(distortion_entry)
    cam_dist_entry = distortion_entry
    if distortion_mode == "un_redistort":
        cam_dist_entry = {**dist_norm, "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0}

    cam = build_camera(row, c2w_train, cam_dist_entry, device, scene_id=scene_id)
    with torch.no_grad():
        out = pipeline.model.get_outputs_for_camera(cam)
    rgb_t = out["rgb"].clamp(0, 1).detach()

    if distortion_mode == "un_redistort":
        out_chw = redistort_to_target(
            rgb_t.permute(2, 0, 1),
            render_fx=float(row.fx), render_fy=float(row.fy),
            render_cx=float(row.cx), render_cy=float(row.cy),
            target_w=int(row.width), target_h=int(row.height),
            target_fx=float(row.fx), target_fy=float(row.fy),
            target_cx=float(row.cx), target_cy=float(row.cy),
            k1=dist_norm["k1"], k2=dist_norm["k2"],
        )
        rgb = out_chw.permute(1, 2, 0).cpu().numpy()
    else:
        rgb = rgb_t.cpu().numpy()
    return Image.fromarray((rgb * 255).astype(np.uint8))


def estimate_gain_from_reference(ref_render: Image.Image, ref_gt: Image.Image,
                                  lo: float = 0.6, hi: float = 1.7, saturation_thresh: float = 0.92):
    """Uoc luong he so hieu chinh mau/phoi sang PER-CHANNEL tu 1 cap
    (render view train, anh GT view train do). Vi CUNG GOC NHIN/CUNG NOI DUNG,
    ti le mean(GT)/mean(render) la uoc luong SACH cua phep hieu chinh ma
    bilateral grid da hoc nhung bi bo qua luc render test (khong bi nhieu boi
    khac biet noi dung/bau troi). Clip de tranh hieu chinh cuc doan.

    ROBUST: loai pixel co GIA TRI KENH TOI DA (o CA render lan GT) vuot
    saturation_thresh (thang 0-1) TRUOC khi tinh mean - vung loa/bao hoa (vd
    kim loai phan chieu tren cot/anten BTS) ma Gaussian Splatting hau nhu
    khong tai tao dung vi tri/cuong do se keo lech ty le gain neu tinh tren
    TOAN BO pixel. Fallback: neu mask con lai <5% tong pixel, dung lai toan
    bo pixel va in canh bao, tranh chia cho tap rong."""
    r = np.asarray(ref_render.convert("RGB")).astype(np.float32) / 255.0
    g = np.asarray(ref_gt.convert("RGB").resize(ref_render.size, Image.LANCZOS)).astype(np.float32) / 255.0
    mask = (r.max(axis=-1) <= saturation_thresh) & (g.max(axis=-1) <= saturation_thresh)
    if mask.sum() < 0.05 * mask.size:
        print("[CANH BAO][estimate_gain_from_reference] Mask loai outlier con lai <5% pixel - "
              "dung lai toan bo pixel (khong loai outlier) de tranh thong ke tren tap rong.")
        mask = np.ones_like(mask)
    gains = []
    for c in range(3):
        rr = float(r[..., c][mask].mean())
        gg = float(g[..., c][mask].mean())
        val = gg / max(rr, 1e-4)
        gains.append(min(max(val, lo), hi))
    return np.array(gains, dtype=np.float32)


def combine_gains(gains: list, weights: np.ndarray) -> np.ndarray:
    """Trung binh co trong so cua nhieu gain array (moi cai tu 1 reference
    khac nhau) - dung khi K>1 (xem find_k_nearest_references)."""
    stacked = np.stack(gains, axis=0)  # [K, 3]
    return (stacked * weights[:, None]).sum(axis=0).astype(np.float32)


def combine_lab_stats(stats_list: list, weights: np.ndarray):
    """Trung binh co trong so cua nhieu (mean, std) LAB stats. LUU Y: trung
    binh co trong so cua std la XAP XI don gian hoa (khong phai pooled-variance
    chinh xac ve mat thong ke) - du dung cho muc dich hieu chinh gan dung."""
    means = np.stack([s[0] for s in stats_list], axis=0)
    stds = np.stack([s[1] for s in stats_list], axis=0)
    mean = (means * weights[:, None]).sum(axis=0)
    std = (stds * weights[:, None]).sum(axis=0).clip(min=1e-6)
    return mean, std


def apply_gain(img: Image.Image, gain: np.ndarray) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    arr = np.clip(arr * gain[None, None, :], 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def build_appearance_context(pipeline, processed_data_root, scene_id, T, scale,
                              distortion_entry, device, render_scale=0.35, appearance_mode="rgb_gain",
                              k_neighbors: int = 1, distortion_mode: str = "rasterizer"):
    """Chuan bi ngu canh cho appearance transfer: danh sach reference + cache
    correction da tinh (moi reference chi render 1 lan du nhieu test pose cung
    tro toi no). appearance_mode: "rgb_gain" (mac dinh, hanh vi cu - 1 he so
    nhan vo huong/kenh RGB) hoac "lab_transfer" (Reinhard 2001 - sua ca
    trung binh lan contrast/phan tan mau trong khong gian LAB, xem
    image_postprocess.py). k_neighbors: so view train gan nhat dung lam
    reference (mac dinh 1 = hanh vi cu, chi 1 view). K>1 (vd 3) lay trung
    binh co trong so nhieu view - on dinh hon truoc anh sang/goc chup bat
    thuong cua 1 view don le. distortion_mode: (SUA LOI NHAT QUAN) reference
    PHAI duoc render BANG DUNG che do voi anh chinh, neu khong gain/LAB-stats
    se uoc luong tu 1 cach render khac voi anh dang can hieu chinh."""
    refs, intr = load_train_reference_frames(processed_data_root, scene_id)
    if not refs or intr is None:
        print(f"[{scene_id}] [appearance] Khong co reference train -> BO QUA appearance transfer scene nay.")
        return None
    return {
        "refs": refs, "intr": intr, "T": T, "scale": scale,
        "distortion_entry": distortion_entry, "device": device,
        "render_scale": render_scale, "appearance_mode": appearance_mode,
        "k_neighbors": k_neighbors, "distortion_mode": distortion_mode, "correction_cache": {},
    }


def _appearance_correction_for_single_ref(pipeline, ctx, ref, scene_id):
    """Render + tinh correction cho 1 reference (co cache theo duong dan GT)."""
    key = str(ref["gt_path"])
    if key in ctx["correction_cache"]:
        return ctx["correction_cache"][key]
    ref_render = render_pose_lowres(pipeline, ref["c2w_ns_raw"], ctx["T"], ctx["scale"],
                                     ctx["intr"], ctx["distortion_entry"], ctx["device"],
                                     scene_id, render_scale=ctx["render_scale"],
                                     distortion_mode=ctx.get("distortion_mode", "rasterizer"))
    ref_gt = Image.open(ref["gt_path"]).convert("RGB")
    if ctx["appearance_mode"] == "lab_transfer":
        correction = (lab_color_transfer_stats(ref_render), lab_color_transfer_stats(ref_gt))
    else:
        correction = estimate_gain_from_reference(ref_render, ref_gt)
    ctx["correction_cache"][key] = correction
    return correction


def appearance_correction_for_test_pose(pipeline, ctx, test_c2w_ns_raw, scene_id, exclude_path=None):
    """Tra ve 'correction' cho 1 pose test, dung K view train gan nhat
    (ctx["k_neighbors"], mac dinh 1) ket hop qua trung binh co trong so
    (xem gain_weights_from_scores):
    - mode "rgb_gain": mang gain per-channel (nhu truoc khi K=1).
    - mode "lab_transfer": tuple (src_stats, dst_stats) cho apply_lab_color_transfer()."""
    k = ctx.get("k_neighbors", 1)
    top_k = find_k_nearest_references(test_c2w_ns_raw, ctx["refs"], k=k, exclude_path=exclude_path)
    if not top_k:
        return np.ones(3, dtype=np.float32) if ctx["appearance_mode"] == "rgb_gain" else None

    if len(top_k) == 1:
        return _appearance_correction_for_single_ref(pipeline, ctx, top_k[0][0], scene_id)

    weights = gain_weights_from_scores([s for _, s in top_k])
    per_ref_corrections = [_appearance_correction_for_single_ref(pipeline, ctx, ref, scene_id)
                            for ref, _ in top_k]
    if ctx["appearance_mode"] == "lab_transfer":
        src_stats = combine_lab_stats([c[0] for c in per_ref_corrections], weights)
        dst_stats = combine_lab_stats([c[1] for c in per_ref_corrections], weights)
        return (src_stats, dst_stats)
    return combine_gains(per_ref_corrections, weights)


def apply_appearance_correction(img: Image.Image, correction, mode: str) -> Image.Image:
    """Dispatch ap dung correction theo dung mode da dung de tinh no."""
    if mode == "lab_transfer":
        if correction is None:
            return img
        src_stats, dst_stats = correction
        return apply_lab_color_transfer(img, src_stats, dst_stats)
    return apply_gain(img, correction)


def render_rows(pipeline, df: pd.DataFrame, distortion_entry: dict, T: np.ndarray, scale: float,
                 out_dir: Path, device: str, scene_id: str = "", supersample_factor: float = 1.0,
                 appearance_ctx=None, post_process: str = "none",
                 denoise_strength: float = 0.3, sharpen_amount: float = 0.5,
                 distortion_mode: str = "rasterizer", resample_prefilter: bool = False,
                 jpeg_quality: int = 95, lowpass_sigma: float = 0.3,
                 despeckle_threshold: float = 0.03):
    """supersample_factor > 1.0: render o do phan giai cao hon target roi downsample bang
    LANCZOS ve dung kich thuoc test_poses.csv yeu cau. Gaussian Splatting doc lap voi do phan
    giai render (chi phu thuoc camera intrinsics), nen ky thuat nay giup giam alias tren chi
    tiet manh (day cap/anten) - bu tot voi rasterize-mode=antialiased da bat luc train, giup
    ca LPIPS (nhay voi alias/artifact) lan SSIM.

    distortion_mode="rasterizer" (mac dinh): gsplat rasterize truc tiep voi distortion_params
    that -> resize LANCZOS thuong ve target_size (hanh vi cu, khong doi).
    distortion_mode="un_redistort": render PINHOLE (distortion=0, cung fx/fy/cx/cy), roi dung
    redistort_to_target() (xem distortion_redistort.py) de warp NGUOC ve dung luoi meo cua
    target - THAY THE cho resize LANCZOS (khong lam ca 2, tranh double-resample). Xem
    docstring distortion_redistort.py de biet ly do co ca 2 che do va gioi han pham vi (chi
    xu ly duoc meo radial k1/k2, khong xu ly p1/p2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_norm = _normalize_intrinsics_entry(distortion_entry)
    zero_dist_entry = {**dist_norm, "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0}

    for _, row in df.iterrows():
        qvec = np.array([row.qw, row.qx, row.qy, row.qz])
        tvec = np.array([row.tx, row.ty, row.tz])
        c2w_ns_raw = row_to_c2w_ns_raw(qvec, tvec)
        c2w_train = apply_dataparser_transform(c2w_ns_raw, T, scale)

        render_row = row
        if supersample_factor != 1.0:
            render_row = row.copy()
            render_row["width"] = int(round(row.width * supersample_factor))
            render_row["height"] = int(round(row.height * supersample_factor))
            render_row["fx"] = row.fx * supersample_factor
            render_row["fy"] = row.fy * supersample_factor
            render_row["cx"] = row.cx * supersample_factor
            render_row["cy"] = row.cy * supersample_factor

        cam_dist_entry = zero_dist_entry if distortion_mode == "un_redistort" else distortion_entry
        camera = build_camera(render_row, c2w_train, cam_dist_entry, device, scene_id=scene_id)

        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)

        rgb_t = outputs["rgb"].clamp(0, 1).detach()

        target_size = (int(row.width), int(row.height))
        if distortion_mode == "un_redistort":
            # Warp truc tiep tu buffer render (co the o do phan giai supersample)
            # ve dung luoi meo cua target trong 1 lan grid_sample - thay resize LANCZOS.
            rgb_chw = rgb_t.permute(2, 0, 1)
            out_chw = redistort_to_target(
                rgb_chw,
                render_fx=float(render_row.fx), render_fy=float(render_row.fy),
                render_cx=float(render_row.cx), render_cy=float(render_row.cy),
                target_w=int(row.width), target_h=int(row.height),
                target_fx=float(row.fx), target_fy=float(row.fy),
                target_cx=float(row.cx), target_cy=float(row.cy),
                k1=dist_norm["k1"], k2=dist_norm["k2"],
                prefilter=resample_prefilter,
            )
            rgb = out_chw.permute(1, 2, 0).cpu().numpy()
        else:
            rgb = rgb_t.cpu().numpy()
        img = Image.fromarray((rgb * 255).astype(np.uint8))

        # (Vu khi canh tranh) Hieu chinh phoi sang/mau luc render test bang
        # view train gan nhat - bu lai phep hieu chinh cua bilateral grid da
        # hoc luc train nhung bi BO QUA hoan toan luc render test (grid chi
        # active khi self.training). Ap TRUOC khi downsample.
        if appearance_ctx is not None:
            correction = appearance_correction_for_test_pose(pipeline, appearance_ctx, c2w_ns_raw, scene_id)
            img = apply_appearance_correction(img, correction, appearance_ctx["appearance_mode"])

        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)

        # (Moi) Hau xu ly 2D (denoise/sharpen) - AP SAU CUNG, ngay truoc khi
        # luu anh se duoc cham diem. Mac dinh post_process="none" = no-op
        # tuyet doi, khong doi hanh vi cu. CHI bat sau khi da verify bang so
        # qua --sanity_check (xem image_postprocess.py de biet ly do).
        if post_process != "none":
            img = enhance_image(img, mode=post_process,
                                 denoise_strength=denoise_strength, sharpen_amount=sharpen_amount,
                                 lowpass_sigma=lowpass_sigma, despeckle_threshold=despeckle_threshold)

        save_submission_image(img, out_dir / row.image_name, jpeg_quality)


def save_submission_image(img: Image.Image, path: Path, jpeg_quality: int = 95):
    """Luu anh nop bai voi chat luong JPEG CHI DINH RO.

    === VI SAO CAN HAM NAY (loi that, do bang so) ===
    Truoc day chi goi img.save(path). Voi duoi .JPG, PIL mac dinh quality=75 va
    chroma subsampling 4:2:0 - tuc la NEN MAT MAT NANG ngay tren chinh nhung
    anh se duoc cham diem. Do tren anh that cua scene HCM0421:
        quality=75 -> 32.31 dB   (dang dung)
        quality=90 -> 36.36 dB
        quality=95 -> 51.38 dB
    Bang luong tu JPEG cua anh nop bai cu = 4638 trong khi anh chup that = 927
    (cang nho cang net), file 318 KB vs 707 KB - xac nhan nen qua manh.
    Cong sai so binh phuong voi PSNR dang do 22.402 dB: bo phan mat mat nay
    tra lai ~0.47 dB PSNR MIEN PHI. Va nhieu KHOI (blocking) do JPEG chinh la
    loai meo LPIPS phat rat nang, ma LPIPS chiem 40% diem so - nen loi ich
    thuc te nhieu kha nang lon hon con so PSNR goi y.

    subsampling=0 ep 4:4:4 (khong giam do phan giai kenh mau) - quan trong voi
    SSIM/LPIPS vi chi tiet mau o canh mai nha/day cap khong bi tron.
    Chi ap cho JPEG; duoi khac (vd .png) giu nguyen hanh vi cu."""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.save(path, quality=jpeg_quality, subsampling=0)
    else:
        img.save(path)


def make_zip(submission_root: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in submission_root.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(submission_root))


def find_latest_config(checkpoints_root: Path, scene_id: str):
    candidates = sorted(checkpoints_root.glob(f"{scene_id}/**/config.yml"))
    return candidates[-1] if candidates else None


def compute_psnr(gt_img: Image.Image, pred_img: Image.Image) -> float:
    gt_arr = np.asarray(gt_img.convert("RGB")).astype(np.float32) / 255.0
    pred_arr = np.asarray(pred_img.convert("RGB")).astype(np.float32) / 255.0
    mse = np.mean((gt_arr - pred_arr) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def _psnr_from_arrays(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    mse = np.mean((gt_arr - pred_arr) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def compute_region_psnr(gt_img: Image.Image, pred_img: Image.Image) -> dict:
    """PSNR rieng cho vung TRUNG TAM (crop 50%x50% giua anh) va vung GOC (hop
    4 crop goc, moi crop 25%x25%) - de phat hien lech chat luong theo VI TRI
    trong khung hinh (vd mo goc do meo/mat do), thay vi chi trung binh toan
    anh nhu compute_psnr(). Vung goc la noi distortion radial manh nhat (ban
    kinh tu tam anh lon nhat) nen dung de doi chieu rasterizer vs un_redistort."""
    gt_arr = np.asarray(gt_img.convert("RGB")).astype(np.float32) / 255.0
    pred_arr = np.asarray(pred_img.convert("RGB")).astype(np.float32) / 255.0
    if pred_arr.shape != gt_arr.shape:
        pred_arr = np.asarray(
            pred_img.convert("RGB").resize(gt_img.size, Image.LANCZOS)
        ).astype(np.float32) / 255.0
    h, w = gt_arr.shape[:2]
    ch, cw = int(h * 0.25), int(w * 0.25)

    center = gt_arr[ch:h - ch, cw:w - cw]
    center_pred = pred_arr[ch:h - ch, cw:w - cw]

    corner_gt = np.concatenate([
        gt_arr[:ch, :cw].reshape(-1, 3), gt_arr[:ch, w - cw:].reshape(-1, 3),
        gt_arr[h - ch:, :cw].reshape(-1, 3), gt_arr[h - ch:, w - cw:].reshape(-1, 3),
    ], axis=0)
    corner_pred = np.concatenate([
        pred_arr[:ch, :cw].reshape(-1, 3), pred_arr[:ch, w - cw:].reshape(-1, 3),
        pred_arr[h - ch:, :cw].reshape(-1, 3), pred_arr[h - ch:, w - cw:].reshape(-1, 3),
    ], axis=0)

    return {
        "center": _psnr_from_arrays(center, center_pred),
        "corner": _psnr_from_arrays(corner_gt, corner_pred),
    }


def sanity_check_with_train_images(scene_id: str, config_path: Path, processed_data_root: Path,
                                    distortion_entry: dict, device: str, num_samples: int = 3,
                                    check_appearance: bool = False, post_process: str = "none",
                                    denoise_strength: float = 0.3, sharpen_amount: float = 0.5,
                                    diagnose_distortion: bool = False, appearance_mode: str = "rgb_gain",
                                    appearance_k_neighbors: int = 1):
    """Render lai vai anh TRAIN da biet ca pose (transform_matrix co san
    trong transforms.json) lan anh goc, roi so PSNR CO/KHONG ap dung
    dataparser_transform. Day la cach kiem chung bang so xem fix quan
    trong nhat co thuc su can thiet va dung khong, thay vi tin theo ly thuyet.

    Neu check_appearance=True: cung do them PSNR khi ap appearance transfer
    (dung view train GAN NHAT KHAC lam reference, gia lap dung tinh huong luc
    render test) - de xac nhan vu khi nay thuc su tang PSNR TRUOC khi bat cho
    test set that.

    Neu post_process != "none": do them PSNR sau khi ap denoise/sharpen (xem
    image_postprocess.py) - BAT BUOC kiem tra bang so o day TRUOC khi bat
    --post_process cho submission that, vi sharpen qua tay co the lam Score
    TE HON thay vi tot hon (rui ro 2 chieu, khong chi "nhin dep hon").

    Neu diagnose_distortion=True: render THEM 1 lan bang che do "un_redistort"
    (xem distortion_redistort.py) tren CUNG anh/pose, so PSNR VUNG (trung tam
    vs goc, xem compute_region_psnr()) giua rasterizer (mac dinh, pred_fix) va
    un_redistort - vung goc la noi distortion manh nhat nen la noi phan biet
    ro nhat 2 gia thuyet "mo do mat do COLMAP thua" vs "mo do rasterizer ap
    meo sai".
    """
    import random

    transforms_path = processed_data_root / scene_id / "transforms.json"
    images_dir = processed_data_root / scene_id / "images"
    if not transforms_path.exists():
        print(f"[sanity_check][{scene_id}] Khong tim thay {transforms_path}, bo qua.")
        return

    with open(transforms_path) as f:
        transforms = json.load(f)
    frames = transforms["frames"]
    sample_frames = random.sample(frames, min(num_samples, len(frames)))

    T, scale = load_dataparser_transform(config_path)
    pipeline = load_pipeline(config_path, device)

    appearance_ctx = None
    if check_appearance:
        # LUU Y: pred_fix trong ham nay LUON render bang rasterizer (xem "CO ap
        # dung fix" ben duoi, khong doi theo tham so distortion_mode) - nen
        # reference cung PHAI dung rasterizer de nhat quan voi pred_fix dang
        # duoc hieu chinh, bat ke --distortion_mode nguoi dung chon cho lan
        # render submission that.
        appearance_ctx = build_appearance_context(
            pipeline, processed_data_root, scene_id, T, scale, distortion_entry, device,
            appearance_mode=appearance_mode, k_neighbors=appearance_k_neighbors,
            distortion_mode="rasterizer")

    psnr_with_fix, psnr_without_fix = [], []
    psnr_appearance = []
    psnr_post_process = []
    corner_rasterizer_list, corner_un_redistort_list = [], []
    dist_norm = _normalize_intrinsics_entry(distortion_entry)
    for frame in sample_frames:
        c2w_ns_raw = np.array(frame["transform_matrix"])
        img_path = images_dir / Path(frame["file_path"]).name
        if not img_path.exists():
            continue
        gt_img = Image.open(img_path).convert("RGB")
        w, h = gt_img.size

        fake_row = pd.Series({
            "fx": transforms["fl_x"], "fy": transforms["fl_y"],
            "cx": transforms["cx"], "cy": transforms["cy"],
            "width": w, "height": h,
        })

        # CO ap dung fix (dung)
        c2w_train = apply_dataparser_transform(c2w_ns_raw, T, scale)
        camera = build_camera(fake_row, c2w_train, distortion_entry, device, scene_id=scene_id)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        rgb = outputs["rgb"].clamp(0, 1).detach().cpu().numpy()
        pred_fix = Image.fromarray((rgb * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
        p_fix = compute_psnr(gt_img, pred_fix)
        psnr_with_fix.append(p_fix)

        # KHONG ap dung fix (nhu cac ban truoc day)
        camera_nofix = build_camera(fake_row, c2w_ns_raw, distortion_entry, device, scene_id=scene_id)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera_nofix)
        rgb = outputs["rgb"].clamp(0, 1).detach().cpu().numpy()
        pred_nofix = Image.fromarray((rgb * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
        p_nofix = compute_psnr(gt_img, pred_nofix)
        psnr_without_fix.append(p_nofix)

        line = (f"[sanity_check][{scene_id}] {img_path.name}: PSNR co-fix={p_fix:.2f}dB  "
                f"PSNR khong-fix={p_nofix:.2f}dB")

        # (Kiem chung vu khi) Appearance transfer: dung reference train GAN
        # NHAT KHAC (exclude chinh anh nay de mo phong dung viec test pose
        # khong co GT cua chinh no) -> uoc luong gain -> ap len pred_fix.
        if appearance_ctx is not None:
            correction = appearance_correction_for_test_pose(pipeline, appearance_ctx, c2w_ns_raw, scene_id,
                                                               exclude_path=img_path)
            pred_app = apply_appearance_correction(pred_fix, correction, appearance_ctx["appearance_mode"])
            p_app = compute_psnr(gt_img, pred_app)
            psnr_appearance.append(p_app)
            if appearance_ctx["appearance_mode"] == "rgb_gain":
                correction_repr = np.asarray(correction).round(3).tolist()
            else:
                correction_repr = "lab_stats"
            line += f"  PSNR appearance={p_app:.2f}dB (correction={correction_repr})"

        if post_process != "none":
            pred_pp = enhance_image(pred_fix, mode=post_process,
                                     denoise_strength=denoise_strength, sharpen_amount=sharpen_amount)
            p_pp = compute_psnr(gt_img, pred_pp)
            psnr_post_process.append(p_pp)
            line += f"  PSNR post_process({post_process})={p_pp:.2f}dB"

        if diagnose_distortion:
            # Render THEM 1 lan bang un_redistort (pinhole + warp nguoc) tren
            # CUNG pose/anh - so PSNR VUNG voi pred_fix (rasterizer, mac dinh).
            zero_dist_entry = {**dist_norm, "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0}
            camera_pinhole = build_camera(fake_row, c2w_train, zero_dist_entry, device, scene_id=scene_id)
            with torch.no_grad():
                outputs_pinhole = pipeline.model.get_outputs_for_camera(camera_pinhole)
            rgb_pinhole = outputs_pinhole["rgb"].clamp(0, 1).detach()
            out_chw = redistort_to_target(
                rgb_pinhole.permute(2, 0, 1),
                render_fx=float(fake_row.fx), render_fy=float(fake_row.fy),
                render_cx=float(fake_row.cx), render_cy=float(fake_row.cy),
                target_w=w, target_h=h,
                target_fx=float(fake_row.fx), target_fy=float(fake_row.fy),
                target_cx=float(fake_row.cx), target_cy=float(fake_row.cy),
                k1=dist_norm["k1"], k2=dist_norm["k2"],
            )
            pred_un_redistort = Image.fromarray(
                (out_chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

            region_rasterizer = compute_region_psnr(gt_img, pred_fix)
            region_un_redistort = compute_region_psnr(gt_img, pred_un_redistort)
            corner_rasterizer_list.append(region_rasterizer["corner"])
            corner_un_redistort_list.append(region_un_redistort["corner"])
            line += (f"  | VUNG-GOC rasterizer={region_rasterizer['corner']:.2f}dB "
                     f"un_redistort={region_un_redistort['corner']:.2f}dB "
                     f"(trung-tam rasterizer={region_rasterizer['center']:.2f}dB)")

        print(line)

    del pipeline
    torch.cuda.empty_cache()

    if psnr_with_fix:
        avg_fix = sum(psnr_with_fix) / len(psnr_with_fix)
        avg_nofix = sum(psnr_without_fix) / len(psnr_without_fix)
        print(f"[sanity_check][{scene_id}] Trung binh: co-fix={avg_fix:.2f}dB  "
              f"khong-fix={avg_nofix:.2f}dB  (chenh lech={avg_fix - avg_nofix:+.2f}dB)")
        if avg_fix < 20 and avg_fix <= avg_nofix:
            print(f"[CANH BAO][{scene_id}] Ca 2 cach deu cho PSNR thap va fix khong giup ich hon - "
                  f"co the intrinsics/convention pose van con sai o cho khac, can kiem tra them.")
        elif avg_fix > avg_nofix + 1:
            print(f"[{scene_id}] Fix dataparser_transform giup PSNR tang ro ret - xac nhan day la "
                  f"buoc quan trong, GIU LAI khi render test set that.")
        else:
            print(f"[{scene_id}] Chenh lech khong lon - co the transform gan identity/scale gan 1 "
                  f"o scene nay, nhung van nen giu fix vi khong lam hai va dung ve mat ly thuyet.")

        if psnr_appearance:
            avg_app = sum(psnr_appearance) / len(psnr_appearance)
            delta = avg_app - avg_fix
            print(f"[sanity_check][{scene_id}] Appearance transfer: {avg_app:.2f}dB "
                  f"(vs co-fix {avg_fix:.2f}dB, chenh lech={delta:+.2f}dB)")
            if delta > 0.3:
                print(f"[{scene_id}] => Appearance transfer TANG PSNR ro ret -> BAT --appearance_transfer "
                      f"khi render test set that (co train/test appearance gap thuc su tu bilateral grid).")
            elif delta < -0.3:
                print(f"[{scene_id}] => Appearance transfer LAM GIAM PSNR -> KHONG bat cho scene nay "
                      f"(phoi sang cac view da dong deu, hoac bilateral grid tat).")
            else:
                print(f"[{scene_id}] => Appearance transfer gan nhu trung tinh o scene nay "
                      f"(gain ~ 1.0) -> bat hay khong deu duoc, uu tien do tren nhieu mau hon neu con phan van.")

        if psnr_post_process:
            avg_pp = sum(psnr_post_process) / len(psnr_post_process)
            delta_pp = avg_pp - avg_fix
            print(f"[sanity_check][{scene_id}] Post-process ({post_process}): {avg_pp:.2f}dB "
                  f"(vs co-fix {avg_fix:.2f}dB, chenh lech={delta_pp:+.2f}dB)")
            if delta_pp > 0.2:
                print(f"[{scene_id}] => Post-process TANG PSNR - co the can nhac bat --post_process "
                      f"{post_process} cho submission that, nhung VAN NEN doi chieu them SSIM/LPIPS thuc "
                      f"qua eval_submission_path.py truoc khi quyet dinh cuoi cung (PSNR khong dai dien "
                      f"day du cho Score).")
            else:
                print(f"[{scene_id}] => Post-process KHONG giup ich (hoac lam giam) PSNR o scene nay - "
                      f"KHONG nen bat --post_process {post_process} cho submission that.")

        if corner_rasterizer_list:
            avg_corner_rast = sum(corner_rasterizer_list) / len(corner_rasterizer_list)
            avg_corner_ur = sum(corner_un_redistort_list) / len(corner_un_redistort_list)
            delta_dist = avg_corner_ur - avg_corner_rast
            print(f"[sanity_check][{scene_id}] VUNG-GOC trung binh: rasterizer={avg_corner_rast:.2f}dB "
                  f"un_redistort={avg_corner_ur:.2f}dB (chenh lech={delta_dist:+.2f}dB)")
            if delta_dist > 1.0:
                print(f"[{scene_id}] => un_redistort THANG RO o vung goc -> NGHI NGO rasterizer dang ap "
                      f"distortion SAI/khong chinh xac tai vung ria - can dieu tra THEM TRUOC khi tiep tuc "
                      f"dau tu GPU vao dense MVS (dense MVS khong sua duoc loi nay).")
            elif delta_dist < -1.0:
                print(f"[{scene_id}] => rasterizer THANG RO o vung goc (dung nhu ky vong ly thuyet) -> "
                      f"mo vung ria/goc nhieu kha nang do MAT DO COLMAP thua, khong phai loi distortion -> "
                      f"tiep tuc uu tien dense_colmap_init.py.")
            else:
                print(f"[{scene_id}] => 2 che do gan nhu ngang nhau o vung goc -> distortion khong phai "
                      f"nguyen nhan chinh cua mo vung ria/goc o scene nay - tiep tuc uu tien dense MVS.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                         help="Thu muc chua cac scene raw, moi scene co test/test_poses.csv")
    parser.add_argument("--processed_data_root", type=str, default="data/processed",
                         help="Thu muc data/processed (chua transforms.json, images/, dung cho --sanity_check "
                              "va de doc lai distortion tu intrinsics_map.json)")
    parser.add_argument("--checkpoints_root", type=str, required=True,
                         help="Thu muc outputs/<scene_id>/splatfacto/<timestamp>/config.yml")
    parser.add_argument("--intrinsics_json", type=str, default=None,
                         help="Duong dan intrinsics_map.json (mac dinh: <processed_data_root>/intrinsics_map.json). "
                              "Dung de tai su dung distortion (k1,k2,p1,p2) cho anh test, vi test_poses.csv "
                              "khong co cot distortion.")
    parser.add_argument("--submission_dir", type=str, default="submission")
    parser.add_argument("--zip_name", type=str, default="submission.zip")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                         help="(Moi - SUA LOI) Chat luong JPEG khi luu anh nop bai. Truoc day khong "
                              "truyen gi nen PIL dung mac dinh 75 = nen mat mat nang ngay tren anh duoc "
                              "cham diem (do duoc: q75 chi dat 32.3 dB vs q95 dat 51.4 dB, tuc vut bo "
                              "~0.47 dB PSNR va them nhieu KHOI ma LPIPS phat rat nang). 95 la muc "
                              "'gan nhu khong mat mat'; 100 lam file rat lon ma loi ich khong dang ke.")
    parser.add_argument("--resample_prefilter", action="store_true",
                         help="(Moi) Chi tac dung voi --distortion_mode un_redistort: loc trung binh vung "
                              "(area filter) ve target size TRUOC khi grid_sample ap meo, thay vi de bilinear "
                              "2x2 tu ha do phan giai (vut bo phan lon thong tin supersample, gay rang cua). "
                              "BAT theo dung ket qua do duoc tu eval_submission_path.py --resample_prefilter.")
    parser.add_argument("--allow_incomplete", action="store_true",
                         help="(Moi) Van tao file zip du submission THIEU scene/thieu anh. Mac dinh la "
                              "DUNG LAI, vi diem thi la trung binh 5 scene - thieu 1 scene la mat trang "
                              "1/5 diem, va truoc day loi nay xay ra AM THAM (chi in '[BO QUA]' roi van zip).")
    parser.add_argument("--scene", type=str, default=None,
                         help="Chi render cac scene chi dinh. Nhieu scene cach nhau dau PHAY, vd "
                              "'HCM0539,chair'. De trong = render TAT CA scene co test_poses.csv. "
                              "Luu y: preflight cuoi chi kiem tra DUNG cac scene liet ke o day, nen "
                              "khi nop bai that hay de trong de no bat duoc scene con thieu.")
    parser.add_argument("--supersample_factor", type=float, default=1.0,
                         help="Render o do phan giai cao hon target (vd 1.5, 2.0) roi downsample LANCZOS "
                              "ve dung kich thuoc test_poses.csv - giam alias, tang PSNR/SSIM/LPIPS. "
                              "Mac dinh 1.0 = tat (render dung target size, hanh vi cu).")
    parser.add_argument("--appearance_transfer", action="store_true",
                         help="(Vu khi canh tranh) Hieu chinh phoi sang/mau tung anh test bang view train "
                              "gan nhat - bu lai phep hieu chinh bilateral grid da hoc luc train nhung bi "
                              "bo qua luc render test. An toan (no-op khi khong lech). NEN chay --sanity_check "
                              "voi co --appearance_transfer TRUOC de xac nhan PSNR tang tren anh train da biet GT.")
    parser.add_argument("--appearance_mode", type=str, default="rgb_gain",
                         choices=["rgb_gain", "lab_transfer"],
                         help="(Moi) Cach hieu chinh khi --appearance_transfer bat. 'rgb_gain' (mac dinh, "
                              "khong doi hanh vi cu): 1 he so nhan vo huong/kenh RGB. 'lab_transfer': bien doi "
                              "affine kieu Reinhard 2001 trong khong gian LAB - sua ca trung binh LAN "
                              "contrast/phan tan mau, xem image_postprocess.py. Chay --sanity_check "
                              "--appearance_transfer --appearance_mode lab_transfer truoc de so PSNR voi "
                              "rgb_gain bang so truoc khi doi mac dinh submission.")
    parser.add_argument("--appearance_k_neighbors", type=int, default=1,
                         help="(Moi) So view train GAN NHAT dung lam reference cho appearance_transfer, "
                              "ket hop qua trung binh co trong so (1/score, xem gain_weights_from_scores). "
                              "Mac dinh 1 (khong doi hanh vi cu - chi 1 view gan nhat). THU 2-3 de on dinh "
                              "hon truoc anh sang/goc chup bat thuong cua 1 view don le.")
    parser.add_argument("--appearance_mode_json", type=str, default=None,
                         help="(Moi) Duong dan JSON {scene_id: 'rgb_gain'|'lab_transfer'} de OVERRIDE "
                              "--appearance_mode THEO TUNG SCENE (xem select_appearance_mode.py de tao file "
                              "nay tu dong qua A/B thuc te). Scene khong co trong JSON dung --appearance_mode chung.")
    parser.add_argument("--sanity_check", action="store_true",
                         help="Chi render vai anh TRAIN da biet ground-truth de so PSNR co/khong fix "
                              "dataparser_transform, khong render test set, khong nen zip.")
    parser.add_argument("--lowpass_sigma", type=float, default=0.3,
                         help="Sigma cua --post_process lowpass (loc thong thap Gaussian rat nhe). 0.3 la "
                              "muc khop pho tan so voi anh chup that nhat - do tren 60 anh nop bai HCM0421 "
                              "so voi 240 anh that: render THUA 1.5-1.7x nang luong tan so cao (needle "
                              "artifact/rang cua), khong phai thieu. Xem image_postprocess.py.")
    parser.add_argument("--despeckle_threshold", type=float, default=0.03,
                         help="Nguong cua --post_process despeckle (va lo thung toi). Xem image_postprocess.py.")
    parser.add_argument("--post_process", type=str, default="none",
                         choices=["none", "denoise", "sharpen", "denoise_sharpen", "lowpass",
                                  "despeckle", "despeckle_lowpass"],
                         help="(Moi) Hau xu ly 2D anh render TRUOC khi luu (= truoc khi cham diem). "
                              "Mac dinh 'none' (an toan, khong doi gi). RUI RO 2 CHIEU THUC SU: "
                              "sharpen qua tay tao ringing khong khop GT co the lam Score TE HON, "
                              "khong chi 'nhin dep hon' - BAT BUOC chay --sanity_check --post_process "
                              "<mode> truoc de xac nhan PSNR tang bang so, TRUOC khi dung cho submission that.")
    parser.add_argument("--denoise_strength", type=float, default=0.3,
                         help="Cuong do bilateral denoise (0-1), chi dung khi post_process co 'denoise'.")
    parser.add_argument("--sharpen_amount", type=float, default=0.5,
                         help="Cuong do unsharp mask, chi dung khi post_process co 'sharpen'.")
    parser.add_argument("--distortion_mode", type=str, default="rasterizer",
                         choices=["rasterizer", "un_redistort"],
                         help="(Moi) 'rasterizer' (mac dinh, khong doi hanh vi cu): gsplat ap "
                              "distortion truc tiep trong phep chieu. 'un_redistort': render pinhole "
                              "roi warp nguoc ve luoi meo bang redistort_to_target() (xem "
                              "distortion_redistort.py). CHAY --sanity_check truoc de so PSNR vung "
                              "goc giua 2 che do TRUOC khi doi mac dinh cho submission that.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Su dung device: {device}")

    data_root = Path(args.data_root)
    processed_data_root = Path(args.processed_data_root)
    checkpoints_root = Path(args.checkpoints_root)
    submission_root = Path(args.submission_dir)

    intrinsics_json_path = Path(args.intrinsics_json) if args.intrinsics_json else processed_data_root / "intrinsics_map.json"
    intrinsics_map = {}
    if intrinsics_json_path.exists():
        with open(intrinsics_json_path) as f:
            intrinsics_map = json.load(f)
    else:
        print(f"[CANH BAO] Khong tim thay {intrinsics_json_path} - se render voi distortion=0 cho moi scene.")

    scene_appearance_overrides = {}
    if args.appearance_mode_json:
        overrides_path = Path(args.appearance_mode_json)
        if overrides_path.exists():
            with open(overrides_path) as f:
                scene_appearance_overrides = json.load(f)
            print(f"[appearance] Da doc override theo scene tu {overrides_path}: {scene_appearance_overrides}")
        else:
            print(f"[CANH BAO] --appearance_mode_json duoc chi dinh nhung khong thay {overrides_path} - "
                  f"dung --appearance_mode chung ({args.appearance_mode}) cho moi scene.")

    if args.sanity_check:
        scenes = ([s.strip() for s in args.scene.split(",") if s.strip()] if args.scene
                   else sorted(p.name for p in processed_data_root.iterdir() if p.is_dir()))
        for scene_id in scenes:
            config_path = find_latest_config(checkpoints_root, scene_id)
            if config_path is None:
                print(f"[sanity_check][BO QUA] Khong tim thay checkpoint cho scene {scene_id}")
                continue
            sanity_check_with_train_images(scene_id, config_path, processed_data_root,
                                            intrinsics_map.get(scene_id), device,
                                            check_appearance=args.appearance_transfer,
                                            post_process=args.post_process,
                                            denoise_strength=args.denoise_strength,
                                            sharpen_amount=args.sharpen_amount,
                                            diagnose_distortion=(args.distortion_mode == "un_redistort"),
                                            appearance_mode=args.appearance_mode,
                                            appearance_k_neighbors=args.appearance_k_neighbors)
        return

    required_cols = {"image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
                      "fx", "fy", "cx", "cy", "width", "height"}

    if args.scene:
        # Ho tro NHIEU scene cach nhau dau phay (vd "HCM0539,chair"). Chay 1 lan
        # cho nhieu scene tot hon chay nhieu lan: preflight kiem tra DUNG tap
        # scene ban dinh nop, va chi dong goi zip 1 luot o cuoi.
        scene_dirs = [data_root / s.strip() for s in args.scene.split(",") if s.strip()]
    else:
        scene_dirs = sorted(
            p for p in data_root.iterdir()
            if p.is_dir() and (p / "test" / "test_poses.csv").exists()
        )

    total_rendered = 0
    expected_per_scene = {}   # scene_id -> so anh test_poses.csv YEU CAU
    skipped_scenes = []       # scene bi bo qua (thieu checkpoint / thieu CSV)
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        csv_path = scene_dir / "test" / "test_poses.csv"
        if not csv_path.exists():
            print(f"[BO QUA] Khong co {csv_path}")
            skipped_scenes.append((scene_id, "thieu test_poses.csv"))
            continue

        df = pd.read_csv(csv_path)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"[{scene_id}] test_poses.csv thieu cot: {missing}")
        expected_per_scene[scene_id] = len(df)

        config_path = find_latest_config(checkpoints_root, scene_id)
        if config_path is None:
            print(f"[BO QUA] Khong tim thay checkpoint cho scene {scene_id}")
            skipped_scenes.append((scene_id, "chua train (khong co config.yml)"))
            continue

        T, scale = load_dataparser_transform(config_path)
        note = "GAN 1.0, transform anh huong nho" if abs(scale - 1.0) < 0.05 else "KHAC 1.0 ro ret - fix nay quan trong voi scene nay"
        print(f"[{scene_id}] dataparser scale = {scale:.6f} ({note})")

        print(f"[{scene_id}] Load checkpoint: {config_path}")
        pipeline = load_pipeline(config_path, device)

        appearance_ctx = None
        if args.appearance_transfer:
            appearance_mode_for_scene = scene_appearance_overrides.get(scene_id, args.appearance_mode)
            appearance_ctx = build_appearance_context(
                pipeline, processed_data_root, scene_id, T, scale,
                intrinsics_map.get(scene_id), device, appearance_mode=appearance_mode_for_scene,
                k_neighbors=args.appearance_k_neighbors, distortion_mode=args.distortion_mode)

        print(f"[{scene_id}] Cau hinh render: supersample_factor={args.supersample_factor} "
              f"distortion_mode={args.distortion_mode} (resample_prefilter={args.resample_prefilter}) "
              f"post_process={args.post_process} appearance_transfer={args.appearance_transfer} "
              f"jpeg_quality={args.jpeg_quality}")

        out_dir = submission_root / scene_id
        render_rows(pipeline, df, intrinsics_map.get(scene_id), T, scale, out_dir, device, scene_id=scene_id,
                    supersample_factor=args.supersample_factor, appearance_ctx=appearance_ctx,
                    post_process=args.post_process, denoise_strength=args.denoise_strength,
                    sharpen_amount=args.sharpen_amount, distortion_mode=args.distortion_mode,
                    resample_prefilter=args.resample_prefilter, jpeg_quality=args.jpeg_quality,
                    lowpass_sigma=args.lowpass_sigma, despeckle_threshold=args.despeckle_threshold)

        print(f"[{scene_id}] Da render {len(df)} anh -> {out_dir}")
        total_rendered += len(df)

        del pipeline
        torch.cuda.empty_cache()

    print(f"\nTong cong {total_rendered} anh da render.")

    # ===== PREFLIGHT: kiem tra DAY DU truoc khi dong goi =====
    # LY DO (rui ro that, khong phai gia dinh): truoc day scene thieu checkpoint
    # chi in "[BO QUA]" roi van chay tiep xuong make_zip() - tao ra file nop bai
    # THIEU HAN 1 hoac nhieu scene ma khong co gi chan lai. Diem thi la TRUNG BINH
    # 5 scene, nen thieu 1 scene = mat trang 1/5 diem, thieu 3 scene = mat 3/5.
    # Kiem tra nay bat loi do TRUOC khi nop, thay vi phat hien sau khi da muon.
    problems = []
    for scene_id, why in skipped_scenes:
        problems.append(f"  - {scene_id}: BI BO QUA ({why})")
    for scene_id, n_expected in sorted(expected_per_scene.items()):
        out_dir = submission_root / scene_id
        n_actual = len(list(out_dir.glob("*"))) if out_dir.is_dir() else 0
        if n_actual != n_expected:
            problems.append(f"  - {scene_id}: co {n_actual} anh nhung test_poses.csv yeu cau {n_expected}")

    print("\n=== KIEM TRA DAY DU SUBMISSION ===")
    for scene_id, n_expected in sorted(expected_per_scene.items()):
        out_dir = submission_root / scene_id
        n_actual = len(list(out_dir.glob("*"))) if out_dir.is_dir() else 0
        mark = "OK " if n_actual == n_expected else "LOI"
        print(f"  [{mark}] {scene_id}: {n_actual}/{n_expected} anh")

    if problems:
        print("\n[CANH BAO NGHIEM TRONG] Submission CHUA DAY DU:")
        for line in problems:
            print(line)
        if not args.allow_incomplete:
            print("\nDA DUNG LAI - KHONG tao file zip, de tranh nop bai thieu scene.")
            print("Neu that su muon dong goi ban thieu (vd de test), chay lai voi --allow_incomplete.")
            return
        print("\n--allow_incomplete duoc bat -> van dong goi ban THIEU nay theo yeu cau.")

    make_zip(submission_root, Path(args.zip_name))
    print(f"File nop bai: {args.zip_name}")


if __name__ == "__main__":
    main()
