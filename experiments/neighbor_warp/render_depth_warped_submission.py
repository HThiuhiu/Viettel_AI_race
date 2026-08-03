"""
Submission dung DEPTH-BASED IMAGE WARPING (DIBR): thay vi dung nguyen anh
THAT cua frame lang gieng (da bi cham 0 diem - xem
build_neighbor_substitution_submission.py, lech pose hinh hoc that su), giu
lai loi ich cua "anh that sac net" nhung SUA DUNG THEO HINH HOC 3D truoc khi
nop:

    1. Render Gaussian model BINH THUONG tai dung pose test yeu cau (PINHOLE,
       chua ap distortion) -> lay ca RGB (fallback) LAN DEPTH.
    2. Tim frame train GAN NHAT theo so thu tu (nhu ban truoc), lay anh THAT
       + pose THAT cua no.
    3. Dung DEPTH cua pose test (buoc 1) + pose cua ca 2 camera -> WARP anh
       that cua frame lang gieng ve DUNG khung hinh pose test (xem
       depth_warp.py - da verify hinh hoc bang so hoc, sai so <0.04 tren canh
       tong hop dac, xem docstring depth_warp.py).
    4. Blend: dung pixel da warp o vung hop le (trong khung anh nguon, khong
       bi che khuat); fallback ve anh Gaussian model render o vung con lai.
    5. Ap redistort_to_target() de dua ve dung luoi meo (distortion that) ma
       test_poses.csv yeu cau.

=== CANH BAO QUAN TRONG - CHUA VERIFY DUOC (khong co GPU o day) ===
Gia dinh "depth" tra ve tu outputs cua Splatfacto (`pipeline.model.
get_outputs_for_camera()["depth"]`) la Z-DEPTH DOC TRUC QUANG HOC (khoang
cach chieu len truc nhin cua camera, KHONG PHAI khoang cach Euclid tam-den-
diem). Day la quy uoc pho bien nhat trong NeRF/3DGS nhung CHUA duoc xac nhan
truc tiep tren dau ra that. TRUOC KHI TIN DUNG:
    - Chay --sanity_check_depth truoc (in vai gia tri depth mau, so sanh
      voi uoc luong khoang cach camera-vat-the bang mat thuong, vd cot BTS
      thuong cach camera ~10-40m tuy scene).
    - Neu depth co ve sai don vi/quy uoc (vd luon ~1.0 hoac am), DUNG lai,
      bao loi ngay, KHONG dung ket qua warp lam sai lech them.

=== VI TRI FILE (thu muc "experiments/neighbor_warp/", TACH BIET hoan toan khoi pipeline train chinh) ===
File nay va 2 file lien quan (build_neighbor_substitution_submission.py,
depth_warp.py) nam trong thu muc con experiments/neighbor_warp/ - KHONG duoc train.sh hay bat ky
script train/tham so nao goi den, chi la cong cu tao thu submission rieng.

Cach dung (chay tu thu muc GOC cua project, khong can cd vao experiments/neighbor_warp/):
    python experiments/neighbor_warp/render_depth_warped_submission.py --sanity_check_depth \
        --checkpoints_root ... --scene HCM0421
    python experiments/neighbor_warp/render_depth_warped_submission.py --data_root . --processed_data_root data/processed \
        --checkpoints_root /kaggle/working/outputs --submission_dir submission_warped --zip_name submission_warped.zip
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

# Script nay nam trong experiments/neighbor_warp/, nhung can import
# render_and_submit.py va distortion_redistort.py nam trong pipeline/ (khong
# phai file thu nghiem - dung chung voi pipeline train that) - them pipeline/
# vao sys.path de import duoc bat ke cach goi script (tu root, tu ben trong
# thu muc nay, hay bang duong dan tuyet doi).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from render_and_submit import (
    row_to_c2w_ns_raw, load_dataparser_transform, apply_dataparser_transform,
    load_pipeline, build_camera, _normalize_intrinsics_entry, find_latest_config,
)
from build_neighbor_substitution_submission import extract_frame_num, build_frame_index, find_neighbor_frame
from depth_warp import warp_by_depth, blend_warped_with_fallback
from distortion_redistort import redistort_to_target

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def load_train_frame_poses(processed_data_root: Path, scene_id: str) -> dict:
    """Doc transforms.json -> dict {so_thu_tu_frame: (c2w_ns_raw, gt_path)}."""
    transforms_path = processed_data_root / scene_id / "transforms.json"
    images_dir = processed_data_root / scene_id / "images"
    with open(transforms_path) as f:
        transforms = json.load(f)
    out = {}
    for fr in transforms.get("frames", []):
        name = Path(fr["file_path"]).name
        num = extract_frame_num(name)
        gt_path = images_dir / name
        if num is not None and gt_path.exists():
            out[num] = (np.array(fr["transform_matrix"]), gt_path)
    return out


def render_pinhole_with_depth(pipeline, c2w_train: np.ndarray, fx, fy, cx, cy, w, h, device):
    """Render camera PINHOLE (distortion=0) tai c2w_train, tra ve (rgb_hwc,
    depth_hw) - CA HAI deu torch.Tensor tren device. depth lay tu
    outputs["depth"] - xem CANH BAO trong docstring file ve quy uoc chua
    verify duoc cua dai luong nay."""
    row = pd.Series({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": int(w), "height": int(h)})
    zero_dist = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0, "camera_model": "OPENCV"}
    camera = build_camera(row, c2w_train, zero_dist, device)
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera)
    rgb = outputs["rgb"].clamp(0, 1).detach()
    if "depth" not in outputs:
        raise KeyError(
            "Khong tim thay key 'depth' trong outputs cua get_outputs_for_camera(). "
            "Cac key thuc te tra ve: " + str(list(outputs.keys())) +
            " - can sua render_pinhole_with_depth() de dung dung ten key."
        )
    depth = outputs["depth"].detach()
    if depth.dim() == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    return rgb, depth


def sanity_check_depth(pipeline, c2w_train: np.ndarray, fx, fy, cx, cy, w, h, device, scene_id: str):
    _, depth = render_pinhole_with_depth(pipeline, c2w_train, fx, fy, cx, cy, w, h, device)
    d = depth.cpu().numpy()
    print(f"[sanity_check_depth][{scene_id}] shape={d.shape} "
          f"min={d.min():.4f} max={d.max():.4f} mean={d.mean():.4f} median={np.median(d):.4f}")
    print(f"[sanity_check_depth][{scene_id}] So sanh voi uoc luong khoang cach that (bang mat thuong, "
          f"vd cot BTS ~10-40m tuy scene) - neu gia tri o tren KHONG hop ly (vd luon ~1.0, am, "
          f"hoac qua nho/qua lon so voi ty le canh), DUNG su dung ket qua warp cho toi khi xac nhan lai.")
    return d


def process_scene(scene_id: str, config_path: Path, processed_data_root: Path, test_csv_path: Path,
                   intrinsics_map: dict, out_dir: Path, device: str, max_neighbor_radius: int = 500):
    T, scale = load_dataparser_transform(config_path)
    pipeline = load_pipeline(config_path, device)
    distortion_entry = intrinsics_map.get(scene_id)
    dist_norm = _normalize_intrinsics_entry(distortion_entry)

    frame_poses = load_train_frame_poses(processed_data_root, scene_id)
    frame_index = {num: gt_path for num, (_, gt_path) in frame_poses.items()}

    df = pd.read_csv(test_csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_fallback_full = 0, 0
    for _, row in df.iterrows():
        target_num = extract_frame_num(row.image_name)
        qvec = np.array([row.qw, row.qx, row.qy, row.qz])
        tvec = np.array([row.tx, row.ty, row.tz])
        c2w_ns_raw_target = row_to_c2w_ns_raw(qvec, tvec)
        c2w_train_target = apply_dataparser_transform(c2w_ns_raw_target, T, scale)

        fx, fy, cx, cy = float(row.fx), float(row.fy), float(row.cx), float(row.cy)
        w, h = int(row.width), int(row.height)

        rgb_fallback, depth_target = render_pinhole_with_depth(
            pipeline, c2w_train_target, fx, fy, cx, cy, w, h, device)

        if target_num is None:
            print(f"[LOI][{scene_id}] Khong parse duoc so thu tu frame tu '{row.image_name}' - "
                  f"dung fallback Gaussian render thuan tuy.")
            final_chw = rgb_fallback.permute(2, 0, 1)
            n_fallback_full += 1
        else:
            offset, neighbor_num = None, None
            for r in range(1, max_neighbor_radius + 1):
                for cand in (target_num - r, target_num + r):
                    if cand in frame_poses:
                        offset, neighbor_num = r if cand > target_num else -r, cand
                        break
                if neighbor_num is not None:
                    break

            if neighbor_num is None:
                print(f"[CANH BAO][{scene_id}] Khong tim thay neighbor cho '{row.image_name}' - "
                      f"dung fallback Gaussian render thuan tuy.")
                final_chw = rgb_fallback.permute(2, 0, 1)
                n_fallback_full += 1
            else:
                neighbor_c2w_ns_raw, neighbor_gt_path = frame_poses[neighbor_num]
                neighbor_c2w_train = apply_dataparser_transform(neighbor_c2w_ns_raw, T, scale)
                neighbor_img = Image.open(neighbor_gt_path).convert("RGB")
                if neighbor_img.size != (w, h):
                    neighbor_img = neighbor_img.resize((w, h), Image.LANCZOS)
                neighbor_chw = torch.from_numpy(np.asarray(neighbor_img).astype(np.float32) / 255.0)
                neighbor_chw = neighbor_chw.permute(2, 0, 1).to(device)

                c2w_train_target_t = torch.from_numpy(c2w_train_target[:3, :4]).float().to(device)
                neighbor_c2w_train_t = torch.from_numpy(neighbor_c2w_train[:3, :4]).float().to(device)

                warped_chw, valid_hw = warp_by_depth(
                    neighbor_chw, neighbor_c2w_train_t, depth_target, c2w_train_target_t,
                    fx, fy, cx, cy)
                final_chw = blend_warped_with_fallback(warped_chw, valid_hw, rgb_fallback.permute(2, 0, 1))
                n_ok += 1

        if dist_norm["k1"] != 0.0 or dist_norm["k2"] != 0.0:
            final_chw = redistort_to_target(
                final_chw, render_fx=fx, render_fy=fy, render_cx=cx, render_cy=cy,
                target_w=w, target_h=h, target_fx=fx, target_fy=fy, target_cx=cx, target_cy=cy,
                k1=dist_norm["k1"], k2=dist_norm["k2"])

        img = Image.fromarray((final_chw.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        img.save(out_dir / row.image_name)

    print(f"[{scene_id}] Da tao {len(df)} anh ({n_ok} dung depth-warp, {n_fallback_full} fallback thuan tuy)")
    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()


def make_zip(submission_root: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in submission_root.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(submission_root))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--processed_data_root", type=str, default="data/processed")
    ap.add_argument("--checkpoints_root", type=str, required=True)
    ap.add_argument("--intrinsics_json", type=str, default=None)
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--submission_dir", type=str, default="submission_warped")
    ap.add_argument("--zip_name", type=str, default="submission_warped.zip")
    ap.add_argument("--max_neighbor_radius", type=int, default=500)
    ap.add_argument("--sanity_check_depth", action="store_true",
                     help="Chi render 1 pose va in thong ke depth de kiem tra quy uoc TRUOC khi tin dung.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)
    processed_data_root = Path(args.processed_data_root)
    checkpoints_root = Path(args.checkpoints_root)

    intrinsics_json_path = Path(args.intrinsics_json) if args.intrinsics_json else processed_data_root / "intrinsics_map.json"
    intrinsics_map = {}
    if intrinsics_json_path.exists():
        with open(intrinsics_json_path) as f:
            intrinsics_map = json.load(f)

    if args.scene:
        scene_dirs = [data_root / args.scene]
    else:
        scene_dirs = sorted(p for p in data_root.iterdir()
                             if p.is_dir() and (p / "test" / "test_poses.csv").exists())

    if args.sanity_check_depth:
        scene_id = scene_dirs[0].name
        config_path = find_latest_config(checkpoints_root, scene_id)
        if config_path is None:
            raise RuntimeError(f"Khong tim thay checkpoint cho scene {scene_id}")
        T, scale = load_dataparser_transform(config_path)
        pipeline = load_pipeline(config_path, device)
        df = pd.read_csv(scene_dirs[0] / "test" / "test_poses.csv")
        row = df.iloc[0]
        qvec = np.array([row.qw, row.qx, row.qy, row.qz])
        tvec = np.array([row.tx, row.ty, row.tz])
        c2w_train = apply_dataparser_transform(row_to_c2w_ns_raw(qvec, tvec), T, scale)
        sanity_check_depth(pipeline, c2w_train, float(row.fx), float(row.fy), float(row.cx), float(row.cy),
                            int(row.width), int(row.height), device, scene_id)
        return

    submission_root = Path(args.submission_dir)
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        config_path = find_latest_config(checkpoints_root, scene_id)
        if config_path is None:
            print(f"[BO QUA] Khong tim thay checkpoint cho scene {scene_id}")
            continue
        process_scene(scene_id, config_path, processed_data_root, scene_dir / "test" / "test_poses.csv",
                      intrinsics_map, submission_root / scene_id, device, args.max_neighbor_radius)

    make_zip(submission_root, Path(args.zip_name))
    print(f"File nop bai: {args.zip_name}")


if __name__ == "__main__":
    main()
