import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np


CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}
CAMERA_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}


SPARSE_FILE_ALIASES = {
    # Ten chuan COLMAP -> danh sach bien the hay gap tren thuc te (thieu 's',
    # sai hoa/thuong...). Luon uu tien ten chuan truoc, cac ten con lai chi
    # la fallback khi ten chuan khong ton tai.
    "cameras.bin": ["cameras.bin", "camera.bin", "Cameras.bin", "Camera.bin"],
    "images.bin": ["images.bin", "image.bin", "Images.bin", "Image.bin"],
    "points3D.bin": ["points3D.bin", "point3D.bin", "Point3D.bin", "points3d.bin", "Points3D.bin"],
    "points3D.ply": ["points3D.ply", "point3D.ply", "Point3D.ply", "points3d.ply", "Points3D.ply"],
    "cameras.txt": ["cameras.txt", "camera.txt", "Cameras.txt", "Camera.txt"],
    "images.txt": ["images.txt", "image.txt", "Images.txt", "Image.txt"],
}


def find_sparse_file(sparse_dir: Path, canonical_name: str):
    """Tim file trong sparse/0/ theo ten chuan COLMAP, chap nhan cac bien the
    hay gap (thieu 's', sai hoa/thuong). Tra ve Path neu tim thay, None neu
    khong. In ra canonical_name khac voi ten thuc te tim duoc de nguoi dung
    biet code dang dung file nao (tranh am tham dung nham file khac)."""
    if not sparse_dir.exists():
        return None
    for alias in SPARSE_FILE_ALIASES.get(canonical_name, [canonical_name]):
        p = sparse_dir / alias
        if p.exists():
            if alias != canonical_name:
                print(f"[LUU Y] Dung '{p}' thay cho ten chuan COLMAP '{canonical_name}' "
                      f"(ten file thuc te khac chuan, van nhan dien duoc).")
            return p
    # fallback cuoi: quet case-insensitive toan bo thu muc
    target_lower = canonical_name.lower()
    for f in sparse_dir.iterdir():
        if f.is_file() and f.name.lower() == target_lower:
            print(f"[LUU Y] Dung '{f}' thay cho ten chuan COLMAP '{canonical_name}' "
                  f"(khac hoa/thuong).")
            return f
    return None


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_txt(path: Path) -> dict:
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            cameras[camera_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": list(map(float, parts[4:])),
            }
    return cameras


def read_cameras_bin(path: Path) -> dict:
    cameras = {}
    with open(path, "rb") as f:
        (num_cameras,) = read_next_bytes(f, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(f, 24, "iiQQ")
            model = CAMERA_MODEL_IDS[model_id]
            params = list(read_next_bytes(f, 8 * CAMERA_NUM_PARAMS[model], "d" * CAMERA_NUM_PARAMS[model]))
            cameras[camera_id] = {
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def read_images_txt(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#") and line.strip()]

    images = []
    for i in range(0, len(lines), 2):
        parts = lines[i].strip().split()
        images.append({
            "image_id": int(parts[0]),
            "qvec": np.array(list(map(float, parts[1:5]))),
            "tvec": np.array(list(map(float, parts[5:8]))),
            "camera_id": int(parts[8]),
            "name": parts[9],
        })
    return images


def read_images_bin(path: Path) -> list:
    images = []
    with open(path, "rb") as f:
        (num_images,) = read_next_bytes(f, 8, "Q")
        for _ in range(num_images):
            image_id = read_next_bytes(f, 4, "i")[0]
            qvec = np.array(read_next_bytes(f, 32, "dddd"))
            tvec = np.array(read_next_bytes(f, 24, "ddd"))
            camera_id = read_next_bytes(f, 4, "i")[0]

            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8")

            (num_points2d,) = read_next_bytes(f, 8, "Q")
            f.seek(num_points2d * 24, 1)
            images.append({
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            })
    return images


def get_intrinsics_and_distortion(camera: dict):
    """Tra ve day du fx,fy,cx,cy,w,h + he so meo ong kinh (k1,k2,k3,k4,p1,p2)
    va camera_model tuong ung de ghi vao transforms.json (dung quy uoc cua
    Nerfstudio: OPENCV dung k1,k2,p1,p2; OPENCV_FISHEYE dung k1..k4).

    Day la diem sua loi chinh so voi ban goc: ban goc chi lay fx,fy,cx,cy
    roi bo qua toan bo params con lai (chinh la he so meo).
    """
    model, width, height, params = camera["model"], camera["width"], camera["height"], camera["params"]
    dist = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0}
    ns_camera_model = "OPENCV"  # default: Nerfstudio OPENCV voi distortion=0 tuong duong pinhole

    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model == "SIMPLE_RADIAL":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        dist["k1"] = params[3]
    elif model == "RADIAL":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        dist["k1"], dist["k2"] = params[3], params[4]
    elif model == "OPENCV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        dist["k1"], dist["k2"], dist["p1"], dist["p2"] = params[4], params[5], params[6], params[7]
    elif model == "OPENCV_FISHEYE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        dist["k1"], dist["k2"], dist["k3"], dist["k4"] = params[4], params[5], params[6], params[7]
        ns_camera_model = "OPENCV_FISHEYE"
    elif model in ("SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        dist["k1"] = params[3]
        if model == "RADIAL_FISHEYE":
            dist["k2"] = params[4]
        ns_camera_model = "OPENCV_FISHEYE"
    elif model in ("FULL_OPENCV", "THIN_PRISM_FISHEYE", "FOV"):
        # Cac model nay co nhieu he so hon Nerfstudio OPENCV ho tro
        # (vd FULL_OPENCV co them k3,k4,k5,k6). Nerfstudio OPENCV chi nhan
        # k1,k2,p1,p2 nen ta lay xap xi 4 he so dau, canh bao phan con lai
        # bi bo qua thay vi am tham mat het nhu ban goc.
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        if len(params) >= 8:
            dist["k1"], dist["k2"], dist["p1"], dist["p2"] = params[4], params[5], params[6], params[7]
        print(f"[CANH BAO] Camera model {model} co nhieu he so meo hon Nerfstudio OPENCV ho tro. "
              f"Chi dung 4 he so dau (k1,k2,p1,p2), cac he so bac cao hon (k3+) bi bo qua.")
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    return fx, fy, cx, cy, width, height, dist, ns_camera_model


def find_points3d_source(scene_dir: Path):
    """Tra ve (Path, is_dense) cho point cloud dung khoi tao Gaussian. Uu tien
    dense/fused.ply (tu dense_colmap_init.py, neu ban da chay MVS) vi day dac
    hon han sparse COLMAP - dac biet quan trong voi be mat kim loai it texture/
    phan chieu (cot/anten BTS) noi COLMAP sparse matching thuong rat thua.
    Fallback ve sparse/0/points3D.ply (hanh vi cu) neu dense chua duoc tao -
    hoan toan tuong thich nguoc, khong bat buoc phai chay dense_colmap_init.py."""
    dense_path = scene_dir / "dense" / "fused.ply"
    if dense_path.exists():
        return dense_path, True
    sparse_path = find_sparse_file(scene_dir / "sparse" / "0", "points3D.ply")
    if sparse_path is not None:
        return sparse_path, False
    return None, False


def load_colmap(scene_dir: Path):
    sparse_dir = scene_dir / "sparse" / "0"
    cameras_path_txt = find_sparse_file(sparse_dir, "cameras.txt")
    images_path_txt = find_sparse_file(sparse_dir, "images.txt")

    if cameras_path_txt:
        cameras = read_cameras_txt(cameras_path_txt)
    else:
        cameras_path_bin = find_sparse_file(sparse_dir, "cameras.bin")
        if cameras_path_bin is None:
            raise FileNotFoundError(
                f"Khong tim thay cameras.bin/cameras.txt (hoac bien the) trong {sparse_dir}. "
                f"Chay validate_colmap_structure.py de kiem tra chi tiet."
            )
        cameras = read_cameras_bin(cameras_path_bin)

    if images_path_txt:
        images = read_images_txt(images_path_txt)
    else:
        images_path_bin = find_sparse_file(sparse_dir, "images.bin")
        if images_path_bin is None:
            raise FileNotFoundError(
                f"Khong tim thay images.bin/images.txt (hoac bien the) trong {sparse_dir}. "
                f"File nay BAT BUOC phai co (luu pose tung anh) - khong the tao transforms.json "
                f"neu thieu. Chay validate_colmap_structure.py de kiem tra chi tiet."
            )
        images = read_images_bin(images_path_bin)

    return cameras, images


def colmap_to_nerfstudio(scene_dir: Path, out_dir: Path, scene_id: str, intrinsics_map: dict, scene_root: Path = None):
    """(Moi) mask_path: co che DUNG de sky-mask hoat dong voi Nerfstudio - da
    xac nhan TRUC TIEP tu source nerfstudio_dataparser.py (KHONG co co CLI
    nao cho mask theo file/thu muc, chi co --mask-color cho 1 mau don). Moi
    frame CO THE khai bao "mask_path" (giong "file_path"), dataparser tu doc
    - nhung BAT BUOC TAT CA frame phai co HOAC KHONG frame nao co (yeu cau
    ro rang tu chinh source, khong phai suy doan). Vi vay logic duoi day
    kiem tra DU 100% truoc khi nhung, neu thieu du chi 1 frame se BO QUA
    HOAN TOAN (khong nhung mask_path cho frame nao), in canh bao ro rang."""
    cameras, images = load_colmap(scene_dir)
    frames = []
    ref = None  # (fx, fy, cx, cy, width, height, dist, ns_camera_model)
    existing_names = {p.name for p in (scene_dir / "images").glob("*")}
    masks_dir = scene_dir / "masks"
    has_masks_dir = masks_dir.is_dir()
    existing_mask_names = {p.name for p in masks_dir.glob("*")} if has_masks_dir else set()
    test_image_names = set()
    if scene_root is not None:
        test_csv_path = scene_root / "test" / "test_poses.csv"
        if test_csv_path.exists():
            try:
                import csv as _csv
                with open(test_csv_path, "r", encoding="utf-8") as f:
                    reader = _csv.DictReader(f)
                    if reader.fieldnames and "image_name" in reader.fieldnames:
                        test_image_names = {row["image_name"] for row in reader}
            except Exception as e:
                print(f"[{scene_id}] [CANH BAO] Khong doc duoc {test_csv_path}: {e}")

    skipped_missing = 0
    skipped_test_leak = 0
    camera_models_seen = set()

    for img in images:
        if img["name"] in test_image_names:
            skipped_test_leak += 1
            continue
        if img["name"] not in existing_names:
            skipped_missing += 1
            continue

        cam = cameras[img["camera_id"]]
        camera_models_seen.add(cam["model"])
        fx, fy, cx, cy, width, height, dist, ns_camera_model = get_intrinsics_and_distortion(cam)
        ref = (fx, fy, cx, cy, width, height, dist, ns_camera_model)

        w2c = np.eye(4)
        w2c[:3, :3] = qvec_to_rotmat(img["qvec"])
        w2c[:3, 3] = img["tvec"]
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1

        frame_entry = {
            "file_path": f"images/{img['name']}",
            "transform_matrix": c2w.tolist(),
        }
        if has_masks_dir:
            frame_entry["_mask_available"] = img["name"] in existing_mask_names
        frames.append(frame_entry)

    if has_masks_dir:
        n_with_mask = sum(1 for fr in frames if fr.get("_mask_available"))
        if n_with_mask == len(frames):
            for fr in frames:
                name = Path(fr["file_path"]).name
                fr["mask_path"] = f"masks/{name}"
                del fr["_mask_available"]
            print(f"[{scene_id}] Sky mask: DU 100% ({n_with_mask}/{len(frames)} frame) - "
                  f"da nhung 'mask_path' vao transforms.json.")
        else:
            for fr in frames:
                fr.pop("_mask_available", None)
            print(f"[{scene_id}] [CANH BAO] Sky mask KHONG DU: chi {n_with_mask}/{len(frames)} frame "
                  f"co mask tuong ung trong '{masks_dir}'. Nerfstudio yeu cau TAT CA frame phai co "
                  f"mask_path HOAC KHONG frame nao co - BO QUA HOAN TOAN mask_path cho scene nay "
                  f"(khong nhung cho frame nao) de tranh loi dataparser.")

    if skipped_missing:
        print(f"[{scene_id}] Bo qua {skipped_missing} frame trong images.bin khong co file anh thuc te "
              f"(vi du: anh thuoc tap test).")
    if skipped_test_leak:
        print(f"[{scene_id}] CHONG LEAKAGE: da loai {skipped_test_leak} anh trung ten voi "
              f"test_poses.csv khoi tap train (khong dua vao transforms.json), du file co the "
              f"van ton tai vat ly trong train/images.")

    if len(camera_models_seen) > 1:
        print(f"[CANH BAO][{scene_id}] Scene co nhieu hon 1 camera model ({camera_models_seen}). "
              f"Script hien chi ghi 1 bo intrinsics dai dien (cua frame cuoi cung duoc xu ly) vao "
              f"transforms.json top-level; neu cac anh dung camera khac nhau ro ret, can sua de ghi "
              f"intrinsics rieng cho tung frame (per-frame fl_x/fl_y/cx/cy trong 'frames').")

    if not frames or ref is None:
        raise RuntimeError(f"No COLMAP images found for scene {scene_id}")

    fx_ref, fy_ref, cx_ref, cy_ref, width_ref, height_ref, dist_ref, ns_camera_model = ref

    transforms = {
        "fl_x": fx_ref,
        "fl_y": fy_ref,
        "cx": cx_ref,
        "cy": cy_ref,
        "w": width_ref,
        "h": height_ref,
        "k1": dist_ref["k1"],
        "k2": dist_ref["k2"],
        "p1": dist_ref["p1"],
        "p2": dist_ref["p2"],
        "camera_model": ns_camera_model,
        "frames": frames,
    }
    if ns_camera_model == "OPENCV_FISHEYE":
        transforms["k3"] = dist_ref["k3"]
        transforms["k4"] = dist_ref["k4"]
        del transforms["p1"]
        del transforms["p2"]
    points3d_path, points3d_is_dense = find_points3d_source(scene_dir)
    if points3d_path is not None:
        transforms["ply_file_path"] = "sparse_pc.ply"
        print(f"[{scene_id}] Dung {'DENSE MVS' if points3d_is_dense else 'sparse COLMAP'} "
              f"point cloud lam Gaussian init: {points3d_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    masks_embedded = has_masks_dir and any("mask_path" in fr for fr in frames)

    intrinsics_map[scene_id] = {
        "fx": fx_ref, "fy": fy_ref, "cx": cx_ref, "cy": cy_ref,
        "w": width_ref, "h": height_ref,
        "k1": dist_ref["k1"], "k2": dist_ref["k2"],
        "k3": dist_ref["k3"], "k4": dist_ref["k4"],
        "p1": dist_ref["p1"], "p2": dist_ref["p2"],
        "camera_model": ns_camera_model,
    }
    return masks_embedded


def discover_scenes(raw_root: Path):
    scene_dirs = []
    for candidate in sorted(raw_root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "train" / "images").exists() and (candidate / "train" / "sparse" / "0").exists():
            # Layout: <root>/<scene>/train/images, test o <root>/<scene>/test
            scene_dirs.append((candidate.name, candidate / "train", candidate))
        elif (candidate / "images").exists() and (candidate / "sparse" / "0").exists():
            # Layout README: <root>/<scene>/images, test o <root>/<scene>/test (neu co)
            scene_dirs.append((candidate.name, candidate, candidate))
    return scene_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    intrinsics_map = {}
    scene_dirs = discover_scenes(raw_root)
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found in {raw_root}")

    for scene_id, scene_dir, scene_root in scene_dirs:
        out_dir = out_root / scene_id
        print(f"[{scene_id}] preprocessing {scene_dir} -> {out_dir}")
        masks_embedded = colmap_to_nerfstudio(scene_dir, out_dir, scene_id, intrinsics_map, scene_root=scene_root)

        src_images = scene_dir / "images"
        dst_images = out_dir / "images"
        # Luon xoa va copy lai de tranh tinh trang copy do dang tu lan chay truoc.
        if dst_images.exists():
            shutil.rmtree(dst_images)
        shutil.copytree(src_images, dst_images)

        # (Moi) Chi copy masks/ neu colmap_to_nerfstudio() da xac nhan nhung
        # "mask_path" cho DU 100% frame (all-or-nothing, xem docstring ham do).
        if masks_embedded:
            src_masks = scene_dir / "masks"
            dst_masks = out_dir / "masks"
            if dst_masks.exists():
                shutil.rmtree(dst_masks)
            shutil.copytree(src_masks, dst_masks)
            print(f"[{scene_id}] Da copy sky mask vao {dst_masks}")

        src_points, src_points_is_dense = find_points3d_source(scene_dir)
        if src_points is not None:
            shutil.copy2(src_points, out_dir / "sparse_pc.ply")
        pc_status = "KHONG CO - dung random init"
        if src_points is not None:
            pc_status = "DENSE MVS" if src_points_is_dense else "sparse COLMAP"
        print(f"[{scene_id}] done (camera_model={intrinsics_map[scene_id]['camera_model']}, "
              f"k1={intrinsics_map[scene_id]['k1']:.5f}, sparse_pc={pc_status})")

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "intrinsics_map.json", "w", encoding="utf-8") as f:
        json.dump(intrinsics_map, f, indent=2)

    print(f"Preprocessed {len(scene_dirs)} scenes.")
    print(f"Intrinsics + distortion saved to {out_root / 'intrinsics_map.json'}")


if __name__ == "__main__":
    main()
