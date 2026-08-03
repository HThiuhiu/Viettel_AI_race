"""
Kiem tra nhanh cau truc du lieu COLMAP truoc khi chay preprocess/train.
KHONG can GPU, KHONG can cai Nerfstudio - chi doc header cua file .bin
de dem so luong camera/anh, chay trong vai giay cho toan bo dataset.

Muc dich: phat hien som scene nao thieu file / sai ten file / so luong
anh trong images.bin lech qua nhieu so voi so anh that trong thu muc
images/ - truoc khi tram GPU quota vao 1 loi co the phat hien trong 5 giay.

Cach dung:
    python tools/validate_colmap_structure.py --raw_root .
    python tools/validate_colmap_structure.py --raw_root /kaggle/working/Viettel
"""

import argparse
import struct
import sys
from pathlib import Path

REQUIRED_IN_SPARSE = ["cameras.bin", "images.bin"]
OPTIONAL_IN_SPARSE = ["points3D.bin", "points3D.ply"]
# Cac bien the ten file duoc CHAP NHAN (khop voi logic find_sparse_file
# trong preprocess_colmap_to_nerfstudio.py ban v2) - neu file ton tai duoi
# 1 trong cac ten nay thi KHONG bi coi la thieu.
SPARSE_FILE_ALIASES = {
    "cameras.bin": ["cameras.bin", "camera.bin", "Cameras.bin", "Camera.bin"],
    "images.bin": ["images.bin", "image.bin", "Images.bin", "Image.bin"],
    "points3D.bin": ["points3D.bin", "point3D.bin", "Point3D.bin", "points3d.bin", "Points3D.bin"],
    "points3D.ply": ["points3D.ply", "point3D.ply", "Point3D.ply", "points3d.ply", "Points3D.ply"],
}


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def count_cameras_bin(path: Path) -> int:
    with open(path, "rb") as f:
        (n,) = read_next_bytes(f, 8, "Q")
    return n


def count_images_bin(path: Path) -> int:
    """Chi doc so luong anh o dau file, khong parse full (nhanh, an toan)."""
    with open(path, "rb") as f:
        (n,) = read_next_bytes(f, 8, "Q")
    return n


def find_sparse_file(sparse_dir: Path, canonical_name: str):
    """Tim file theo ten chuan hoac cac bien the duoc chap nhan. Tra ve Path
    neu tim thay (bat ke ten chuan hay bien the), None neu khong."""
    if not sparse_dir.exists():
        return None
    for alias in SPARSE_FILE_ALIASES.get(canonical_name, [canonical_name]):
        p = sparse_dir / alias
        if p.exists():
            return p
    target_lower = canonical_name.lower()
    for f in sparse_dir.iterdir():
        if f.is_file() and f.name.lower() == target_lower:
            return f
    return None


def validate_scene(scene_name: str, train_dir: Path, sparse_dir: Path) -> list:
    problems = []
    images_dir = train_dir / "images"

    if not images_dir.exists():
        problems.append(f"KHONG CO thu muc images/ tai {images_dir}")
        n_images_on_disk = 0
    else:
        n_images_on_disk = sum(1 for _ in images_dir.glob("*") if _.is_file())
        if n_images_on_disk == 0:
            problems.append(f"Thu muc images/ ton tai nhung RONG: {images_dir}")

    if not sparse_dir.exists():
        problems.append(f"KHONG CO thu muc sparse/0/ tai {sparse_dir}")
        return problems

    for required in REQUIRED_IN_SPARSE:
        found = find_sparse_file(sparse_dir, required)
        if found is None:
            problems.append(f"THIEU FILE BAT BUOC: {sparse_dir / required} (da thu ca cac bien "
                             f"the {SPARSE_FILE_ALIASES.get(required, [required])})")
        elif found.name != required:
            print(f"  [{scene_name}] Dung '{found.name}' thay cho ten chuan '{required}' - OK, "
                  f"preprocess_colmap_to_nerfstudio.py (v2) nhan dien duoc bien the nay.")

    for optional in OPTIONAL_IN_SPARSE:
        found = find_sparse_file(sparse_dir, optional)
        if found is None:
            print(f"  [{scene_name}] [tuy chon] Khong co {optional} (hoac bien the) - "
                  f"se dung random init Gaussian thay vi COLMAP points")
        elif found.name != optional:
            print(f"  [{scene_name}] Dung '{found.name}' thay cho ten chuan '{optional}' - OK.")

    cameras_bin = find_sparse_file(sparse_dir, "cameras.bin")
    images_bin = find_sparse_file(sparse_dir, "images.bin")
    if cameras_bin is not None:
        try:
            n_cam = count_cameras_bin(cameras_bin)
            print(f"  [{scene_name}] cameras.bin: {n_cam} camera(s)")
        except Exception as e:
            problems.append(f"Doc {cameras_bin} loi (file co the bi hong): {e}")

    if images_bin is not None:
        try:
            n_img_bin = count_images_bin(images_bin)
            print(f"  [{scene_name}] images.bin: {n_img_bin} pose(s), "
                  f"thu muc images/ co {n_images_on_disk} file anh")
            if n_img_bin < n_images_on_disk:
                problems.append(
                    f"images.bin chi co {n_img_bin} pose nhung images/ co {n_images_on_disk} anh "
                    f"-> se co anh KHONG train duoc vi thieu pose (kiem tra lai COLMAP reconstruction)"
                )
        except Exception as e:
            problems.append(f"Doc {images_bin} loi (file co the bi hong): {e}")

    return problems


def discover_scenes(raw_root: Path):
    scenes = []
    for candidate in sorted(raw_root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "train").exists():
            scenes.append((candidate.name, candidate / "train"))
        elif (candidate / "images").exists() and (candidate / "sparse").exists():
            scenes.append((candidate.name, candidate))
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    scenes = discover_scenes(raw_root)

    if not scenes:
        print(f"[LOI] Khong tim thay scene nao trong {raw_root} "
              f"(can co <scene>/train/ hoac <scene>/images + <scene>/sparse)")
        sys.exit(1)

    print(f"Tim thay {len(scenes)} scene: {[s[0] for s in scenes]}\n")

    all_problems = {}
    for scene_name, train_dir in scenes:
        sparse_dir = train_dir / "sparse" / "0"
        print(f"=== [{scene_name}] ===")
        problems = validate_scene(scene_name, train_dir, sparse_dir)
        if problems:
            all_problems[scene_name] = problems
        print()

    print("=" * 60)
    if all_problems:
        print(f"CO VAN DE o {len(all_problems)}/{len(scenes)} scene:\n")
        for scene_name, problems in all_problems.items():
            print(f"[{scene_name}]")
            for p in problems:
                print(f"  - {p}")
        print(f"\n=> Sua cac van de tren TRUOC KHI chay preprocess/train tren Kaggle, "
              f"de khong ton quota GPU vao loi co the phat hien ngay tren may local.")
        sys.exit(1)
    else:
        print(f"OK - {len(scenes)}/{len(scenes)} scene co du file bat buoc. "
              f"Co the chay preprocess_colmap_to_nerfstudio.py.")


if __name__ == "__main__":
    main()
