"""
Dense MVS init cho Gaussian Splatting - thay sparse COLMAP points (thuong rat
thua o 2 loai vung: (1) be mat kim loai it texture/phan chieu nhu cot/anten BTS,
va (2) VUNG RIA KHUNG HINH - noi drone chi bay ngang qua voi it goc nhin chong
lan hon han vung trung tam quy dao) bang point cloud DAY DAC hon tu COLMAP dense
stereo (patch_match_stereo + stereo_fusion).

BANG CHUNG THUC TE (anh render that, checkpoint 10k): vung trung tam anh (toa
nha, bon nuoc, cot ang-ten) sac net, nhung goc/ria khung hinh mo han han - dung
du doan cua gia thuyet "it goc nhin chong lan o ria" ben tren. Day la ly do chinh
de nang cap script nay thay vi tiep tuc chinh loss/regularization (2 huong da
thu truoc do deu cho ket qua "cang can thiep manh cang te hon").

QUAN TRONG ve he toa do: script nay CHI dung pose/intrinsics DA CO SAN trong
sparse/0/ (cameras.bin, images.bin) de undistort anh va chay dense stereo -
KHONG chay lai SfM, KHONG doi pose/camera. Vi vay point cloud dau ra van nam
DUNG he toa do cu, tuong thich hoan toan voi test_poses.csv va
dataparser_transforms.json - khac voi viec chay lai toan bo COLMAP tu dau (se
ra 1 he toa do moi, PHA VO alignment voi test poses).

Script nay hoan toan OPT-IN: preprocess_colmap_to_nerfstudio.py se tu dong uu
tien dung ket qua cua script nay (<scene>/dense/fused.ply) neu no ton tai, va
fallback ve sparse points3D.ply neu ban khong chay script nay - khong bat buoc.

=== NANG CAP SO VOI BAN TRUOC ===
1. --timeout_minutes: patch_match_stereo co the chay HANG GIO khong bao truoc -
   rui ro that voi quota Kaggle/Colab co han. Het gio se DUNG AN TOAN, giu
   nguyen sparse fallback, KHONG lam hong du lieu da co.
2. Bao cao SO SANH so diem sparse vs dense sau khi xong - biet ngay dense co
   thuc su day hon dang ke khong, thay vi doan.
3. Tham so chat luong stereo_fusion (--min_num_pixels, --max_reproj_error,
   --max_depth_error) duoc dua ra tunable thay vi an trong code - anh huong
   truc tiep den viec giu lai bao nhieu diem o vung ria it chong lan (nguong
   qua chat se loai het diem o dung vung can cuu).
4. --gpu_index rieng cho buoc dense (doc lap voi GPU dung de train, huu ich
   neu chay 2 scene song song tren 2 GPU khac nhau).

Yeu cau:
- Binary `colmap` phai co san (Kaggle: apt-get install -y colmap - can bat
  Internet cho Kaggle notebook). Dense stereo (patch_match_stereo) can COLMAP
  ban co ho tro CUDA - neu ban cai qua apt va thay chay rat cham/loi CUDA,
  nhieu kha nang ban apt khong co CUDA, can build tu source hoac dung Docker
  image co san COLMAP+CUDA.
- Ton them thoi gian GPU dang ke (tuy so anh/do phan giai) - dung
  --max_image_size de gioi han do phan giai lam viec cua patch_match_stereo
  (chi anh huong buoc stereo, KHONG anh huong anh goc dung de train - van
  giu resolution=1 that su cho qua trinh train, dung mac dinh cua pipeline).

Cach dung:
    python pipeline/dense_colmap_init.py --raw_root .
    python pipeline/dense_colmap_init.py --raw_root . --scene HCM0421
    python pipeline/dense_colmap_init.py --raw_root . --max_image_size 1600 --skip_existing
    python pipeline/dense_colmap_init.py --raw_root . --scene HCM0674 --timeout_minutes 90
"""

import argparse
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path


def discover_scenes(raw_root: Path):
    """Giong het discover_scenes() trong preprocess_colmap_to_nerfstudio.py -
    tra ve (scene_id, scene_dir) voi scene_dir la thu muc chua images/ va
    sparse/0/ truc tiep (vd <root>/<scene>/train hoac <root>/<scene>)."""
    scene_dirs = []
    for candidate in sorted(raw_root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "train" / "images").exists() and (candidate / "train" / "sparse" / "0").exists():
            scene_dirs.append((candidate.name, candidate / "train"))
        elif (candidate / "images").exists() and (candidate / "sparse" / "0").exists():
            scene_dirs.append((candidate.name, candidate))
    return scene_dirs


def count_ply_points(ply_path: Path) -> int:
    """Doc header PLY (ASCII, ke ca khi phan data la binary) de lay so diem -
    khong can load toan bo file, du lon van nhanh."""
    try:
        with open(ply_path, "rb") as f:
            while True:
                line = f.readline().decode("ascii", errors="replace").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[2])
                if line == "end_header" or not line:
                    return -1
    except Exception:
        return -1


def count_bin_points(bin_path: Path) -> int:
    """So diem trong points3D.bin (COLMAP binary sparse format): 8 byte dau la
    so luong diem (uint64), giong cach doc images.bin/cameras.bin."""
    try:
        with open(bin_path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            return n
    except Exception:
        return -1


def run_step(cmd, scene_id: str, step_name: str, timeout_sec) -> bool:
    print(f"[{scene_id}] Chay {step_name}: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        elapsed_min = (time.time() - t0) / 60
        print(f"[LOI][{scene_id}] {step_name} VUOT QUA THOI GIAN CHO PHEP "
              f"({elapsed_min:.1f} phut) - DUNG AN TOAN, giu nguyen sparse "
              f"points lam fallback (khong lam hong du lieu da co).")
        return False
    if result.returncode != 0:
        print(f"[LOI][{scene_id}] {step_name} that bai (exit {result.returncode}). "
              f"Bo qua scene nay, giu nguyen sparse points lam fallback.")
        return False
    elapsed_min = (time.time() - t0) / 60
    print(f"[{scene_id}] {step_name} xong sau {elapsed_min:.1f} phut.")
    return True


def dense_init_one_scene(scene_id: str, scene_dir: Path, colmap_bin: str, max_image_size: int,
                          skip_existing: bool, timeout_min: float, gpu_index: str,
                          min_num_pixels: int, max_reproj_error: float, max_depth_error: float) -> bool:
    dense_dir = scene_dir / "dense"
    fused_path = dense_dir / "fused.ply"

    if skip_existing and fused_path.exists():
        print(f"[{scene_id}] {fused_path} da ton tai va --skip_existing duoc bat -> bo qua")
        return True

    sparse_dir = scene_dir / "sparse" / "0"
    images_dir = scene_dir / "images"
    if not sparse_dir.exists() or not images_dir.exists():
        print(f"[BO QUA][{scene_id}] Thieu {sparse_dir} hoac {images_dir}")
        return False

    # Bao cao mat do sparse TRUOC KHI chay - lam moc so sanh voi dense sau nay.
    n_sparse = -1
    for name in ("points3D.ply", "points3D.bin"):
        p = sparse_dir / name
        if p.exists():
            n_sparse = count_ply_points(p) if name.endswith(".ply") else count_bin_points(p)
            if n_sparse >= 0:
                print(f"[{scene_id}] Sparse hien tai ({name}): {n_sparse:,} diem")
                break

    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True)

    timeout_sec = timeout_min * 60 if timeout_min > 0 else None
    gpu_args = ["--PatchMatchStereo.gpu_index", gpu_index] if gpu_index else []

    ok = run_step(
        [colmap_bin, "image_undistorter",
         "--image_path", str(images_dir),
         "--input_path", str(sparse_dir),
         "--output_path", str(dense_dir),
         "--output_type", "COLMAP"],
        scene_id, "image_undistorter", timeout_sec,
    )
    if not ok:
        return False

    ok = run_step(
        [colmap_bin, "patch_match_stereo",
         "--workspace_path", str(dense_dir),
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true",
         "--PatchMatchStereo.max_image_size", str(max_image_size),
         *gpu_args],
        scene_id, "patch_match_stereo (dense stereo - buoc lau nhat)", timeout_sec,
    )
    if not ok:
        return False

    ok = run_step(
        [colmap_bin, "stereo_fusion",
         "--workspace_path", str(dense_dir),
         "--workspace_format", "COLMAP",
         "--input_type", "geometric",
         "--output_path", str(fused_path),
         "--StereoFusion.min_num_pixels", str(min_num_pixels),
         "--StereoFusion.max_reproj_error", str(max_reproj_error),
         "--StereoFusion.max_depth_error", str(max_depth_error)],
        scene_id, "stereo_fusion", timeout_sec,
    )
    if not ok:
        return False

    if not fused_path.exists():
        print(f"[LOI][{scene_id}] stereo_fusion bao thanh cong nhung khong thay {fused_path}")
        return False

    n_dense = count_ply_points(fused_path)
    print(f"[{scene_id}] Xong dense MVS -> {fused_path}")
    if n_sparse >= 0 and n_dense >= 0:
        ty_le = n_dense / max(n_sparse, 1)
        print(f"[{scene_id}] SO SANH: sparse={n_sparse:,} diem -> dense={n_dense:,} diem "
              f"(gap {ty_le:.1f}x)")
        if ty_le < 3:
            print(f"[{scene_id}] [CANH BAO] Dense chi day hon sparse duoi 3 lan - loi ich co "
                  f"the khong dang ke so voi thoi gian bo ra. Thu ha --max_reproj_error hoac "
                  f"--min_num_pixels neu nghi ngo nguong dang loc qua chat.")
    print(f"[{scene_id}] (preprocess_colmap_to_nerfstudio.py se tu uu tien dung file nay)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--scene", type=str, default=None, help="Chi chay 1 scene (de test truoc khi chay het)")
    parser.add_argument("--colmap_bin", type=str, default="colmap", help="Duong dan binary colmap")
    parser.add_argument("--max_image_size", type=int, default=1600,
                         help="Do phan giai lam viec cua patch_match_stereo (KHONG phai anh goc dung de "
                              "train) - giam so nay de chay nhanh hon neu het gio quota GPU Kaggle")
    parser.add_argument("--skip_existing", action="store_true",
                         help="Bo qua scene da co dense/fused.ply tu lan chay truoc")
    parser.add_argument("--timeout_minutes", type=float, default=120.0,
                         help="Thoi gian toi da cho MOI buoc COLMAP (image_undistorter/"
                              "patch_match_stereo/stereo_fusion). Vuot qua se DUNG AN TOAN va giu "
                              "sparse fallback, tranh chiem het quota GPU Kaggle/Colab. 0 = khong gioi han.")
    parser.add_argument("--gpu_index", type=str, default="",
                         help="Chi dinh GPU rieng cho patch_match_stereo (vd '0' hoac '1') - doc lap "
                              "voi CUDA_VISIBLE_DEVICES dung cho train, huu ich khi chay 2 scene song song.")
    parser.add_argument("--min_num_pixels", type=int, default=5,
                         help="So anh toi thieu phai dong y ve 1 diem 3D moi giu lai (stereo_fusion). "
                              "Ha xuong (vd 3) cho vung ria khung hinh it goc nhin chong lan - nhung ha "
                              "qua thap se lay ca nhieu/diem sai.")
    parser.add_argument("--max_reproj_error", type=float, default=2.0,
                         help="Sai so tai chieu toi da (pixel) de giu 1 diem. Mac dinh COLMAP la 2.0 - "
                              "tang len (vd 4.0) neu --skip_existing bao dense khong day hon sparse dang "
                              "ke, co the nguong dang loc qua chat dung vung can cuu (ria khung hinh).")
    parser.add_argument("--max_depth_error", type=float, default=1.0,
                         help="Sai so do sau toi da (ty le so voi do sau) de giu 1 diem. Mac dinh COLMAP 1.0.")
    args = parser.parse_args()

    if shutil.which(args.colmap_bin) is None:
        print(f"[LOI] Khong tim thay binary '{args.colmap_bin}' trong PATH.\n"
              f"  Tren Kaggle: bat 'Internet' cho notebook roi chay:\n"
              f"    apt-get update && apt-get install -y colmap\n"
              f"  Dense stereo (patch_match_stereo) can COLMAP co ho tro CUDA - neu ban thay chay\n"
              f"  cuc cham hoac bao loi lien quan CUDA, ban apt co the la bien dich CPU-only.",
              file=sys.stderr)
        sys.exit(1)

    raw_root = Path(args.raw_root)
    scene_dirs = discover_scenes(raw_root)
    if args.scene:
        scene_dirs = [(sid, sdir) for sid, sdir in scene_dirs if sid == args.scene]
    if not scene_dirs:
        raise RuntimeError(f"Khong tim thay scene nao trong {raw_root}" +
                            (f" khop ten '{args.scene}'" if args.scene else ""))

    results = {}
    for scene_id, scene_dir in scene_dirs:
        print(f"\n=== [{scene_id}] Bat dau dense MVS init ===")
        results[scene_id] = dense_init_one_scene(
            scene_id, scene_dir, args.colmap_bin, args.max_image_size,
            args.skip_existing, args.timeout_minutes, args.gpu_index,
            args.min_num_pixels, args.max_reproj_error, args.max_depth_error,
        )

    print("\n=== Tong ket ===")
    for scene_id, ok in results.items():
        print(f"[{scene_id}] {'OK - da co dense/fused.ply' if ok else 'THAT BAI - se dung sparse fallback'}")
    print("\nBuoc tiep theo: chay lai preprocess_colmap_to_nerfstudio.py binh thuong - "
          "no se tu nhan ra va uu tien dung dense/fused.ply cho cac scene vua xong.")


if __name__ == "__main__":
    main()
