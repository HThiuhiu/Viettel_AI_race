"""
Export Gaussian splat (.ply) tu checkpoint da train, roi loc bot floater
bang statistical outlier removal (open3d) de dung cho VIEWER Digital Twin
(Potree, SuperSplat, Three.js, ...).

LUU Y QUAN TRONG: script nay KHONG anh huong diem submission/render cua
cuoc thi, vi render_and_submit.py render truc tiep tu pipeline dang train
(gauss_params trong bo nho), khong doc lai file .ply nay. Muc dich cua
script la tao ra 1 asset .ply "sach" hon de dung cho phan trung bay
Digital Twin (neu de bai/giam khao co xem xet phan nay), hoac de tu kiem
tra truc quan chat luong scene bang mat thuong trong 1 web viewer.

Cach dung:
    python tools/clean_export_gaussians.py \
        --load_config outputs/HCM0421/HCM0421/splatfacto/2024-01-01_000000/config.yml \
        --output_dir exports/HCM0421 \
        --nb_neighbors 20 --std_ratio 2.0

Yeu cau: pip install open3d --break-system-packages
"""

import argparse
import subprocess
import sys
from pathlib import Path


def export_ply(config_path: Path, output_dir: Path) -> Path:
    """Goi ns-export gaussian-splat de xuat .ply tu checkpoint da train."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(output_dir),
    ]
    print(f"[export_ply] Chay: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    candidates = sorted(output_dir.glob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"Khong tim thay file .ply trong {output_dir} sau khi export.")
    return candidates[0]


def clean_ply(ply_path: Path, out_path: Path, nb_neighbors: int, std_ratio: float):
    import open3d as o3d

    print(f"[clean_ply] Doc {ply_path} ...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_before = len(pcd.points)

    cleaned, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    n_after = len(cleaned.points)
    removed = n_before - n_after
    pct = 100.0 * removed / max(n_before, 1)
    print(f"[clean_ply] Da loai {removed}/{n_before} Gaussian ({pct:.1f}%) duoc coi la floater/outlier.")

    if pct > 15:
        print("[CANH BAO] Ty le loai bo > 15%. Co the nb_neighbors/std_ratio dang qua chat, "
              "hoac scene thuc su co nhieu floater (nen xem lai hyperparameter luc train: "
              "camera-optimizer, strategy mcmc, densify_grad_thresh). VOI SCENE BTS: ty le cao "
              "cung co the la dau hieu SOR dang xoa nham day cap/anten mong (mat do diem thap "
              "tu nhien doc theo cau truc manh, khong phai floater that). Kiem tra truc quan "
              "file ply truoc va sau khi loc (vd trong CloudCompare hoac web viewer) - dac biet "
              "zoom vao vung day cap/anten - truoc khi dung ban da loc lam ban chinh thuc.")

    o3d.io.write_point_cloud(str(out_path), cleaned)
    print(f"[clean_ply] Da ghi file da loc vao {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", type=str, required=True, help="Duong dan config.yml")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--nb_neighbors", type=int, default=20,
                         help="So lang gieng gan nhat dung de tinh khoang cach trung binh. "
                              "Voi scene BTS co day cap/anten mong, KHONG nen tang qua cao "
                              "(vd >30) vi se buoc SOR so sanh Gaussian tren day cap voi ca "
                              "Gaussian o be mat rong, de xoa nham.")
    parser.add_argument("--std_ratio", type=float, default=2.0,
                         help="Nguong do lech chuan; nho hon = loc gat gao hon nhung de xoa "
                              "nham Gaussian day cap/anten mong. Voi scene BTS, KHONG nen ha "
                              "xuong duoi 1.5; neu can loc floater manh hon, uu tien tinh chinh "
                              "lai hyperparameter luc train (cull-scale-thresh, densify-grad-thresh) "
                              "thay vi ha std_ratio o day.")
    args = parser.parse_args()

    try:
        import open3d  # noqa: F401
    except ImportError:
        print("[LOI] Chua cai open3d. Chay: pip install open3d --break-system-packages", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    ply_path = export_ply(Path(args.load_config), output_dir)
    clean_ply(ply_path, output_dir / f"{ply_path.stem}_cleaned.ply", args.nb_neighbors, args.std_ratio)


if __name__ == "__main__":
    main()