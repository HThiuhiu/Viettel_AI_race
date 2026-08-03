"""
A/B `rgb_gain` vs `lab_transfer` (xem image_postprocess.py, render_and_submit.py)
THEO TUNG SCENE thay vi chon 1 mode co dinh cho toan bo submission - 2 co che
co the phu hop khac nhau tuy dac diem anh sang/mau cua tung scene.

Tai su dung TRUC TIEP eval_submission_path.py (subprocess, khong viet lai
logic load pipeline/tinh Score) - chay 2 lan/scene (rgb_gain, lab_transfer),
so Score that trong CSV ket qua, chon mode thang, ghi ra 1 file JSON de
render_and_submit.py doc va tu dong ap dung DUNG mode cho DUNG scene qua
--appearance_mode_json.

Cach dung:
    python tools/select_appearance_mode.py --checkpoints_root /kaggle/working/outputs
    python tools/select_appearance_mode.py --checkpoints_root ... --scene HCM0674 \
        --supersample_factor 1.5 --distortion_mode un_redistort --appearance_k_neighbors 3

    Sau do render submission that:
    python pipeline/render_and_submit.py ... --appearance_transfer \
        --appearance_mode_json appearance_mode_per_scene.json
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def find_configs(checkpoints_root: Path) -> dict:
    """Tra ve dict scene_id -> config_path (config.yml MOI NHAT) cho moi
    scene duoi checkpoints_root/<scene_id>/**/config.yml."""
    configs = {}
    for scene_dir in sorted(checkpoints_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        candidates = sorted(scene_dir.glob("**/config.yml"))
        if candidates:
            configs[scene_dir.name] = candidates[-1]
    return configs


def run_eval(config_path: Path, mode: str, extra_args: list, out_csv: Path) -> bool:
    pipeline_dir = Path(__file__).resolve().parent.parent / "pipeline"
    cmd = [sys.executable, str(pipeline_dir / "eval_submission_path.py"), "--config", str(config_path),
           "--appearance_transfer", "--appearance_mode", mode, "--out_csv", str(out_csv)] + extra_args
    print("Chay:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return False
    print(result.stdout[-500:])
    return True


def read_last_score(csv_path: Path):
    if not csv_path.exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_root", type=str, required=True)
    ap.add_argument("--scene", type=str, default=None, help="Chi chay 1 scene (de test nhanh)")
    ap.add_argument("--supersample_factor", type=float, default=1.5)
    ap.add_argument("--distortion_mode", type=str, default="rasterizer",
                     choices=["rasterizer", "un_redistort"])
    ap.add_argument("--appearance_k_neighbors", type=int, default=1)
    ap.add_argument("--out_json", type=str, default="appearance_mode_per_scene.json")
    ap.add_argument("--tmp_csv_dir", type=str, default=".")
    args = ap.parse_args()

    checkpoints_root = Path(args.checkpoints_root)
    configs = find_configs(checkpoints_root)
    if args.scene:
        configs = {k: v for k, v in configs.items() if k == args.scene}
    if not configs:
        raise RuntimeError(f"Khong tim thay checkpoint nao duoi {checkpoints_root}" +
                            (f" khop ten '{args.scene}'" if args.scene else ""))

    extra_args = ["--supersample_factor", str(args.supersample_factor),
                  "--distortion_mode", args.distortion_mode,
                  "--appearance_k_neighbors", str(args.appearance_k_neighbors)]

    results = {}
    for scene_id, config_path in configs.items():
        print(f"\n=== [{scene_id}] A/B rgb_gain vs lab_transfer ===")
        scores = {}
        ok_all = True
        for mode in ("rgb_gain", "lab_transfer"):
            tmp_csv = Path(args.tmp_csv_dir) / f"_tmp_select_appearance_{scene_id}_{mode}.csv"
            tmp_csv.unlink(missing_ok=True)
            if not run_eval(config_path, mode, extra_args, tmp_csv):
                print(f"[CANH BAO][{scene_id}] eval_submission_path.py that bai voi mode={mode}, bo qua scene nay.")
                ok_all = False
                break
            scores[mode] = read_last_score(tmp_csv)
            tmp_csv.unlink(missing_ok=True)
        if not ok_all or None in scores.values():
            continue
        winner = max(scores, key=scores.get)
        print(f"[{scene_id}] rgb_gain={scores['rgb_gain']:.5f}  lab_transfer={scores['lab_transfer']:.5f}  "
              f"-> THANG: {winner}")
        results[scene_id] = winner

    out_path = Path(args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi {out_path}: {results}")
    print(f"Dung file nay voi: python pipeline/render_and_submit.py --appearance_transfer "
          f"--appearance_mode_json {out_path} ...")


if __name__ == "__main__":
    main()
