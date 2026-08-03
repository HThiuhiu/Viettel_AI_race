"""
Quet TAT CA checkpoint da luu cua 1 lan train (khong chi checkpoint cuoi cung)
va do Score THAT (qua eval_submission_path.py, dung duong render nop bai) cho
tung checkpoint, roi bao cao checkpoint tot nhat.

VI SAO CAN SCRIPT NAY: nerfstudio/eval_submission_path.py mac dinh luon load
checkpoint MOI NHAT trong load_dir. Nhung da co bang chung thuc te (config B,
scene HCM0421): checkpoint step 20000 cho Score TOT HON han checkpoint cuoi
25000 - tuc la train lau hon KHONG dam bao tot hon (co the do dao dong loss,
overfit cuc bo, hay 1 vong densify/cull "xau" xay ra gan cuoi). --save-only-
latest-checkpoint False trong train.sh da duoc bat san nen cac checkpoint
trung gian VAN CON TREN DISK, chi la chua ai do dung Score that cho tung cai.

Cach dung:
    python pipeline/select_best_checkpoint.py --config <...>/config.yml \
        --supersample_factor 2.0 --appearance_transfer \
        --appearance_mode lab_transfer --appearance_k_neighbors 3 \
        --distortion_mode un_redistort

Co the truyen them bat ky flag nao eval_submission_path.py ho tro qua
--extra_args "--post_process denoise_sharpen ..." (chuoi don, se duoc tach
bang shlex).

Ket qua: in bang so sanh Score/PSNR/SSIM/LPIPS theo tung step, va ghi
<config_dir>/checkpoint_sweep_scores.csv de doi chieu sau nay.
"""

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Khop CHINH XAC khoi tom tat cuoi cung cua eval_submission_path.py (chu HOA,
# "PSNR  = " co dau cach truoc "="), KHONG duoc them re.IGNORECASE - cac dong
# tung anh dung dang chu thuong lien "psnr=0.xxx" (khong dau cach) nen se KHONG
# bi khop nham neu giu phan biet hoa/thuong nay.
SCORE_LINE_RE = re.compile(
    r"PSNR\s*=\s*([\d.]+).*?SSIM\s*=\s*([\d.]+).*?LPIPS\s*=\s*([\d.]+).*?SCORE\s*=\s*([\d.]+)",
    re.DOTALL,
)


def find_checkpoint_steps(config_path: Path):
    ckpt_dir = config_path.parent / "nerfstudio_models"
    if not ckpt_dir.is_dir():
        print(f"[LOI] Khong thay thu muc checkpoint {ckpt_dir}", file=sys.stderr)
        sys.exit(1)
    steps = []
    for p in ckpt_dir.glob("step-*.ckpt"):
        m = re.search(r"step-0*(\d+)\.ckpt", p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def run_eval_for_step(script_dir: Path, config_path: Path, step: int, extra_args):
    cmd = [
        sys.executable, str(script_dir / "eval_submission_path.py"),
        "--config", str(config_path),
        "--load_step", str(step),
    ] + extra_args
    print(f"\n=== Dang do Score cho checkpoint step={step} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(f"[CANH BAO] step={step} that bai:\n{proc.stderr[-2000:]}", file=sys.stderr)
        return None
    m = SCORE_LINE_RE.search(proc.stdout)
    if not m:
        print(f"[CANH BAO] Khong parse duoc Score tu output cho step={step}", file=sys.stderr)
        return None
    psnr, ssim, lpips, score = (float(x) for x in m.groups())
    return {"step": step, "psnr": psnr, "ssim": ssim, "lpips": lpips, "score": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="config.yml cua lan train can quet")
    ap.add_argument("--steps", type=str, default=None,
                     help="Danh sach step can do, cach nhau boi dau phay (vd '10000,20000,25000'). "
                          "Neu khong dat, tu dong quet TAT CA checkpoint co tren disk.")
    ap.add_argument("--extra_args", type=str, default="",
                     help="Cac flag them cho eval_submission_path.py, vd "
                          "'--supersample_factor 2.0 --appearance_transfer --distortion_mode un_redistort'")
    ap.add_argument("--out_csv", type=str, default=None,
                     help="Mac dinh: <thu_muc_config>/checkpoint_sweep_scores.csv")
    args = ap.parse_args()

    config_path = Path(args.config)
    script_dir = Path(__file__).resolve().parent
    extra_args = shlex.split(args.extra_args)

    if args.steps:
        steps = sorted(int(s) for s in args.steps.split(",") if s.strip())
    else:
        steps = find_checkpoint_steps(config_path)
    if not steps:
        print("[LOI] Khong tim thay checkpoint nao de quet.", file=sys.stderr)
        sys.exit(1)
    print(f"Se do Score cho {len(steps)} checkpoint: {steps}")

    rows = []
    for step in steps:
        row = run_eval_for_step(script_dir, config_path, step, extra_args)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[LOI] Khong co checkpoint nao do thanh cong.", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: r["score"], reverse=True)
    out_csv = Path(args.out_csv) if args.out_csv else config_path.parent / "checkpoint_sweep_scores.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "psnr", "ssim", "lpips", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== BANG XEP HANG CHECKPOINT (Score giam dan) ===")
    for r in rows:
        print(f"  step={r['step']:>7}  psnr={r['psnr']:.3f}  ssim={r['ssim']:.4f}  "
              f"lpips={r['lpips']:.4f}  score={r['score']:.4f}")
    best = rows[0]
    print(f"\n=> Checkpoint TOT NHAT: step={best['step']} (score={best['score']:.4f})")
    print(f"Da ghi bang so sanh: {out_csv}")


if __name__ == "__main__":
    main()
