

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path


CONFIG_OVERRIDES = {
    "A": {"DENSIFY_GRAD_THRESH": "0.0004", "CULL_ALPHA_THRESH": "0.0015"},
    "B": {"DENSIFY_GRAD_THRESH": "0.0008", "CULL_ALPHA_THRESH": "0.0025"},
}

TRAIN_ENV = {
    "STRATEGY": "default",         
    "SH_DEGREE": "3",              
    "RASTERIZE_MODE": "antialiased",
    "NUM_DOWNSCALES": "0",
    "RESOLUTION_SCHEDULE": "1200",
    "APPLY_MIP_FILTER_PATCH": "1",
    "DENSIFY_GRAD_THRESH": "0.0005",
    "CULL_ALPHA_THRESH": "0.0020",
    "CULL_SCALE_THRESH": "0.30",
    "RESET_ALPHA_EVERY": "60",
    "USE_SCALE_REGULARIZATION": "1",
    "MAX_GAUSS_RATIO": "5.0",
    "ROBUST_DROP_PERCENTILE": "0.05",
    "ROBUST_RAMP_START_FRAC": "0.15",
    "ROBUST_RAMP_END_FRAC": "0.35",
    "ROBUST_SOFT_WEIGHT": "0.15",
    "PERCEPTUAL_LOSS_MAX_WEIGHT": "0.08",
    "PERCEPTUAL_LOSS_RAMP_START_FRAC": "0.6",  
    "SSIM_LAMBDA": "0.3",           
    "FREQ_LOSS_WEIGHT": "0.0",     
    "TRAIN_SPLIT_FRACTION": "0.9",  
    "STEPS_PER_SAVE": "5000",       
    "MAX_JOBS": "4",
}

ITERATIONS = 30000
TRAIN_ENV["ITERATIONS"] = str(ITERATIONS)


def build_env(config_name: str, iterations: int, outputs: str) -> dict:
    env = dict(TRAIN_ENV)
    env.update(CONFIG_OVERRIDES.get(config_name, {}))
    env["ITERATIONS"] = str(iterations)
    env["OUTPUT_ROOT"] = outputs
    env["TORCH_EXTENSIONS_DIR"] = f"/kaggle/working/torch_ext_{config_name}"
    return env

RENDER_ARGS = ("--supersample_factor 2.5 --distortion_mode un_redistort "
               "--jpeg_quality 95 --appearance_transfer --appearance_mode rgb_gain "
               "--appearance_k_neighbors 3")

FILTER_CANDIDATES = [
    ("none", ""),
    ("despeckle", "--post_process despeckle"),
    ("despeckle+lowpass", "--post_process despeckle_lowpass --lowpass_sigma 0.35"),
]

SCORE_RE = re.compile(r"SCORE\s*=\s*([\d.]+)")

COMPETITION_SCENES = []
# ==========================================================================


def sh(cmd, capture=False):
    print(f"\n$ {cmd}\n", flush=True)
    if capture:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(r.stdout[-4000:])
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
        return r
    return subprocess.run(cmd, shell=True)


def find_configs(outputs_root: Path) -> dict:
    out = {}
    for c in sorted(outputs_root.glob("*/**/config.yml")):
        scene = c.relative_to(outputs_root).parts[0]
        out[scene] = c          # glob da sap xep -> giu ban moi nhat
    return out


def phase_check(a):
    print("=" * 70 + "\nPHASE CHECK - kiem tra moi truong (khong dung GPU)\n" + "=" * 70)
    ok = True

    train_sh = Path(a.work_dir) / "train.sh"
    if not train_sh.exists():
        print(f"[LOI] Khong thay {train_sh}"); return False
    txt = train_sh.read_text(encoding="utf-8", errors="ignore").
    if "use-scale-regularization" not in txt:
        print("[CANH BAO] train.sh KHONG truyen --pipeline.model.use-scale-regularization.")
        print("           => MAX_GAUSS_RATIO=7.0 se VO HIEU (khong chong duoc needle).")
        print("           Sua train.sh, hoac bo 2 bien do khoi cau hinh cho do ao tuong.")
        ok = False
    else:
        print("[OK] train.sh co noi use-scale-regularization")

    for f, needle in [("pipeline/render_and_submit.py", "save_submission_image"),
                       ("pipeline/eval_submission_path.py", "--jpeg_quality"),
                       ("pipeline/image_postprocess.py", "despeckle_dark"),
                       ("pipeline/distortion_redistort.py", "uu + 0.5 - target_cx")]:
        p = Path(a.work_dir) / f
        if not p.exists() or needle not in p.read_text(encoding="utf-8", errors="ignore"):
            print(f"[LOI] {f} chua duoc cap nhat (thieu '{needle}') - upload lai truoc khi chay")
            ok = False
        else:
            print(f"[OK] {f}")

    have = find_configs(Path(a.outputs))
    print(f"\nScene THI (can nop): {COMPETITION_SCENES}")
    print(f"Scene DA co checkpoint: {sorted(have) or '(chua co scene nao)'}")
    missing = [s for s in COMPETITION_SCENES if s not in have]
    if missing:
        print(f"\n>>> CON THIEU {len(missing)}/5 scene: {missing}")
        print(f">>> Moi scene thieu keo diem trung binh xuong ~0.144 - uu tien cao nhat.")
        print(f">>> Neu ngan sach GPU eo hep: ha --iterations va train DU 5 scene van hon "
              f"train that ky 1 scene.")
    else:
        print("\n[OK] Du ca 5 scene thi.")
    return ok


def phase_train(a):
    print("=" * 70 + f"\nPHASE TRAIN - config {a.config} - GPU {a.gpu}\n" + "=" * 70)
    envd = build_env(a.config, a.iterations, a.outputs)
    env = " ".join(f"{k}={v}" for k, v in envd.items())
    scene_arg = f"--scene {a.scene} " if a.scene else ""
    log = f"/kaggle/working/train_{a.config}_{a.scene or 'ALL'}.txt"

    print(f"OUTPUT_ROOT          = {envd['OUTPUT_ROOT']}")
    print(f"TORCH_EXTENSIONS_DIR = {envd['TORCH_EXTENSIONS_DIR']}")
    print(f"DENSIFY_GRAD_THRESH  = {envd['DENSIFY_GRAD_THRESH']}")
    print(f"CULL_ALPHA_THRESH    = {envd['CULL_ALPHA_THRESH']}")
    print(f"ITERATIONS           = {envd['ITERATIONS']}  (2 lich trinh ramp dung con so nay)")

    cmd = (f"cd {a.work_dir} && {env} nohup bash {a.work_dir}/train.sh "
           f"--raw-root {a.work_dir} --data-root {a.data_root} {scene_arg}"
           f"--iterations {a.iterations} --downscale 1 --gpu {a.gpu} > {log} 2>&1 &")
    sh(cmd)
    print(f"\nTrain da chay nen. Theo doi:  !tail -30 {log}")
    print("KHONG chay phase eval cho den khi train KET THUC HAN - nap checkpoint")
    print("dang ghi do tung lam 2 lan do cung cau hinh lech nhau 0.027 diem.")
    return True


def phase_eval(a):
    """Chon checkpoint tot nhat, roi chot bo loc hau xu ly - tren tung scene."""
    print("=" * 70 + "\nPHASE EVAL - chon checkpoint + chot bo loc\n" + "=" * 70)
    configs = find_configs(Path(a.outputs))
    if not configs:
        print("[LOI] Chua co checkpoint nao."); return False

    scenes = [a.scene] if a.scene else sorted(configs)
    for scene in scenes:
        cfg = configs.get(scene)
        if cfg is None:
            print(f"[BO QUA] {scene}: chua co checkpoint"); continue
        print(f"\n{'#' * 70}\n# {scene}\n{'#' * 70}")

        sh(f'cd {a.work_dir} && python pipeline/select_best_checkpoint.py --config {cfg} '
           f'--extra_args "{RENDER_ARGS}"')

        sweep = cfg.parent / "checkpoint_sweep_scores.csv"
        best_step = None
        if sweep.exists():
            rows = sweep.read_text(encoding="utf-8").strip().splitlines()[1:]
            if rows:
                best_step = rows[0].split(",")[0]    
                print(f"\n=> Checkpoint tot nhat cua {scene}: step {best_step}")

        step_arg = f"--load_step {best_step}" if best_step else ""
        results = []
        for name, pp in FILTER_CANDIDATES:
            r = sh(f"cd {a.work_dir} && python pipeline/eval_submission_path.py --config {cfg} "
                   f"{step_arg} {RENDER_ARGS} {pp} "
                   f"--out_csv /kaggle/working/final_{scene}.csv", capture=True)
            m = SCORE_RE.search(r.stdout or "")
            results.append((name, float(m.group(1)) if m else None, pp))

        valid = [(n, s, pp) for n, s, pp in results if s is not None]
        if valid:
            base = dict((n, s) for n, s, _ in valid).get("none")
            print(f"\n--- Bo loc cho {scene} ---")
            for n, s, _ in valid:
                d = f"{s - base:+.4f}" if base is not None else "  n/a"
                print(f"  {n:20} score={s:.4f}  ({d} so voi none)")
            best = max(valid, key=lambda x: x[1])
            if base is not None and best[1] - base > 0.005:
                print(f"  => DUNG '{best[0]}' (thang ro ret)")
            else:
                print("  => Giu 'none': khong cai nao thang qua 0.005, phan con lai la nhieu.")
    return True


def phase_render(a):
    print("=" * 70 + "\nPHASE RENDER + KIEM TRA\n" + "=" * 70)
    scene_list = a.scene if a.scene else ",".join(COMPETITION_SCENES)
    sh(f"cd {a.work_dir} && python pipeline/render_and_submit.py "
       f"--data_root {a.data_root} --processed_data_root {a.processed} "
       f"--checkpoints_root {a.outputs} --scene {scene_list} {RENDER_ARGS} {a.extra_render} "
       f"--submission_dir /kaggle/working/submission "
       f"--zip_name /kaggle/working/submission.zip")

    print("\n--- Kiem tra anh da luu ---")
    imgs = sorted(glob.glob("/kaggle/working/submission/*/*"))
    print(f"Tong so anh: {len(imgs)}  (can 60/scene)")
    if imgs:
        try:
            from PIL import Image
            im = Image.open(imgs[0])
            q = sum(sum(v) for v in im.quantization.values()) if im.quantization else None
            print(f"Bang luong tu JPEG = {q}  (anh chup that = 927; bai nop CU bi loi = 4638)")
            if q and q > 2000:
                print(">>> CANH BAO: van dang nen q75! Kiem tra --jpeg_quality da duoc truyen chua.")
        except Exception as e:
            print(f"(khong doc duoc anh de kiem tra: {e})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                     choices=["check", "train", "eval", "render", "all"])
    ap.add_argument("--work_dir", default="/kaggle/working/Viettel")
    ap.add_argument("--data_root", default="/kaggle/working/data/processed")
    ap.add_argument("--processed", default="/kaggle/working/data/processed")
    ap.add_argument("--outputs", default="/kaggle/working/outputs")
    ap.add_argument("--scene", default=None,
                     help="De TRONG de train/nop TAT CA scene (khuyen nghi - diem la trung binh 5 scene)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--config", default="B", choices=sorted(CONFIG_OVERRIDES),
                     help="A = phu kin (densify manh, cull toi thieu); B = vua phai "
                          "(giu nguong da tung validate). Loss GIONG HET nhau o ca 2 - "
                          "chi khac truc densify/cull de phep so sanh dien giai duoc.")
    ap.add_argument("--iterations", type=int, default=ITERATIONS,
                     help="Phai khop voi thuc te vi 2 lich trinh ramp (robust, perceptual) "
                          "dung no de tinh tien trinh.")
    ap.add_argument("--extra_render", default="",
                     help="Vd '--post_process despeckle' neu phase eval cho thay no thang")
    a = ap.parse_args()

    phases = ["check", "train", "eval", "render"] if a.phase == "all" else [a.phase]
    for p in phases:
        ok = {"check": phase_check, "train": phase_train,
              "eval": phase_eval, "render": phase_render}[p](a)
        if p == "check" and not ok and a.phase == "all":
            print("\n[DUNG] Phase check that bai - sua truoc khi chay tiep.")
            return
        if p == "train" and a.phase == "all":
            print("\n[DUNG] Train dang chay NEN. Doi no ket thuc roi chay:")
            print("       python run_final.py --phase eval")
            return


if __name__ == "__main__":
    main()
