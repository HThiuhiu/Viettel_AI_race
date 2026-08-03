"""
Chan doan: xac nhan STRATEGY thuc su duoc dung (mcmc hay am tham fallback ve
default/ADC) va SO GAUSSIAN THUC TE trong checkpoint da train - doc truc tiep
tu config.yml + checkpoint bang chinh co che eval_setup() ma render_and_submit.py
dang dung (dang tin cay, khong doan qua parse YAML tay).

Ly do can script nay: neu MCMC khong thuc su active (vd Nerfstudio cai qua PyPI
thieu field strategy/max_gs_num), ADC mac dinh se chay thay - KHONG CO TRAN CUNG
cho so Gaussian, va thuong phan bo kem hieu qua hon (nhieu diem o vung de, thieu
diem o vung kho nhu kim loai it texture/phan chieu) - dung trieu chung "nhieu
Gaussian nhung van mo 1 so vung" ma MCMC (voi trong so + gioi han cung
max_gs_num) duoc thiet ke de tranh.

Cach dung:
    python pipeline/check_gs_count.py --checkpoints_root /kaggle/working/outputs
    python pipeline/check_gs_count.py --checkpoints_root /kaggle/working/outputs --scene HCM0421
"""

import argparse
import sys
from pathlib import Path

import torch

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def find_latest_config(checkpoints_root: Path, scene_id: str):
    candidates = sorted(checkpoints_root.glob(f"{scene_id}/**/config.yml"))
    return candidates[-1] if candidates else None


def check_one_scene(scene_id: str, config_path: Path, device: str):
    from nerfstudio.utils.eval_utils import eval_setup

    config, pipeline, _, _ = eval_setup(config_path, test_mode="inference")
    model_config = config.pipeline.model

    strategy = getattr(model_config, "strategy", None)
    max_gs_num = getattr(model_config, "max_gs_num", None)
    num_gs = pipeline.model.means.shape[0]

    print(f"\n[{scene_id}] config: {config_path}")
    if strategy is None:
        print(f"  [CANH BAO] Ban Nerfstudio nay KHONG CO field 'strategy' trong SplatfactoModelConfig -> "
              f"MCMC KHONG THE active du STRATEGY=mcmc duoc yeu cau luc train. Da chay ADC mac dinh "
              f"(khong co tran cung so Gaussian) - day rat co the la nguyen nhan so Gaussian vuot "
              f"MAX_GS_NUM va phan bo kem hieu qua (mo o vung kho).")
    else:
        print(f"  strategy da luu trong config: {strategy}")
        if strategy != "mcmc":
            print(f"  [LUU Y] strategy khong phai 'mcmc' - kiem tra lai STRATEGY env var luc train.")

    print(f"  max_gs_num da cau hinh: {max_gs_num if max_gs_num is not None else '(khong co field nay)'}")
    print(f"  SO GAUSSIAN THUC TE trong checkpoint: {num_gs:,}")

    if max_gs_num is not None and num_gs > max_gs_num:
        print(f"  [XAC NHAN BUG] So Gaussian thuc te ({num_gs:,}) VUOT max_gs_num da cau hinh "
              f"({max_gs_num:,}) -> MCMC KHONG thuc su gioi han duoc, chac chan dang chay ADC "
              f"(hoac strategy khong duoc ap dung dung). Can cai lai Nerfstudio tu "
              f"NERFSTUDIO_INSTALL_SPEC (xem kaggle_train_t4x2.sh) va train lai.")
    elif max_gs_num is not None:
        print(f"  [OK] So Gaussian nam trong gioi han max_gs_num -> MCMC co ve dang hoat dong dung.")

    return num_gs, strategy, max_gs_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_root", type=str, required=True)
    parser.add_argument("--scene", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints_root = Path(args.checkpoints_root)

    if args.scene:
        scene_ids = [args.scene]
    else:
        scene_ids = sorted(p.name for p in checkpoints_root.iterdir() if p.is_dir())

    if not scene_ids:
        print(f"[LOI] Khong tim thay scene nao trong {checkpoints_root}", file=sys.stderr)
        sys.exit(1)

    for scene_id in scene_ids:
        config_path = find_latest_config(checkpoints_root, scene_id)
        if config_path is None:
            print(f"[{scene_id}] [BO QUA] Khong tim thay checkpoint")
            continue
        try:
            check_one_scene(scene_id, config_path, device)
        except Exception as e:
            print(f"[{scene_id}] [LOI] Khong doc duoc checkpoint: {e}")


if __name__ == "__main__":
    main()
