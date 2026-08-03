"""
Patch Splatfacto them FreGS-style PROGRESSIVE FREQUENCY REGULARIZATION vao
get_loss_dict() (Zhang et al., "FreGS: 3D Gaussian Splatting with Progressive
Frequency Regularization", CVPR 2024).

=== NGUON CONG THUC (da tra cuu ban HTML day du cua paper, khong suy doan) ===
FreGS bien doi anh render/GT sang mien tan so qua FFT 2D, tach amplitude +
phase:
    |F(u,v)| = sqrt(Re^2 + Im^2)        (amplitude)
    angle(F(u,v)) = arctan(Im/Re)       (phase)
Dung 1 low-pass filter TINH (static, ban kinh D0 co dinh) va 1 high-pass
filter DONG (dynamic, ban kinh/dai D_t MO RONG dan theo buoc train):
    LF(u,v) = F(u,v) * H_low(u,v)
    HF(u,v) = F(u,v) * H_high(u,v)
Loss:
    L_f = w_l*(d_la + d_lp)                          neu 0 < t <= T0
    L_f = w_l*(d_la + d_lp) + w_h*(d_ha + d_hp)       neu t > T0
(d_la/d_lp = chenh lech L1 amplitude/phase o vung LOW; d_ha/d_hp = o vung
HIGH). Dai tan so cua high-pass filter mo rong TUYEN TINH theo buoc train:
    D0 < D_t < (t - T0)*(D - D0)/(T - T0) + D0
=> Chien luoc "coarse-to-fine": tu buoc 0 den T0, CHI phat sai lech vung tan
so THAP (cau truc tho); sau T0, dan dan THEM ca vung tan so CAO (chi tiet
min), day rong dan cho den buoc T (paper dung T = luc densification ket
thuc, vd 15000). Bo sung cho L1+SSIM/D-SSIM da dung suot qua trinh train,
KHONG thay the.

=== DON GIAN HOA CO CHU DICH SO VOI PAPER (ghi ro, khong am tham) ===
1. Paper dung 2 trong so rieng w_l/w_h - patch nay dung 1 FREQ_LOSS_WEIGHT
   chung cho de tune (it tham so hon). Neu can tach rieng sau, sua
   FREQ_LOSS_WEIGHT_HIGH lam trong so rieng cho nhanh high-pass.
2. T0/T (buoc bat dau/ket thuc mo rong dai high-pass) o day la bien moi
   truong TUYET DOI (FREQ_LOSS_T0, FREQ_LOSS_T_END) thay vi gan chet vao
   iteration densify_until_iter cua paper goc (Splatfacto/Nerfstudio dung
   STOP_SPLIT_AT, khong phai densify_until_iter cua INRIA) - mac dinh
   FREQ_LOSS_T_END doc tu STOP_SPLIT_AT (da duoc train.sh export san) neu
   co, khop dung tinh than "ket thuc cung luc densification" cua paper.
3. Ban nay CHUA verify tren du lieu 3DGS THAT (khong co GPU o day) - da
   verify BANG SO HOC (FFT tren tensor tong hop) rang cong thuc chay dung,
   mask low/high dung vi tri, gia tri huu han - nhung TAC DONG THAT len
   Score PHAI duoc do bang eval_submission_path.py, khong suy doan.

=== AN TOAN TUYET DOI MAC DINH ===
FREQ_LOSS_WEIGHT=0.0 (khong dat bien moi truong) => nhanh nay khong bao gio
chay, get_loss_dict() giong het ban goc.

Cach dung:
    python patches/patch_splatfacto_frequency_loss.py

    Sau do train voi:
    FREQ_LOSS_WEIGHT=0.01 bash train.sh --scene <ten_scene> --iterations 20000
    (0.01 la gia tri khoi diem than trong - FFT tren toan anh co the co bien
    do lon hon L1/SSIM, THU gia tri nho truoc, A/B qua eval_submission_path.py
    truoc khi tang, dung ky luat 1-bien/1-lan-do da theo suot du an.)

Khoi phuc: python patches/patch_splatfacto_frequency_loss.py --restore
"""

import argparse
import os
import re
import shutil
import sys


def find_splatfacto_path():
    import nerfstudio
    return os.path.join(os.path.dirname(nerfstudio.__file__), "models", "splatfacto.py")


HELPERS = '''
# ===== [FREQUENCY LOSS PATCH] chen boi patch_splatfacto_frequency_loss.py =====
import os as _fq_os
import torch as _fq_torch


def _fq_fft_amp_phase(gray_hw):
    """FFT 2D (norm="ortho") + fftshift (dua tan so 0 ve tam anh, dung quy
    uoc ban kinh trong _fq_radial_mask). Tra ve (amplitude, phase) cung
    shape [H,W].

    QUAN TRONG (loi da phat hien va sua qua kiem thu thuc te, khong phai tu
    dau): torch.fft.fft2 MAC DINH (norm=None, "backward") KHONG chuan hoa -
    bien do FFT (dac biet thanh phan tan so thap/DC) ti le thuan voi TONG SO
    PIXEL (H*W), khong phai trung binh. Voi anh that (494x660, gia tri
    0-1), da do duoc bien do sai lech trung binh ~15x LON HON L1 loss ngay
    ca voi FREQ_LOSS_WEIGHT nho (0.01) - du de lan at hoan toan L1+SSIM va
    lam mat y nghia "bo sung" cua loss nay (dung norm="ortho" giam ve ~1.1x
    L1, dung thang do mong muon)."""
    F = _fq_torch.fft.fftshift(_fq_torch.fft.fft2(gray_hw, norm="ortho"))
    return F.abs(), _fq_torch.angle(F)


def _fq_radial_mask(h, w, device, r_frac_lo, r_frac_hi):
    """Mask hinh vanh khan (annulus) chon vung tan so co ban kinh chuan hoa
    trong [r_frac_lo, r_frac_hi] * ban kinh toi da - dung chung cho ca low-pass
    (lo=0) va high-pass (lo=D0, hi=D_t dang mo rong)."""
    yy, xx = _fq_torch.meshgrid(
        _fq_torch.arange(h, dtype=_fq_torch.float32, device=device),
        _fq_torch.arange(w, dtype=_fq_torch.float32, device=device), indexing="ij")
    cy, cx = h / 2.0, w / 2.0
    dist = _fq_torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = _fq_torch.sqrt(_fq_torch.tensor(cy ** 2 + cx ** 2, device=device))
    lo, hi = r_frac_lo * max_r, r_frac_hi * max_r
    return (dist >= lo) & (dist <= hi)


def _fq_frequency_loss(model, gt_img, pred_img):
    """FreGS-style progressive frequency regularization (xem docstring file
    patch de biet cong thuc goc + don gian hoa co chu dich). An toan: chi
    duoc goi khi FREQ_LOSS_WEIGHT > 0 (kiem tra o diem goi trong
    get_loss_dict), ham nay khong tu kiem tra lai weight."""
    step = int(model.step)
    total_iters = float(_fq_os.environ.get("ITERATIONS", "30000"))
    t0 = float(_fq_os.environ.get("FREQ_LOSS_T0", "500"))
    stop_split_at_env = _fq_os.environ.get("STOP_SPLIT_AT")
    default_t_end = float(stop_split_at_env) if stop_split_at_env else total_iters * 0.75
    t_end = float(_fq_os.environ.get("FREQ_LOSS_T_END", str(default_t_end)))
    low_frac = float(_fq_os.environ.get("FREQ_LOSS_LOW_RADIUS_FRAC", "0.1"))
    high_frac = float(_fq_os.environ.get("FREQ_LOSS_HIGH_RADIUS_FRAC", "1.0"))

    gt_gray = gt_img.mean(dim=-1)
    pred_gray = pred_img.mean(dim=-1)
    gt_amp, gt_phase = _fq_fft_amp_phase(gt_gray)
    pred_amp, pred_phase = _fq_fft_amp_phase(pred_gray)

    h, w = gt_gray.shape
    device = gt_gray.device
    low_mask = _fq_radial_mask(h, w, device, 0.0, low_frac)
    d_la = (gt_amp - pred_amp).abs()[low_mask].mean()
    d_lp = (gt_phase - pred_phase).abs()[low_mask].mean()
    loss = d_la + d_lp

    if step > t0:
        progress = (step - t0) / max(t_end - t0, 1.0)
        progress = min(max(progress, 0.0), 1.0)
        cur_hi = low_frac + (high_frac - low_frac) * progress
        high_mask = _fq_radial_mask(h, w, device, low_frac, cur_hi)
        if high_mask.any():
            d_ha = (gt_amp - pred_amp).abs()[high_mask].mean()
            d_hp = (gt_phase - pred_phase).abs()[high_mask].mean()
            loss = loss + d_ha + d_hp
    return loss
# ===== [FREQUENCY LOSS PATCH] het phan chen =====
'''

# QUAN TRONG: HELPERS chi chua import + def ham (module-level), PHAI chen o
# NGOAI class - giong dung anchor da verify voi cac patch truoc (chen truoc 1
# method se pha vo indentation cua class, gay IndentationError).
ANCHOR_MODULE_LEVEL = "from nerfstudio.models.base_model import Model"

INJECTION_LOSS = '''
        # ==== [FREQUENCY LOSS PATCH] ====
        _fq_weight = float(_fq_os.environ.get("FREQ_LOSS_WEIGHT", "0.0"))
        _fq_every_n = max(1, int(_fq_os.environ.get("FREQ_LOSS_EVERY_N", "5")))
        if self.training and _fq_weight > 0.0 and int(self.step) % _fq_every_n == 0:
            loss_dict["frequency_loss"] = _fq_frequency_loss(self, gt_img, pred_img) * _fq_weight
            if not hasattr(self, "_fq_logged"):
                self._fq_logged = True
                print(f"===[FREQUENCY LOSS PATCH] dang active (FreGS-style), weight={_fq_weight}, "
                      f"every_n={_fq_every_n}===")
        # ========================================
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerfstudio_path", type=str, default=None,
                     help="Duong dan truc tiep toi splatfacto.py, neu khong tu dong tim duoc")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    splatfacto_path = args.nerfstudio_path or find_splatfacto_path()
    backup_path = splatfacto_path + ".frequency_loss_ORIGINAL_backup"

    if not os.path.exists(splatfacto_path):
        print(f"[LOI] Khong thay {splatfacto_path}", file=sys.stderr)
        sys.exit(1)

    if args.restore:
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, splatfacto_path)
            print(f"Da khoi phuc {splatfacto_path} tu ban goc.")
        else:
            print("[LOI] Khong co backup de khoi phuc.", file=sys.stderr)
        return

    if not os.path.exists(backup_path):
        shutil.copyfile(splatfacto_path, backup_path)
        print(f"Da tao backup ban GOC: {backup_path}")

    with open(backup_path, "r", encoding="utf-8") as f:
        code = f.read()

    if ANCHOR_MODULE_LEVEL not in code:
        print(f"[LOI] Khong tim thay '{ANCHOR_MODULE_LEVEL}'. Ban Nerfstudio khac voi "
              f"ban da verify - DUNG lai, khong patch mu quang.", file=sys.stderr)
        sys.exit(1)

    func_marker = "def get_loss_dict"
    func_start = code.find(func_marker)
    if func_start == -1:
        print(f"[LOI] Khong tim thay 'def get_loss_dict'. DUNG lai.", file=sys.stderr)
        sys.exit(1)

    rest = code[func_start + len(func_marker):]
    next_def = re.search(r"\n    def \w+\(|\nclass \w+", rest)
    func_end = func_start + len(func_marker) + (next_def.start() if next_def else len(rest))
    func_body = code[func_start:func_end]

    if "return loss_dict" not in func_body:
        print(f"[LOI] Khong tim thay 'return loss_dict' trong get_loss_dict. Cau truc ham "
              f"co the da doi - DUNG lai, khong patch mu quang.", file=sys.stderr)
        sys.exit(1)
    if "gt_img" not in func_body or "pred_img" not in func_body:
        print(f"[LOI] Khong tim thay bien 'gt_img'/'pred_img' trong get_loss_dict - ten bien "
              f"co the da doi o ban Nerfstudio nay. DUNG lai.", file=sys.stderr)
        sys.exit(1)

    new_func_body = func_body.replace("return loss_dict", INJECTION_LOSS + "\n        return loss_dict", 1)
    code = code[:func_start] + new_func_body + code[func_end:]

    code = code.replace(ANCHOR_MODULE_LEVEL, HELPERS + "\n" + ANCHOR_MODULE_LEVEL, 1)

    with open(splatfacto_path, "w", encoding="utf-8") as f:
        f.write(code)

    print("\n[HOAN TAT] Da patch splatfacto.py: frequency_loss (FreGS-style progressive) da them.")
    print("Mac dinh AN TOAN (FREQ_LOSS_WEIGHT=0 = giong het ban goc).")
    print("Kich hoat: FREQ_LOSS_WEIGHT=0.01 bash train.sh --scene <ten_scene> --iterations 20000")
    print("Khoi phuc ve ban goc: python patches/patch_splatfacto_frequency_loss.py --restore")


if __name__ == "__main__":
    main()
