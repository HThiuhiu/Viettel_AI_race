"""
Patch Splatfacto them ROBUST L1 LOSS kieu RobustNeRF (Sabour et al., CVPR 2023) -
bo qua % pixel co residual (|GT-render|) LON NHAT khi tinh L1 loss, coi do la
NOI DUNG TRANSIENT (khong nhat quan giua cac goc nhin: phan chieu doi theo goc
tren be mat kim loai, bong di chuyen theo gio chup) thay vi cau truc 3D that.

=== VI SAO PATCH NAY, DUA TREN BANG CHUNG THAT DA DO ===

3 lan do that (khong doan):
  1. LPIPS tuong doi on (77% toi da) trong khi PSNR/SSIM te (38%/56%) - dau hieu
     kinh dien cua transient content: anh "nhin hop ly ve cau truc" (LPIPS - so
     dac trung sau, dung sai khong gian tot) nhung "sai gia tri pixel cu the o
     1 goc nhin" (PSNR/SSIM - so tung pixel tuyet doi).
  2. Misalignment DA BI LOAI qua --diagnose_misalignment (gain <0.5dB ca 2 lan do).
  3. Tang MCMC_SCALE_REG/OPACITY_REG lam MOI THU te hon (0.02 -> 0.05: Score giam
     0.5077 -> 0.4332) - da xac nhan tu source that (splatfacto.py get_loss_dict):
     2 tham so nay phat BLANKET, KHONG DIEU KIEN len TAT CA Gaussian, khong phai
     cong cu chi sua vung loi cuc bo - loai ca huong nay.

=> Nghi pham con lai kho nhat, khop toan bo bang chung: mot phan anh KHONG NHAT
   QUAN giua cac view (transient), khong phai sai hinh dang/vi tri Gaussian.
   RobustNeRF giai quyet dung van de nay bang cach KHONG ep model hoc noi dung
   khong nhat quan do vao cau truc 3D tinh.

=== CO CHE (ban 2: co RAMP va SOFT-WEIGHT) ===
Trong get_loss_dict(), thay:
    Ll1 = torch.abs(gt_img - pred_img).mean()
bang trung binh CO TRONG SO cua residual, trong do pixel bi nghi la transient
chi bi HA trong so chu khong bi bo han:
    pct    = ROBUST_DROP_PERCENTILE * ramp(step)      # ramp tuyen tinh 15%->35%
    nguong = phan vi (1 - pct) cua residual
    w      = 1.0 neu residual <= nguong, nguoc lai = ROBUST_SOFT_WEIGHT (0.15)
    Ll1    = sum(residual * w) / sum(w)

Hai thay doi so voi ban 1 va ly do:
  * RAMP: giai doan dau MOI pixel deu sai nhieu (model chua hoi tu), nen "5%
    te nhat" luc do khong phai transient ma la chi tiet kho (day cap, anten,
    nen xa). Bo chung tu buoc 0 = triet tieu gradient o dung cho kho nhat.
  * SOFT-WEIGHT: bo han (trong so 0) khien vung bi phan loai NHAM khong bao gio
    hoi phuc duoc vi mat sach gradient. 0.15 giu lai duong hoi phuc.

An toan: ROBUST_DROP_PERCENTILE=0 (mac dinh neu khong dat bien moi truong) =
Ll1 y het ban goc. ROBUST_SOFT_WEIGHT=0.0 = hanh vi bo han cua ban 1.

Bien lien quan (deu co mac dinh hop ly, khong bat buoc dat):
    ROBUST_RAMP_START_FRAC  (0.15)  bat dau ramp tai 15% tien trinh
    ROBUST_RAMP_END_FRAC    (0.35)  dat pct day du tai 35% tien trinh
    ROBUST_SOFT_WEIGHT      (0.15)  trong so cho pixel nghi la transient
    ITERATIONS              (30000) tong so buoc - dung chung voi patch perceptual

=== CACH DUNG ===
    python patches/patch_robust_transient_loss.py --gs_repo <khong dung - day la Nerfstudio>
    Thuc ra doi tuong patch la file splatfacto.py cua CHINH MOI TRUONG dang chay,
    khong phai 1 repo rieng - xem --nerfstudio_path ben duoi.

    python patches/patch_robust_transient_loss.py

    Sau do train voi:
    ROBUST_DROP_PERCENTILE=0.06 bash train.sh --scene HCM0674 ...
    (0.06 = bo qua 6% pixel te nhat moi anh moi buoc - gia tri khoi diem hop ly,
    paper goc dung ~5-10% tuy scene)

Khoi phuc: python patches/patch_robust_transient_loss.py --restore
"""

import argparse
import os
import shutil
import sys


def find_splatfacto_path():
    import nerfstudio
    return os.path.join(os.path.dirname(nerfstudio.__file__), "models", "splatfacto.py")


HELPERS = '''
# ===== [ROBUST TRANSIENT LOSS] chen boi patch_robust_transient_loss.py =====
import os as _rtl_os
import torch as _rtl_torch


def _rtl_ramp_pct(step: int) -> float:
    """Ty le pixel bi coi la TRANSIENT tai buoc `step`, tang TUYEN TINH tu 0 (o
    moc ROBUST_RAMP_START_FRAC) len ROBUST_DROP_PERCENTILE (o moc
    ROBUST_RAMP_END_FRAC) roi giu nguyen.

    VI SAO CAN RAMP: giai doan dau train, MOI pixel deu sai nhieu vi model chua
    hoi tu - luc do "5% pixel te nhat" KHONG phai transient ma chi la vung chua
    kip hoc (thuong la chi tiet tan so cao: day cap, anten, nen xa). Bo chung
    ngay tu buoc 0 se khien model khong bao gio duoc nhan gradient o dung nhung
    vung kho nhat. Ramp tu 15%->35% tien trinh cho model kip hoc cau truc tho
    truoc, roi moi bat dau loc transient khi residual lon THUC SU mang y nghia
    "khong nhat quan giua cac view".

    Dung chung bien ITERATIONS voi patch perceptual loss de 2 lich trinh cung
    tham chieu 1 nguon tong so buoc."""
    target = float(_rtl_os.environ.get("ROBUST_DROP_PERCENTILE", "0.0"))
    if target <= 0.0:
        return 0.0
    start = float(_rtl_os.environ.get("ROBUST_RAMP_START_FRAC", "0.15"))
    end = float(_rtl_os.environ.get("ROBUST_RAMP_END_FRAC", "0.35"))
    total = float(_rtl_os.environ.get("ITERATIONS", "30000"))
    progress = step / max(total, 1.0)
    if end <= start:
        ramp = 1.0 if progress >= start else 0.0
    else:
        ramp = (progress - start) / (end - start)
        ramp = min(max(ramp, 0.0), 1.0)
    return target * ramp


def _rtl_robust_l1(model, gt_img, pred_img):
    """L1 loss kieu RobustNeRF nhung dung SOFT-WEIGHT thay vi bo han.

    Pixel co residual vuot nguong phan vi (1 - pct) khong bi loai bo hoan toan
    ma chi bi HA TRONG SO xuong ROBUST_SOFT_WEIGHT (mac dinh 0.15).

    VI SAO SOFT thay vi HARD: bo han (trong so 0) lam gradient tai nhung vung do
    BIEN MAT hoan toan. Neu phan loai sai - tuc pixel do that ra la cau truc that
    chua hoc xong chu khong phai phan chieu - thi model khong bao gio sua duoc
    nua, vi khong con tin hieu nao keo no ve. Trong so 0.15 giu lai mot phan
    gradient: du nho de khong bi noi dung transient dan dat, nhung du khac 0 de
    vung bi phan loai nham van co duong hoi phuc.

    Cac gia tri bien: ROBUST_SOFT_WEIGHT=1.0 => y het L1 thuong; =0.0 => y het
    hanh vi bo han cu. An toan: ROBUST_DROP_PERCENTILE khong dat hoac =0 =>
    L1 THUONG, giong het ban goc."""
    residual_full = _rtl_torch.abs(gt_img - pred_img)
    pct = _rtl_ramp_pct(int(getattr(model, "step", 0)))
    if pct <= 0.0:
        return residual_full.mean()
    soft_w = float(_rtl_os.environ.get("ROBUST_SOFT_WEIGHT", "0.15"))
    residual = residual_full.mean(dim=-1)  # gop 3 kenh mau, con [H, W]
    n = residual.numel()
    k = max(1, int(n * (1.0 - pct)))
    threshold = residual.reshape(-1).kthvalue(k).values
    w = _rtl_torch.where(residual <= threshold,
                          _rtl_torch.ones_like(residual),
                          _rtl_torch.full_like(residual, soft_w))
    # Trung binh CO TRONG SO: chia cho tong trong so, khong phai so pixel
    return (residual * w).sum() / w.sum().clamp_min(1e-8)
# ===== [ROBUST TRANSIENT LOSS] het phan chen =====
'''

# QUAN TRONG: HELPERS chua ham module-level (import + def) nen PHAI chen o
# ngoai class, khong duoc chen truoc 1 method (se pha vo indentation cua class,
# gay IndentationError - da tu bat loi nay qua kiem thu truoc khi giao code).
# Dung lai dung anchor da verify voi patch_splatfacto_mip_filter.py truoc do.
ANCHOR_MODULE_LEVEL = "from nerfstudio.models.base_model import Model"
OLD_LL1 = "        Ll1 = torch.abs(gt_img - pred_img).mean()"
NEW_LL1 = "        Ll1 = _rtl_robust_l1(self, gt_img, pred_img)  # [ROBUST TRANSIENT LOSS]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerfstudio_path", type=str, default=None,
                     help="Duong dan truc tiep toi splatfacto.py, neu khong tu dong tim duoc")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    splatfacto_path = args.nerfstudio_path or find_splatfacto_path()
    backup_path = splatfacto_path + ".robust_loss_ORIGINAL_backup"

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
    if OLD_LL1 not in code:
        print(f"[LOI] Khong tim thay dong 'Ll1 = torch.abs(gt_img - pred_img).mean()' "
              f"dung 1 khoang trang nhu da verify tu source. Co the indentation khac ban "
              f"da kiem chung - DUNG lai.", file=sys.stderr)
        sys.exit(1)

    # HELPERS (module-level: import + def ham) chen truoc anchor module-level -
    # KHONG chen truoc method get_loss_dict (se nam trong than class, sai indentation).
    code = code.replace(ANCHOR_MODULE_LEVEL, HELPERS + "\n" + ANCHOR_MODULE_LEVEL, 1)
    code = code.replace(OLD_LL1, NEW_LL1, 1)

    with open(splatfacto_path, "w", encoding="utf-8") as f:
        f.write(code)

    print("\n[HOAN TAT] Da patch splatfacto.py: Ll1 gio dung robust loss (RobustNeRF-style).")
    print("Mac dinh AN TOAN (ROBUST_DROP_PERCENTILE=0 = giong het ban goc).")
    print("Kich hoat: ROBUST_DROP_PERCENTILE=0.06 bash train.sh --scene <ten_scene> ...")
    print("Khoi phuc ve ban goc: python patches/patch_robust_transient_loss.py --restore")


if __name__ == "__main__":
    main()
