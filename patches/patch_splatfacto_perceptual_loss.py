"""
Patch Splatfacto them PERCEPTUAL LOSS (LPIPS-as-training-loss, kieu Johnson et
al. 2016 "Perceptual Losses for Real-Time Style Transfer") vao get_loss_dict(),
theo LICH TRINH RAMP (curriculum: L1+SSIM truoc, LPIPS chi them dan o cuoi
qua trinh train) thay vi trong so hang so.

=== VI SAO PATCH NAY (+ CAP NHAT: co so tu paper, khong con chi ly thuyet) ===
Score cham diem = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm - LPIPS chiem trong
so CAO NHAT (40%). Nhung get_loss_dict() GOC cua Splatfacto (da verify tu
source o vong truoc) CHI dung:
    Ll1 = torch.abs(gt_img - pred_img).mean()
    main_loss = (1 - ssim_lambda) * Ll1 + ssim_lambda * (1 - simloss)
`self.lpips` chi duoc goi LUC EVAL, KHONG BAO GIO duoc dung lam tin hieu
gradient. Da search va xac nhan: cong thuc `L = (1-λ1-λ2)*L1 + λ1*L_SSIM +
λ2*L_LPIPS` la dang duoc dung trong nhieu pipeline 3DGS gan day; rieng biet,
"Drop-In Perceptual Optimization for 3D Gaussian Splatting" (Apple ML
Research, 2026) xac nhan huong "them perceptual loss" giup LPIPS/DISTS/FID
tot hon dang ke so voi loss L1+SSIM thuan tuy cua 3DGS goc (ho dung Wasserstein
Distortion, ban nang cao hon LPIPS thuan - patch nay dung LPIPS co san lam
diem khoi dau don gian hon, WD-R la huong nang cap tiem nang sau).

=== TAI SAO DOI TU HANG SO SANG RAMP (curriculum) ===
Ap LPIPS ngay tu buoc dau (khi Gaussian con hon loan, moi init tu sparse
points) co the tao tin hieu gradient nhieu/khong on dinh - dung ly thuyet
curriculum learning (hoc tu de/thô truoc, kho/tinh sau). Ramp dan LPIPS len o
GIAI DOAN CUOI (sau khi hinh hoc da on dinh qua L1+SSIM) hop ly hon ap dung
tu dau.

=== CO CHE ===
    progress = self.step / ITERATIONS   (ITERATIONS doc tu bien moi truong,
                                          train.sh da export san)
    ramp = clamp((progress - PERCEPTUAL_LOSS_RAMP_START_FRAC) /
                  (1 - PERCEPTUAL_LOSS_RAMP_START_FRAC), 0, 1)
    weight = PERCEPTUAL_LOSS_MAX_WEIGHT * ramp
    if self.training and weight > 0 and self.step % PERCEPTUAL_LOSS_EVERY_N == 0:
        loss_dict["perceptual_loss"] = self.lpips(gt_bchw, pred_bchw) * weight

AN TOAN TUYET DOI mac dinh: PERCEPTUAL_LOSS_MAX_WEIGHT=0.0 (khong dat bien moi
truong) => weight luon = 0, nhanh nay khong bao gio chay, get_loss_dict()
giong het ban goc. Tuong thich nguoc: neu chi dat PERCEPTUAL_LOSS_WEIGHT (ten
bien cu, ban khong-ramp truoc do) ma khong dat PERCEPTUAL_LOSS_MAX_WEIGHT,
gia tri do duoc dung lam max_weight (van ramp theo start_frac mac dinh 0.7).

PERCEPTUAL_LOSS_RAMP_START_FRAC (mac dinh 0.7): bat dau ramp tu 70% tien
trinh train - tuong duong "30% cuoi" (nam trong khoang de xuat "10-20% cuoi"
nhung dung ramp TUYEN TINH mmuot thay vi bat dot ngot, tranh cu-soc gradient
tai 1 buoc). Dat gan 1.0 hon (vd 0.85-0.9) neu muon giong sat hon de xuat
"10-20% cuoi".

=== VI SAO CHI TINH MOI N BUOC (PERCEPTUAL_LOSS_EVERY_N, mac dinh 5) ===
LPIPS forward+backward qua mang VGG/AlexNet ton compute DANG KE hon han
L1/SSIM hien co. Da xac nhan qua tra cuu: nhieu pipeline thuc te chi tinh
LPIPS "moi 20 buoc" de giam tai - mac dinh o day (5) an toan hon (tin hieu
day dac hon) nhung co the TANG len 10-20 neu can train nhanh hon, co can cu
thuc te chu khong phai doan.

=== YEU CAU KIEM TRA TRUOC KHI PATCH (khong patch mu quang) ===
Anchor-check BAT BUOC xac nhan `self.lpips` TON TAI nhu 1 thuoc tinh cua model
(cung module da duoc dung thanh cong trong eval_submission_path.py qua
`pipeline.model.lpips(gt_t, pred_t)` voi input NCHW [0,1]) - neu ban
Nerfstudio khac khong co thuoc tinh nay, DUNG lai thay vi tao code chet.

Cach dung:
    python patches/patch_splatfacto_perceptual_loss.py

    Sau do train voi (ITERATIONS da duoc train.sh export tu dong):
    PERCEPTUAL_LOSS_MAX_WEIGHT=0.05 bash train.sh --scene <ten_scene> --iterations 20000
    (0.05 la gia tri khoi diem hop ly - THU vai gia tri nho 0.02-0.1, A/B qua
    eval_submission_path.py --distortion_mode un_redistort truoc khi chot,
    dung ky luat A/B da theo suot du an: 1 bien/1 lan do. Co the dieu chinh
    them PERCEPTUAL_LOSS_RAMP_START_FRAC neu muon ramp som/muon hon.)

Khoi phuc: python patches/patch_splatfacto_perceptual_loss.py --restore
"""

import argparse
import os
import shutil
import sys


def find_splatfacto_path():
    import nerfstudio
    return os.path.join(os.path.dirname(nerfstudio.__file__), "models", "splatfacto.py")


HELPERS = '''
# ===== [PERCEPTUAL LOSS PATCH] chen boi patch_splatfacto_perceptual_loss.py =====
import os as _pl_os


def _pl_perceptual_loss(model, gt_img, pred_img):
    """LPIPS-as-training-loss (Johnson et al. 2016 style). Chi duoc goi khi
    weight (da tinh qua lich trinh ramp) > 0 - xem _pl_ramp_weight()."""
    gt_bchw = gt_img.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    pred_bchw = pred_img.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    return model.lpips(gt_bchw, pred_bchw)


def _pl_ramp_weight(step: int):
    """Lich trinh curriculum: 0 truoc moc PERCEPTUAL_LOSS_RAMP_START_FRAC cua
    tien trinh train, ramp TUYEN TINH len PERCEPTUAL_LOSS_MAX_WEIGHT sau do.
    Tuong thich nguoc: neu chi dat PERCEPTUAL_LOSS_WEIGHT (ten bien cu, ban
    khong-ramp) ma khong dat PERCEPTUAL_LOSS_MAX_WEIGHT, dung gia tri do lam
    max_weight. An toan tuyet doi: khong dat gi ca => max_weight=0 => luon 0."""
    max_weight = float(_pl_os.environ.get(
        "PERCEPTUAL_LOSS_MAX_WEIGHT", _pl_os.environ.get("PERCEPTUAL_LOSS_WEIGHT", "0.0")))
    if max_weight <= 0.0:
        return 0.0
    start_frac = float(_pl_os.environ.get("PERCEPTUAL_LOSS_RAMP_START_FRAC", "0.7"))
    total_iters = float(_pl_os.environ.get("ITERATIONS", "30000"))
    progress = step / max(total_iters, 1.0)
    if start_frac >= 1.0:
        ramp = 1.0 if progress >= 1.0 else 0.0
    else:
        ramp = (progress - start_frac) / (1.0 - start_frac)
        ramp = min(max(ramp, 0.0), 1.0)
    return max_weight * ramp
# ===== [PERCEPTUAL LOSS PATCH] het phan chen =====
'''

# QUAN TRONG: HELPERS chi chua import + def ham (module-level), PHAI chen o
# NGOAI class - giong dung anchor da verify voi 2 patch truoc (chen truoc 1
# method se pha vo indentation cua class, gay IndentationError).
ANCHOR_MODULE_LEVEL = "from nerfstudio.models.base_model import Model"

INJECTION_LOSS = '''
        # ==== [PERCEPTUAL LOSS PATCH] ====
        _pl_weight = _pl_ramp_weight(int(self.step))
        _pl_every_n = max(1, int(_pl_os.environ.get("PERCEPTUAL_LOSS_EVERY_N", "5")))
        if self.training and _pl_weight > 0.0 and hasattr(self, "lpips") and int(self.step) % _pl_every_n == 0:
            loss_dict["perceptual_loss"] = _pl_perceptual_loss(self, gt_img, pred_img) * _pl_weight
            if not hasattr(self, "_pl_logged"):
                self._pl_logged = True
                print(f"===[PERCEPTUAL LOSS PATCH] dang active, weight={_pl_weight:.4f}, "
                      f"every_n={_pl_every_n}, perceptual_loss se xuat hien dinh ky trong loss dict===")
        # ========================================
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerfstudio_path", type=str, default=None,
                     help="Duong dan truc tiep toi splatfacto.py, neu khong tu dong tim duoc")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    splatfacto_path = args.nerfstudio_path or find_splatfacto_path()
    backup_path = splatfacto_path + ".perceptual_loss_ORIGINAL_backup"

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

    if "self.lpips" not in code:
        print("[LOI] Khong tim thay 'self.lpips' trong splatfacto.py - ban Nerfstudio nay "
              "co the khong co thuoc tinh lpips tren model (khac ban da verify dung trong "
              "eval_submission_path.py). DUNG lai, khong tao code chet goi thuoc tinh khong "
              "ton tai.", file=sys.stderr)
        sys.exit(1)

    func_marker = "def get_loss_dict"
    func_start = code.find(func_marker)
    if func_start == -1:
        print(f"[LOI] Khong tim thay 'def get_loss_dict'. DUNG lai.", file=sys.stderr)
        sys.exit(1)

    import re
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

    print("\n[HOAN TAT] Da patch splatfacto.py: perceptual_loss (LPIPS-as-training-loss, co ramp) da them.")
    print("Mac dinh AN TOAN (PERCEPTUAL_LOSS_MAX_WEIGHT=0 = giong het ban goc).")
    print("Kich hoat: PERCEPTUAL_LOSS_MAX_WEIGHT=0.05 bash train.sh --scene <ten_scene> --iterations 20000")
    print("Tuy chinh diem bat dau ramp: PERCEPTUAL_LOSS_RAMP_START_FRAC=0.7 (mac dinh)")
    print("Khoi phuc ve ban goc: python patches/patch_splatfacto_perceptual_loss.py --restore")


if __name__ == "__main__":
    main()
