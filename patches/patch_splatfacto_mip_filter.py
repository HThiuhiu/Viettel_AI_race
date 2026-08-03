"""
Patch Splatfacto (file cai dat, vi ns-train chay o subprocess rieng nen khong
the monkeypatch trong bo nho tu notebook - phai sua truc tiep file .py) de them
MIP-FILTER kieu Mip-Splatting (Yu et al., CVPR'24): theo doi kich thuoc pixel
nho nhat tung quan sat duoc cho tung Gaussian qua cac goc nhin train, roi phat
(khong ep cung) neu scale nho hon nguong do - chong "kim/needle artifact" va
alias khi render tu khoang cach khac luc train.

BAN DA SUA so voi ban V8 goc (2 diem duoc danh gia la co van de ky thuat):

1. DA XOA `degenerate_scale_guard` - xac nhan TRUNG LAP hoan toan voi co che
   NATIVE da co san trong Splatfacto (`use_scale_regularization` +
   `max_gauss_ratio`, xem get_loss_dict trong source goc):
       scale_reg = (max(ratio, max_gauss_ratio) - max_gauss_ratio) * 0.1
   Ban V8 them 1 lop phat THU HAI (nguong 12.0, trong so 0.02, cong thuc
   log-ratio khac) chong len co che goc (nguong 10.0, trong so 0.1) dang duoc
   dieu khien qua --pipeline.model.max-gauss-ratio trong kaggle_train_t4x2.sh -
   2 co che cung active se tao duong cong phat khong ai chu dich thiet ke, dung
   ngay vung day cap/anten da duoc can chinh can than. Muon doi nguong/trong so,
   sua truc tiep --pipeline.model.max-gauss-ratio (da co san, khong can patch).

2. MIP-FILTER DOI TU HARD CLAMP SANG SOFT LOSS - ban V8 goc ghi de truc tiep
   `scales_param.data` moi 100 buoc (bo qua autograd). Adam optimizer giu
   momentum (exp_avg/exp_avg_sq) rieng cho tung phan tu tensor; sua `.data`
   "chui" ma khong dong bo lai state nay se khien vai buoc ngay sau moi lan
   clamp (lap lai dinh ky suot qua trinh train) update theo huong/do lon tinh
   tu GIA TRI CU (truoc clamp) - rui ro dao dong dinh ky. Splatfacto goc khi
   densify/prune (thao tac tuong tu: sua tensor ngoai luong gradient) co xu ly
   optimizer state di kem chinh vi ly do nay. Ban sua nay thay hard-clamp bang
   1 loss term kha vi (hinge-squared, giong cach degenerate_scale_guard cu da
   lam) - de optimizer tu dieu chinh qua gradient binh thuong, khong co xung
   dot voi Adam state.

QUAN TRONG - patch nay tac dong TOAN CUC len file cai dat, va se BI GHI DE moi
khi moi truong duoc cai lai (vd Kaggle notebook chay lai tu dau voi
NERFSTUDIO_INSTALL_SPEC). Phai chay lai script nay MOI SESSION, SAU buoc cai
Nerfstudio va TRUOC khi goi ns-train. Khong co canh bao tu dong neu ban quen -
tu kiem tra bang cach: dòng "===[SPLATFACTO MIP-FILTER PATCH]===" phai xuat
hien trong output cua ns-train (duoc in ra tu injection ben duoi) o nhung buoc
dau tien.

Cach dung (truoc khi chay ns-train / kaggle_train_t4x2.sh):
    python patches/patch_splatfacto_mip_filter.py
"""

import os
import re
import shutil
import sys

import nerfstudio

splatfacto_path = os.path.join(os.path.dirname(nerfstudio.__file__), "models", "splatfacto.py")
backup_path = splatfacto_path + ".mip_filter_patch_ORIGINAL_backup"

# ==============================================================================
# BUOC 0: BACKUP TU DONG (idempotent - luon patch tu ban GOC, khong patch chong patch)
# ==============================================================================
if not os.path.exists(backup_path):
    shutil.copyfile(splatfacto_path, backup_path)
    print(f"[OK] Da tao backup file GOC tai {backup_path}")
else:
    print(f"[OK] Backup file GOC da ton tai, se patch lai tu ban nay.")

shutil.copyfile(backup_path, splatfacto_path)
with open(splatfacto_path, "r") as f:
    code = f.read()

# ==============================================================================
# KHOI 1: HAM MIP-FILTER (chi con phan MOI thuc su - da xoa degenerate_scale_guard
# vi trung lap voi --pipeline.model.max-gauss-ratio/use-scale-regularization co san)
# ==============================================================================
injection_funcs = '''
import torch

def _spf_mip_update(model, camera):
    """Theo doi kich thuoc the gioi thuc cua 1 pixel (tai vi tri tung Gaussian)
    qua tung goc nhin train - luu gia tri NHO NHAT tung quan sat duoc. Chay
    duoi torch.no_grad() (chi thong ke, khong tinh gradient) nen an toan de goi
    moi buoc train."""
    with torch.no_grad():
        try:
            cam_pos = camera.camera_to_worlds[0, :3, 3].to(model.device)
            cam_forward = camera.camera_to_worlds[0, :3, 2].to(model.device)
            means = model.means
            vec = means - cam_pos[None, :]

            z_depth = torch.abs(torch.sum(vec * cam_forward[None, :], dim=-1)).clamp_min(1e-6)
            fx = float(camera.fx.flatten()[0])
            pixel_world_size = z_depth / max(fx, 1e-6)

            if not hasattr(model, "_mip_min_pixel_size") or model._mip_min_pixel_size.shape[0] != means.shape[0]:
                model._mip_min_pixel_size = pixel_world_size.clone()
            else:
                model._mip_min_pixel_size = torch.minimum(model._mip_min_pixel_size, pixel_world_size)
        except Exception:
            pass


def _spf_mip_loss(model, filter_coeff=0.15):
    """(SUA so voi ban V8 goc) Soft, kha vi - phat (khong ep cung/khong ghi de
    .data) khi scale nho hon nguong pixel-footprint nho nhat tung quan sat
    duoc, nhan filter_coeff. Dung hinge-squared (giong huong tiep can cua
    use-scale-regularization co san trong Splatfacto) de gradient tu dieu
    chinh binh thuong qua Adam, tranh xung dot optimizer-state ma hard clamp
    truoc day gap phai."""
    if not hasattr(model, "_mip_min_pixel_size"):
        return torch.tensor(0.0, device=model.means.device)
    min_allowed = (model._mip_min_pixel_size * filter_coeff).clamp_min(1e-6)
    actual_scales = torch.exp(torch.clamp(model.gauss_params["scales"], max=15.0))
    deficit = torch.relu(min_allowed.unsqueeze(-1) - actual_scales)
    return (deficit ** 2).mean()
'''

marker = "from nerfstudio.models.base_model import Model"
if marker not in code:
    print(f"[LOI] Khong tim thay anchor '{marker}'. Ban Nerfstudio hien tai co the "
          f"khac cau truc voi ban da test patch nay - DUNG lai, khong patch mu quang.", file=sys.stderr)
    sys.exit(1)
code = code.replace(marker, injection_funcs + "\n" + marker, 1)

# ==============================================================================
# KHOI 2: LUU CAMERA HIEN TAI DE MIP-FILTER DUNG (trong get_outputs - method NAY,
# khong phai get_outputs_for_camera - da verify thu tu trong file: get_outputs
# xuat hien TRUOC get_outputs_for_camera nen tim substring dung ham can patch)
# ==============================================================================
outputs_func_marker = "def get_outputs"
outputs_func_start = code.find(outputs_func_marker)
if outputs_func_start == -1:
    print(f"[LOI] Khong tim thay 'def get_outputs' trong splatfacto.py. DUNG lai.", file=sys.stderr)
    sys.exit(1)

def_line_end = code.find(":\n", outputs_func_start)
if def_line_end == -1:
    print(f"[LOI] Khong parse duoc dong dinh nghia ham get_outputs. DUNG lai.", file=sys.stderr)
    sys.exit(1)

insert_pos = def_line_end + 2
after_def = code[insert_pos:]
docstring_match = re.match(r'\s*("""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')\s*\n', after_def)
if docstring_match:
    insert_pos += docstring_match.end()

stash_line = "        self._spf_last_camera = camera  # [SPLATFACTO MIP-FILTER PATCH]\n"
if "_spf_last_camera = camera" not in code:
    code = code[:insert_pos] + stash_line + code[insert_pos:]

# ==============================================================================
# KHOI 3: GAN LOSS (chi con mip-filter - da xoa degenerate_scale_guard)
# ==============================================================================
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
          f"co the da doi (vd nhieu diem return) - DUNG lai, khong patch mu quang.", file=sys.stderr)
    sys.exit(1)

injection_loss = '''
        # ==== [SPLATFACTO MIP-FILTER PATCH] ====
        if self.training:
            _spf_cam = getattr(self, "_spf_last_camera", None)
            if _spf_cam is not None:
                _spf_mip_update(self, _spf_cam)
            loss_dict["mip_filter_loss"] = _spf_mip_loss(self) * 0.1
            if not hasattr(self, "_spf_mip_logged"):
                self._spf_mip_logged = True
                print("===[SPLATFACTO MIP-FILTER PATCH] dang active, "
                      "mip_filter_loss se xuat hien trong loss dict moi buoc===")
        # ========================================
'''

new_func_body = func_body.replace("return loss_dict", injection_loss + "\n        return loss_dict", 1)
code = code[:func_start] + new_func_body + code[func_end:]

with open(splatfacto_path, "w") as f:
    f.write(code)

print("\n[HOAN TAT] Splatfacto da duoc patch: mip-filter (soft loss) da them, "
      "KHONG con degenerate_scale_guard trung lap (dung --pipeline.model.max-gauss-ratio "
      "co san neu can chinh nguong ty le truc Gaussian).")
