#!/usr/bin/env bash
set -euo pipefail

# (Sua loi quan trong) Resolve THU MUC CHUA train.sh (khong phai thu muc dang
# dung khi goi lenh) - cac script Python con (dense_colmap_init.py,
# preprocess_colmap_to_nerfstudio.py, ...) o CUNG thu muc voi train.sh, nhung
# neu goi "bash /path/to/train.sh" tu 1 cwd KHAC (vd Kaggle notebook goi tu
# /kaggle/working trong khi repo nam o /kaggle/working/Viettel), duong dan
# TUONG DOI "python dense_colmap_init.py" se that bai (No such file or
# directory) vi no tim trong cwd hien tai, khong phai thu muc repo. Dung
# SCRIPT_DIR (tuyet doi) cho MOI loi goi script Python con ben duoi.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RAW_ROOT="${RAW_ROOT:-.}"
DATA_ROOT="${DATA_ROOT:-data/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/outputs}"
SCENE="${SCENE:-}"
ITERATIONS="${ITERATIONS:-30000}"
DOWNSCALE="${DOWNSCALE:-1}"                                   # (Sua loi quan trong) Anh train da san o dung do phan giai
                                                                # test_poses.csv yeu cau (vd 1320x989) - downscale=2 truoc day
                                                                # lam model chi hoc duoc chi tiet o nua do phan giai can nop
                                                                # bai, gay mo anh va bi phat nang boi LPIPS (40% diem so).
GPU_ID="${GPU_ID:-0}"
INSTALL_NERFSTUDIO="${INSTALL_NERFSTUDIO:-1}"
NERFSTUDIO_INSTALL_SPEC="${NERFSTUDIO_INSTALL_SPEC:-git+https://github.com/nerfstudio-project/nerfstudio.git@50e0e3c70c775e89333256213363badbf074f29d}"
                                                                # (Sua loi quan trong) Ban PyPI moi nhat (1.1.5, 11/2024) KHONG
                                                                # co field strategy/max_gs_num (MCMC) - cai tu GitHub (commit
                                                                # da verify co MCMC) de STRATEGY=mcmc ben duoi thuc su co hieu luc,
                                                                # thay vi am tham fallback ve ADC mac dinh nhu truoc day.
MODEL="${MODEL:-splatfacto}"                                 # splatfacto | splatfacto-big (chay cuoi neu con thoi gian)
DENSE_INIT="${DENSE_INIT:-0}"                                 # (Moi, opt-in) 1 = chay dense COLMAP MVS init truoc preprocess
DENSE_MAX_IMAGE_SIZE="${DENSE_MAX_IMAGE_SIZE:-1600}"          # Do phan giai lam viec cua patch_match_stereo neu DENSE_INIT=1
DENSE_TIMEOUT_MINUTES="${DENSE_TIMEOUT_MINUTES:-120}"        # (Moi) Toi da bao nhieu phut cho MOI buoc COLMAP cua dense
                                                              # init - het gio se DUNG AN TOAN, fallback ve sparse, tranh
                                                              # an het quota GPU Kaggle/Colab vao patch_match_stereo treo.
DENSE_GPU_INDEX="${DENSE_GPU_INDEX:-}"                        # (Moi) GPU rieng cho patch_match_stereo, doc lap voi --gpu
                                                              # dung de train - de trong = de COLMAP tu chon.
DENSE_MIN_NUM_PIXELS="${DENSE_MIN_NUM_PIXELS:-5}"            # (Moi) So anh toi thieu dong y ve 1 diem 3D moi giu (stereo_fusion)
DENSE_MAX_REPROJ_ERROR="${DENSE_MAX_REPROJ_ERROR:-2.0}"      # (Moi) Sai so tai chieu toi da (pixel) de giu 1 diem. TANG
                                                              # (vd 3.0-4.0) neu log dense_colmap_init.py bao dense KHONG
                                                              # day hon sparse dang ke - dau hieu nguong dang loc mat dung
                                                              # diem o vung RIA khung hinh it goc nhin chong lan (chinh
                                                              # vung dang bi mo trong anh checkpoint 10k da xem).
DENSE_MAX_DEPTH_ERROR="${DENSE_MAX_DEPTH_ERROR:-1.0}"        # (Moi) Sai so do sau toi da (ty le so voi do sau) de giu diem
export MAX_JOBS="${MAX_JOBS:-2}"                              # gioi han job bien dich song song luc JIT-compile
                                                                # CUDA kernel cua gsplat lan dau -> tranh OOM RAM tren Kaggle
STEPS_PER_LOG="${STEPS_PER_LOG:-10}"                          # so iteration giua 2 lan tinh stats (khong anh huong toc do/chat luong)
LOCAL_WRITER_ENABLE="${LOCAL_WRITER_ENABLE:-True}"           # (Sua doi) Truoc day la False -> Nerfstudio VAN tinh
                                                              # PSNR/SSIM/LPIPS dinh ky trong luc train (moi
                                                              # STEPS_PER_EVAL_ALL_IMAGES buoc) nhung KHONG in ra console
                                                              # vi writer bi tat. Bat lai de thay duoc so ngay trong log
                                                              # train, khong phai cho den khi ns-eval chay sau cung.

# --- Hyperparameter mat/chi tiet (giu nguyen tu ban goc) ---
DENSIFY_GRAD_THRESH="${DENSIFY_GRAD_THRESH:-0.00015}"        # (Nâng cấp) Hạ cực thấp để bắt chi tiết mỏng manh nhất như ăng-ten
CULL_ALPHA_THRESH="${CULL_ALPHA_THRESH:-0.003}"              # (Dieu chinh lai) Tra ve 0.003 (thay vi 0.005): voi MCMC, tran
                                                              # so Gaussian da duoc MAX_GS_NUM kiem soat cung, khong can siet
                                                              # alpha manh de tiet kiem VRAM nua - trong khi 0.005 co rui ro
                                                              # xoa nham Gaussian mo/mong bieu dien day cap, anten BTS, dieu
                                                              # ma LPIPS (40% diem so) phat rat nang vi mat chi tiet cau truc.
                                                              # Neu can A/B, so sanh 0.003 vs 0.005 qua metrics_summary.csv.
RESET_ALPHA_EVERY="${RESET_ALPHA_EVERY:-30}"                 # (Moi, ADC-only) Dinh ky (moi reset_alpha_every x refine_every
                                                              # buoc) reset opacity TAT CA Gaussian ve thap, buoc "kiem lai"
                                                              # qua gradient - Gaussian du thua/vo dung (sinh ra luc bung no
                                                              # dau train) khong du tin hieu de hoi phuc va bi CULL_ALPHA_THRESH
                                                              # don tu nhien. Van hoat dong SAU KHI stop-split-at da dung sinh
                                                              # Gaussian moi (cull khong can splitting). Ha xuong (vd 15-20)
                                                              # neu muon don rac tich cuc hon, sat voi STOP_SPLIT_RATIO thap.
WARMUP_LENGTH="${WARMUP_LENGTH:-500}"                        # (Moi) So buoc dau KHONG densify gi ca, de Gaussian on dinh
                                                              # vi tri truoc khi bat dau tach. Tang len (vd 1000-1500) de
                                                              # tranh bung no so luong ngay tu vai tram buoc dau - dung
                                                              # hien tuong "5 trieu Gaussian o step 2600" da gap.
REFINE_EVERY="${REFINE_EVERY:-100}"                          # (Moi) Tan suat chay vong densify/cull. Tang len (vd 150-200)
                                                              # lam cham nhip tang truong tong the ma khong can doi nguong
                                                              # gradient. Luu y: chu ky reset opacity thuc te =
                                                              # RESET_ALPHA_EVERY x REFINE_EVERY buoc, nen tang REFINE_EVERY
                                                              # cung keo dai khoang cach giua 2 lan reset alpha.
NUM_DOWNSCALES="${NUM_DOWNSCALES:-2}"                        # (Moi - QUAN TRONG cho do net TOAN ANH) Splatfacto train
                                                              # coarse-to-fine: do phan giai = 1/2^max(NUM_DOWNSCALES -
                                                              # step/RESOLUTION_SCHEDULE, 0). Mac dinh 2 => bat dau o 1/4
                                                              # do phan giai, len 1/2 tai step 3000, full tai step 6000.
                                                              # VAN DE: o 1/4 res, chi tiet tan so cao (soi tham, nen xa,
                                                              # vung ngoai vi) KHONG TON TAI duoi dang gradient -> khong
                                                              # Gaussian nao duoc cap cho chung, trong khi vat the chinh
                                                              # (to, tan so thap) van duoc cap du. Ket qua: "chu the net,
                                                              # xung quanh nhoe". Dat 0 de train full res ngay tu dau
                                                              # (net deu hon, ton VRAM/thoi gian hon), hoac 1 de dung hoa.
RESOLUTION_SCHEDULE="${RESOLUTION_SCHEDULE:-3000}"           # (Moi) So buoc giua 2 lan tang do phan giai. Ha xuong (vd
                                                              # 1000-1500) de dat full res SOM hon, dam bao phan lon qua
                                                              # trinh densify dien ra khi model da nhin thay chi tiet that.
                                                              # LUU Y phoi hop: full res dat duoc tai buoc
                                                              # NUM_DOWNSCALES x RESOLUTION_SCHEDULE - con so nay PHAI nho
                                                              # hon nhieu so voi STOP_SPLIT_AT, neu khong Gaussian se ngung
                                                              # duoc sinh truoc khi model kip thay chi tiet tan so cao.
DENSIFY_SIZE_THRESH="${DENSIFY_SIZE_THRESH:-0.0075}"           # (Moi, ADC-only) Nguong kich thuoc quyet dinh SPLIT hay CLONE:
                                                              # Gaussian LON hon nguong -> bi tach doi (split); NHO hon ->
                                                              # nhan ban (clone). Tuong ung voi --percent-dense cua ban
                                                              # INRIA goc. Ha xuong (vd 0.004) => nhieu Gaussian bi split
                                                              # hon => chi tiet min hon nhung so luong tang nhanh hon.
CULL_SCALE_THRESH="${CULL_SCALE_THRESH:-0.15}"               # Giữ nguyên để bảo vệ dây cáp
MAX_GAUSS_RATIO="${MAX_GAUSS_RATIO:-10.0}"                   # (Moi) Gioi han ty le truc dai/ngan cua tung Gaussian (mac
                                                              # dinh Splatfacto la 10.0, truoc day chua tung dat tuong
                                                              # minh). Day la tham so dieu chinh "needle Gaussian" - lien
                                                              # quan truc tiep toi day cap/anten mong: qua thap se cat cut
                                                              # cac Gaussian dai that su can co de bieu dien day cap (mat
                                                              # chi tiet, hai LPIPS); qua cao de sinh artifact "kim" gay
                                                              # aliasing khi nhin tu goc khac (cung hai LPIPS/SSIM). Giu
                                                              # mac dinh 10.0 tru khi co ket qua A/B ro rang.
STOP_SPLIT_RATIO="${STOP_SPLIT_RATIO:-0.7}"                 # (Nâng cấp) Ép mô hình sinh điểm đến tận 95% thời gian train (thay vì 80%), tận dụng tối đa 30k iters
STOP_SPLIT_AT="${STOP_SPLIT_AT:-}"                         # Neu de trong, se tinh dong ben duoi = STOP_SPLIT_RATIO x ITERATIONS

# --- Sky masking & checkpoint/eval dinh ky (moi) ---
MASK_DIR_NAME="${MASK_DIR_NAME:-masks}"                       
STEPS_PER_SAVE="${STEPS_PER_SAVE:-10000}"                    # (Nâng cấp) Tăng khoảng cách save để giảm thắt cổ chai I/O trên Kaggle
STEPS_PER_EVAL_ALL_IMAGES="${STEPS_PER_EVAL_ALL_IMAGES:-10000}" # Chay eval (PSNR/SSIM) dinh ky moi N buoc trong luc train
TRAIN_SPLIT_FRACTION="${TRAIN_SPLIT_FRACTION:-0.9}"          # (Moi, truoc day AN NGAM) Ty le anh train thuc su dung de
                                                              # toi uu Gaussian; phan con lai (mac dinh 10%) Nerfstudio
                                                              # tu giu lam eval noi bo, phan bo deu theo thu tu anh.
                                                              # RUI RO can biet: 10% cua 240 anh ~ 24 anh/scene - mau
                                                              # nho, LPIPS/SSIM/PSNR do duoc co nhieu lay mau dang ke.
                                                              # Neu dung compute_score.py de A/B nhieu cau hinh LAP DI
                                                              # LAP LAI tren dung 24 anh co dinh nay, co rui ro "chon
                                                              # cau hinh dep tren dung tap eval do" ma khong chac tong
                                                              # quat hoa len 60 anh test that (BTC giu kin, khac han).
                                                              # Goi y: HA xuong (vd 0.8-0.85) trong GIAI DOAN DO THAM SO
                                                              # de co eval-split lon hon, dang tin hon khi so sanh A/B;
                                                              # roi TANG LAI (vd 0.9-0.95) cho lan train CUOI CUNG dung
                                                              # de nop bai, danh nhieu anh hon cho train thuc su.

# --- Hyperparameter moi cho hackathon tuning ---
CAMERA_OPTIMIZER_MODE="${CAMERA_OPTIMIZER_MODE:-SO3xR3}"   
RASTERIZE_MODE="${RASTERIZE_MODE:-antialiased}"             
USE_BILATERAL_GRID="${USE_BILATERAL_GRID:-True}"            
STRATEGY="${STRATEGY:-mcmc}"                                 # (Nâng cấp) Bắt buộc ưu tiên MCMC nếu Nerfstudio hỗ trợ, kiểm soát VRAM cực tốt
if [[ "$STRATEGY" != "mcmc" && "$STRATEGY" != "default" ]]; then
  echo "[CANH BAO] STRATEGY='$STRATEGY' khong phai 'mcmc' hay 'default' (phan biet hoa/thuong," \
       "vd 'MCMC' hay co khoang trang thua se KHONG khop) -> se AM THAM roi vao nhanh ADC" \
       "(else) o train_one_scene() ma khong bao loi. Kiem tra lai gia tri bien STRATEGY."
fi
MAX_GS_NUM="${MAX_GS_NUM:-2500000}"                          # (Dieu chinh theo nghien cuu MCMC) 2 trieu la muc du
                                                              # dung cho chat luong tot voi phan bo Gaussian dung dan
                                                              # (MCMC relocation, khong phai chi tang so luong). Neu
                                                              # check_gs_count.py bao so Gaussian THUC TE vuot con so
                                                              # nay -> MCMC KHONG thuc su active, dang chay ADC khong
                                                              # tran (xem CANH BAO tu check_gs_count.py sau moi scene).
MCMC_NOISE_LR="${MCMC_NOISE_LR:-5e5}"                        # (Moi) He so nhieu injection cua MCMC (SGLD-style) - giup
                                                              # thoat local minima, gia tri mac dinh tu paper 3DGS-MCMC.
                                                              # Nhan voi learning-rate dang giam dan nen tu dong "nguoi"
                                                              # (anneal) ve cuoi qua trinh train, khong can tu chinh lich.
MCMC_OPACITY_REG="${MCMC_OPACITY_REG:-0.008}"                 # (Moi) Phat opacity de Gaussian "vo dung" (dong gop gan 0
                                                              # vao anh) thuc su giam ve 0 -> duoc tai dinh vi (relocate)
                                                              # sang vung con thieu, thay vi ton tai lo lung lam mat ngan
                                                              # sach ngan sach 2M Gaussian vao cho khong dong gop.
MCMC_SCALE_REG="${MCMC_SCALE_REG:-0.008}"                     # (Moi) Phat Gaussian phinh to/mo de "an gian" giam loss
                                                              # re thay vi dung nhieu Gaussian nho sac net - day chinh la
                                                              # co che truc tiep tranh trieu chung "nhieu Gaussian nhung
                                                              # van mo" ban gap phai. Neu con mo sau khi xac nhan MCMC
                                                              # da active (qua check_gs_count.py), thu tang gia tri nay
                                                              # (vd 0.02-0.05) truoc khi nghi ngo huong khac.
SH_DEGREE="${SH_DEGREE:-4}"                                   # 4 (thay vi 3) de mo phong phan xa kim loai cot/anten BTS
SSIM_LAMBDA="${SSIM_LAMBDA:-0.3}"                             # (Nang cap) Nang tu 0.2 mac dinh len 0.3: cong thuc loss chinh
                                                               # cua Splatfacto la (1-ssim_lambda)*L1 + ssim_lambda*(1-SSIM);
                                                               # SSIM chiem 30% diem so nen can trong so cao hon mac dinh.
                                                               # (Da xoa USE_LPIPS/--pipeline.model.use-lpips: flag nay KHONG
                                                               # ton tai o bat ky ban Nerfstudio nao - da verify truc tiep tu
                                                               # source, se lam ns-train crash ngay neu truyen vao.)
APPLY_MIP_FILTER_PATCH="${APPLY_MIP_FILTER_PATCH:-0}"        # (Moi, opt-in) 1 = tu dong ap patch_splatfacto_mip_filter.py
                                                              # (soft mip-filter loss, chong needle/alias artifact) truoc train.
ROBUST_DROP_PERCENTILE="${ROBUST_DROP_PERCENTILE:-0.0}"      # (Moi, doc boi patch_robust_transient_loss.py - LUON duoc ap
                                                              # dung nhung mac dinh no-op) Ty le % pixel residual lon nhat
                                                              # bi bo qua khoi L1 loss moi anh moi buoc (RobustNeRF-style).
                                                              # 0.0 = tat, giong het ban goc. Thu 0.03-0.08 neu nghi ngo
                                                              # noi dung transient (phan chieu doi theo goc, bong di dong).
PERCEPTUAL_LOSS_MAX_WEIGHT="${PERCEPTUAL_LOSS_MAX_WEIGHT:-0.0}" # (Moi, doc boi patch_splatfacto_perceptual_loss.py -
                                                              # LUON duoc ap dung nhung mac dinh no-op) Trong so TOI DA
                                                              # cua LPIPS lam tin hieu gradient luc train (LPIPS chiem
                                                              # 40% Score nhung goc chi dung L1+SSIM). 0.0 = tat, giong
                                                              # het ban goc. Thu 0.02-0.1. Trong so THUC TE duoc ramp
                                                              # dan len qua PERCEPTUAL_LOSS_RAMP_START_FRAC (curriculum:
                                                              # L1+SSIM truoc, LPIPS chi them dan o cuoi train - da xac
                                                              # nhan qua tra cuu la cach lam pho bien trong 3DGS gan day).
PERCEPTUAL_LOSS_RAMP_START_FRAC="${PERCEPTUAL_LOSS_RAMP_START_FRAC:-0.7}" # (Moi) Bat dau ramp LPIPS tu X% tien trinh
                                                              # train (0.7 = 30% cuoi). Dat gan 1.0 hon (vd 0.85-0.9)
                                                              # de giong sat hon "10-20% cuoi" neu muon.
PERCEPTUAL_LOSS_EVERY_N="${PERCEPTUAL_LOSS_EVERY_N:-5}"      # (Moi) Chi tinh perceptual loss moi N buoc (LPIPS ton
                                                              # compute hon han L1/SSIM) - tang N (vd 10-20, theo thuc te
                                                              # nhieu pipeline 3DGS da tra cuu) de train nhanh hon,
                                                              # giam N de tin hieu gradient perceptual day dac hon.
FREQ_LOSS_WEIGHT="${FREQ_LOSS_WEIGHT:-0.0}"                  # (Moi, doc boi patch_splatfacto_frequency_loss.py - LUON
                                                              # duoc ap dung nhung mac dinh no-op) FreGS-style (CVPR'24)
                                                              # progressive frequency regularization - so amplitude+phase
                                                              # FFT giua render/GT, mo rong dai high-pass dan theo tien
                                                              # trinh train (coarse-to-fine). 0.0 = tat, giong het ban
                                                              # goc. Thu 0.01 truoc (FFT loss co the co bien do lon hon
                                                              # L1/SSIM), A/B qua eval_submission_path.py truoc khi tang.
FREQ_LOSS_T0="${FREQ_LOSS_T0:-500}"                          # (Moi) Truoc buoc nay: CHI phat sai lech tan so THAP.
FREQ_LOSS_T_END="${FREQ_LOSS_T_END:-}"                       # (Moi) Buoc dai high-pass dat rong toi da. De trong = tu
                                                              # dong dung STOP_SPLIT_AT (khop tinh than paper: ket thuc
                                                              # cung luc densification dung sinh Gaussian moi).
FREQ_LOSS_LOW_RADIUS_FRAC="${FREQ_LOSS_LOW_RADIUS_FRAC:-0.1}" # (Moi) Ban kinh low-pass co dinh (ty le ban kinh toi da).
FREQ_LOSS_HIGH_RADIUS_FRAC="${FREQ_LOSS_HIGH_RADIUS_FRAC:-1.0}" # (Moi) Ban kinh high-pass toi da (dai mo rong toi).
FREQ_LOSS_EVERY_N="${FREQ_LOSS_EVERY_N:-5}"                  # (Moi) Chi tinh frequency loss moi N buoc (FFT ton compute
                                                              # hon L1/SSIM, tuy re hon LPIPS/VGG).
SKY_MASK_INIT="${SKY_MASK_INIT:-0}"                          # (Moi, opt-in) 1 = chay generate_sky_masks.py (heuristic,
                                                              # khong can mang/model tai ve) truoc buoc preprocess, ghi
                                                              # mask vao <scene>/train/masks/ de train bo qua vung troi.
                                                              # MAC DINH TAT: mask cung co the tao artifact tai bien
                                                              # troi/day cap - chi bat de A/B co kiem chung qua CSV.

# --- Danh gia & song song ---
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"                    # 1 = tu chay ns-eval sau train

usage() {
  cat <<USAGE
Usage:
  bash train.sh [options]

Options co ban:
  --raw-root PATH        Raw project/data root. Default: .
  --data-root PATH       Processed Nerfstudio data root. Default: data/processed
  --output-root PATH     Output root. Default: /kaggle/working/outputs
  --scene NAME           Train only one scene. Empty means all scenes.
  --iterations N         Max iterations. Default: 30000
  --downscale N          Nerfstudio downscale factor. Default: 1 (khop dung do
                         phan giai test_poses.csv yeu cau nop bai)
  --gpu ID               CUDA_VISIBLE_DEVICES value. Default: 0
  --no-install           Skip pip install nerfstudio
  --no-eval              Bo qua ns-eval tu dong sau khi train
  -h, --help             Show this help

Bien moi truong quan trong (xem CHANGELOG o dau file):
  CAMERA_OPTIMIZER_MODE (off|SO3xR3|SE3), RASTERIZE_MODE (classic|antialiased),
  USE_BILATERAL_GRID (True|False), STRATEGY (default|mcmc), MAX_GS_NUM,
  MODEL (splatfacto|splatfacto-big), MAX_JOBS (gioi han job compile CUDA JIT)
  SSIM_LAMBDA (mac dinh 0.3, trong so SSIM trong loss - SSIM chiem 30% diem so),
  MAX_GAUSS_RATIO (mac dinh 10.0, gioi han ty le truc dai/ngan Gaussian -
    HA xuong 4-6 cho scene indoor (chair/bonsai) de chan needle artifact tren
    kinh/be mat phang; GIU 10.0 cho scene BTS vi day cap/anten CAN Gaussian dai),
  WARMUP_LENGTH (mac dinh 500) so buoc dau khong densify - tang len 1000-1500
    de tranh bung no so luong Gaussian ngay dau train,
  REFINE_EVERY (mac dinh 100) tan suat chay densify/cull - tang len 150-200
    de lam cham nhip tang truong Gaussian,
  NUM_DOWNSCALES (mac dinh 2) / RESOLUTION_SCHEDULE (mac dinh 3000) - lich
    coarse-to-fine. QUAN TRONG cho do net TOAN ANH: mac dinh khien densify
    dien ra chu yeu o do phan giai thap -> vung nen/ngoai vi khong duoc cap
    Gaussian. Dat NUM_DOWNSCALES=0 (full res ngay tu dau) hoac ha
    RESOLUTION_SCHEDULE de dat full res truoc khi STOP_SPLIT_AT xay ra,
  CULL_SCALE_THRESH (mac dinh 0.15, giu Gaussian day cap/anten mong),
  STOP_SPLIT_RATIO (mac dinh 0.95, dung neu STOP_SPLIT_AT khong duoc chi dinh),
  MASK_DIR_NAME (mac dinh "masks", ten thu muc sky mask trong tung scene),
  STEPS_PER_SAVE / STEPS_PER_EVAL_ALL_IMAGES (mac dinh 10000, checkpoint/eval dinh ky),
  NERFSTUDIO_INSTALL_SPEC (mac dinh: commit GitHub co MCMC, xem CHANGELOG)
  DENSE_INIT=1 (moi, opt-in) chay dense COLMAP MVS init truoc preprocess -
    thay sparse points bang point cloud day dac hon (xem dense_colmap_init.py),
    ton them thoi gian GPU dang ke. DENSE_MAX_IMAGE_SIZE (mac dinh 1600) chinh
    do phan giai lam viec cua buoc nay. Cac bien di kem:
      DENSE_TIMEOUT_MINUTES (mac dinh 120) - toi da cho MOI buoc COLMAP, het
        gio se DUNG AN TOAN va fallback ve sparse (tranh an het quota GPU),
      DENSE_GPU_INDEX (mac dinh rong) - GPU rieng cho patch_match_stereo,
      DENSE_MIN_NUM_PIXELS (mac dinh 5) / DENSE_MAX_REPROJ_ERROR (mac dinh 2.0)
        / DENSE_MAX_DEPTH_ERROR (mac dinh 1.0) - nguong loc diem cua
        stereo_fusion. TANG DENSE_MAX_REPROJ_ERROR (vd 3.0-4.0) neu log
        dense_colmap_init.py bao dense KHONG day hon sparse dang ke - dau
        hieu nguong dang loc mat dung diem o vung RIA khung hinh it goc nhin
        chong lan (vung quan sat thay bi mo trong anh checkpoint 10k thuc te).
  APPLY_MIP_FILTER_PATCH=1 (moi, opt-in) tu dong ap patch_splatfacto_mip_filter.py
    (soft mip-filter loss chong needle/alias artifact) truoc khi ns-train chay.
  ROBUST_DROP_PERCENTILE (moi, mac dinh 0.0) - patch_robust_transient_loss.py
    LUON duoc ap dung tu dong (an toan: 0.0 = y het ban goc), dat 0.03-0.08 de
    bo qua % pixel residual lon nhat moi buoc (nghi ngo noi dung transient:
    phan chieu doi theo goc, bong di dong - khac voi cau truc 3D that).
  SKY_MASK_INIT=1 (moi, opt-in) chay generate_sky_masks.py (heuristic, KHONG
    can mang/tai model) truoc preprocess, ghi mask vao <scene>/train/masks/.
    MAC DINH TAT: mask cung co the tao artifact tai bien troi/day cap mong -
    chi bat de A/B co kiem chung qua metrics_summary.csv, khong nen bat mac dinh.
  PERCEPTUAL_LOSS_MAX_WEIGHT (moi, mac dinh 0.0) - patch_splatfacto_perceptual_loss.py
    LUON duoc ap dung tu dong (an toan: 0.0 = y het ban goc), dat 0.02-0.1 de
    them LPIPS lam tin hieu gradient luc train (LPIPS chiem 40% Score nhung
    loss goc chi dung L1+SSIM). Trong so ramp dan tu 0 len gia tri nay theo
    PERCEPTUAL_LOSS_RAMP_START_FRAC (mac dinh 0.7 = bat dau ramp tu 70% tien
    trinh, curriculum L1+SSIM truoc/LPIPS sau - xem patch de biet co so tra
    cuu). PERCEPTUAL_LOSS_EVERY_N (mac dinh 5) - chi tinh perceptual loss moi
    N buoc de tiet kiem compute (LPIPS ton hon han L1/SSIM), tang len 10-20
    neu train qua cham (khoang pho bien trong thuc te da tra cuu).
  FREQ_LOSS_WEIGHT (moi, mac dinh 0.0) - patch_splatfacto_frequency_loss.py
    LUON duoc ap dung tu dong (an toan: 0.0 = y het ban goc). FreGS-style
    (CVPR'24) progressive frequency regularization: so amplitude+phase FFT
    giua render/GT, dai high-pass mo rong dan tu FREQ_LOSS_T0 (mac dinh 500)
    den FREQ_LOSS_T_END (mac dinh = STOP_SPLIT_AT). Thu 0.01 truoc, A/B qua
    eval_submission_path.py truoc khi tang - xem docstring patch de biet
    cong thuc goc + cac don gian hoa co chu dich so voi paper.
  TRAIN_SPLIT_FRACTION (moi, mac dinh 0.9) ty le anh train thuc su dung de
    toi uu Gaussian; phan con lai la eval noi bo. HA xuong (vd 0.8) khi dang
    do A/B nhieu cau hinh de eval-split lon hon, dang tin hon; TANG LAI (vd
    0.9-0.95) cho lan train CUOI CUNG dung de nop bai.

Vi du:
  bash train.sh --scene HCM0421 --iterations 1000
  bash train.sh --scene HCM0421 --iterations 30000
  STRATEGY=mcmc MAX_GS_NUM=1500000 bash train.sh --scene HCM0421 --iterations 30000
  MODEL=splatfacto-big bash train.sh --scene HCM0421 --iterations 30000
  bash train.sh --iterations 30000
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-root) RAW_ROOT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --downscale) DOWNSCALE="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --no-install) INSTALL_NERFSTUDIO="0"; shift ;;
    --no-eval) EVAL_AFTER_TRAIN="0"; shift ;;
    --parallel) echo "[LUU Y] --parallel da bi vo hieu hoa trong ban nay - luon chay 1 tien trinh, khong chia GPU."; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Cong thuc dong: neu nguoi dung KHONG tu truyen STOP_SPLIT_AT ro rang, tinh
# = STOP_SPLIT_RATIO x ITERATIONS (thay vi hard-code 20000). Dat SAU khi da
# parse xong --iterations o tren, de dam bao dung gia tri ITERATIONS cuoi cung.
if [[ -z "$STOP_SPLIT_AT" ]]; then
  STOP_SPLIT_AT=$(awk -v it="$ITERATIONS" -v r="$STOP_SPLIT_RATIO" 'BEGIN{printf "%d", it*r}')
  echo "=== stop-split-at duoc tinh dong = ${STOP_SPLIT_AT} (= ${STOP_SPLIT_RATIO} x ${ITERATIONS} iterations) ==="
else
  echo "=== stop-split-at duoc nguoi dung chi dinh truc tiep = ${STOP_SPLIT_AT} ==="
fi

# (Moi) FREQ_LOSS_T_END mac dinh = STOP_SPLIT_AT (khop tinh than paper FreGS:
# dai high-pass mo rong toi da dung luc densification ket thuc) - chi tinh
# khi nguoi dung KHONG tu truyen gia tri ro rang.
if [[ -z "$FREQ_LOSS_T_END" ]]; then
  FREQ_LOSS_T_END="$STOP_SPLIT_AT"
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export TORCH_HOME="/kaggle/working/.cache/torch"
export WANDB_MODE=disabled
export COMET_DISABLE_AUTO_LOGGING=1

mkdir -p "$OUTPUT_ROOT" "$TORCH_HOME"

echo "=== Kaggle GPU check ==="
nvidia-smi || true
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

DETECTED_GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "=== So GPU phat hien duoc: ${DETECTED_GPU_COUNT} (ban nay luon chay 1 tien trinh tren --gpu ${GPU_ID}) ==="

if [[ "$INSTALL_NERFSTUDIO" == "1" ]]; then
  if command -v ns-train >/dev/null 2>&1 && ns-train splatfacto --help 2>&1 | grep -q -- "--pipeline.model.strategy"; then
    echo "=== Nerfstudio da cai san va ho tro MCMC (--pipeline.model.strategy) - bo qua cai lai ==="
  else
    # (Sua loi quan trong) KHONG cai qua PyPI ("pip install nerfstudio") - ban PyPI
    # moi nhat (1.1.5) KHONG co field strategy/max_gs_num nen STRATEGY=mcmc se am
    # tham fallback ve ADC. Cai tu GitHub (commit da verify co MCMC) de dung dung
    # chien luoc da chon trong CHANGES_FINAL.md.
    echo "=== Installing Nerfstudio tu ${NERFSTUDIO_INSTALL_SPEC} (de co MCMC strategy) ==="
    python -m pip install -U pip
    python -m pip install "$NERFSTUDIO_INSTALL_SPEC"
  fi
fi

if [[ "$APPLY_MIP_FILTER_PATCH" == "1" ]]; then
  echo "=== APPLY_MIP_FILTER_PATCH=1: dang patch splatfacto.py (mip-filter soft loss) ==="
  python "$SCRIPT_DIR/patches/patch_splatfacto_mip_filter.py" || echo "[CANH BAO] patch_splatfacto_mip_filter.py that bai - tiep tuc voi splatfacto.py chua patch."
fi

# (Moi) Luon chay patch nay - AN TOAN TUYET DOI vi hanh vi chi doi khi
# ROBUST_DROP_PERCENTILE > 0 (mac dinh 0.0 = y het ban goc). Chay vo dieu
# kien de loai bo rui ro "quen patch truoc khi train" da tung gap phai.
echo "=== Ap dung patch_robust_transient_loss.py (no-op neu ROBUST_DROP_PERCENTILE=0) ==="
python "$SCRIPT_DIR/patches/patch_robust_transient_loss.py" || echo "[CANH BAO] patch_robust_transient_loss.py that bai - tiep tuc voi splatfacto.py chua patch (Ll1 mac dinh)."

echo "=== Ap dung patch_splatfacto_perceptual_loss.py (no-op neu PERCEPTUAL_LOSS_MAX_WEIGHT=0) ==="
python "$SCRIPT_DIR/patches/patch_splatfacto_perceptual_loss.py" || echo "[CANH BAO] patch_splatfacto_perceptual_loss.py that bai - tiep tuc khong co perceptual loss."

echo "=== Ap dung patch_splatfacto_frequency_loss.py (no-op neu FREQ_LOSS_WEIGHT=0) ==="
python "$SCRIPT_DIR/patches/patch_splatfacto_frequency_loss.py" || echo "[CANH BAO] patch_splatfacto_frequency_loss.py that bai - tiep tuc khong co frequency loss."

if [[ "$DENSE_INIT" == "1" ]]; then
  # (Moi, opt-in) Dense MVS init - thay sparse COLMAP points bang point cloud
  # day dac hon tu patch_match_stereo, dac biet co ich cho be mat kim loai it
  # texture/phan chieu (cot/anten BTS). Ton them thoi gian GPU dang ke - chi
  # bat khi con du quota. KHONG doi he toa do nen an toan voi test_poses.csv.
  echo "=== DENSE_INIT=1: chay dense COLMAP MVS init (co the mat nhieu thoi gian) ==="
  if ! command -v colmap >/dev/null 2>&1; then
    echo "=== Khong thay binary 'colmap', thu cai qua apt (can bat Internet cho Kaggle notebook) ==="
    apt-get update && apt-get install -y colmap || echo "[CANH BAO] Cai colmap qua apt that bai - se bo qua dense init, dung sparse fallback."
  fi
  dense_gpu_args=()
  if [[ -n "$DENSE_GPU_INDEX" ]]; then
    dense_gpu_args=(--gpu_index "$DENSE_GPU_INDEX")
  fi
  python "$SCRIPT_DIR/pipeline/dense_colmap_init.py" --raw_root "$RAW_ROOT" \
    --max_image_size "${DENSE_MAX_IMAGE_SIZE:-1600}" \
    --timeout_minutes "$DENSE_TIMEOUT_MINUTES" \
    --min_num_pixels "$DENSE_MIN_NUM_PIXELS" \
    --max_reproj_error "$DENSE_MAX_REPROJ_ERROR" \
    --max_depth_error "$DENSE_MAX_DEPTH_ERROR" \
    "${dense_gpu_args[@]}" \
    --skip_existing || \
    echo "[CANH BAO] dense_colmap_init.py that bai/loi mot phan - preprocess ben duoi se tu fallback ve sparse points cho cac scene chua co dense/fused.ply."
fi

if [[ "$SKY_MASK_INIT" == "1" ]]; then
  echo "=== SKY_MASK_INIT=1: chay generate_sky_masks.py (heuristic, opt-in) ==="
  python "$SCRIPT_DIR/pipeline/generate_sky_masks.py" --raw_root "$RAW_ROOT" || \
    echo "[CANH BAO] generate_sky_masks.py that bai/loi mot phan - train se tiep tuc khong co sky mask cho cac scene loi."
fi

if [[ "$SKY_MASK_INIT" == "1" ]]; then
  # (Moi) SKY_MASK_INIT=1 BAT BUOC chay lai preprocess du transforms.json cu
  # da ton tai tu lan chay truoc - neu khong, ban transforms.json CACHE
  # (chua co "mask_path" vi duoc tao truoc khi mask ton tai) se duoc dung
  # lai, sky mask se AM THAM khong co tac dung gi ca du generate_sky_masks.py
  # da chay thanh cong (dung loi da gap voi cach tim sai duong dan truoc do).
  echo "=== SKY_MASK_INIT=1: bat buoc chay lai preprocess de nhung mask_path vao transforms.json ==="
  python "$SCRIPT_DIR/pipeline/preprocess_colmap_to_nerfstudio.py" --raw_root "$RAW_ROOT" --out_root "$DATA_ROOT"
elif [[ ! -d "$DATA_ROOT" ]] || ! find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -name transforms.json | grep -q .; then
  echo "=== Preprocessing data ==="
  python "$SCRIPT_DIR/pipeline/preprocess_colmap_to_nerfstudio.py" --raw_root "$RAW_ROOT" --out_root "$DATA_ROOT"
fi

METRICS_CSV="$OUTPUT_ROOT/metrics_summary.csv"
if [[ ! -f "$METRICS_CSV" ]]; then
  mkdir -p "$OUTPUT_ROOT"
  echo "scene,gpu,strategy,camera_optimizer,rasterize_mode,bilateral_grid,iterations,downscale,ssim_lambda,cull_alpha_thresh,dense_init,robust_drop_percentile,mip_filter_patch,perceptual_loss_weight,freq_loss_weight,psnr,ssim,lpips,config_path" > "$METRICS_CSV"
fi


SUPPORTS_STRATEGY_FLAG="0"
if ns-train splatfacto --help 2>&1 | grep -q -- "--pipeline.model.strategy"; then
  SUPPORTS_STRATEGY_FLAG="1"
  echo "=== Nerfstudio ho tro --pipeline.model.strategy (MCMC) ==="
else
  echo "=== [LUU Y] Ban Nerfstudio dang cai KHONG ho tro --pipeline.model.strategy ==="
  echo "    (tinh nang MCMC chi co o ban moi hon). Se BO QUA flag nay hoan toan,"
  echo "    kem ca khi STRATEGY=mcmc duoc yeu cau - dung ADC mac dinh thay the."
  if [[ "$STRATEGY" == "mcmc" ]]; then
    echo "    [CANH BAO] STRATEGY=mcmc duoc yeu cau nhung KHONG duoc ho tro tren ban nay -> fallback ve default."
  fi
fi
export SUPPORTS_STRATEGY_FLAG

train_one_scene() {
  local scene_dir="$1"
  local gpu_id="$2"

  if [[ ! -f "$scene_dir/transforms.json" ]]; then
    echo "[SKIP] Missing transforms.json: $scene_dir"
    return 0
  fi

  local scene_id
  scene_id="$(basename "$scene_dir")"
  echo "=== [$scene_id] Training on CUDA_VISIBLE_DEVICES=$gpu_id ==="
  echo "iterations=$ITERATIONS downscale=$DOWNSCALE strategy=$STRATEGY camera_optimizer=$CAMERA_OPTIMIZER_MODE output=$OUTPUT_ROOT/$scene_id"

  # (Sua loi quan trong - LAN 3) Da xac nhan truc tiep tu source
  # nerfstudio_dataparser.py: KHONG CO co CLI nao cho mask theo file/thu muc
  # (chi co --mask-color cho 1 mau don, khac hoan toan muc dich). Co che DUNG
  # la nhung "mask_path" duoc nhung SAN trong tung frame cua transforms.json
  # (dataparser tu doc, yeu cau TAT CA frame phai co hoac KHONG frame nao co -
  # xem preprocess_colmap_to_nerfstudio.py, ham colmap_to_nerfstudio()) - vi
  # vay o day KHONG con truyen co CLI nao ca, chi con log de biet co
  # SKY_MASK_INIT dang bat hay khong (viec nhung mask_path da xay ra luc
  # preprocess, truoc khi vao ham nay).
  if [[ "$SKY_MASK_INIT" == "1" ]]; then
    echo "[$scene_id] SKY_MASK_INIT=1 - mask_path (neu du dieu kien) da duoc nhung vao transforms.json luc preprocess."
  fi

  # (Thiet ke lai) Tach rieng tham so cua ADC vs MCMC thay vi tron lan nhu
  # truoc - 2 co che nay dieu khien qua trinh sinh/loai bo Gaussian hoan toan
  # khac nhau (ADC: split/clone theo gradient + cull theo alpha/scale; MCMC:
  # relocate Gaussian "chet" (opacity thap) sang vung con thieu + noise
  # injection kieu SGLD, khong dung khai niem densify/cull nhu ADC). Truyen ca
  # 2 bo tham so cung luc gay nham lan ve viec knob nao thuc su dang co hieu
  # luc - gio chi truyen dung bo tham so cua strategy DANG THUC SU CHAY.
  local strategy_args=()
  local adc_args=()
  local mcmc_args=()
  if [[ "$SUPPORTS_STRATEGY_FLAG" == "1" && "$STRATEGY" == "mcmc" ]]; then
    strategy_args=(--pipeline.model.strategy mcmc)
    mcmc_args=(
      --pipeline.model.max-gs-num "$MAX_GS_NUM"
      --pipeline.model.noise-lr "$MCMC_NOISE_LR"
      --pipeline.model.mcmc-opacity-reg "$MCMC_OPACITY_REG"
      --pipeline.model.mcmc-scale-reg "$MCMC_SCALE_REG"
    )
  else
    if [[ "$SUPPORTS_STRATEGY_FLAG" == "1" ]]; then
      strategy_args=(--pipeline.model.strategy default)
    fi
    adc_args=(
      --pipeline.model.cull-alpha-thresh "$CULL_ALPHA_THRESH"
      --pipeline.model.cull-scale-thresh "$CULL_SCALE_THRESH"
      --pipeline.model.densify-grad-thresh "$DENSIFY_GRAD_THRESH"
      --pipeline.model.densify-size-thresh "$DENSIFY_SIZE_THRESH"
      --pipeline.model.stop-split-at "$STOP_SPLIT_AT"
      --pipeline.model.reset-alpha-every "$RESET_ALPHA_EVERY"
    )
  fi
  # Neu SUPPORTS_STRATEGY_FLAG=0 (Nerfstudio qua cu), ca strategy_args va
  # mcmc_args deu rong, adc_args van duoc dung - dung hanh vi ADC mac dinh
  # cua ban Nerfstudio do, khong truyen flag lang nghia gay loi cu phap.

  set +e
  CUDA_VISIBLE_DEVICES="$gpu_id" ns-train "$MODEL" \
    --data "$scene_dir" \
    --output-dir "$OUTPUT_ROOT/$scene_id" \
    --max-num-iterations "$ITERATIONS" \
    --pipeline.model.sh-degree "$SH_DEGREE" \
    --pipeline.model.ssim-lambda "$SSIM_LAMBDA" \
    --pipeline.model.warmup-length "$WARMUP_LENGTH" \
    --pipeline.model.refine-every "$REFINE_EVERY" \
    --pipeline.model.num-downscales "$NUM_DOWNSCALES" \
    --pipeline.model.resolution-schedule "$RESOLUTION_SCHEDULE" \
    --pipeline.model.use-scale-regularization True \
    --pipeline.model.max-gauss-ratio "$MAX_GAUSS_RATIO" \
    --pipeline.model.camera-optimizer.mode "$CAMERA_OPTIMIZER_MODE" \
    --pipeline.model.rasterize-mode "$RASTERIZE_MODE" \
    --pipeline.model.use-bilateral-grid "$USE_BILATERAL_GRID" \
    --pipeline.model.color-corrected-metrics True \
    "${strategy_args[@]}" \
    "${adc_args[@]}" \
    "${mcmc_args[@]}" \
    --logging.steps-per-log "$STEPS_PER_LOG" \
    --logging.local-writer.enable "$LOCAL_WRITER_ENABLE" \
    --save-only-latest-checkpoint False \
    --steps-per-save "$STEPS_PER_SAVE" \
    --steps-per-eval-batch 500 \
    --steps-per-eval-all-images "$STEPS_PER_EVAL_ALL_IMAGES" \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --eval-mode fraction \
    --train-split-fraction "$TRAIN_SPLIT_FRACTION" \
    --downscale-factor "$DOWNSCALE"
  train_status=$?
  set -e

  if [[ $train_status -ne 0 ]]; then
    echo "[LOI][$scene_id] ns-train that bai (exit $train_status). Bo qua scene nay, tiep tuc scene tiep theo."
    return 0
  fi

  echo "=== [$scene_id] Done training ==="

  # (Moi) Xac nhan MCMC co thuc su active va so Gaussian thuc te co nam trong
  # MAX_GS_NUM khong - phat hien ngay neu ADC am tham chay thay MCMC (khong co
  # tran cung -> de ra qua nhieu Gaussian phan bo kem hieu qua, van mo o vung kho).
  CUDA_VISIBLE_DEVICES="$gpu_id" python "$SCRIPT_DIR/pipeline/check_gs_count.py" --checkpoints_root "$OUTPUT_ROOT" --scene "$scene_id" || \
    echo "[CANH BAO][$scene_id] check_gs_count.py that bai, bo qua kiem tra nay."

  if [[ "$EVAL_AFTER_TRAIN" == "1" ]]; then
    local config_path
    config_path=$(find "$OUTPUT_ROOT/$scene_id" -name config.yml | sort | tail -n 1)
    if [[ -n "$config_path" ]]; then
      echo "=== [$scene_id] Chay ns-eval (PSNR/SSIM/LPIPS tren tap eval giu lai tu train) ==="
      local metrics_json="$OUTPUT_ROOT/$scene_id/eval_metrics.json"
      CUDA_VISIBLE_DEVICES="$gpu_id" ns-eval --load-config "$config_path" --output-path "$metrics_json" || {
        echo "[CANH BAO][$scene_id] ns-eval that bai, bo qua ghi metrics."
        return 0
      }
      python - "$scene_id" "$gpu_id" "$config_path" "$metrics_json" "$METRICS_CSV" <<'PY'
import json, sys, csv
scene_id, gpu_id, config_path, metrics_json, csv_path = sys.argv[1:6]
with open(metrics_json) as f:
    m = json.load(f)
results = m.get("results", m)
psnr = results.get("psnr", results.get("cc_psnr", ""))
ssim = results.get("ssim", results.get("cc_ssim", ""))
lpips = results.get("lpips", results.get("cc_lpips", ""))
import os
row = [
    scene_id, gpu_id, os.environ.get("STRATEGY", ""), os.environ.get("CAMERA_OPTIMIZER_MODE", ""),
    os.environ.get("RASTERIZE_MODE", ""), os.environ.get("USE_BILATERAL_GRID", ""),
    os.environ.get("ITERATIONS", ""), os.environ.get("DOWNSCALE", ""),
    os.environ.get("SSIM_LAMBDA", ""), os.environ.get("CULL_ALPHA_THRESH", ""),
    os.environ.get("DENSE_INIT", ""), os.environ.get("ROBUST_DROP_PERCENTILE", ""),
    os.environ.get("APPLY_MIP_FILTER_PATCH", ""), os.environ.get("PERCEPTUAL_LOSS_MAX_WEIGHT", ""),
    os.environ.get("FREQ_LOSS_WEIGHT", ""),
    psnr, ssim, lpips, config_path,
]
with open(csv_path, "a", newline="") as f:
    csv.writer(f).writerow(row)
print(f"[{scene_id}] psnr={psnr} ssim={ssim} lpips={lpips} -> ghi vao {csv_path}")
PY
    fi
  fi
}
export -f train_one_scene
export OUTPUT_ROOT ITERATIONS DOWNSCALE SH_DEGREE SSIM_LAMBDA MAX_GAUSS_RATIO CULL_ALPHA_THRESH CULL_SCALE_THRESH \
  DENSIFY_GRAD_THRESH DENSIFY_SIZE_THRESH \
  STOP_SPLIT_AT RESET_ALPHA_EVERY WARMUP_LENGTH REFINE_EVERY NUM_DOWNSCALES RESOLUTION_SCHEDULE \
  CAMERA_OPTIMIZER_MODE RASTERIZE_MODE USE_BILATERAL_GRID STRATEGY MAX_GS_NUM \
  MCMC_NOISE_LR MCMC_OPACITY_REG MCMC_SCALE_REG TRAIN_SPLIT_FRACTION \
  EVAL_AFTER_TRAIN METRICS_CSV MODEL SUPPORTS_STRATEGY_FLAG STEPS_PER_LOG LOCAL_WRITER_ENABLE \
  MASK_DIR_NAME STEPS_PER_SAVE STEPS_PER_EVAL_ALL_IMAGES \
  DENSE_INIT ROBUST_DROP_PERCENTILE APPLY_MIP_FILTER_PATCH SKY_MASK_INIT \
  PERCEPTUAL_LOSS_MAX_WEIGHT PERCEPTUAL_LOSS_RAMP_START_FRAC PERCEPTUAL_LOSS_EVERY_N \
  FREQ_LOSS_WEIGHT FREQ_LOSS_T0 FREQ_LOSS_T_END FREQ_LOSS_LOW_RADIUS_FRAC FREQ_LOSS_HIGH_RADIUS_FRAC FREQ_LOSS_EVERY_N

scene_dirs=()
if [[ -n "$SCENE" ]]; then
  scene_dirs+=("$DATA_ROOT/$SCENE")
else
  while IFS= read -r -d '' d; do
    scene_dirs+=("$d")
  done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

# (Don gian hoa) Luon chay tuan tu, 1 tien trinh tren 1 GPU duy nhat
# (--gpu, mac dinh 0) - da bo nhanh chia 2 GPU song song (--parallel bi vo
# hieu hoa o phan parse argument ben tren).
echo "=== Chay tuan tu ${#scene_dirs[@]} scene tren --gpu ${GPU_ID} ==="
for scene_dir in "${scene_dirs[@]}"; do
  train_one_scene "$scene_dir" "$GPU_ID"
done

echo "=== Tong hop config da train ==="
find "$OUTPUT_ROOT" -name config.yml -print | sort

if [[ -f "$METRICS_CSV" ]]; then
  echo "=== Bang tong hop PSNR/SSIM/LPIPS (metrics_summary.csv) ==="
  column -s, -t "$METRICS_CSV" 2>/dev/null || cat "$METRICS_CSV"
fi