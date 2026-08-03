"""
Hau xu ly 2D cho anh render TRUOC khi luu submission (dung chung giua
render_and_submit.py va eval_submission_path.py - tranh lap code lan thu 3
sau khi 2 file nay da tung lap logic apply_gain/LANCZOS resize rieng).

QUAN TRONG: diem so duoc cham TREN CHINH anh da ghi ra submission/ - nen bat
ky xu ly nao chen truoc buoc save/tinh metric deu anh huong TRUC TIEP toi
Score, khong chi anh huong cach nhin truc quan. Vi vay module nay THIET KE
AN TOAN: mac dinh mode="none" (no-op tuyet doi, tra ve anh y het dau vao),
CHI kich hoat khi nguoi dung chu dong chon --post_process, va PHAI duoc
verify bang so qua --sanity_check (xem render_and_submit.py) TRUOC khi bat
cho submission that - khong suy doan tu ly thuyet.

=== CO SO KY THUAT (thu tu co dinh: denoise TRUOC, sharpen SAU) ===

1. Bilateral filter (Tomasi & Manduchi, 1998) - khu nhieu/sparkle BAO TOAN
   BIEN. 3DGS rasterize truc tiep (khong Monte-Carlo) nen nhieu chu yeu la
   alias/sparkle o bien cac Gaussian nho chong lan, khong phai nhieu ngau
   nhien deu nhu anh chup thieu sang - bilateral phu hop hon Gaussian blur
   thuong vi khong lam mo bien THAT (canh mai nha, day cap).

2. Unsharp masking (ky thuat tang cuong bien co dien, nen tang cho hau het
   pipeline "detail enhancement" hau xu ly anh) - bu lai tan so cao co the
   bi mat qua downsample LANCZOS/do mo von co cua Gaussian splatting (moi
   Gaussian la 1 blob mem, khong sac canh tuyet doi nhu pixel rieng le).
   Dung nguong (threshold) de KHONG khuech dai nhieu o vung phang (troi,
   tuong phang) - chi tang cuong o vung DA CO canh that.

RUI RO PHAI Y THUC: sharpen qua tay tao halo/ringing KHONG khop GT se lam
LPIPS/SSIM/PSNR TE HON, khong phai chi "nhin dep hon". Day la ky thuat xu ly
anh 2D co dien, khong phai nghien cuu 3DGS chuyen biet - luon giu amount/
strength o muc NHE va verify bang --sanity_check truoc.
"""

import cv2
import numpy as np
from PIL import Image


def bilateral_denoise(img: Image.Image, strength: float = 0.3) -> Image.Image:
    """Khu nhieu/sparkle bao toan bien. strength trong [0, 1], map tuyen tinh
    sang sigmaColor/sigmaSpace cua cv2.bilateralFilter. strength=0 => no-op."""
    if strength <= 0.0:
        return img
    sigma = max(1.0, strength * 80.0)  # strength=0.3 -> sigma~24 (nhe)
    arr = np.asarray(img.convert("RGB"))
    d = 5  # duong kinh cua so loc - nho de giu chi tiet manh (day cap/anten)
    out = cv2.bilateralFilter(arr, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    return Image.fromarray(out)


def unsharp_mask(img: Image.Image, amount: float = 0.5, threshold: int = 2) -> Image.Image:
    """Tang cuong bien: sharpened = img + amount*(img - blur(img)), CHI ap
    dung o pixel co chenh lech > threshold (tranh khuech dai nhieu vung phang).
    amount=0 => no-op."""
    if amount <= 0.0:
        return img
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.2)
    diff = arr - blurred
    mask = (np.abs(diff).max(axis=-1, keepdims=True) > threshold).astype(np.float32)
    sharpened = arr + amount * diff * mask
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)


def gaussian_lowpass(img: Image.Image, sigma: float = 0.3) -> Image.Image:
    """Loc thong thap Gaussian RAT NHE. sigma=0 => no-op.

    === VI SAO CAN, VA VI SAO KHONG DUNG bilateral_denoise CHO VIEC NAY ===
    Do pho tan so tren 60 anh nop bai that (HCM0421, ckpt 25000) so voi 240 anh
    chup that cung scene cho thay anh render THUA nang luong tan so cao:
        do net (Lap.var)  1.51x    pho dai 2/3/5:  1.70x / 1.54x / 1.61x
    Da loai tru nen JPEG la nguyen nhan (ep nen q=75 len anh that chi lam pho
    doi x1.00-1.03). Vay phan thua nay la ARTIFACT RENDER that: needle Gaussian,
    rang cua, canh Gaussian cung. Vi no KHONG co trong anh that, no la SAI SO -
    ha bot se giam MSE (loi PSNR) chu khong lam mat chi tiet that.

    Quet sigma cho thay sigma ~ 0.3 giam sai lech pho tu 0.4424 xuong 0.2196
    (giam ~50%) - do la ly do mac dinh 0.3.

    KHONG dung bilateral (bao toan bien) cho muc dich nay: chinh cac "bien" gia
    cua needle artifact la thu can ha, ma bilateral duoc thiet ke de GIU chung.
    Day cung la ly giai vi sao do thuc te truoc day cho thay denoise_sharpen
    lam RỚT diem (0.723 -> 0.703): thanh phan sharpen di NGUOC huong can di.

    VAN PHAI DO BANG SO truoc khi bat cho submission that - khop pho khong dam
    bao khop metric."""
    if sigma <= 0.0:
        return img
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    out = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def despeckle_dark(img: Image.Image, threshold: float = 0.03, ksize: int = 3,
                    abs_dark_max: float = 0.05) -> Image.Image:
    """Vá các diem anh TOI BAT THUONG so voi lan can (lo thung khi render).
    threshold=0 => no-op.

    === CO SO DO DUOC TREN DU LIEU THAT ===
    Do tren 60 anh nop bai HCM0421 (ckpt 25000) vs 240 anh chup that cung scene:
        diem gan-den (<0.02):  render 0.941%  vs  anh that 0.207%  (thua 4.5x)
        so cum roi rac      :  render 6176    vs  anh that 1582    (thua 3.9x)
        ty le cum <= 4px    :  render 93.5%   vs  anh that 96.7%
    Ca 2 ben deu la DOM NHO, nhung render thua gap ~4 lan. Phan bo DEU khap anh
    (0.58-1.41% moi o luoi 5x5), khong tu o vien => KHONG phai loi warp/padding
    ma la LO THUNG tai tao: cho Gaussian phu khong kin, mau nen (den) lo ra.

    Uoc luong thiet hai (mo phong dam 0.73% diem den roi rac vao anh that):
    chiem toi ~36% tong sai so hien tai. LUU Y: day la CAN TREN, vi no gia dinh
    GT sang binh thuong tai dung nhung diem do - thuc te mot phan trong so do
    la vung von da toi that (khe giua cac nha). Phai DO tren tap eval co GT
    (eval_submission_path.py) moi biet loi ich thuc, dung tin con so can tren.

    === VI SAO AN TOAN: PHAI THOA CA HAI DIEU KIEN ===
    (1) toi hon trung vi cuc bo qua `threshold` -> nhieu "pepper" co dien, VA
    (2) ban than diem do phai toi tuyet doi (< `abs_dark_max`) -> dung la lo
        thung lo mau nen, khong phai chi la mot diem hoi toi trong vung sang.

    Dieu kien (2) la BAT BUOC, do bang so: neu chi dung (1), bo loc cham vao qua
    nhieu noi dung hop le - ap len anh CHUP THAT (von khong co lo thung) chi dat
    30-34 dB, tuc tu gay ra sai so bang ~17% tong sai so hien tai, trieu tieu
    gan het loi ich. Them (2) thu hep vung tac dong ve dung cac diem gan-den.

    Khong dung toi diem sang, khong dung toi vung toi LIEN MACH (bong do that
    co trung vi cuc bo cung toi nen dieu kien (1) khong kich hoat)."""
    if threshold <= 0.0:
        return img
    arr = np.asarray(img.convert("RGB")).astype(np.uint8)
    med = cv2.medianBlur(arr, ksize).astype(np.float32)
    arr_f = arr.astype(np.float32)
    lum, lum_med = arr_f.mean(axis=2), med.mean(axis=2)
    mask = ((lum_med - lum) > (threshold * 255.0)) & (lum < (abs_dark_max * 255.0))
    out = np.where(mask[..., None], med, arr_f)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


_VALID_MODES = ("none", "denoise", "sharpen", "denoise_sharpen", "lowpass",
                "despeckle", "despeckle_lowpass")


def enhance_image(img: Image.Image, mode: str = "none",
                   denoise_strength: float = 0.3, sharpen_amount: float = 0.5,
                   lowpass_sigma: float = 0.3, despeckle_threshold: float = 0.03) -> Image.Image:
    """mode in {"none", "denoise", "sharpen", "denoise_sharpen", "lowpass",
    "despeckle", "despeckle_lowpass"}.
    Thu tu CO DINH denoise -> sharpen (chuan phuc hoi anh, tranh khuech dai
    nhieu thanh ringing khi sharpen). "lowpass"/"despeckle" la cac che do RIENG
    (khong ket hop voi sharpen - nguoc huong nhau, xem gaussian_lowpass).
    "despeckle_lowpass" = va lo thung TRUOC roi loc thong thap - dung thu tu
    phuc hoi anh chuan (sua khuyet diem cuc bo truoc, lam min tong the sau).
    mode="none" (mac dinh) tra ve anh y het dau vao."""
    if mode not in _VALID_MODES:
        raise ValueError(f"mode khong hop le: {mode} (hop le: {_VALID_MODES})")
    if mode == "none":
        return img
    if mode == "lowpass":
        return gaussian_lowpass(img, sigma=lowpass_sigma)
    if mode == "despeckle":
        return despeckle_dark(img, threshold=despeckle_threshold)
    if mode == "despeckle_lowpass":
        return gaussian_lowpass(despeckle_dark(img, threshold=despeckle_threshold),
                                 sigma=lowpass_sigma)
    out = img
    if mode in ("denoise", "denoise_sharpen"):
        out = bilateral_denoise(out, strength=denoise_strength)
    if mode in ("sharpen", "denoise_sharpen"):
        out = unsharp_mask(out, amount=sharpen_amount)
    return out


def lab_color_transfer_stats(img: Image.Image, lightness_lo: float = 10.0, lightness_hi: float = 245.0):
    """Tinh mean+std tung kenh L,a,b (khong gian mau LAB - tach do sang khoi
    mau sac, phu hop hieu chinh phoi sang/mau tong the hon RGB gain thuan tuy
    vi RGB gain chi sua duoc TRUNG BINH tung kenh, khong sua duoc contrast/do
    bao hoa). Tra ve (mean[3], std[3]) float64.

    ROBUST: loai pixel co kenh L (lightness, thang 0-255 trong LAB 8-bit cua
    OpenCV) NAM NGOAI [lightness_lo, lightness_hi] TRUOC khi tinh mean/std -
    day la vung cháy sang/den kich (vd loa kim loai tren cot/anten BTS,
    Gaussian Splatting hau nhu khong tai tao dung vi tri/cuong do cac diem
    sang dac trung nay) se keo lech thong ke neu tinh tren TOAN BO pixel.
    Fallback: neu mask loai con lai <5% tong pixel (scene qua nhieu vung
    cuc tri), dung lai TOAN BO pixel va in canh bao, tranh tinh tren tap rong."""
    arr = np.asarray(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float64)
    flat = lab.reshape(-1, 3)
    mask = (flat[:, 0] >= lightness_lo) & (flat[:, 0] <= lightness_hi)
    if mask.sum() < 0.05 * flat.shape[0]:
        print("[CANH BAO][lab_color_transfer_stats] Mask loai outlier con lai <5% pixel - "
              "dung lai toan bo pixel (khong loai outlier) de tranh thong ke tren tap rong.")
        mask = np.ones(flat.shape[0], dtype=bool)
    filtered = flat[mask]
    mean = filtered.mean(axis=0)
    std = filtered.std(axis=0).clip(min=1e-6)
    return mean, std


def apply_lab_color_transfer(img: Image.Image, src_stats, dst_stats) -> Image.Image:
    """Bien doi affine kieu Reinhard et al. 2001 "Color Transfer between
    Images": (pixel - mean_src)/std_src * std_dst + mean_dst, tung kenh LAB
    doc lap - sua duoc CA trung binh (nhu RGB gain) LAN do tuong phan/phan
    tan mau, khac RGB gain (chi la 1 he so nhan vo huong/kenh)."""
    src_mean, src_std = src_stats
    dst_mean, dst_std = dst_stats
    arr = np.asarray(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab = (lab - src_mean) / src_std * dst_std + dst_mean
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)
