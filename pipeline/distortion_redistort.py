"""
"Un-redistort": render Gaussian o camera PINHOLE (distortion=0) roi warp NGUOC
ve dung luoi anh CO MEO (radial) ma test_poses.csv/COLMAP yeu cau, thay vi de
rasterizer (gsplat) tu ap distortion truc tiep trong phep chieu ("rasterizer-
native", cach render_and_submit.py dang dung mac dinh).

Port truc tiep tu inria/render_test_inria.py (undistort_normalized(),
redistort_render() - da verify bang test tong hop: sai so round-trip
undistort->distort ~1.1e-4 px), tong quat hoa de dung chung cho nhanh
Nerfstudio (render/target intrinsics tach biet ro rang, ho tro supersample).

=== VI SAO CO CA 2 CHE DO, KHONG CHI DUNG 1 ===
Rasterizer-native VE LY THUYET tot hon (khong co phep resample/warp THEM nen
khong mat chi tiet tan so cao) - NHUNG chua tung duoc verify bang so tren GPU
that. Vung ria/goc khung hinh (dang bi mo trong anh checkpoint 10k da xem)
chinh la noi distortion manh nhat (ban kinh tu tam anh lon nhat) - neu
rasterizer ap meo sai o do, trieu chung se giong het gia thuyet "mat do COLMAP
thua". Co ca 2 che do de A/B THAT bang so (--distortion_mode trong
render_and_submit.py/eval_submission_path.py) thay vi tin theo ly thuyet.

=== GIOI HAN PHAM VI (BAT BUOC BIET TRUOC KHI DUNG) ===
Ham nghich dao CHI xu ly meo RADIAL (k1, k2) - giong het pham vi da chap nhan
o nhanh INRIA. KHONG nghich dao duoc thanh phan tangential (p1, p2). Ly do
chap nhan duoc: COLMAP SIMPLE_RADIAL (model pho bien cho anh drone) khong co
p1/p2; k1 do thuc te tren du lieu BTS ~0.008-0.009 (nho). Neu scene dung model
OPENCV/OPENCV_FISHEYE voi p1/p2 dang ke, che do un_redistort o day se KHONG
chinh xac hoan toan - van dung duoc de so sanh tuong doi nhung dung ket luan
qua chac chan cho cac scene do.
"""

import torch


_PREFILTER_WARNED = False


def _warn_non_integer_prefilter(sx: float):
    """In canh bao DUNG 1 LAN (tranh spam 60 dong khi render ca scene)."""
    global _PREFILTER_WARNED
    if _PREFILTER_WARNED:
        return
    _PREFILTER_WARNED = True
    print(f"[CANH BAO] --resample_prefilter bi BO QUA vi ty le supersample ({sx:.3f}) "
          f"khong phai so nguyen. Da do bang so: o ty le LE, prefilter lam TE HON "
          f"(vd ss=2.5: 37.3 dB co prefilter vs 43.1 dB khong co). Prefilter chi co loi "
          f"o ty le NGUYEN >= 2 (2.0/3.0/4.0) - hay dung supersample_factor nguyen.")


def undistort_normalized(xd: torch.Tensor, yd: torch.Tensor, k1: float, k2: float = 0.0,
                          iters: int = 8):
    """Nghich dao meo radial: tim (x,y) sao cho distort(x,y) = (xd,yd).
    COLMAP SIMPLE_RADIAL/RADIAL: xd = x*(1 + k1*r^2 + k2*r^4), r^2 = x^2+y^2.
    Khong co nghiem dong kin -> lap diem bat dong (hoi tu rat nhanh voi k1 nho,
    da verify hoi tu trong <=8 buoc lap voi k1~0.01 o test tong hop truoc)."""
    x, y = xd.clone(), yd.clone()
    for _ in range(iters):
        r2 = x * x + y * y
        f = 1.0 + k1 * r2 + k2 * r2 * r2
        x = xd / f
        y = yd / f
    return x, y


def redistort_to_target(render_rgb: torch.Tensor, render_fx: float, render_fy: float,
                         render_cx: float, render_cy: float, target_w: int, target_h: int,
                         target_fx: float, target_fy: float, target_cx: float, target_cy: float,
                         k1: float, k2: float = 0.0, prefilter: bool = False) -> torch.Tensor:
    """render_rgb: [C,H,W] float [0,1], da render PINHOLE (distortion=0) o do
    phan giai render (co the khac target vi supersample). Tra ve [C,target_h,
    target_w] - anh tren DUNG luoi co meo ma target intrinsics mo ta.

    THAY THE cho buoc resize LANCZOS thong thuong (vua doi do phan giai vua ap
    meo trong 1 lan grid_sample, tranh 2 lan resample chong nhau).

    Neu k1==0 va k2==0 (khong co meo/da undistort roi), CHI resize (bilinear)
    ve target size ma khong sinh loi so hoc thua - hanh vi tuong duong resize
    thuong khi khong co meo.

    === prefilter (MOI - sua loi khu rang cua khi supersample) ===
    VAN DE: grid_sample(mode="bilinear") chi doc 2x2 diem anh lan can quanh moi
    diem lay mau. Khi supersample_factor = 2.5, moi diem anh dau ra ung voi
    ~2.5x2.5 ~ 6.25 diem anh nguon, nhung bilinear VAN chi doc 4 diem - phan lon
    thong tin supersample bi VUT BO va tan so cao gap nguoc lai thanh RANG CUA
    (aliasing). Day la ly thuyet lay mau co ban: truoc khi giam do phan giai
    PHAI loc thong thap khop voi ty le giam (mipmap/area filter, Williams 1983);
    bilinear 2x2 chi du cho ty le giam <= 2.
    BANG CHUNG KHOP: do thuc te cho thay Score tang manh tu supersample 1.5 ->
    2.0 -> 2.5 roi CHUNG LAI o 2.5-3.0 (0.7236 vs 0.7232). Do chinh la dau hieu
    cua bo loc bi gioi han, khong phai supersample da bao hoa that.
    CACH SUA: prefilter=True lay trung binh vung bang box filter CHINH XAC
    (avg_pool2d) ve dung target size TRUOC, roi moi grid_sample de ap meo (luc
    nay ty le 1:1 nen bilinear la du). Intrinsics render duoc quy doi tuong ung.

    === KET QUA DO BANG SO HOC TONG HOP (anh nhieu tan so cao, CPU, khong can GPU) ===
    So sanh voi "anh chan ly" = tich phan canh vat lien tuc tren dien tich tung
    diem anh o do phan giai target. Tran so hoc cua phep do la 48.84 dB.

        ss    prefilter=False   prefilter=True
        1.0        48.84            48.84       (khong giam do phan giai)
        1.5        43.92            35.35       <- ty le LE: prefilter HAI
        2.0        48.84            48.84       <- bilinear 2x2 = box 2x2 chinh xac
        2.5        43.10            37.32       <- ty le LE: prefilter HAI
        3.0        38.49            48.84       <- prefilter DAT TRAN
        4.0        39.88            48.84       <- prefilter DAT TRAN

    KET LUAN AP DUNG: dung supersample NGUYEN. Tot nhat la ss=3 hoac 4 KEM
    prefilter (dat tran so hoc), hoac ss=2 (tu no da chinh xac). TUYET DOI
    khong dung ty le le kem prefilter. Ham tu bo qua prefilter + canh bao 1 lan
    neu ty le khong nguyen, nen khong the vo tinh roi vao o xau.
    Mac dinh False = hanh vi cu y nguyen; bat qua --resample_prefilter."""
    device = render_rgb.device
    C, Hr, Wr = render_rgb.shape

    if prefilter and (Wr > target_w or Hr > target_h):
        sx = Wr / float(target_w)
        sy = Hr / float(target_h)
        # Chon bo loc theo TY LE GIAM (da do bang so hoc tong hop, xem ket qua
        # trong docstring): ty le NGUYEN -> avg_pool2d la box filter CHINH XAC
        # (dat toi da ly thuyet). Ty le LE -> mode="area" cua PyTorch KHONG con
        # la box filter sach (no la adaptive-pool, do rong cua so khong deu) va
        # do that su TE HON ca khong loc; dung interpolate(antialias=True) - bo
        # loc co ho tro dan theo ty le, dung ho hang voi LANCZOS cua PIL.
        int_x = abs(sx - round(sx)) < 1e-6 and round(sx) >= 2
        int_y = abs(sy - round(sy)) < 1e-6 and round(sy) >= 2
        if int_x and int_y:
            render_rgb = torch.nn.functional.avg_pool2d(
                render_rgb.unsqueeze(0), kernel_size=(int(round(sy)), int(round(sx)))).squeeze(0)
            # Quy doi intrinsics: trong he toa do LIEN TUC, thu nho anh tu Wr ve
            # target_w chi la phep co gian tuyen tinh [0,Wr] -> [0,target_w], nen
            # moi toa do lien tuc chia deu cho s: fx' = fx/s, cx' = cx/s (khong
            # co +-0.5 nao - chung chi xuat hien khi lam viec voi CHI SO).
            render_fx = render_fx / sx
            render_fy = render_fy / sy
            render_cx = render_cx / sx
            render_cy = render_cy / sy
            C, Hr, Wr = render_rgb.shape
        else:
            # Do bang so: voi ty le LE, prefilter roi resample lan 2 luon TE HON
            # la de nguyen (vd ss=2.5: 37.3 dB co prefilter vs 43.1 dB khong co).
            # Nen BO QUA prefilter thay vi lam hong am tham, va noi ro cach sua.
            _warn_non_integer_prefilter(sx)

    # (SUA LOI LECH DUOI-PIXEL - da kiem chung bang so hoc tong hop)
    # TRUOC DAY: xd = (uu - target_cx)/target_fx voi uu = arange(w), tuc coi CHI SO
    # diem anh la vi tri lay mau. Nhung cx/cy theo quy uoc toa do LIEN TUC (cx=660.0
    # = 1320/2, cy=494.5 = 989/2 - dung tam anh), trong do tam diem anh chi so u nam
    # o u+0.5. Thieu 0.5 nay gay LECH HE THONG = 0.5*(ss-1)/ss diem anh target, LON
    # DAN theo supersample: 0.17px @ss1.5, 0.25px @ss2.0, 0.30px @ss2.5, 0.38px @ss4.
    # Do bang so tren anh tong hop nhieu tan so cao: o ss=2.5 mat ~9.6 dB PSNR chi vi
    # 0.3 diem anh lech. --diagnose_misalignment KHONG BAO GIO thay duoc loi nay vi no
    # chi do dich chuyen NGUYEN diem anh (nen truoc day ket luan "khong co misalignment").
    # Day cung la ly do Score CHUNG LAI o supersample 2.5-3.0: cang supersample, lech
    # cang lon, triet tieu dan loi ich cua supersample.
    vv, uu = torch.meshgrid(
        torch.arange(target_h, dtype=torch.float32, device=device),
        torch.arange(target_w, dtype=torch.float32, device=device), indexing="ij")
    xd = (uu + 0.5 - target_cx) / target_fx
    yd = (vv + 0.5 - target_cy) / target_fy

    if k1 == 0.0 and k2 == 0.0:
        x, y = xd, yd
    else:
        x, y = undistort_normalized(xd, yd, k1, k2)

    su = x * render_fx + render_cx
    sv = y * render_fy + render_cy

    # su/sv gio la toa do LIEN TUC trong anh render (khong phai chi so), nen doi
    # sang [-1,1] cua grid_sample(align_corners=False) la su*2/Wr - 1, KHONG cong
    # them 0.5 nua (0.5 do da nam san trong "uu + 0.5" o tren - cong 2 lan chinh
    # la nguon lech duoi-pixel cu).
    # Kiem chung: khi render/target trung intrinsics va ty le ss, su = (u+0.5)*ss
    # => gx = (u+0.5)*2/target_w - 1, dung bang tam vet chieu cua diem anh target u.
    gx = su * 2.0 / Wr - 1.0
    gy = sv * 2.0 / Hr - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(
        render_rgb.unsqueeze(0), grid, mode="bilinear",
        padding_mode="border", align_corners=False)
    return out.squeeze(0).clamp(0.0, 1.0)
