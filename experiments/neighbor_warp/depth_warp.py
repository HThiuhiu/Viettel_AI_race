"""
Depth-based Image Warping (DIBR - Depth Image Based Rendering): chieu anh
THAT cua 1 camera nguon (neighbor frame trong train) sang dung goc nhin cua
1 camera dich (pose test yeu cau), dung DEPTH MAP ma chinh Gaussian model
uoc luong tai pose dich - thay the cho "trick" dung nguyen anh lang gieng
khong sua gi (da bi cham 0 diem vi lech pose hinh hoc that su, khong chi
lech mau).

"""

import numpy as np
import torch


def backproject_camera_space(uu: torch.Tensor, vv: torch.Tensor, depth: torch.Tensor,
                              fx: float, fy: float, cx: float, cy: float):
    """Tu toa do pixel (uu,vv) + z-depth -> toa do 3D trong khong gian CAMERA
    (quy uoc OpenGL: X phai, Y len, camera nhin theo -Z). z_cam = -depth vi
    camera nhin ve huong -Z, vat truoc camera co z am."""
    z_cam = -depth
    x_cam = (uu - cx) / fx * depth
    y_cam = -(vv - cy) / fy * depth
    return torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [..., 3]


def project_camera_space(xyz_cam: torch.Tensor, fx: float, fy: float, cx: float, cy: float):
    """Nguoc lai backproject_camera_space: toa do 3D camera-space -> pixel
    (u,v) + z-depth. Tra ve (u, v, depth, valid) - valid=False neu diem o SAU
    lung camera (z_cam >= 0, tuc dang o phia sau, khong the nhin thay)."""
    x_cam, y_cam, z_cam = xyz_cam[..., 0], xyz_cam[..., 1], xyz_cam[..., 2]
    depth = -z_cam
    valid = depth > 1e-6
    safe_depth = depth.clamp_min(1e-6)
    u = x_cam / safe_depth * fx + cx
    v = -y_cam / safe_depth * fy + cy
    return u, v, depth, valid


def transform_points(xyz: torch.Tensor, mat_3x4: torch.Tensor) -> torch.Tensor:
    """Ap dung 1 phep bien doi affine 3x4 (dang c2w hoac w2c) len tap diem
    [...,3] -> [...,3]. mat_3x4[:, :3] la phan xoay, mat_3x4[:, 3] la tinh tien."""
    R = mat_3x4[:, :3]
    t = mat_3x4[:, 3]
    return xyz @ R.T + t


def invert_c2w(c2w_3x4: torch.Tensor) -> torch.Tensor:
    """Nghich dao ma tran camera-to-world (3x4) -> world-to-camera (3x4)."""
    R = c2w_3x4[:, :3]
    t = c2w_3x4[:, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=-1)


def warp_by_depth(source_img_chw: torch.Tensor, source_c2w_3x4: torch.Tensor,
                   target_depth_hw: torch.Tensor, target_c2w_3x4: torch.Tensor,
                   fx: float, fy: float, cx: float, cy: float):
    """Chieu anh source_img_chw (anh THAT cua camera nguon, vd frame lang
    gieng trong train) sang dung goc nhin cua target_c2w, dung target_depth_hw
    (depth Gaussian model uoc luong TAI POSE DICH) de biet moi pixel dich
    tuong ung diem 3D nao.

    Tra ve (warped_img_chw, valid_mask_hw) - valid_mask=False o pixel: (a)
    diem 3D tuong ung roi ra ngoai khung hinh nguon, hoac (b) nam SAU lung
    camera nguon (bi che khuat / ngoai tam nhin) - CAC PIXEL NAY PHAI duoc
    fallback ve anh Gaussian model tu render binh thuong, KHONG dung anh
    warp (khong co du lieu that de warp)."""
    device = target_depth_hw.device
    H, W = target_depth_hw.shape
    vv, uu = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device), indexing="ij")

    xyz_cam_target = backproject_camera_space(uu, vv, target_depth_hw, fx, fy, cx, cy)
    xyz_world = transform_points(xyz_cam_target.reshape(-1, 3), target_c2w_3x4).reshape(H, W, 3)

    w2c_source = invert_c2w(source_c2w_3x4)
    xyz_cam_source = transform_points(xyz_world.reshape(-1, 3), w2c_source).reshape(H, W, 3)

    u_src, v_src, depth_src, valid_front = project_camera_space(xyz_cam_source, fx, fy, cx, cy)

    Hs, Ws = source_img_chw.shape[1], source_img_chw.shape[2]
    valid_bounds = (u_src >= 0) & (u_src <= Ws - 1) & (v_src >= 0) & (v_src <= Hs - 1)
    valid = valid_front & valid_bounds

    gx = (u_src + 0.5) * 2.0 / Ws - 1.0
    gy = (v_src + 0.5) * 2.0 / Hs - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    warped = torch.nn.functional.grid_sample(
        source_img_chw.unsqueeze(0), grid, mode="bilinear",
        padding_mode="border", align_corners=False).squeeze(0)

    return warped, valid


def blend_warped_with_fallback(warped_chw: torch.Tensor, valid_hw: torch.Tensor,
                                fallback_chw: torch.Tensor) -> torch.Tensor:
    """Dung pixel warp o vung valid, fallback ve anh Gaussian model render o
    vung khong valid (che khuat/ngoai khung anh nguon)."""
    mask = valid_hw.unsqueeze(0).float()
    return warped_chw * mask + fallback_chw * (1.0 - mask)
