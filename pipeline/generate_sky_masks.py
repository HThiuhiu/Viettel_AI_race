"""
Sinh sky mask HEURISTIC (khong tai model, khong can mang luc chay - quan trong
vi notebook Kaggle luc train thuong tat Internet) cho tung anh train, ghi vao
<scene>/train/masks/<ten_anh> - dung dinh dang nhi phan (255 = giu lai/khong
phai troi, 0 = bo qua khi tinh loss) - duoc `preprocess_colmap_to_nerfstudio.py`
doc va nhung thanh "mask_path" cho tung frame trong transforms.json (KHONG
dung co CLI, da xac nhan khong ton tai qua kiem thu that tren Kaggle).

=== VI SAO CAN CANH GIAC VOI TINH NANG NAY ===
Mo ta tham khao ma user cung cap ("khong sky-mask trong ban cuoi") ham y 1
pipeline doi chung DA THU sky mask va CHU DONG BO trong cau hinh cuoi cung -
nhieu kha nang vi mask cung tao VIEN ARTIFACT tai bien troi/vat the, dac biet
nguy hiem cho day cap/anten BTS MONG cat ngang bau troi (dung chi tiet LPIPS
phat nang nhat neu mask cat nham 1 phan anten vao vung "bo qua"). VI VAY:
mac dinh TAT trong train.sh (SKY_MASK_INIT=0), chi bat de A/B CO KIEM CHUNG
qua metrics_summary.csv - khong duoc coi day la cai thien chac chan.

=== HEURISTIC (khong dung model, 3 tin hieu ket hop) ===
1. Vi tri: uu tien nua tren khung hinh (troi hau nhu luon o phia tren).
2. Mau sac: troi thuong co do bao hoa (saturation) THAP va do sang (value) CAO
   trong khong gian mau HSV (kha thi voi troi trong/nhieu may, kem chinh xac
   hon voi troi hoang hon nhieu mau - chap nhan duoc vi day la fallback tho).
3. Gradient/texture: troi PHANG (gradient thap), khac voi mai nha/day cap co
   nhieu canh net. Loc bang Laplacian variance cuc bo (cua so nho).

3 tin hieu ket hop bang AND (phai thoa CA 3 moi coi la troi) - uu tien AN TOAN
(mask it/nho hon that) hon la mask lem sang vat the chinh, vi mask lem 1 phan
ang-ten se bi LPIPS phat nang hon la bo sot 1 it troi.

Cach dung:
    python pipeline/generate_sky_masks.py --raw_root .
    python pipeline/generate_sky_masks.py --raw_root . --scene HCM0421
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def discover_scenes(raw_root: Path):
    scene_dirs = []
    for candidate in sorted(raw_root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "train" / "images").exists():
            scene_dirs.append((candidate.name, candidate / "train"))
        elif (candidate / "images").exists():
            scene_dirs.append((candidate.name, candidate))
    return scene_dirs


def compute_sky_mask(img_bgr: np.ndarray, top_fraction: float = 0.6,
                      sat_thresh: int = 60, val_thresh: int = 140,
                      texture_thresh: float = 1500.0) -> np.ndarray:
    """Tra ve mask uint8 (255 = GIU LAI de train, 0 = BO QUA vi la troi).
    Nguoc voi "sky mask" truc quan - day la mask NGUOC duoc nhung vao
    "mask_path" cua tung frame trong transforms.json (xem
    preprocess_colmap_to_nerfstudio.py - KHONG con dung co CLI --mask-dir,
    da xac nhan khong ton tai qua kiem thu that tren Kaggle).

    (Sua loi quan trong, phat hien qua kiem tra truc quan tren anh that)
    texture_thresh MAC DINH CU (80.0) qua thap: bau troi CO MAY (khong phai
    troi trong xanh phang lang) co gradient/texture ro ret o vien may, bi
    nguong 80.0 loai KHOI mask du van la troi that - chi phu duoc ~31% vung
    troi thuc te tren anh mau (kiem tra bang mat + do dac tren HCM0421). Tang
    len 1500.0: phu duoc ~47% (tang dang ke), da xac nhan KHONG gay false-
    positive tren 4 anh nadir (chup thang dung, khong co troi) kiem tra
    rieng - ranh gioi van bam sat duong chan troi nho connected-component
    filter co san, khong lem vao anten/cong trinh."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # Phuong sai cuc bo (cua so 15x15) cua Laplacian - do "phang" cuc bo, dung
    # blur binh phuong thay vi vong lap tung pixel de nhanh.
    lap_sq_mean = cv2.blur(laplacian ** 2, (15, 15))
    local_texture = lap_sq_mean  # xap xi variance cuc bo (mean da gan 0)

    is_top_region = np.zeros((h, w), dtype=bool)
    is_top_region[: int(h * top_fraction), :] = True

    is_sky_color = (sat < sat_thresh) & (val > val_thresh)
    is_flat = local_texture < texture_thresh

    is_sky = is_top_region & is_sky_color & is_flat

    # Chi giu vung troi LON NHAT lien thong (loai bo cac mang nho rai rac -
    # thuong la mai ton sang mau/kinh phan chieu bi nham thanh troi).
    is_sky_u8 = (is_sky.astype(np.uint8)) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(is_sky_u8, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        if areas.max() > 0.02 * h * w:  # bo qua neu vung lon nhat qua nho (khong dang tin)
            is_sky = labels == largest_idx
        else:
            is_sky = np.zeros((h, w), dtype=bool)

    # Lam min bien mask (giam rang cua tu connected components/nguong cung)
    mask_keep = (~is_sky).astype(np.uint8) * 255
    mask_keep = cv2.GaussianBlur(mask_keep, (9, 9), 0)
    mask_keep = (mask_keep > 127).astype(np.uint8) * 255
    return mask_keep


def process_scene(scene_id: str, scene_dir: Path, skip_existing: bool) -> int:
    images_dir = scene_dir / "images"
    masks_dir = scene_dir / "masks"
    if not images_dir.exists():
        print(f"[BO QUA][{scene_id}] Khong thay {images_dir}")
        return 0

    masks_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix in exts)
    for img_path in image_paths:
        mask_path = masks_dir / img_path.name
        if skip_existing and mask_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[LOI][{scene_id}] Khong doc duoc {img_path}")
            continue
        mask = compute_sky_mask(img_bgr)
        cv2.imwrite(str(mask_path), mask)
        n_done += 1

    print(f"[{scene_id}] Da ghi {n_done}/{len(image_paths)} mask vao {masks_dir}")
    return n_done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    scene_dirs = discover_scenes(raw_root)
    if args.scene:
        scene_dirs = [(sid, sdir) for sid, sdir in scene_dirs if sid == args.scene]
    if not scene_dirs:
        raise RuntimeError(f"Khong tim thay scene nao trong {raw_root}" +
                            (f" khop ten '{args.scene}'" if args.scene else ""))

    print("[CANH BAO] Sky mask la heuristic THO, mac dinh TAT trong train.sh - chi bat de")
    print("A/B co kiem chung. Neu nghi mask lem vao day cap/anten, kiem tra truc quan vai")
    print("anh trong <scene>/train/masks/ truoc khi tin tuong dung cho full train.")

    total = 0
    for scene_id, scene_dir in scene_dirs:
        total += process_scene(scene_id, scene_dir, args.skip_existing)
    print(f"\n=== Tong: {total} mask duoc ghi ===")


if __name__ == "__main__":
    main()
