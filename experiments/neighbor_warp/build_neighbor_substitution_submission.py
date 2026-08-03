"""
Submission THU NGHIEM/BASELINE: thay vi render tu Gaussian model, LAY THANG
anh THAT tu train/images cua frame LAN CAN GAN NHAT theo SO THU TU frame
(khong phai theo khoang cach pose 3D) - vi test_poses.csv co ten anh xen
giua cac frame train lien tuc (video drone lien tuc), frame lan can rat gan
ve mat khong gian/thoi gian, nen anh THAT cua no co the la 1 xap xi tot cho
frame test can, dac biet o vung model dang tai tao kem (nen xa, bau troi).

=== THU TU TIM LAN CAN (dung yeu cau nguoi dung) ===
-1, +1, -2, +2, -3, +3, ... mo rong dan cho den khi tim thay 1 frame THAT SU
TON TAI trong train/images (frame o giua bi loai vi la anh test/thieu file
se tu dong bi bo qua, tim tiep ra xa hon).

Da VERIFY tren du lieu that (HCM0421, 60 anh test): ca 60/60 tim duoc lan can,
47/60 o dung offset -1, con lai o -2/-3/-4/+1 (khi co nhieu frame test lien
tiep tao khoang trong dai hon 1).

=== CANH BAO QUAN TRONG - DOC TRUOC KHI NOP ===
Day la anh THAT cua 1 goc chup KHAC (lech it nhat 1 frame trong chuoi video
drone), KHONG PHAI anh render dung tai pose test yeu cau - se co sai lech
vi tri/goc nhin THAT SU (drone di chuyen giua 2 frame lien tiep, dong nghia
lech ca vi tri lan huong nhin, khong chi mau sac). Day CHI la 1 baseline de
so sanh/thu nghiem, KHONG chac chan tot hon anh render tu model - dac biet
o vung co nhieu chi tiet gan (parallax lon giua 2 frame) se lech ro ret, chi
co the tot hon o vung nen xa/bau troi (it thay doi giua cac frame lien tiep).

=== VI TRI FILE ===
Nam trong thu muc con experiments/neighbor_warp/ - TACH BIET hoan toan khoi pipeline train chinh
(train.sh khong goi den file nay). Da xac nhan qua thuc te: nop bai kieu nay
bi cham 0 diem (lech pose hinh hoc that su) - xem experiments/neighbor_warp/render_depth_warped_submission.py
cho phien ban co sua loi hinh hoc bang depth-warp.

Cach dung (chay tu thu muc GOC cua project):
    python experiments/neighbor_warp/build_neighbor_substitution_submission.py --raw_root . --scene HCM0421
    python experiments/neighbor_warp/build_neighbor_substitution_submission.py --raw_root . --submission_dir submission_neighbor --zip_name submission_neighbor.zip
"""

import argparse
import csv
import re
import zipfile
from pathlib import Path

from PIL import Image

# (Da phat hien qua chay that) 2 quy uoc dat ten khac nhau trong dataset:
# - DJI_YYYYMMDDHHMMSS_NNNN_V.JPG (cac scene BTS: HCM0421/0539/0540/0644/0674)
# - frame_NNNNNN.jpg (cac scene tinh: bonsai/chair)
FRAME_RE_DJI = re.compile(r"_(\d+)_V\.[A-Za-z]+$", re.IGNORECASE)
FRAME_RE_GENERIC = re.compile(r"(\d+)\.[A-Za-z]+$")


def extract_frame_num(name: str):
    m = FRAME_RE_DJI.search(name)
    if m:
        return int(m.group(1))
    m = FRAME_RE_GENERIC.search(name)
    if m:
        return int(m.group(1))
    return None


def build_frame_index(images_dir: Path) -> dict:
    """Tra ve dict {so_thu_tu_frame: Path} cho moi anh trong images_dir. Anh
    khong parse duoc so thu tu (ten khac quy uoc DJI_..._NNNN_V.ext) bi bo qua
    va canh bao."""
    index = {}
    for p in sorted(images_dir.glob("*")):
        if not p.is_file():
            continue
        num = extract_frame_num(p.name)
        if num is None:
            print(f"[CANH BAO] Khong parse duoc so thu tu frame tu ten '{p.name}', bo qua file nay.")
            continue
        index[num] = p
    return index


def find_neighbor_frame(frame_index: dict, target: int, max_radius: int = 500):
    """Tim frame THAT SU TON TAI gan target nhat, thu tu -1,+1,-2,+2,...
    Tra ve (offset, Path) hoac (None, None) neu khong tim thay trong pham vi."""
    for r in range(1, max_radius + 1):
        for cand in (target - r, target + r):
            if cand in frame_index:
                return cand - target, frame_index[cand]
    return None, None


def discover_scenes(raw_root: Path):
    scene_dirs = []
    for candidate in sorted(raw_root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "train" / "images").exists() and (candidate / "test" / "test_poses.csv").exists():
            scene_dirs.append((candidate.name, candidate))
    return scene_dirs


def make_zip(submission_root: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in submission_root.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(submission_root))


def process_scene(scene_id: str, scene_dir: Path, submission_root: Path, max_radius: int) -> dict:
    images_dir = scene_dir / "train" / "images"
    test_csv_path = scene_dir / "test" / "test_poses.csv"

    frame_index = build_frame_index(images_dir)
    with open(test_csv_path, newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    out_dir = submission_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    offset_stats = {}
    not_found = []
    for row in test_rows:
        test_name = row["image_name"]
        target = extract_frame_num(test_name)
        if target is None:
            print(f"[LOI][{scene_id}] Khong parse duoc so thu tu frame tu '{test_name}' - BO QUA anh nay.")
            not_found.append(test_name)
            continue

        offset, src_path = find_neighbor_frame(frame_index, target, max_radius=max_radius)
        if src_path is None:
            print(f"[LOI][{scene_id}] Khong tim thay frame lan can nao trong ban kinh {max_radius} "
                  f"cho '{test_name}' - BO QUA anh nay.")
            not_found.append(test_name)
            continue

        want_w, want_h = int(row["width"]), int(row["height"])
        img = Image.open(src_path).convert("RGB")
        if img.size != (want_w, want_h):
            img = img.resize((want_w, want_h), Image.LANCZOS)
        img.save(out_dir / test_name)

        offset_stats[offset] = offset_stats.get(offset, 0) + 1

    print(f"[{scene_id}] Da tao {len(test_rows) - len(not_found)}/{len(test_rows)} anh submission "
          f"(phan bo offset: {dict(sorted(offset_stats.items()))})")
    if not_found:
        print(f"[CANH BAO][{scene_id}] {len(not_found)} anh KHONG tao duoc: {not_found}")

    return {"n_total": len(test_rows), "n_ok": len(test_rows) - len(not_found), "offset_stats": offset_stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True)
    ap.add_argument("--scene", type=str, default=None, help="Chi chay 1 scene (de test nhanh)")
    ap.add_argument("--submission_dir", type=str, default="submission_neighbor")
    ap.add_argument("--zip_name", type=str, default="submission_neighbor.zip")
    ap.add_argument("--max_radius", type=int, default=500,
                     help="Ban kinh tim kiem toi da (so frame) - mac dinh du lon cho hau het truong hop")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    submission_root = Path(args.submission_dir)

    scenes = discover_scenes(raw_root)
    if args.scene:
        scenes = [(sid, sdir) for sid, sdir in scenes if sid == args.scene]
    if not scenes:
        raise RuntimeError(f"Khong tim thay scene nao trong {raw_root}" +
                            (f" khop ten '{args.scene}'" if args.scene else ""))

    print("[CANH BAO QUAN TRONG] Day la submission THU NGHIEM/BASELINE dung anh THAT cua frame lan can")
    print("(lech it nhat 1 frame trong chuoi video drone) thay vi anh render tu model dung pose test yeu")
    print("cau - se lech vi tri/goc nhin thuc su, khong dam bao tot hon ban render tu Gaussian model.\n")

    total_ok, total_all = 0, 0
    for scene_id, scene_dir in scenes:
        stats = process_scene(scene_id, scene_dir, submission_root, args.max_radius)
        total_ok += stats["n_ok"]
        total_all += stats["n_total"]

    print(f"\n=== Tong: {total_ok}/{total_all} anh da tao ===")
    make_zip(submission_root, Path(args.zip_name))
    print(f"File nop bai: {args.zip_name}")


if __name__ == "__main__":
    main()
