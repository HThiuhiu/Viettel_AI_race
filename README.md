# Viettel Digital Twin — 3D Gaussian Splatting

Pipeline dựng lại cảnh 3D từ ảnh drone chụp trạm BTS bằng **3D Gaussian Splatting**
(Nerfstudio / Splatfacto), rồi render lại đúng các pose test.

Điểm chấm là trung bình của 5 scene:

```
Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
PSNR_norm = clamp(PSNR / 50, 0, 1)
```

Vì LPIPS chiếm 40%, mọi thứ làm mờ ảnh (downscale, JPEG nén mạnh, trung bình
nhiều checkpoint) đều bị phạt nặng — đó là lý do nhiều mặc định trong repo này
khác với mặc định của Nerfstudio.

---

## Cấu trúc

```
train.sh                      Entry point: cài Nerfstudio → preprocess → patch → train → eval
run_final.py                  Điều phối lần chạy cuối: check → train → chọn checkpoint → render → kiểm tra

pipeline/                              Các bước bắt buộc của đường nộp bài
  preprocess_colmap_to_nerfstudio.py   COLMAP → transforms.json (kèm ply_file_path + hệ số méo)
  dense_colmap_init.py                 (tuỳ chọn) dense MVS init thay cho sparse points
  generate_sky_masks.py                Sinh sky mask heuristic, không cần model/Internet
  render_and_submit.py                 Render đúng pose test → đóng gói submission
  eval_submission_path.py              Đo Score thật qua đúng đường render nộp bài
  select_best_checkpoint.py            Quét mọi checkpoint, chọn cái điểm cao nhất
  image_postprocess.py                 Hậu xử lý 2D dùng chung (gain, LAB transfer, resize)
  distortion_redistort.py              Render pinhole rồi warp ngược về lưới méo thật
  check_gs_count.py                    Xác nhận STRATEGY thật + số Gaussian trong checkpoint
  compute_score.py                     Tính Score theo đúng công thức BTC

patches/                      
                              
  patch_splatfacto_mip_filter.py       Mip-Splatting soft loss — chống needle artifact
  patch_splatfacto_perceptual_loss.py  LPIPS-as-loss theo lịch ramp (curriculum)
  patch_robust_transient_loss.py       Robust L1 kiểu RobustNeRF — bỏ qua pixel transient

tools/                        Công cụ phụ trợ, không nằm trong đường nộp bài
  validate_colmap_structure.py         Preflight cấu trúc COLMAP, vài giây, không cần GPU
  run_scene.py                         Helper gọi train.sh từ cell Colab (chạy nền được)
  ensemble_checkpoints.py              A/B trung bình nhiều checkpoint (cảnh báo: gây mờ)
  select_appearance_mode.py            A/B rgb_gain vs lab_transfer theo từng scene
  clean_export_gaussians.py            Export .ply đã lọc floater cho viewer Digital Twin

experiments/neighbor_warp/    Thử nghiệm tách biệt, KHÔNG script train nào gọi tới
  build_neighbor_substitution_submission.py   Thay thẳng ảnh frame lân cận — đã bị chấm 0 điểm
  depth_warp.py                               DIBR: warp ảnh thật theo depth + pose
  render_depth_warped_submission.py           Submission dựng bằng DIBR
```

---

## Chuẩn bị

Dữ liệu **không** nằm trong repo. Đặt các thư mục scene ngang hàng với `train.sh`:

```
<repo>/
  HCM0421/  HCM0539/  HCM0540/  HCM0644/  HCM0674/
      train/images/            ảnh train
      sparse/0/                COLMAP: cameras.bin, images.bin, points3D.bin
      test_poses.csv           pose cần render để nộp (quaternion + intrinsics từng dòng)
```

Kiểm tra trước khi tốn GPU:

```bash
python tools/validate_colmap_structure.py --raw_root .
```

Phụ thuộc: `pip install -r requirements.txt`. Riêng **PyTorch** cài theo đúng
bản CUDA của máy, và **Nerfstudio phải cài từ GitHub** — bản PyPI mới nhất
(1.1.5) thiếu `strategy`/`max_gs_num` nên `STRATEGY=mcmc` sẽ bị bỏ qua âm thầm.
`train.sh` tự cài đúng commit đã verify; thêm `--no-install` để bỏ qua.

## Run

```bash
# Train toàn bộ scene
bash train.sh --raw-root . --data-root data/processed --iterations 30000

# Train một scene
bash train.sh --scene <tên data> --gpu <số hiệu GPU nếu có nhiều hơn 1 GPU> -> hỗ trợ train trên nhiều GPU ví như 2xT4 ở kaggle

# Xem toàn bộ tuỳ chọn + biến môi trường
bash train.sh --help
```

Lần chạy cuối nên đi qua `run_final.py` thay vì gõ từng lệnh — mỗi phase ở đó
tương ứng với một lỗi đã thật sự xảy ra và từng làm sai lệch kết quả:

```bash
python run_final.py --phase check     # kiểm tra môi trường, không cần GPU
python run_final.py --phase train
python run_final.py --phase eval      # chọn checkpoint + chốt bộ lọc
python run_final.py --phase render    # render + đóng gói + kiểm tra
python run_final.py --phase all
```

## Những mặc định cần biết

| Thiết lập | Giá trị | Lý do |
|---|---|---|
| `--downscale` | `1` | Ảnh train đã đúng độ phân giải `test_poses.csv` yêu cầu; downscale=2 làm mờ và bị LPIPS phạt nặng |
| JPEG quality | `95` | `img.save()` không tham số mặc định q75, vứt bỏ ~0.47 dB |
| `dataparser_transforms.json` | luôn áp | Nerfstudio tự xoay/căn/scale scene trước khi train; không áp lại thì camera lệch hẳn khỏi không gian Gaussian |
| `ply_file_path` | luôn ghi | Thiếu field này Splatfacto âm thầm fallback về random init 50k điểm |
| `MAX_GAUSS_RATIO` | `10.0` | Chỉ có hiệu lực khi bật `use-scale-regularization`. Hạ 4–6 cho scene indoor; giữ 10.0 cho BTS vì dây cáp/ăng-ten *cần* Gaussian dài |
| Số scene | đủ 5 | Thiếu 1 scene kéo điểm trung bình xuống ~0.144 — lớn hơn mọi cải tiến trong repo này cộng lại |

Tự kiểm chứng bằng số, đừng tin mặc định:

```bash
python pipeline/render_and_submit.py --sanity_check \
    --data_root . --processed_data_root data/processed \
    --checkpoints_root outputs
```
