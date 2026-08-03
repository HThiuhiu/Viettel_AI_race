
import subprocess
ADC_ONLY_VARS = {"CULL_ALPHA_THRESH", "CULL_SCALE_THRESH", "DENSIFY_GRAD_THRESH",
                  "STOP_SPLIT_AT", "STOP_SPLIT_RATIO"}


def run_scene(work_dir: str, data_processed: str, scene: str, iterations: int, gpu_id: int = 0,
              downscale: int = 1, env_overrides: dict = None, log_dir: str = "/content",
              pytorch_cuda_alloc_conf: str = "expandable_segments:True", background: bool = False):
    """background=True: chay train o TIEN TRINH NEN, ham tra lai NGAY LAP TUC
    (khong cho train xong) - cell Colab goi ham nay se chay xong tuc thi, cho
    phep chay cell khac (vd tail log, kiem tra nvidia-smi) trong luc train van
    dang tiep dien ben duoi. Log LUON duoc ghi ra file (ca 2 che do) de cell
    khac co the doc duoc bat ke che do nao.

    Tra ve:
        (Popen, log_path) neu background=True - dung proc.poll() (None = con
            chay) hoac proc.wait() de cho neu can, proc.returncode sau khi xong.
        (returncode:int, log_path) neu background=False - hanh vi cu, cho den
            khi train xong moi tra quyen dieu khien lai cho cell.
    """
    env_overrides = dict(env_overrides or {})

    strategy = env_overrides.get("STRATEGY", "mcmc")
    if strategy == "mcmc":
        ignored = ADC_ONLY_VARS & env_overrides.keys()
        if ignored:
            print(f"[CANH BAO] STRATEGY dang la '{strategy}' - cac bien sau se BI BO QUA am tham "
                  f"vi chi ap dung cho ADC (STRATEGY=default): {sorted(ignored)}")

    if pytorch_cuda_alloc_conf:
        # Ten dung la PYTORCH_CUDA_ALLOC_CONF (khong phai PYTORCH_ALLOC_CONF) -
        # sai ten se bi PyTorch am tham bo qua, khong bao loi.
        env_overrides.setdefault("PYTORCH_CUDA_ALLOC_CONF", pytorch_cuda_alloc_conf)

    env_str = " ".join(f"{k}={v}" for k, v in env_overrides.items())
    log_path = f"{log_dir}/train_GPU{gpu_id}_{scene}_log.txt"

    cmd = (
        f"{env_str} "
        f"bash {work_dir}/train.sh "
        f"--raw-root {work_dir} --data-root {data_processed} "
        f"--scene {scene} --iterations {iterations} --downscale {downscale} --gpu {gpu_id} "
        f"2>&1 | tee {log_path}"
    )
    print(f"[run_scene] Lenh se chay ({'NEN, khong cho' if background else 'CHO den khi xong'}):\n{cmd}\n")

    if background:
        # start_new_session=True (~ setsid): tach tien trinh khoi session cua
        # notebook, giong pattern "nohup ... &" - train tiep tuc chay du cell
        # goc da "xong". stdout/stderr cua Popen tu than khong can giu (tee o
        # trong cmd da tu ghi ra log_path doc lap voi viec Popen doc no hay
        # khong), nen dua ve DEVNULL de tranh treo neu buffer day.
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        print(f"[run_scene] Da chay NEN (pid={proc.pid}). Cell nay da xong - chay duoc cell khac ngay.")
        print(f"[run_scene] Log dang duoc ghi tai: {log_path}")
        print(f"[run_scene] Xem log tu 1 cell KHAC: !tail -n 50 {log_path}")
        print(f"[run_scene] Theo doi lien tuc tu 1 cell KHAC: !tail -f {log_path}")
        print(f"[run_scene] Kiem tra con chay khong: proc.poll() (None = dang chay)")
        return proc, log_path

    result = subprocess.run(cmd, shell=True)
    print(f"[run_scene] Log da luu tai: {log_path}")
    if result.returncode != 0:
        print(f"[CANH BAO] Lenh thoat voi ma loi {result.returncode} - kiem tra log ben tren.")
    return result.returncode, log_path


if __name__ == "__main__":
    run_scene(
        work_dir="/content/Viettel",
        data_processed="/content/Viettel/data/processed",
        scene="HCM0540",
        iterations=20000,
        gpu_id=0,
        env_overrides={
            "SSIM_LAMBDA": "0.35",
            "MCMC_SCALE_REG": "0.03",
            "USE_BILATERAL_GRID": "False",
            "STEPS_PER_EVAL_ALL_IMAGES": "5000",
            "RASTERIZE_MODE": "classic",
        },
    )
