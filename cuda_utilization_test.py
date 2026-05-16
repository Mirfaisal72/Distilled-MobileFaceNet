import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("------------------")
import time
import subprocess
import threading
import sys

# Adjustable params
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# default size: 4096 -> matrix of shape (4096, 4096). Increase if you have more VRAM.
SIZE = 4096
# seconds to run each heavy operation
ITER_SECONDS = 0.8
# how many heavy ops to run (None for infinite)
ITERATIONS = 30

def nvidia_smi_read():
    """Return (util_percent, mem_used_mb, mem_total_mb) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL
        ).decode().strip()
        # Example out: "12 %, 2048 MiB, 4096 MiB" or "12, 2048, 4096"
        parts = [p.strip().replace(" %", "") for p in out.split(",")]
        if len(parts) >= 3:
            util = int(parts[0])
            used = int(parts[1])
            total = int(parts[2])
            return util, used, total
        else:
            return None
    except Exception:
        return None

def monitor_loop(stop_event, interval=1.0):
    """Print GPU stats every `interval` seconds until stop_event is set."""
    while not stop_event.is_set():
        smi = nvidia_smi_read()
        if smi:
            util, used, total = smi
            # torch memory stats (for current process)
            allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            reserved  = torch.cuda.memory_reserved()  if torch.cuda.is_available() else 0
            print(f"[GPU] util%={util} | driver-MiB used={used}/{total} | "
                  f"torch_alloc={allocated//1024**2}MiB reserved={reserved//1024**2}MiB")
        else:
            print("[GPU] nvidia-smi failed or not installed; torch.cuda.is_available() =", torch.cuda.is_available())
        time.sleep(interval)

def heavy_compute_loop(size, seconds_per_iter=0.8, iterations=None):
    """Perform heavy matrix multiplies creating load on GPU."""
    dtype = torch.float32
    print(f"Creating matrices of shape ({size}, {size}) on {DEVICE} ...")
    # Allocate two matrices and keep them on GPU
    a = torch.randn((size, size), device=DEVICE, dtype=dtype)
    b = torch.randn((size, size), device=DEVICE, dtype=dtype)
    # warmup
    print("Warmup matmul ...")
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    iter_count = 0
    try:
        while iterations is None or iter_count < iterations:
            t0 = time.time()
            # run heavy ops for roughly `seconds_per_iter` seconds
            # we repeatedly do matmul until time is up to create sustained load
            ops = 0
            while time.time() - t0 < seconds_per_iter:
                c = torch.matmul(a, b)            # heavy kernel
                # optionally do a small elementwise op to avoid lazy optimization
                c = c * 1.000001
                ops += 1
            # force synchronization so host sees accurate timing and GPU work completed
            torch.cuda.synchronize()
            iter_count += 1
            print(f"[Work] Iter {iter_count} completed: {ops} matmuls in {seconds_per_iter:.2f}s")
            # keep `c` around for one loop to occupy some memory then delete
            del c
            # small pause to let monitor show changes
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # free memory
        del a, b
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Finished heavy compute loop, freed GPU memory.")

if __name__ == "__main__":
    if DEVICE == "cpu":
        print("CUDA not available. This script requires a CUDA-enabled PyTorch and a visible GPU.")
        sys.exit(1)

    print("Using device:", DEVICE)
    print("PyTorch built for CUDA:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0))
    # start monitor thread
    stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_loop, args=(stop_event, 1.0), daemon=True)
    monitor.start()

    try:
        heavy_compute_loop(SIZE, seconds_per_iter=ITER_SECONDS, iterations=ITERATIONS)
    finally:
        stop_event.set()
        monitor.join(timeout=2.0)
        # final snapshot
        final = nvidia_smi_read()
        if final:
            util, used, total = final
            print(f"[FINAL GPU] util%={util} | driver-MiB used={used}/{total}")
        print("Script complete.")
