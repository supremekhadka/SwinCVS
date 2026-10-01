"""Background sampler for GPU-only memory and power draw during inference.

Shared (copied) across the cvs, CVS-AdaptNet, SwinCVS and endoscapes repos so
all of them report resources the same way; see ResourceMonitor.
"""

import re
import shutil
import subprocess
import threading
import time

import torch

try:
    import psutil
except ImportError:
    psutil = None


class ResourceMonitor:
    """Background sampler for GPU-only memory and power draw during a run.

    Memory is always read from CUDA's own allocator counters
    (torch.cuda.memory_allocated / max_memory_allocated) on both backends,
    so it reflects only what this process actually holds on the GPU, not
    system-wide RAM — including on Jetson's unified memory, where the GPU
    and CPU share physical DRAM but CUDA allocations are still tracked
    separately by the allocator. On a CPU-only run, memory falls back to
    the process RSS via psutil (there is no GPU to isolate).

    Power backend is auto-detected:
    - tegrastats (Jetson, incl. Orin Nano / JetPack 7.2): Jetson has no
      NVML-backed nvidia-smi. Orin Nano's tegrastats exposes no rail for
      the GPU alone — VDD_IN is total module power (CPU+GPU+everything
      else) and VDD_CPU_GPU_CV lumps the GPU in with the CPU and the deep
      learning accelerator cores. VDD_CPU_GPU_CV is used here as the
      closest available proxy to GPU-only power; some carrier boards may
      not expose it, in which case power stays empty.
    - nvidia-smi (desktop/server GPUs): power comes from nvidia-smi's
      power.draw, which is genuinely GPU-only (the discrete card's own
      power rail).

    If neither binary is on PATH, power samples stay empty.
    """

    _TEGRASTATS_POWER_RE = re.compile(r"VDD_CPU_GPU_CV (\d+)mW")

    def __init__(self, device: torch.device, interval: float = 0.2):
        self.device = device
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None
        self.samples = []  # list of {"time_s", "memory_mb", "power_w"}
        self._backend = "tegrastats" if shutil.which("tegrastats") else ("nvidia-smi" if shutil.which("nvidia-smi") else None)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _sample_tegrastats_power(self):
        try:
            proc = subprocess.Popen(
                ("tegrastats", "--interval", "100"),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            try:
                out = proc.stdout.readline()
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            return None
        power_match = self._TEGRASTATS_POWER_RE.search(out)
        return int(power_match.group(1)) / 1000.0 if power_match else None

    def _sample_nvidia_smi_power(self):
        try:
            out = subprocess.check_output(
                ("nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"),
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            )
            return float(out.strip().splitlines()[0])
        except Exception:
            return None

    def _sample_memory_mb(self):
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        if psutil is not None:
            return psutil.Process().memory_info().rss / (1024 ** 2)
        return None

    def _run(self):
        while not self._stop_event.is_set():
            mem_mb = self._sample_memory_mb()
            if self._backend == "tegrastats":
                power_w = self._sample_tegrastats_power()
            elif self._backend == "nvidia-smi":
                power_w = self._sample_nvidia_smi_power()
            else:
                power_w = None
            self.samples.append({
                "time_s": time.perf_counter() - self._start_time,
                "memory_mb": mem_mb,
                "power_w": power_w,
            })
            self._stop_event.wait(self.interval)

    def start(self):
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def peak_memory_mb(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        memory_samples = [s["memory_mb"] for s in self.samples if s["memory_mb"] is not None]
        return max(memory_samples) if memory_samples else None

    def peak_power_w(self):
        power_samples = [s["power_w"] for s in self.samples if s["power_w"] is not None]
        return max(power_samples) if power_samples else None
