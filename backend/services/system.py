import os
import platform
import subprocess
from pathlib import Path

def _nvidia_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader"
        ], stderr=subprocess.STDOUT, shell=True, text=True)
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpus.append({"name": parts[0], "memory_total": parts[1]})
        return gpus
    except Exception:
        return []

def _openvino_devices():
    try:
        from openvino.runtime import Core
        core = Core()
        return core.available_devices
    except Exception:
        return []

def get_info():
    devices = _openvino_devices()
    nvidia = _nvidia_info()
    accelerators = []
    has_npu = any(d.startswith("NPU") for d in devices)
    has_gpu = any(d.startswith("GPU") for d in devices)
    has_cpu = "CPU" in devices
    if has_npu:
        accelerators.append({"id": "NPU", "label": "Intel NPU"})
    if has_gpu:
        accelerators.append({"id": "GPU", "label": "Intel GPU"})
    if has_cpu:
        accelerators.append({"id": "CPU", "label": "CPU"})
    if nvidia:
        accelerators.append({"id": "NVIDIA", "label": "NVIDIA GPU"})
    # cooperative acceleration options
    combos = []
    if has_npu and has_gpu:
        combos.append({"id": "MULTI:NPU,GPU", "label": "Intel NPU+GPU (协同)"})
    if has_npu and has_cpu:
        combos.append({"id": "MULTI:NPU,CPU", "label": "Intel NPU+CPU (协同)"})
    if has_npu and has_gpu and has_cpu:
        combos.append({"id": "MULTI:NPU,GPU,CPU", "label": "Intel NPU+GPU+CPU (协同)"})
    accelerators = combos + accelerators
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "openvino_devices": devices,
        "nvidia_gpus": nvidia,
        "accelerators": accelerators,
        "cwd": str(Path.cwd()),
    }