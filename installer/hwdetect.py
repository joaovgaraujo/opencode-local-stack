"""Cross-platform hardware detection: OS, GPU vendor/VRAM, free/total RAM, free disk.

Stdlib only (no pip deps), so the installer stays a zero-dependency checkout.
Detection is best-effort: it shells out to vendor tools (nvidia-smi, rocm-smi)
and falls back to platform-native memory APIs. Every field can be None if it
couldn't be determined - callers must handle that (see catalog.fit_verdict).
"""
import ctypes
import platform
import re
import shutil
import subprocess


class Hardware:
    def __init__(self):
        self.os_name = platform.system()          # 'Windows' | 'Linux' | 'Darwin'
        self.gpu_vendor = None                    # 'nvidia' | 'amd' | 'apple' | 'other' | None
        self.gpu_name = None
        self.vram_total_gb = None
        self.vram_free_gb = None
        self.compute_cap = None                   # NVIDIA only
        self.ram_total_gb = None
        self.ram_free_gb = None                   # best-effort on macOS - see _detect_ram_macos
        self.disk_free_gb = None
        self.is_apple_silicon = False              # arm64 Mac (unified memory, no discrete VRAM)

    def summary_lines(self):
        lines = [f"OS: {self.os_name}"]
        if self.gpu_vendor == "apple":
            lines.append(f"GPU: {self.gpu_name or 'Apple Silicon'} (unified memory - see RAM below)")
        elif self.gpu_vendor:
            gpu = f"GPU: {self.gpu_name or self.gpu_vendor}"
            if self.vram_total_gb is not None:
                gpu += f"  ({self.vram_total_gb:.1f} GB VRAM"
                if self.vram_free_gb is not None:
                    gpu += f", {self.vram_free_gb:.1f} GB free"
                gpu += ")"
            lines.append(gpu)
        else:
            lines.append("GPU: none detected (CPU-only inference)")
        if self.ram_total_gb is not None:
            lines.append(f"RAM: {self.ram_total_gb:.1f} GB total, "
                          f"{self.ram_free_gb:.1f} GB free" if self.ram_free_gb is not None
                          else f"RAM: {self.ram_total_gb:.1f} GB total")
        if self.disk_free_gb is not None:
            lines.append(f"Disk free: {self.disk_free_gb:.1f} GB")
        return lines


def _run(args):
    try:
        # encoding/errors explicit: plain text=True decodes with the console's
        # active code page on Windows, which can crash on non-ASCII bytes in
        # a GPU name or locale-dependent tool output.
        return subprocess.run(args, capture_output=True, text=True, timeout=15,
                               encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None


def _detect_nvidia(hw):
    r = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap",
              "--format=csv,noheader,nounits"])
    if not r or r.returncode != 0 or not r.stdout.strip():
        return False
    # First GPU line if multiple are present.
    line = r.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return False
    hw.gpu_vendor = "nvidia"
    hw.gpu_name = parts[0]
    try:
        hw.vram_total_gb = float(parts[1]) / 1024
        hw.vram_free_gb = float(parts[2]) / 1024
    except ValueError:
        pass
    hw.compute_cap = parts[3]
    return True


def _detect_amd(hw):
    r = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
    if not r or r.returncode != 0 or not r.stdout.strip():
        return False
    hw.gpu_vendor = "amd"
    name_match = re.search(r"Card series:\s*(.+)", r.stdout) or re.search(r"GPU\[0\].*?:\s*(.+)", r.stdout)
    hw.gpu_name = name_match.group(1).strip() if name_match else "AMD GPU (ROCm)"
    total_match = re.search(r"VRAM Total Memory \(B\).*?(\d+)", r.stdout)
    used_match = re.search(r"VRAM Total Used Memory \(B\).*?(\d+)", r.stdout)
    if total_match:
        total_bytes = int(total_match.group(1))
        hw.vram_total_gb = total_bytes / 1e9
        if used_match:
            hw.vram_free_gb = max(0.0, hw.vram_total_gb - int(used_match.group(1)) / 1e9)
    return True


def _detect_gpu_windows_generic(hw):
    """Fallback when neither nvidia-smi nor rocm-smi are present: ask Windows
    for a video controller name via CIM/WMI so we can at least flag 'GPU
    present, vendor unknown' and steer toward the Vulkan build."""
    r = _run(["powershell", "-NoProfile", "-Command",
              "(Get-CimInstance Win32_VideoController | "
              "Select-Object -First 1 -ExpandProperty Name)"])
    if r and r.returncode == 0 and r.stdout.strip():
        hw.gpu_vendor = "other"
        hw.gpu_name = r.stdout.strip()
        return True
    return False


def _detect_gpu_linux_generic(hw):
    r = _run(["lspci"])
    if not r or r.returncode != 0:
        return False
    for line in r.stdout.splitlines():
        if re.search(r"VGA compatible controller|3D controller", line, re.I):
            hw.gpu_vendor = "other"
            hw.gpu_name = line.split(":", 2)[-1].strip()
            return True
    return False


def _detect_ram_windows(hw):
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return
    hw.ram_total_gb = stat.ullTotalPhys / (1024 ** 3)
    hw.ram_free_gb = stat.ullAvailPhys / (1024 ** 3)


def _detect_ram_linux(hw):
    try:
        with open("/proc/meminfo") as f:
            info = f.read()
    except OSError:
        return
    total_match = re.search(r"MemTotal:\s*(\d+)\s*kB", info)
    avail_match = re.search(r"MemAvailable:\s*(\d+)\s*kB", info)
    if total_match:
        hw.ram_total_gb = int(total_match.group(1)) / (1024 ** 2)
    if avail_match:
        hw.ram_free_gb = int(avail_match.group(1)) / (1024 ** 2)


def macos_available_ram_gb():
    """A fresh vm_stat-based read of unified memory that could be handed to the
    GPU right now (free + inactive + speculative + purgeable pages), in GB, or
    None off macOS / on parse failure.

    detect() caches ram_free_gb once at startup; a memory preflight run just
    before `rapid-mlx serve` needs the value *now* (a large model may have been
    downloaded in the meantime, or another app may have grabbed memory), so this
    re-samples instead of reusing the cached figure."""
    if platform.system() != "Darwin":
        return None
    r = _run(["vm_stat"])
    if not (r and r.returncode == 0):
        return None
    page_size_match = re.search(r"page size of (\d+) bytes", r.stdout)
    page_size = int(page_size_match.group(1)) if page_size_match else 4096
    pages = 0
    found = False
    for label in ("Pages free", "Pages inactive", "Pages speculative",
                  "Pages purgeable"):
        m = re.search(rf"{label}:\s*(\d+)", r.stdout)
        if m:
            pages += int(m.group(1))
            found = True
    if not found:
        return None
    return pages * page_size / (1024 ** 3)


def _detect_macos(hw):
    """Apple Silicon uses unified memory - there's no discrete VRAM, RAM *is*
    the GPU-accessible pool (see catalog.py's mlx_fit_verdict for how that
    changes the fit estimate vs the discrete-GPU llama.cpp path)."""
    hw.is_apple_silicon = (platform.machine() == "arm64")
    if hw.is_apple_silicon:
        hw.gpu_vendor = "apple"
        r = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        hw.gpu_name = r.stdout.strip() if r and r.returncode == 0 and r.stdout.strip() else "Apple Silicon"
    else:
        # Intel Mac: has a discrete/integrated non-Apple GPU. rapid-mlx (MLX)
        # requires Apple Silicon, so this machine isn't a rapidmlx target -
        # see hwdetect.pick_engine.
        hw.gpu_vendor = "other"
        hw.gpu_name = "Intel Mac (non-Apple-Silicon GPU)"

    r = _run(["sysctl", "-n", "hw.memsize"])
    if r and r.returncode == 0 and r.stdout.strip().isdigit():
        hw.ram_total_gb = int(r.stdout.strip()) / (1024 ** 3)

    # macOS has no direct "free RAM" equivalent (aggressive disk-cache/compression
    # make MemAvailable-style figures meaningless) - vm_stat's free+inactive page
    # counts are the closest approximation, treated as a lower bound, not exact.
    r = _run(["vm_stat"])
    if r and r.returncode == 0:
        page_size_match = re.search(r"page size of (\d+) bytes", r.stdout)
        page_size = int(page_size_match.group(1)) if page_size_match else 4096
        free_match = re.search(r"Pages free:\s*(\d+)", r.stdout)
        inactive_match = re.search(r"Pages inactive:\s*(\d+)", r.stdout)
        if free_match and inactive_match:
            pages = int(free_match.group(1)) + int(inactive_match.group(1))
            hw.ram_free_gb = pages * page_size / (1024 ** 3)


def detect(root_path="."):
    hw = Hardware()

    if hw.os_name == "Windows":
        if not _detect_nvidia(hw):
            _detect_gpu_windows_generic(hw)
        _detect_ram_windows(hw)
    elif hw.os_name == "Darwin":
        _detect_macos(hw)
    else:
        if not _detect_nvidia(hw):
            if not _detect_amd(hw):
                _detect_gpu_linux_generic(hw)
        _detect_ram_linux(hw)

    try:
        hw.disk_free_gb = shutil.disk_usage(root_path).free / (1024 ** 3)
    except OSError:
        pass

    return hw


def pick_engine(hw):
    """Return 'rapidmlx' (Apple Silicon Mac) or 'llamacpp' (everything else).
    rapid-mlx requires Apple Silicon (M1+) - an Intel Mac falls back to
    'llamacpp', which will then hit pick_llamacpp_backend's unsupported-OS
    error (no prebuilt llama.cpp macOS-x64 handling is wired up here; only
    the arm64 rapidmlx path is - see install.py)."""
    if hw.os_name == "Darwin" and hw.is_apple_silicon:
        return "rapidmlx"
    return "llamacpp"


def pick_llamacpp_backend(hw):
    """Return one of 'cuda', 'vulkan', 'rocm', 'cpu' - the backend the
    installer should fetch a prebuilt llama.cpp release for. Linux+NVIDIA
    intentionally resolves to 'vulkan': llama.cpp does not publish a
    prebuilt Linux CUDA binary (only Windows), and building CUDA from
    source is not something this installer automates - see docs/DEPLOY.md."""
    if hw.gpu_vendor == "nvidia":
        return "cuda" if hw.os_name == "Windows" else "vulkan"
    if hw.gpu_vendor == "amd":
        return "rocm"
    if hw.gpu_vendor == "other":
        return "vulkan"
    return "cpu"
