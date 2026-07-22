"""Resumable HTTP downloads and llama.cpp release asset selection.
Pure stdlib (urllib) so no extra dependency is needed on either OS."""
import json
import os
import re
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile

LLAMACPP_LATEST_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# (backend, os_name) -> list of regexes to try in order against release asset
# names; the first asset matched wins. Verified against a real "latest"
# release listing - see docs/MODELS.md for the snapshot.
ASSET_PATTERNS = {
    ("cuda", "Windows"): [r"^llama-.*-bin-win-cuda-12\.\d+-x64\.zip$",
                          r"^llama-.*-bin-win-cuda-\d+.*-x64\.zip$"],
    ("vulkan", "Windows"): [r"^llama-.*-bin-win-vulkan-x64\.zip$"],
    ("rocm", "Windows"): [r"^llama-.*-bin-win-hip-radeon-x64\.zip$"],
    ("cpu", "Windows"): [r"^llama-.*-bin-win-cpu-x64\.zip$"],
    ("vulkan", "Linux"): [r"^llama-.*-bin-ubuntu-vulkan-x64\.tar\.gz$"],
    ("rocm", "Linux"): [r"^llama-.*-bin-ubuntu-rocm-[\d.]+-x64\.tar\.gz$"],
    ("cpu", "Linux"): [r"^llama-.*-bin-ubuntu-x64\.tar\.gz$"],
}
# CUDA runtime redistributable, only needed alongside the Windows CUDA build.
CUDART_PATTERNS = [r"^cudart-llama-bin-win-cuda-12\.\d+-x64\.zip$",
                    r"^cudart-llama-bin-win-cuda-\d+.*-x64\.zip$"]


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "opencode-local-installer"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def resolve_llamacpp_assets(backend, os_name):
    """Return dict {'binary': asset_dict, 'cudart': asset_dict|None} for the
    given backend/os, or raise RuntimeError if nothing matches."""
    if os_name not in ("Windows", "Linux"):
        raise RuntimeError(f"No prebuilt llama.cpp release for OS {os_name!r}. "
                            f"Build from source: https://github.com/ggml-org/llama.cpp")
    patterns = ASSET_PATTERNS.get((backend, os_name))
    if not patterns:
        raise RuntimeError(f"No known llama.cpp release asset for backend={backend} os={os_name}")

    release = _fetch_json(LLAMACPP_LATEST_RELEASE_API)
    assets = release.get("assets", [])

    def match_first(pats):
        for pat in pats:
            for a in assets:
                if re.match(pat, a["name"]):
                    return a
        return None

    binary = match_first(patterns)
    if not binary:
        raise RuntimeError(f"Could not find a llama.cpp release asset matching {patterns} "
                            f"in release {release.get('tag_name')}. "
                            f"Check https://github.com/ggml-org/llama.cpp/releases/latest")
    cudart = match_first(CUDART_PATTERNS) if backend == "cuda" else None
    return {"binary": binary, "cudart": cudart, "tag": release.get("tag_name")}


def download_file(url, dest_path, progress_cb=None, resume=True):
    """Download url to dest_path with Range-based resume. progress_cb(done_bytes,
    total_bytes) is called periodically if given. Raises on failure; caller
    decides whether to retry."""
    tmp_path = dest_path + ".part"
    existing = 0
    if resume and os.path.exists(tmp_path):
        existing = os.path.getsize(tmp_path)

    req = urllib.request.Request(url, headers={"User-Agent": "opencode-local-installer"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")

    with urllib.request.urlopen(req, timeout=60) as resp:
        status = getattr(resp, "status", 200)
        if existing and status != 206:
            # Server ignored our Range request - restart from scratch.
            existing = 0
            mode = "wb"
        else:
            mode = "ab" if existing else "wb"

        total = existing + int(resp.headers.get("Content-Length", 0) or 0)
        done = existing
        chunk_size = 1024 * 1024
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
        with open(tmp_path, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)

    os.replace(tmp_path, dest_path)


def _check_safe_path(name, dest_dir, dest_real):
    """Reject an archive entry whose name would escape dest_dir (zip-slip:
    '../' traversal or an absolute path)."""
    target = os.path.realpath(os.path.join(dest_dir, name))
    if not (target == dest_real or target.startswith(dest_real + os.sep)):
        raise RuntimeError(f"Refusing to extract unsafe archive entry: {name!r}")


def extract_archive(archive_path, dest_dir):
    """Extract a .zip or .tar.gz release asset into dest_dir. Validates every
    entry before extracting anything: rejects path traversal ('../', absolute
    paths) for both formats, and additionally rejects symlink/hardlink members
    in tar archives (a symlink entry can point outside dest_dir, then a later
    member written "through" it lands wherever the link points - checking
    each name's own path in isolation, as zip-slip protection normally does,
    doesn't catch that). A llama.cpp release archive has no legitimate reason
    to contain symlinks, so rejecting them outright costs nothing."""
    os.makedirs(dest_dir, exist_ok=True)
    dest_real = os.path.realpath(dest_dir)
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for name in zf.namelist():
                _check_safe_path(name, dest_dir, dest_real)
            zf.extractall(dest_dir)
    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path) as tf:
            members = tf.getmembers()
            for m in members:
                if m.issym() or m.islnk():
                    raise RuntimeError(f"Refusing to extract symlink/hardlink archive "
                                        f"entry: {m.name!r} -> {m.linkname!r}")
                _check_safe_path(m.name, dest_dir, dest_real)
            tf.extractall(dest_dir, members=members)
    else:
        raise RuntimeError(f"Don't know how to extract: {archive_path}")


def download_with_retries(url, dest_path, progress_cb=None, attempts=5):
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            download_file(url, dest_path, progress_cb=progress_cb, resume=True)
            return
        except (urllib.error.URLError, OSError) as e:
            last_err = e
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}\nLast error: {last_err}")
