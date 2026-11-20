Linux and macOS get the same one-step install as Windows, and Linux with an NVIDIA GPU gets llama.cpp's CUDA build, as a download or compiled from source.

## Install

- **Windows:** download **OpenCodeLocal-Setup.exe** below and run it.
- **Linux and macOS:**
  ```
  curl -fsSL https://github.com/joaovgaraujo/opencode-local-stack/releases/latest/download/opencode-local-installer.sh | bash
  ```
  This installs into `~/opencode-local` and asks the same questions as on Windows: model, context and thinking. Nothing uses sudo. Afterwards run `opencode-local` in a project folder.

## Linux

- **NVIDIA GPUs use CUDA.** The installer downloads llama.cpp's prebuilt CUDA build that matches the CUDA version your driver supports (12.8 or 13.5) and puts the CUDA runtime next to `llama-server`. The prebuilt needs Ubuntu 24.04 or newer (glibc 2.38).
- **Building from source.** Choose it in the wizard or pass `--build-llamacpp` to compile llama.cpp with CUDA instead: 5 to 20 minutes, using the newest CUDA toolkit installed and building for your GPU only. On older systems where the prebuilt can't run, the installer builds from source automatically when git, cmake, a compiler and the CUDA toolkit are present.
- **GPU check.** After installing llama.cpp, the installer checks that it starts and sees the GPU.
- **No admin rights needed.** Node.js comes from nodejs.org into the install folder, and OpenCode goes into `~/.local` when the system's npm folder needs root.

## macOS (Apple Silicon)

- **rapid-mlx 1.0.5** (was 0.10.15). It needs Python 3.12 or 3.13: the installer uses one the Mac has, or gets a private one with `uv`, so neither Homebrew nor admin rights are needed.
- **Node.js** comes from nodejs.org when Homebrew isn't installed.
- **`opencode-local` works on Macs**, with the same thinking choices (off sets rapid-mlx's reasoning effort to none) and the 10-minute idle unload.
- **Memory check.** It compares against the Mac's total memory. It used to stop the install when *free* memory was low, even though macOS frees cached memory when the model loads.

## Tested on real hardware

| Machine | Setup | Result |
|---|---|---|
| Ubuntu 26.04, RTX 3500 Ada laptop (12 GB), 61 GB RAM | Qwen3.6-35B-A3B uncensored (HauhauCS) from its Hugging Face link, 64K context, prebuilt CUDA 12.8 | all tests and the OpenCode agentic test pass; 45 tok/s generation, 1,746 tok/s prefill |
| same machine | the same, llama.cpp compiled from source with CUDA 13.3 | all tests pass; 43-48 tok/s |
| Mac mini M4, 16 GB, macOS 27 | Qwen3.5-9B 4-bit (MLX), rapid-mlx 1.0.5 | all tests and the OpenCode agentic test pass |
| Windows 11, RTX 4090 | existing install updated to this version | `opencode-local` works unchanged |

## Fixes found by those tests

- **Source build mixing CUDA versions.** Without `CUDAToolkit_ROOT`, CMake found CUDA 13 headers and compiler but linked Ubuntu's CUDA 12 libraries. The GPU then reported 55 GB free on a 12 GB card and crashed on the first request. The build now uses one toolkit and a fresh build folder.
- **Source build failing on the web UI download.** A shallow git clone made llama.cpp ask for the web UI of release "b1".
- **Bad free-memory figures.** If a GPU reports more free memory than it has, `--fit` is now capped at the card's real size.
- **Brand-new Python.** Python 3.15 is skipped for rapid-mlx because tiktoken has no build for it yet.
