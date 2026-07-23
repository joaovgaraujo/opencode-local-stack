# Docker: llama.cpp CUDA + TurboQuant KV cache

A containerized version of the Linux CUDA + TurboQuant configuration
([`docs/TURBOQUANT.md`](TURBOQUANT.md)). No published image existed for the
selected fork, so this project builds its own from a fork under the user's
account.

## Where things live

- **Source repo (fork):** <https://github.com/joaovgaraujo/llama-cpp-turboquant>,
  branch `feature/turboquant-kv-cache`, a GitHub fork of the reviewed
  `TheTom/llama-cpp-turboquant` at revision `c26cbdf`, plus:
  - `.github/workflows/docker-ghcr.yml`: builds the `server` target of
    `.devops/cuda.Dockerfile` and pushes to GHCR
    (CUDA archs `80;86;89;90;120`, `linux/amd64`).
  - `.devops/cuda.Dockerfile`: gained an `EXTRA_CMAKE_ARGS` build-arg; the
    workflow passes `-DLLAMA_BUILD_UI=OFF`, matching the reviewed bare-metal
    build and avoiding the fork's fragile web-UI asset download/embed step
    (this stack uses OpenCode against the API, not the web UI).
- **Image:** `ghcr.io/joaovgaraujo/llama-cpp-turboquant:server-cuda`
  (plus a `server-cuda-<commit>` tag per build).

## Run it

Host requirements: NVIDIA driver + [NVIDIA container toolkit], Docker with the
`nvidia` runtime, and a GGUF in `./models/` (the installer's download works
fine; the container only mounts the directory read-only).

```bash
docker compose up -d          # uses ./docker-compose.yml
curl http://127.0.0.1:8080/health
```

The compose file mirrors the validated bare-metal configuration: Qwen3.6-35B
Q4_K_M, 65,536 context, flash attention, all MoE experts on CPU
(`--cpu-moe`), KV cache `q8_0`/`q8_0`.

Point OpenCode at it exactly as with the bare-metal server. `opencode.json`
does not care whether the endpoint is a container.

## Building locally instead

```bash
docker build -f vendor/thetom-llama-cpp-turboquant/.devops/cuda.Dockerfile \
  --target server \
  --build-arg CUDA_DOCKER_ARCH=89 \
  --build-arg EXTRA_CMAKE_ARGS=-DLLAMA_BUILD_UI=OFF \
  -t llama-turboquant:server-cuda-local \
  vendor/thetom-llama-cpp-turboquant
```

`CUDA_DOCKER_ARCH=89` (Ada) keeps the local build fast; the published image
covers `80;86;89;90;120`.

## KV-cache type: use q8_0; the fork's turbo4-V CUDA decode is broken

Measured + root-caused on this machine, 2026-07-23, CUDA fork build
`c26cbdf`, Qwen3.6-35B Q4_K_M at 65,536 context:

- `q8_0`/`turbo4` passed one-shot validations and even the 30k needle, but
  failed the OpenCode agentic smoke test **7 out of 7 attempts** (reasoning
  spirals, hallucinated paths, truncated commands). `q8_0`/`q8_0` passed the
  same test repeatedly in under a minute.
- Root cause is **not quantization quality**: fork commit `77ab7e988` routes
  turbo4-V through a miscomputing "wide-V" flash-attention decode path
  (`ggml/src/ggml-cuda/fattn-vec.cuh`), corrupting attention output on every
  decode. Greedy logit probes diverge from the q8_0 reference at the second
  token, at every context length, and 3-bit turbo3 (which kept the old code
  path) tracks q8_0 far more closely than 4-bit turbo4 does. This matches the
  fork's open issue #207; a one-line fix (restoring `TURBO4_0` to the
  4-rows-per-thread branch) took the agentic test from 0/7 to 3/3.
- One-shot corruption symptoms differ by type: `turbo3`'s code-gen failures
  (0 functions, 2/2) are consistent with genuine 3-bit quality loss.
- Even with the fix, turbo4-V saves only ~175 MiB at 64k context on this
  model (hybrid attention: 10 of 50 layers carry KV) with no speed gain.

Keep `q8_0`/`q8_0`. Do not enable `turbo4` on stock fork builds in any
configuration (single-slot included, the kernel bug fires on every decode).

[NVIDIA container toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
