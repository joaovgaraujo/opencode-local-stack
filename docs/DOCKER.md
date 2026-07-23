# Docker: llama.cpp CUDA + TurboQuant KV cache

A containerized version of the Linux CUDA + TurboQuant configuration
([`docs/TURBOQUANT.md`](TURBOQUANT.md)). No published image existed for the
selected fork, so this project builds its own from a fork under the user's
account.

## Where things live

- **Source repo (fork):** <https://github.com/joaovgaraujo/llama-cpp-turboquant>,
  branch `feature/turboquant-kv-cache` — a GitHub fork of the reviewed
  `TheTom/llama-cpp-turboquant` at revision `c26cbdf`, plus:
  - `.github/workflows/docker-ghcr.yml` — builds the `server` target of
    `.devops/cuda.Dockerfile` and pushes to GHCR
    (CUDA archs `80;86;89;90;120`, `linux/amd64`).
  - `.devops/cuda.Dockerfile` — gained an `EXTRA_CMAKE_ARGS` build-arg; the
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

Point OpenCode at it exactly as with the bare-metal server — `opencode.json`
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

## KV-cache type: use q8_0 for agentic work; TurboQuant V-cache is not safe there

Measured on this machine, 2026-07-23, CUDA fork build `c26cbdf`, Qwen3.6-35B
Q4_K_M at 65,536 context:

- `q8_0`/`turbo3`: full llama-bench speed, but **failed the code-gen
  validation twice in a row** (parseable output, 0 functions, no docstring).
- `q8_0`/`turbo4`: passed all four one-shot validations (33-39 tok/s, slightly
  faster than the q8_0 baseline's 30-34), but **the OpenCode agentic smoke
  test spiraled past its 360 s timeout 4 out of 4 attempts** — the reasoning
  model decoded 6,000+ thinking tokens per request without converging.
- `q8_0`/`q8_0`: agentic smoke test **passed in 48 s**.

Earlier results had only ever throughput-benchmarked the turbo V-cache types
and functionally validated turbo4 with one-shot tests; none of that catches
the agentic/reasoning degradation. The VRAM saving from turbo V was modest
here anyway (~150-200 MiB at 60k context, K dominates). Keep `q8_0`/`q8_0`
for OpenCode; consider `turbo4` only for non-agentic long-context serving.

[NVIDIA container toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
