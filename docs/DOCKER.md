# Docker: llama.cpp CUDA + TurboQuant KV cache

A containerized version of the Linux CUDA + TurboQuant configuration
([`docs/TURBOQUANT.md`](TURBOQUANT.md)). No published image existed for the
selected fork, so this project builds its own from a fork under the user's
account.

## Where things live

- **Source repo (fork):** <https://github.com/joaovgaraujo/llama-cpp-turboquant>,
  branch `feature/turboquant-kv-cache`, a GitHub fork of the reviewed
  `TheTom/llama-cpp-turboquant` at revision `c26cbdf`, plus:
  - The **turbo4 V-cache decode fix** (#207) merged in - so the image serves
    `--cache-type-v turbo4` correctly (see the KV section below).
  - `.github/workflows/docker-ghcr.yml`: builds the `server` target of
    `.devops/cuda.Dockerfile` and pushes to GHCR
    (CUDA archs `80;86;89;90;120`, `linux/amd64`).
  - `.devops/cuda.Dockerfile`: base image bumped to **nvidia/cuda 13.3.0 on
    Ubuntu 26.04** (GeForce Ada `sm_89`; matches the toolkit the fork was
    validated with on bare metal). Gained an `EXTRA_CMAKE_ARGS` build-arg; the
    workflow passes `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`,
    matching the reviewed bare-metal build and avoiding the fork's fragile
    web-UI asset download/embed step (this stack uses OpenCode against the
    API, not the web UI).
- **Image:** `ghcr.io/joaovgaraujo/llama-cpp-turboquant:server-cuda`
  (plus a `server-cuda-<commit>` tag per build). ~6.3 GB; CUDA 13.3.29 runtime.

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
  --build-arg "EXTRA_CMAKE_ARGS=-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF" \
  -t llama-turboquant:server-cuda-local \
  vendor/thetom-llama-cpp-turboquant
```

Both UI flags are required: `LLAMA_BUILD_UI=OFF` skips the npm build and
`LLAMA_USE_PREBUILT_UI=OFF` skips the Hugging Face asset download, which
otherwise fails the build in a clean container.

`CUDA_DOCKER_ARCH=89` (Ada) keeps the local build fast; the published image
covers `80;86;89;90;120`.

## KV-cache type: this image includes the turbo4 fix

This image carries the turbo4 V-cache decode fix, so all three KV modes behave
correctly. The published image passed the OpenCode agentic test with
`--cache-type-v turbo4` in **47 s**.

Which KV type to use:
- **`q8_0` (compose default): best decode speed.** turbo4 is never faster - a
  touch slower at depth - so keep q8_0 unless you specifically need more
  context.
- **`turbo4`: only for maximum context on a small card.** Its sole benefit is
  a smaller KV footprint. Measured on an 8 GB budget it doubled reachable
  context for some models (Qwen3.5-4B 131k -> 262k, Gemma 26B 131k -> 262k on
  CUDA) and freed ~0.5-1.6 GB on models already at their 262k limit.
- **`turbo3`: avoid for coding.** Separate from the decode bug, 3-bit turbo3
  has genuine quality loss (failed code-gen validation 2/2).

Background: the stock fork (`c26cbdf`) shipped a CUDA bug - commit `77ab7e988`
routed turbo4-V through a miscomputing "wide-V" flash-attention decode path
(`ggml/src/ggml-cuda/fattn-vec.cuh`), corrupting attention output on every
decode. On the buggy build `q8_0`/`turbo4` passed one-shot validations and the
30k needle but failed the OpenCode agentic test **7/7** (reasoning spirals,
hallucinated paths); greedy logit probes diverged from the q8_0 reference at
the second decoded token, at every context length. This matched the fork's
open issue #207. The one-line fix (restoring `TURBO4_0` to the
4-rows-per-thread branch) is merged into the image branch. **If you build from
stock `c26cbdf` yourself instead of using this image, do not enable `turbo4`.**

[NVIDIA container toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
