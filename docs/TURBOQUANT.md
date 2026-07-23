# TurboQuant (experimental)

TurboQuant (Google Research, ICLR 2026) applies a random Hadamard rotation
before scalar quantization, giving noticeably better quality-per-bit than
standard GGUF quants at very low bit widths — appealing for a 35B MoE model
you're trying to squeeze onto an 8 GB card.

**It is not part of official llama.cpp.** Support exists only as independent
community forks, none merged upstream as of when this doc was written. That
matters for two reasons:

1. **Format**: standard llama.cpp builds cannot load a TurboQuant GGUF at all —
   they don't recognize the quant type.
2. **Binaries**: a few forks publish prebuilt CUDA binaries, but they're
   compiled by individual community members, not the ggml-org project.
   `install.py` deliberately **does not download or execute** any of them —
   downloading and silently running a third-party compiled binary with GPU
   driver access is a real supply-chain risk, and picking one fork over
   another isn't a call this installer should make for you.

## What `install.py` does for you

TurboQuant is opt-in and requires an explicit, reviewed runtime and its real
backend. Before it downloads weights or starts anything, `install.py` verifies
that `--backend` is not `auto`, `--bin-dir` was supplied, and the directory
actually contains `llama-server`. It never substitutes stock llama.cpp and
never downloads a community binary. This prevents an invalid TQ3_1S model from
falling through to a runtime that cannot load it or being mislabeled as the
platform's automatic backend.

## Selected implementation and Linux CUDA build

After reviewing four community implementations, this project selected
[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)
for the current experiment because it implements the catalog's exact
`TQ3_1S` weight type and includes Linux CUDA kernels. Revision `c26cbdf` on
`feature/turboquant-kv-cache` was built locally from source, bare-metal, with
CUDA 13.3 for NVIDIA the GPU arch. That local build result does **not** make the
fork official or generally trusted: it remains third-party code, is not
downloaded by this installer, and must be reviewed and built by each user.

Other reviewed candidates were `turbo-tan/llama.cpp-tq3`,
`spiritbuun/buun-llama-cpp`, and `atomicmilkshake/llama-cpp-turboquant`.
Their formats, accelerated targets, and release policies differ; a binary
that merely says “TurboQuant” is not proof that it supports this GGUF's exact
weight type.

## To actually test it

1. Review and build a TQ3_1S-capable fork. For the selected fork on Linux with
   NVIDIA CUDA, its normal llama.cpp build flags apply:
   ```bash
   cmake -S vendor/thetom-llama-cpp-turboquant \
       -B vendor/thetom-llama-cpp-turboquant/build-native \
       -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build vendor/thetom-llama-cpp-turboquant/build-native \
       --config Release -j
   ```
   Install a compatible CUDA Toolkit and compiler for your machine first; the
   installer deliberately does not automate system toolchain changes.

2. Point the installer at that reviewed build and select its real backend:
   ```bash
   python install.py --model qwen3.6-35b-a3b \
       --quant qwen3.6-35b-a3b-instruct-TQ3_1S.gguf \
       --profile conservative --backend cuda \
       --bin-dir vendor/thetom-llama-cpp-turboquant/build-native/bin \
       --non-interactive
   ```
   The installer will reject a missing or empty `--bin-dir` before downloading
   the 16+ GB model. It cannot prove that an arbitrary `llama-server` actually
   implements TQ3_1S, so reviewing the source/build remains your responsibility.

3. The TQ3_1S file is a **weight** format and needs no special runtime flag in
   this fork. TurboQuant KV-cache formats are separate and optional. Start
   conservatively; for example, the fork recommends keeping K at q8_0 and only
   then trying a turbo V cache. Persist optional flags safely with one token per
   repeated option:
   ```bash
   --extra-server-arg=--cache-type-k --extra-server-arg q8_0 \
   --extra-server-arg=--cache-type-v --extra-server-arg turbo3
   ```
   Values beginning with `-` require the equals form so `argparse` does not
   mistake them for installer options. Both generated launchers preserve the
   exact token list.

4. Compare the TQ model against a standard quant with `tests/validate.py` while
   separately sampling process RSS, system available RAM, and NVIDIA VRAM.

## Measured Linux CUDA result

A bare-metal 12 GB-class laptop GPU (12 GB), CUDA 13.3, and the selected
`c26cbdf` build successfully served the 17,581,938,944-byte TQ3_1S model at
32,768 context. With `--n-cpu-moe 18`, q8_0 K/V cache, one slot, and thinking
disabled for bounded tests, short completion, Python generation, tool calling,
and a 31,549-token needle test all passed. The long prompt processed at
332.0 tok/s; generation ranged from 2.35 to 2.78 tok/s. Peak observed server
VRAM was 10,584 MiB (10,769 MiB total GPU use), peak RSS was 8,258 MiB, and
at least 40.4 GiB system RAM remained available. These are measurements from
one laptop, not portable estimates; see the local `RESULTS.md` for the full
comparison and timeout details.

If you find a verified TurboQuant GGUF repo for one of the other five models
in the catalog, adding it is a one-entry change in `installer/catalog.py` —
just verify the exact filename/size against the HF API first (see
[`MODELS.md`](MODELS.md)).
