# Independent benchmark scores for the catalog models

Only real published scores, on a like-for-like basis. Where no independent
number exists for a model, it's left out, not estimated. Solid bars are
independent measurements; shaded bars are vendor-claimed and unverified.

```
█ measured (independent)   ▒ vendor-claimed
```

## Code generation: LiveCodeBench v6 (%)

```
Qwen3.6-35B-A3B  ████████████████████████████████████████  80  measured
Gemma 4 26B-A4B  ██████████████████████████████████████    77  measured
Gemma 4 12B      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒      72  vendor
Qwen3.5-9B       █████████████████████████████████         66  measured
Qwen3.5-4B       ████████████████████████████              56  measured
```
<sub>Gemma 4 E4B has no published LiveCodeBench score.</sub>

## Reasoning: GPQA Diamond (%)

```
Qwen3.6-35B-A3B  ███████████████████████████████████████████  86  measured
Gemma 4 26B-A4B  █████████████████████████████████████████    82  measured
Qwen3.5-9B       █████████████████████████████████████████    82  measured
Gemma 4 12B      ████████████████████████████████████████     79  measured
Qwen3.5-4B       ██████████████████████████████████████       76  measured
Gemma 4 E4B      ██████████████████████████████               59  measured
```

## Tool use: tau2-bench, Airline variant (%)

```
Qwen3.6-35B-A3B  ████████████████████████████████████  72  measured
Gemma 4 26B-A4B  ██████████████████████████████████    68  measured
Qwen3.5-9B       ██████████████████████████████████    68  measured
```
<sub>Only these three report the Airline variant; the other models use
different tau2 variants that aren't comparable to it.</sub>

## SWE-bench Verified and Terminal-Bench are not charted

Autonomous-coding benchmarks have an independent score for only one catalog
model: `Qwen3.6-35B-A3B` at 73.4 on SWE-bench Verified. The small dense models
(Qwen3.5-9B/4B, Gemma 4 12B/E4B) have no independent SWE-bench or Terminal-Bench
number, so a bar chart would compare one model against blanks.

Among the models that fit a 16 GB Mac, `Qwen3.5-9B` posts the strongest numbers
where data exists: 66 on LiveCodeBench, 82 on GPQA, 68 on tau2 Airline. It also
holds real context on 16 GB, while `Gemma 4 12B` 503s above ~1,200 tokens there
(see MACOS.md). `Qwen3.6-35B-A3B` leads every column but needs 32 GB of unified
memory to run.

<sub>Scores compiled July 2026 from LiveCodeBench, GPQA, and tau2-bench via
independent aggregators. They're harness- and quant-dependent; the local 4-bit
quants score a few points below these full-precision numbers.</sub>

## Local measurements on a 16 GB Apple M4

Measured on one machine with rapid-mlx 0.10.15, 4bit MLX unless noted,
`--pflash off`. tok/s does not transfer between machines. The
OpenCode column is a single smoke test (write `calc.py`, run it, report stdout),
so a pass means the tool-calling loop works at all, not that the model handles
real multi-step edits.

| Model | Footprint | Decode | Usable context | OpenCode smoke test |
|---|---:|---:|---:|---|
| qwen3.5-2b | 1.3 GB | 77 tok/s | 262k | fail (prints code as text) |
| qwen3.5-2b 8bit | 2.25 GB | 46 tok/s | 262k | fail (same as 4bit) |
| qwen3.5-4b | 2.7 GB | 36 tok/s | 262k | fail (prints code as text) |
| gemma-4-e2b | 3.0 GB | 61 tok/s | 130k | pass |
| gemma-4-e4b | 4.5 GB | 32 tok/s | 130k | pass |
| qwen3.5-9b | 5.2 GB | 20 tok/s | 262k | pass |
| gemma-4-12b | 6.8 GB | 13 tok/s | ~1,200 tok | fail (memory, see below) |

Two different reasons a model fails the smoke test:

- **Capability.** Qwen 2B and 4B answer with a ` ```python ` block instead of
  calling the write and run tools. Raising qwen3.5-2b to 8-bit did not change
  this, so it's the model, not the quantization. Qwen needs 9B here.
- **Memory.** gemma-4-12b tool-calls fine, but on 16 GB its usable context is
  ~1,200 tokens, smaller than OpenCode's system prompt, so the run 503s before
  it starts. It passes on 24 GB and up.

Tool-calling doesn't track size: gemma-4-e2b drives OpenCode at 2B while
qwen3.5-4b can't. The difference is the model's tool-call training and the
`gemma4` parser rapid-mlx auto-selects for Gemma.

A text-only GGUF of gemma-4-E2B (vision tower removed) does not save memory over
the MLX build: it ran at 3.4 GB RSS via llama.cpp versus 3.0 GB for the MLX
version, because MLX under `--no-mllm` already loads text-only.
