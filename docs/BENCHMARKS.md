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
