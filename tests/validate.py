#!/usr/bin/env python3
"""Validation for a local llama-server (OpenAI-compatible endpoint), any model
in installer/catalog.py. Runs: short completion, code-gen+ast.parse,
tool-calling, long-context needle. No third-party deps (urllib only). Prints
a pass/fail table + tok/s.

Reads LLAMA_BASE_URL / LLAMA_MODEL from the environment (set by install.py);
falls back to the qwen3.6-35b-a3b defaults for a manual `python validate.py`
run against run.ps1/run.sh with no arguments."""
import ast, json, os, sys, time, urllib.request

BASE = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("LLAMA_MODEL", "qwen3.6-35b-a3b")
results = []  # (name, passed, detail, toks_per_s)

def chat(messages, tools=None, tool_choice=None, max_tokens=2048, temperature=0.2):
    # NOTE: every model in the catalog is a reasoning model - it emits <think> tokens
    # (returned in a separate reasoning_content field) BEFORE the final content.
    # max_tokens must cover BOTH, or content comes back empty with finish_reason=length.
    # The defaults below were bumped after finding Qwen3.5-4B needs more thinking budget
    # than Qwen3.6-35B-A3B did; a small/fast model isn't necessarily a terse one.
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    if tools: body["tools"] = tools
    if tool_choice: body["tool_choice"] = tool_choice
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=data,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        j = json.load(r)
    dt = time.time() - t0
    usage = j.get("usage", {})
    tim = j.get("timings", {})
    ctoks = usage.get("completion_tokens", 0)
    ptoks = usage.get("prompt_tokens", 0)
    tps = tim.get("predicted_per_second") or (ctoks / dt if dt else 0)
    return j, {"dt": dt, "ctoks": ctoks, "ptoks": ptoks, "tps": tps,
               "prompt_tps": tim.get("prompt_per_second")}

def record(name, passed, detail, meta=None):
    tps = round(meta["tps"], 1) if meta else None
    results.append((name, passed, detail, tps))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}"
          + (f"  ({tps} tok/s gen)" if tps else ""))

# --- Test 1: short completion ---
try:
    j, m = chat([{"role": "user", "content": "Answer in one short sentence: why is the sky blue?"}],
                max_tokens=2048)
    txt = j["choices"][0]["message"]["content"].strip()
    ok = len(txt) > 15 and any(w in txt.lower() for w in ["scatter", "light", "blue", "wavelength"])
    record("short-completion", ok, repr(txt[:160]), m)
except Exception as e:
    record("short-completion", False, f"ERROR {e}")

# --- Test 2: code gen + ast.parse ---
try:
    j, m = chat([{"role": "user", "content": "Write a Python function `fib(n)` that returns the nth "
                  "Fibonacci number. Include a docstring. Reply with only a ```python code block."}],
                max_tokens=2048)
    txt = j["choices"][0]["message"]["content"]
    code = txt.split("```python")[1].split("```")[0] if "```python" in txt else \
           (txt.split("```")[1].split("```")[0] if "```" in txt else txt)
    tree = ast.parse(code)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    has_doc = any(ast.get_docstring(f) for f in fns)
    ok = len(fns) >= 1 and has_doc
    record("code-gen", ok, f"parsed OK, {len(fns)} func(s), docstring={has_doc}", m)
except Exception as e:
    record("code-gen", False, f"ERROR {e}")

# --- Test 3: tool calling ---
try:
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["location"]}}}]
    j, m = chat([{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
                tools=tools, tool_choice="auto", max_tokens=200)
    msg = j["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    ok = False; detail = "no tool_calls emitted"
    if tcs:
        tc = tcs[0]["function"]
        args = json.loads(tc["arguments"])
        ok = tc["name"] == "get_weather" and "location" in args and "paris" in str(args["location"]).lower()
        detail = f"name={tc['name']} args={args}"
    record("tool-calling", ok, detail, m)
except Exception as e:
    record("tool-calling", False, f"ERROR {e}")

# --- Test 4: long-context needle (~30k tokens, needle at 60%) ---
try:
    needle = "REMEMBER THIS: the secret vault access code is PURPLE-WOMBAT-4291."
    filler_line = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
                   "reviews configuration files and verifies cache behavior. ")
    # ~30k tokens: build many numbered lines, target ~120k chars.
    lines, target_chars = [], 145000   # ~30k prompt tokens with this filler's ~4.75 chars/token
    i, cur = 0, 0
    while cur < target_chars:
        s = f"[line {i:05d}] {filler_line}"
        lines.append(s); cur += len(s) + 1; i += 1
    insert_at = int(len(lines) * 0.60)
    lines.insert(insert_at, needle)
    document = "\n".join(lines)
    prompt = ("Below is a long document. Somewhere inside it is a secret vault access code. "
              "Read carefully and tell me ONLY the exact code (format WORD-WORD-NNNN).\n\n"
              f"=== DOCUMENT START ===\n{document}\n=== DOCUMENT END ===\n\n"
              "What is the secret vault access code?")
    j, m = chat([{"role": "user", "content": prompt}], max_tokens=3072, temperature=0.0)
    txt = j["choices"][0]["message"]["content"]
    ok = "PURPLE-WOMBAT-4291" in txt.upper()
    record("long-context-needle", ok,
           f"prompt_tokens={m['ptoks']} needle@60% found={ok} ans={txt.strip()[:80]!r}", m)
    # emit machine-readable line for the bench wrapper
    print(f"##NEEDLE_META## prompt_tokens={m['ptoks']} gen_tps={m['tps']:.2f} "
          f"prompt_tps={m['prompt_tps']} dt={m['dt']:.1f}")
except Exception as e:
    record("long-context-needle", False, f"ERROR {e}")

print("\n=== PASS/FAIL TABLE ===")
allpass = True
for name, passed, detail, tps in results:
    allpass = allpass and passed
    print(f"  {'PASS' if passed else 'FAIL':4}  {name:22} {('%.1f tok/s'%tps) if tps else '':>12}")
sys.exit(0 if allpass else 1)
