"""Install any GGUF model from a Hugging Face link.

    python install.py --hf https://huggingface.co/OWNER/NAME
    python install.py --hf OWNER/NAME --hf-file NAME-Q4_K_M.gguf

Accepts a repo id or any huggingface.co URL (repo page, /tree/..., /blob/...
or /resolve/... file links; a file link also picks that file). The repo is
turned into the same model/quant dicts the curated catalog uses, so download,
serving, tests and the OpenCode config run unchanged.

From the Hugging Face API it reads the file list with exact sizes and the GGUF
architecture, which decides MoE (experts can live in system RAM) vs dense.
Nothing here is curated: the fit estimate is the catalog's heuristic, and the
installer's tests are what tell you whether the model drives OpenCode well.
"""
import json
import re
import urllib.parse
import urllib.request

HF_API = "https://huggingface.co/api/models/{repo}"
UA = {"User-Agent": "opencode-local-installer"}

# Repos offered in the wizard. Both load in stock llama.cpp (GGUF arch
# qwen35moe). Only the first was run through this project's tests.
SUGGESTED = [
    ("HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
     "Qwen3.6-35B-A3B uncensored (HauhauCS) - tested with this installer"),
    ("mradermacher/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF",
     "Qwen3.6-35B-A3B abliterated (huihui-ai, GGUF by mradermacher) - not tested here"),
]

# GGUFs in a repo that are not standalone chat weights.
_SKIP_FILE = re.compile(r"(mmproj|imatrix|dflash|draft|mtp[-_.]?head|patch)", re.I)
_SPLIT = re.compile(r"-(\d+)-of-(\d+)\.gguf$", re.I)
# "-A3B", "-A10B", "-A2.5B": active-parameter suffix of an MoE model name.
_MOE_NAME = re.compile(r"[-_]A\d+(\.\d+)?B\b", re.I)
_MOE_ARCH = re.compile(r"moe|gpt-oss|deepseek2|deepseek3|glm4moe|qwen3next|mixtral|llama4", re.I)
# Default quant: the same Q4_K_M class the catalog defaults to, then neighbours.
_QUANT_PREF = ["Q4_K_M", "Q4_K_XL", "Q4_K_P", "Q4_K", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q4_0",
               "Q5_K_M", "Q5_K_XL", "Q5_K", "Q3_K_XL", "Q3_K_M", "Q3_K_L", "Q6_K", "Q8_0"]
# Weight types only the TurboQuant fork can load.
_TQ_WEIGHT = re.compile(r"[-_.]TQ\d_\d?S?\b|[-_.]TTQ\b", re.I)


def parse_link(link):
    """Return (repo, file_or_None) from a repo id or a huggingface.co URL."""
    s = link.strip().strip('"').strip("'")
    s = re.sub(r"^https?://(www\.)?(huggingface\.co|hf\.co)/", "", s)
    s = s.split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = s.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Not a Hugging Face model link or OWNER/NAME id: {link!r}")
    repo = f"{parts[0]}/{parts[1]}"
    hf_file = None
    if len(parts) >= 5 and parts[2] in ("blob", "resolve", "tree"):
        rest = "/".join(parts[4:])
        if rest.lower().endswith(".gguf"):
            hf_file = urllib.parse.unquote(rest)
    return repo, hf_file


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def repo_info(repo):
    q = urllib.parse.urlencode([("expand[]", "gguf"), ("expand[]", "tags")])
    return _get_json(HF_API.format(repo=repo) + "?" + q)


def list_ggufs(repo):
    """Chat-weight GGUFs as [{"file", "size_gb", "shards"}], smallest first.
    Split files are listed once (first shard, total size); companion files
    (mmproj vision adapters, imatrix, draft heads, patches) are dropped."""
    tree = _get_json(HF_API.format(repo=repo) + "/tree/main?recursive=true")
    files = {e["path"]: e.get("size", 0) for e in tree
             if e.get("type") == "file" and e["path"].lower().endswith(".gguf")}
    out = []
    for path, size in files.items():
        if _SKIP_FILE.search(path):
            continue
        m = _SPLIT.search(path)
        shards = 1
        if m:
            if int(m.group(1)) != 1:
                continue
            shards = int(m.group(2))
            stem = path[:m.start()]
            size = sum(s for p, s in files.items() if p.startswith(stem) and _SPLIT.search(p))
        out.append({"file": path, "size_gb": round(size / 1e9, 2), "shards": shards,
                    "turboquant": bool(_TQ_WEIGHT.search(path))})
    return sorted(out, key=lambda q: q["size_gb"])


def quant_tag(path):
    """Short quant name from a file name, e.g. 'Q4_K_M', for display."""
    name = path.rsplit("/", 1)[-1]
    m = re.search(r"[-_.]((?:UD-)?I?Q\d[A-Z0-9_]*|F16|BF16|F32|MXFP4[A-Z0-9_]*)\.gguf$", name, re.I)
    return m.group(1) if m else name


def pick_default(ggufs):
    usable = [q for q in ggufs if q["shards"] == 1 and not q["turboquant"]]
    for tag in _QUANT_PREF:
        for q in usable:
            if re.search(rf"[-_.]{re.escape(tag)}[-_.]", q["file"].rsplit("/", 1)[-1] + ".", re.I):
                return q
    return usable[len(usable) // 2] if usable else None


def guess_arch(repo, info):
    arch = ((info.get("gguf") or {}).get("architecture") or "").lower()
    tags = [t.lower() for t in info.get("tags", [])]
    if _MOE_ARCH.search(arch) or "moe" in tags or _MOE_NAME.search(repo):
        return "moe"
    return "dense"


def _slug(repo):
    name = repo.split("/", 1)[-1].lower()
    name = re.sub(r"-gguf$", "", name)
    return re.sub(r"[^a-z0-9.]+", "-", name).strip("-")[:64]


def build(repo, gguf):
    """(model, quant) dicts in the catalog's shape for one GGUF of a repo."""
    info = repo_info(repo)
    arch = guess_arch(repo, info)
    total = (info.get("gguf") or {}).get("total")
    model = {
        "id": _slug(repo),
        "display_name": repo.split("/", 1)[-1],
        "arch": arch,
        "family": None,
        "total_params_b": round(total / 1e9) if total else 0,
        "active_params_b": None,
        "reasoning": True,
        "custom": True,
        "notes": f"From Hugging Face: {repo} (GGUF arch "
                 f"{(info.get('gguf') or {}).get('architecture') or 'unknown'}). Not curated.",
        "mlx": [],
    }
    quant = {"label": quant_tag(gguf["file"]), "file": gguf["file"], "repo": repo,
             "size_gb": gguf["size_gb"], "default": True}
    model["quants"] = [quant]
    return model, quant


def build_offline(model, gguf):
    """Same model, another file of its repo (no network call)."""
    quant = {"label": quant_tag(gguf["file"]), "file": gguf["file"],
             "repo": model["quants"][0]["repo"], "size_gb": gguf["size_gb"], "default": True}
    return dict(model, quants=[quant]), quant


def resolve(link, hf_file=None):
    """Repo link (+ optional file) -> (model, quant, ggufs). Raises ValueError
    with a readable message when the repo can't be used."""
    repo, link_file = parse_link(link)
    hf_file = hf_file or link_file
    try:
        ggufs = list_ggufs(repo)
    except urllib.error.HTTPError as e:
        raise ValueError(f"Hugging Face says {e.code} for {repo} (typo, private or gated "
                         f"repo?)") from e
    if not ggufs:
        raise ValueError(f"{repo} has no GGUF model files. Look for a repo whose name ends "
                         f"in -GGUF (e.g. by bartowski, unsloth or mradermacher).")
    if hf_file:
        gguf = next((q for q in ggufs if q["file"] == hf_file or
                     q["file"].rsplit("/", 1)[-1] == hf_file.rsplit("/", 1)[-1]), None)
        if gguf is None:
            raise ValueError(f"{hf_file!r} is not in {repo}. Available:\n" + "\n".join(
                f"  {q['file']}  {q['size_gb']:.1f} GB" for q in ggufs))
    else:
        gguf = pick_default(ggufs)
        if gguf is None:
            raise ValueError(f"{repo} only has split or TurboQuant-only GGUFs, which this "
                             f"installer can't download. Pick another repo.")
    if gguf["shards"] > 1:
        raise ValueError(f"{gguf['file']} is split into {gguf['shards']} files; pick a "
                         f"single-file quant.")
    if gguf["turboquant"]:
        raise ValueError(f"{gguf['file']} uses TurboQuant weights that official llama.cpp "
                         f"can't load; pick a normal quant (Q4_K_M etc.).")
    model, quant = build(repo, gguf)
    return model, quant, ggufs
