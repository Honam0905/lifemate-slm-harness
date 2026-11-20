"""Write kaggle/asr_bakeoff.ipynb: zero-shot ASR bake-off on the 887 dev clips (Nemotron 3.5 ASR 0.6B, Cohere
Transcribe 2B) against our Qwen3-ASR numbers, with the same WER function as scripts/build_asr_cache.py.

    python3 kaggle/make_bakeoff_notebook.py
"""
import json
from pathlib import Path

MD_INTRO = """# LifeMate ASR bake-off: should we replace Qwen3-ASR?

Zero-shot transcription of the **887 dev clips** (250 dev conversations: history turns + final requests) with
- `nvidia/nemotron-3.5-asr-streaming-0.6b` (0.6B, RNNT, transformers ≥ 5.13, language `vi-VN`)
- `CohereLabs/cohere-transcribe-03-2026` (2B, encoder-decoder, transformers ≥ 5.4, language `vi`; **gated**)

WER uses exactly the function of `scripts/build_asr_cache.py` (lowercase, punctuation stripped, word edit distance),
so it compares directly with our Qwen3-ASR numbers on the same clips:

| Qwen3-ASR-1.7B | all clips | final requests |
|---|---|---|
| base, no context (fair zero-shot comparison) | 28.10 | 28.33 |
| base, names context | 26.97 | 28.10 |
| + our LoRA A-s100, names context (used in v1, website 34.35) | **20.53** | 22.25 |

**Setup (once):**
1. Right panel: **Accelerator GPU T4** (×1 or ×2), **Internet: On**.
2. **Add Input:** your datasets `lifemate-data` and `lifemate-code` (paths below).
3. **Cohere is gated:** open https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 while logged in and accept the conditions.
   Then **Add-ons → Secrets → add `HF_TOKEN`** (a Hugging Face *read* token) and tick it for this notebook.
   Without it the Cohere part is skipped; Nemotron still runs.
4. First a quick run: set `LIMIT = 24` in the next cell and **Run All** (~5 min). If both models print sensible
   Vietnamese, set `LIMIT = 0` and **Run All** again (~15–25 min); already transcribed clips are reused.

**Output:** `/kaggle/working/bakeoff.zip` (Output panel → download), with `SUMMARY.md`, `results.json`, `per_clip.tsv` and
transcript caches in our cache format (`asr/<name>/train.jsonl`), so the agent can be scored on them later without
transcribing again."""

CELL_CONFIG = '''# ---- configuration -------------------------------------------------------------------------------
DATA = "/kaggle/input/datasets/<your-kaggle-user>/lifemate-data"   # train/train.jsonl, train/audio/<id>/<file>.wav
CODE = "/kaggle/input/datasets/<your-kaggle-user>/lifemate-code"   # artifacts/splits/v1.json (the frozen dev split)
OUT = "/kaggle/working/bakeoff"
LIMIT = 24                 # 24 = quick check (~5 min); 0 = all 887 dev clips
RUN = ["nemotron", "cohere"]
BATCH = {"nemotron": 16, "cohere": 16}
DTYPE = "float32"          # T4 has no bfloat16; float16 is faster but riskier for accuracy
MODEL_ID = {"nemotron": "nvidia/nemotron-3.5-asr-streaming-0.6b", "cohere": "CohereLabs/cohere-transcribe-03-2026"}
REFS = {  # Qwen3-ASR-1.7B on the same 887 clips, same WER function (computed 5 Nov from the Kaggle/Colab caches)
    "qwen3-asr base, no context": (28.10, 28.33),
    "qwen3-asr base, names context": (26.97, 28.10),
    "qwen3-asr + LoRA A-s100, names context (v1)": (20.53, 22.25),
}'''

CELL_ENV = '''import os, sys, json, time, re, unicodedata, traceback, shutil
from pathlib import Path
import numpy as np
import torch, transformers
from packaging.version import Version

print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available(), "no GPU: set Accelerator to GPU T4"
if Version(transformers.__version__) < Version("5.13.0"):
    raise SystemExit("transformers >= 5.13 needed for Nemotron 3.5 ASR: run  !pip install -q 'transformers>=5.13'  then restart")
DT = getattr(torch, DTYPE)

HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print(f"no Kaggle secret HF_TOKEN ({type(e).__name__}): Cohere (gated) will be skipped")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Hugging Face login OK (token not printed)")
Path(OUT).mkdir(parents=True, exist_ok=True)'''

CELL_DATA = '''import librosa

splits = json.load(open(f"{CODE}/artifacts/splits/v1.json", encoding="utf-8"))
dev = set(splits["dev"])
samples = [json.loads(l) for l in open(f"{DATA}/train/train.jsonl", encoding="utf-8")]
samples = [s for s in samples if s["id"] in dev]
gold, clips, finals = {}, [], set()
for s in samples:                              # same order as lifemate.data.loader.user_turn_audio
    for t in s["history"]:
        if t["role"] == "user":
            gold[t["audio"]] = t["audio_text"]; clips.append(t["audio"])
    gold[s["audio"]] = s["audio_text"]; clips.append(s["audio"]); finals.add(s["audio"])
assert len(samples) == 250 and len(clips) == 887, (len(samples), len(clips))
missing = [p for p in clips if not Path(DATA, "train", p).exists()]
assert not missing, f"{len(missing)} audio files missing, e.g. {missing[:3]}"
if LIMIT:
    clips = clips[:LIMIT]

t0 = time.time()
audio = {p: librosa.load(str(Path(DATA, "train", p)), sr=16000, mono=True)[0].astype(np.float32) for p in clips}
dur = {p: len(a) / 16000 for p, a in audio.items()}
print(f"{len(clips)} clips ({sum(p in finals for p in clips)} final requests), {sum(dur.values()) / 60:.1f} min of audio, "
      f"longest {max(dur.values()):.1f}s, loaded in {time.time() - t0:.0f}s")'''

CELL_WER = '''def words(t: str) -> list:
    return re.findall(r"[^\\W_]+", unicodedata.normalize("NFC", t or "").lower())


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(hyp: dict, subset) -> float:
    errs = n = 0
    for p in subset:
        g = words(gold[p]); errs += edit_distance(g, words(hyp.get(p, ""))); n += len(g)
    return round(100 * errs / max(n, 1), 2)


def run_model(name: str, transcribe_batch, bs: int, cache_name: str) -> dict:
    """Transcribe every clip (longest first, so a memory problem shows in the first batch); the cache is appended
    after every batch, so a re-run continues where it stopped."""
    path = Path(OUT, "asr", cache_name, "train.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            r = json.loads(l); done[r["audio"]] = r["text"]
    todo = sorted((p for p in clips if p not in done), key=lambda p: -dur[p])
    print(f"[{name}] {len(clips)} clips, {len(done)} cached, {len(todo)} to do, batch {bs}", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), bs):
        part = todo[i: i + bs]
        try:
            texts = transcribe_batch([audio[p] for p in part])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{name}] out of memory at batch {bs}: retrying this batch one clip at a time", flush=True)
            texts = [transcribe_batch([audio[p]])[0] for p in part]
        assert len(texts) == len(part), (len(texts), len(part))
        with open(path, "a", encoding="utf-8") as f:
            for p, t in zip(part, texts):
                f.write(json.dumps({"audio": p, "text": t}, ensure_ascii=False) + "\\n"); done[p] = t
        if (i // bs) % 10 == 0 or i + bs >= len(todo):
            el = time.time() - t0
            print(f"[{name}] {i + len(part)}/{len(todo)} clips, {el:.0f}s, {el / (i + len(part)):.3f}s/clip", flush=True)
    sec = (time.time() - t0) / max(len(todo), 1) if todo else None
    return {"hyp": done, "sec_per_clip": sec, "cache": str(path)}


def show(name: str, hyp: dict, k: int = 4):
    for p in clips[:k]:
        print(f"  GOLD: {gold[p]}\\n  {name.upper()}: {hyp.get(p)}\\n")


results, status = {}, {}'''

CELL_NEMOTRON = '''name = "nemotron"
if name in RUN:
    try:
        from huggingface_hub import model_info
        from transformers import AutoModelForRNNT, AutoProcessor
        mid = MODEL_ID[name]
        sha = model_info(mid).sha
        proc = AutoProcessor.from_pretrained(mid, revision=sha)
        model = AutoModelForRNNT.from_pretrained(mid, revision=sha, dtype=DT).to("cuda:0").eval()
        sr = proc.feature_extractor.sampling_rate
        print(f"[{name}] {mid} @ {sha[:7]}: {sum(p.numel() for p in model.parameters()):,} params, sampling rate {sr}")

        def nemotron_batch(arrs):
            inputs = proc(arrs, sampling_rate=sr, language="vi-VN", return_tensors="pt")
            inputs = inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                out = model.generate(**inputs, return_dict_in_generate=True)
            seqs = out.sequences
            texts = proc.batch_decode(seqs, skip_special_tokens=True) if hasattr(proc, "batch_decode") else proc.decode(seqs, skip_special_tokens=True)
            return [texts] if isinstance(texts, str) else [t.strip() for t in texts]

        print(f"[{name}] sanity, 1 clip:", nemotron_batch([audio[clips[0]]]), "| gold:", gold[clips[0]])
        try:                                   # does the processor batch (with padding)?
            assert len(nemotron_batch([audio[p] for p in clips[:2]])) == 2
            bs = BATCH[name]
        except Exception as e:
            print(f"[{name}] batching failed ({type(e).__name__}: {e}); using batch 1"); bs = 1
        results[name] = run_model(name, nemotron_batch, bs, f"nemotron35-asr-0.6b-{sha[:7]}")
        results[name]["revision"] = sha
        show(name, results[name]["hyp"])
        status[name] = "ok"
    except Exception as e:
        status[name] = f"FAILED: {type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        for v in ("model", "proc"):
            globals().pop(v, None)
        torch.cuda.empty_cache()
    print(f"[{name}] {status[name]}")'''

CELL_COHERE = '''name = "cohere"
if name in RUN:
    try:
        if not HF_TOKEN:
            raise RuntimeError("no HF_TOKEN secret: accept the model conditions on Hugging Face and add the secret (see the top)")
        from huggingface_hub import model_info
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        mid = MODEL_ID[name]
        sha = model_info(mid).sha
        proc = AutoProcessor.from_pretrained(mid, revision=sha)
        model = CohereAsrForConditionalGeneration.from_pretrained(mid, revision=sha, dtype=DT).to("cuda:0").eval()
        print(f"[{name}] {mid} @ {sha[:7]}: {sum(p.numel() for p in model.parameters()):,} params")

        def cohere_batch(arrs):
            inputs = proc(arrs, sampling_rate=16000, return_tensors="pt", language="vi")
            idx = inputs.get("audio_chunk_index")
            inputs = inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=256)
            texts = proc.decode(out, skip_special_tokens=True, audio_chunk_index=idx, language="vi")
            return [texts] if isinstance(texts, str) else [t.strip() for t in texts]

        print(f"[{name}] sanity, 1 clip:", cohere_batch([audio[clips[0]]]), "| gold:", gold[clips[0]])
        results[name] = run_model(name, cohere_batch, BATCH[name], f"cohere-transcribe-2b-{sha[:7]}")
        results[name]["revision"] = sha
        show(name, results[name]["hyp"])
        status[name] = "ok"
    except Exception as e:
        status[name] = f"FAILED: {type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        for v in ("model", "proc"):
            globals().pop(v, None)
        torch.cuda.empty_cache()
    print(f"[{name}] {status[name]}")'''

CELL_REPORT = '''fin = [p for p in clips if p in finals]
hist = [p for p in clips if p not in finals]
digits = lambda t: bool(re.search(r"\\d", t or ""))
g_dig = [p for p in clips if digits(gold[p])]
table = []
for name, r in results.items():
    h = r["hyp"]
    table.append({"model": f"{name} ({MODEL_ID[name]}@{r['revision'][:7]})", "wer_all": wer(h, clips), "wer_final": wer(h, fin),
                  "wer_history": wer(h, hist), "sec_per_clip_T4": None if r["sec_per_clip"] is None else round(r["sec_per_clip"], 3),
                  "keeps_digits_when_gold_has_them": f"{sum(digits(h.get(p)) for p in g_dig)}/{len(g_dig)}",
                  "empty_outputs": sum(not (h.get(p) or "").strip() for p in clips)})

lines = [f"# ASR bake-off on {len(clips)} dev clips ({len(fin)} final requests){' — QUICK CHECK, set LIMIT = 0 for the real numbers' if LIMIT else ''}", "",
         "| model | WER all | WER final requests | WER history turns | s/clip (T4) | digits kept | empty |", "|---|---|---|---|---|---|---|"]
for t in table:
    lines.append(f"| {t['model']} | **{t['wer_all']}** | {t['wer_final']} | {t['wer_history']} | {t['sec_per_clip_T4']} | {t['keeps_digits_when_gold_has_them']} | {t['empty_outputs']} |")
for k, (a, f_) in REFS.items():
    lines.append(f"| {k} (reference, all 887 clips) | {a} | {f_} | | | | |")
lines += ["", "Status: " + json.dumps(status, ensure_ascii=False), "",
          "Rule of thumb: a zero-shot WER near or below ~22 (our fine-tuned Qwen is 20.53) is worth fine-tuning and an agent",
          "dev-C eval; 22-25 is borderline; above ~25 it is not worth switching (Qwen base with names context is 26.97).",
          "WER only guides: the decision is the agent's dev C score (numbers written as words cost WER but may not cost score)."]
if not LIMIT:
    for t in table:
        v = t["wer_all"]
        lines.append(f"- {t['model']}: " + ("worth fine-tuning + agent eval" if v <= 22 else "borderline" if v <= 25 else "not worth switching"))
summary = "\\n".join(lines)
print(summary)

names = list(results)
with open(Path(OUT, "per_clip.tsv"), "w", encoding="utf-8") as f:
    f.write("\\t".join(["audio", "final", "seconds", "gold"] + names) + "\\n")
    for p in clips:
        f.write("\\t".join([p, str(p in finals), f"{dur[p]:.1f}", gold[p]] + [(results[n]["hyp"].get(p) or "").replace("\\t", " ") for n in names]) + "\\n")
Path(OUT, "SUMMARY.md").write_text(summary + "\\n", encoding="utf-8")
Path(OUT, "results.json").write_text(json.dumps({"limit": LIMIT, "n_clips": len(clips), "status": status, "table": table,
                                                 "refs": REFS, "torch": torch.__version__, "transformers": transformers.__version__},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
print("\\nclips where the two models disagree most with gold (first 8):")
for p in sorted(clips, key=lambda p: -max(edit_distance(words(gold[p]), words(results[n]["hyp"].get(p, ""))) for n in names) if names else 0)[:8]:
    print(f"  GOLD: {gold[p]}")
    for n in names:
        print(f"  {n:9s}: {results[n]['hyp'].get(p)}")
    print()
shutil.make_archive("/kaggle/working/bakeoff", "zip", OUT)
print("wrote /kaggle/working/bakeoff.zip  (Output panel -> download)")'''


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "cells": [md(MD_INTRO), code(CELL_CONFIG), code(CELL_ENV), code(CELL_DATA), code(CELL_WER),
                md("## Nemotron 3.5 ASR 0.6B"), code(CELL_NEMOTRON), md("## Cohere Transcribe 2B (gated)"), code(CELL_COHERE),
                md("## Report"), code(CELL_REPORT)]}
out = Path(__file__).resolve().parent / "asr_bakeoff.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}")
