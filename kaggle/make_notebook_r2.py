"""Write kaggle/lifemate_asr_round2.ipynb: Qwen3-ASR LoRA round 2 (attention + MLP, 3 epochs), built from the proven
round-1 notebook (kaggle/make_notebook.py) with these changes patched in:
  * LoRA on attention AND MLP of the 28 decoder layers (196 modules; the audio encoder stays frozen). The training script
    is patched in the notebook's working copy, so the Kaggle code dataset does not need a new upload.
  * Model A2: 600 steps (3 epochs over the 3,212 train-partition clips), saves 200/400/600 with holdout loss.
  * Model B2: same recipe on all 5,036 clips: 941 steps, saves 314/627/941 (same epochs as A2).
  * dev transcripts for every A2 save; public transcripts for the B2 saves matching the 2 best A2 saves by dev WER.
  * out-of-memory during training -> retried once with gradient checkpointing.
  * MODE defaults to "full" (commit directly); "smoke" still exists for an interactive check.

    python3 kaggle/make_notebook_r2.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nb1", HERE / "make_notebook.py")
nb1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb1)                         # fills nb1.CELLS with the round-1 cells
cells = [list(c) for c in nb1.CELLS]


def patch(i: int, old: str, new: str) -> None:
    assert old in cells[i][1], f"cell {i}: pattern not found: {old[:80]!r}"
    cells[i][1] = cells[i][1].replace(old, new, 1)


def find(snippet: str) -> int:
    hits = [i for i, (_, s) in enumerate(cells) if snippet in s]
    assert len(hits) == 1, (snippet, hits)
    return hits[0]


cells[0][1] = r"""
# LifeMate — Qwen3-ASR LoRA **round 2** on 2 × T4 (attention + MLP, 3 epochs)

**Settings (right panel):** Accelerator **GPU T4 ×2** · Internet **On** · add the two private datasets
(`lifemate-data`, `lifemate-code`). `HF_TOKEN` secret optional (Qwen3-ASR is not gated).

**Run:** `MODE = "full"` is already set → **Save Version → Save & Run All (Commit)**, then close the tab. It runs on
Kaggle's servers (~2.5 h; the 12 h limit is far away). Download `out/` from the version's **Output** tab afterwards.

**What it does** (round 1 = yesterday: attention-only LoRA, 200 steps, dev WER 26.97 → 20.53 at A-s100, 19.36 at A-s200):
1. **Model A2:** LoRA r16 on attention **and MLP** of the 28 decoder layers, 3,212 train-partition clips (354 held out),
   **600 steps = 3 epochs**, adapters + holdout loss at 200 / 400 / 600.
2. **Dev transcripts** (887 clips) with each A2 adapter → `out/asr/<cache>/train.jsonl` + WER.
3. **Model B2:** same recipe on **all** 5,036 clips, 941 steps, adapters at 314 / 627 / 941 (same epochs as A2).
   Final-fit candidate only: **never score Model B2 on dev or audit.**
4. **Public transcripts** (1,933 clips) with the B2 adapters that match the **2 best A2 adapters by dev WER**.

If training runs out of GPU memory it is retried once with gradient checkpointing (slower, same result).
"""

i = find('MODE = "smoke"          #')
patch(i, 'MODE = "smoke"          # "smoke" = quick end-to-end check (run interactively first) | "full" = the overnight run',
      'MODE = "full"           # "full" = round 2 (commit directly) | "smoke" = quick interactive end-to-end check')
patch(i, "STEPS_A, SAVE_A = 200, [100, 200]", "STEPS_A, SAVE_A = 600, [200, 400, 600]   # 3 epochs over the 3,212 train-partition clips")
patch(i, "GRAD_CKPT = False       # set True only if the smoke run reports out-of-memory",
      "GRAD_CKPT = False       # switched on automatically if training runs out of memory")
patch(i, "PUBLIC = True           # transcribe the public test clips with both Model B checkpoints",
      "PUBLIC = True           # transcribe the public test clips with the best Model B2 checkpoints\n"
      "PUBLIC_TOP = 2          # how many B2 checkpoints get public transcripts (ranked by the matching A2 dev WER)\n"
      "TARGETS = \"attn+mlp\"    # round 2: LoRA on self-attention q/k/v/o AND mlp gate/up/down (round 1: attention only)\n"
      "NAME_A, NAME_B, TAG = \"A2\", \"B2\", \"2\"   # output folders out/A2, out/B2; transcript tags ftA2-s*, ftB2-s*")

i = find("shutil.copytree(CODE_SRC, CODE")
patch(i, 'ART = CODE / "artifacts"',
      'ATTN_MLP = r"thinker\\.model\\.layers\\.\\d+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))"\n'
      'if TARGETS == "attn+mlp":   # patch the working copy only; the training script reads TARGET_MODULES from here\n'
      '    _tl = CODE / "scripts" / "train_asr_lora.py"\n'
      '    _src = _tl.read_text()\n'
      '    _old = \'TARGET_MODULES = r"thinker\\\\.model\\\\.layers\\\\.\\\\d+\\\\.self_attn\\\\.(q_proj|k_proj|v_proj|o_proj)"\'\n'
      '    assert _old in _src, "TARGET_MODULES line not found in train_asr_lora.py"\n'
      '    _tl.write_text(_src.replace(_old, "TARGET_MODULES = r\\"" + ATTN_MLP + "\\""))\n'
      '    print("LoRA targets patched to attention + MLP:", ATTN_MLP)\n'
      'ART = CODE / "artifacts"')

i = find("CACHE_CFG = {")
patch(i, 'CACHE_CFG = {"kgbase": "kaggle_kgbase_names", "ftA-s100": "kaggle_ftA_s100_names", "ftA-s200": "kaggle_ftA_s200_names",\n'
         '             f"ftB-s{SAVE_B[0]}": f"kaggle_ftB_s{SAVE_B[0]}_names", f"ftB-s{SAVE_B[-1]}": f"kaggle_ftB_s{SAVE_B[-1]}_names"}',
      'CACHE_CFG = {**{f"ftA{TAG}-s{s}": None for s in SAVE_A}, **{f"ftB{TAG}-s{s}": None for s in SAVE_B}}')
patch(i, 'CACHE_NAME = {k: (backend_name(load_config(CODE / "configs" / f"{v}.yaml")) if (CODE / "configs" / f"{v}.yaml").exists()\n'
         '                  else _base.replace("-ctxnames", f"-{k}-ctxnames")) for k, v in CACHE_CFG.items()}',
      'CACHE_NAME = {k: _base.replace("-ctxnames", f"-{k}-ctxnames") for k in CACHE_CFG}   # = backend_name with asr.tag k')

i = find("def train(name, manifest, steps, save")
patch(i, 'RESULTS = {"wer_dev": {}, "notes": []}',
      'def train_safe(name, manifest, steps, save, extra=()):\n'
      '    """Train; if the GPU runs out of memory, retry once from scratch with gradient checkpointing."""\n'
      '    global GRAD_CKPT\n'
      '    try:\n'
      '        train(name, manifest, steps, save, extra)\n'
      '    except RuntimeError:\n'
      '        log = (LOGS / f"train_{name}.log").read_text(errors="replace")\n'
      '        if GRAD_CKPT or "out of memory" not in log.lower():\n'
      '            raise\n'
      '        print(f"{name}: out of GPU memory -> retrying with gradient checkpointing", flush=True)\n'
      '        RESULTS["notes"].append(f"{name}: OOM without gradient checkpointing, retried with it")\n'
      '        GRAD_CKPT = True\n'
      '        shutil.rmtree(OUT / name, ignore_errors=True)\n'
      '        (LOGS / f"train_{name}.log").rename(LOGS / f"train_{name}.oom.log")\n'
      '        train(name, manifest, steps, save, extra)\n'
      '\n'
      'RESULTS = {"wer_dev": {}, "notes": [], "round": 2, "targets": TARGETS, "steps_a": STEPS_A, "save_a": SAVE_A}')

i = find('section("Model A training (train partition)")')
patch(i, 'section("Model A training (train partition)")\n    train("A", MAN_A, STEPS_A, SAVE_A)',
      'section(f"Model {NAME_A} training (train partition, {TARGETS})")\n    train_safe(NAME_A, MAN_A, STEPS_A, SAVE_A)')

i = find('for tag, adapter in [("kgbase", None)]')
patch(i, 'for tag, adapter in [("kgbase", None)] + [(f"ftA-s{s}", OUT / "A" / f"step{s}") for s in SAVE_A]:',
      'for tag, adapter in [(f"ftA{TAG}-s{s}", OUT / NAME_A / f"step{s}") for s in SAVE_A]:   # base / round 1 known already')
patch(i, 'print("dev WER:", RESULTS["wer_dev"], "| best A by WER:", best, "(the agent\'s dev score decides, tomorrow)")',
      'print("dev WER:", RESULTS["wer_dev"], "| best A2 by WER:", best, "| round 1: base 26.97, ftA-s100 20.53, ftA-s200 19.36")')

i = find('section("Model B training (ALL train clips)")')
patch(i, 'section("Model B training (ALL train clips)")\n    train("B", MAN_B, STEPS_B, SAVE_B, extra=["--no_holdout"])',
      'section(f"Model {NAME_B} training (ALL train clips, {TARGETS})")\n    train_safe(NAME_B, MAN_B, STEPS_B, SAVE_B, extra=["--no_holdout"])')

i = find("# ---- 4. public test transcripts")
patch(i, '    for s in SAVE_B:\n        tag = f"ftB-s{s}"\n        section(f"public transcripts: {tag}")\n'
         '        transcribe(tag, OUT / "B" / f"step{s}", "public_test", "all", OUT / "asr" / CACHE_NAME[tag] / "public_test.jsonl")',
      '    rank = sorted(range(len(SAVE_A)), key=lambda k: RESULTS["wer_dev"][f"ftA{TAG}-s{SAVE_A[k]}"])   # best A2 first\n'
      '    RESULTS["public_b"] = []\n'
      '    for k in rank[:PUBLIC_TOP]:\n'
      '        s = SAVE_B[k]\n'
      '        tag = f"ftB{TAG}-s{s}"\n'
      '        section(f"public transcripts: {tag} (matches ftA{TAG}-s{SAVE_A[k]}, dev WER {RESULTS[\'wer_dev\'][f\'ftA{TAG}-s{SAVE_A[k]}\']})")\n'
      '        transcribe(tag, OUT / NAME_B / f"step{s}", "public_test", "all", OUT / "asr" / CACHE_NAME[tag] / "public_test.jsonl")\n'
      '        RESULTS["public_b"].append({"tag": tag, "matches": f"ftA{TAG}-s{SAVE_A[k]}"})\n'
      '        (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))')

i = find("# ---- summary")
patch(i, 'for name in ("A", "B"):', 'for name in (NAME_A, NAME_B):')
patch(i, 'lines = [f"# Kaggle ASR run ({MODE}) — {time.strftime(\'%Y-%m-%d %H:%M\')}", "",',
      'lines = [f"# Kaggle ASR round 2 ({MODE}, LoRA {TARGETS}, {STEPS_A} steps) — {time.strftime(\'%Y-%m-%d %H:%M\')}", "",')
patch(i, 'f"- best A by dev WER: {RESULTS.get(\'best_A_by_dev_wer\')}",',
      'f"- best A2 by dev WER: {RESULTS.get(\'best_A_by_dev_wer\')} (round 1: base 26.97, ftA-s100 20.53, ftA-s200 19.36)",\n'
      '         f"- public transcripts: {RESULTS.get(\'public_b\')}", f"- notes: {RESULTS.get(\'notes\')}",')
patch(i, '"Next (Colab): copy out/asr/* to artifacts/asr/, then run_eval --config configs/kaggle_<tag>_names.yaml "\n'
         '         "--model artifacts/models/v1-epoch2 --conditions C for kgbase, ftA-s100, ftA-s200; compare with "\n'
         '         "lifemate.eval.report ... --vs (paired bootstrap).",',
      '"Next: put out/asr/* on the Modal volume under /artifacts/asr/ and score the v2 agent on dev C with the best ftA2 "\n'
      '         "transcripts (configs/v2_ftA2_s<step>.yaml); adopt round 2 only if dev C beats the ftA-s100 transcripts.",')

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for kind, src in cells:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
out = HERE / "lifemate_asr_round2.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
