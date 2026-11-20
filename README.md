# LifeMate voice agent — SLM harness (Zalo AI Challenge 2026)

Our own end-to-end pipeline for **LifeMate – Agentic AI That Turns Your Voice Into Action** (Zalo AI Challenge 2026, Task 1,
SLM category: the whole system under 6B parameters, offline, 30 s per sample). It turns a spoken Vietnamese request plus the
conversation history into tool calls on the organisers' `ToolEnv`.

**Public leaderboard: 43.10** (submission of 7 Nov 2026, `configs/v2r_all_molab.yaml` agent + ASR LoRA `B2-s941`).

> The team's final pipeline later moved to a teammate's stronger stack (Qwen3.5-4B + Gipformer, see
> [Datyth/ZaloAI-Challenge-lifemate](https://github.com/Datyth/ZaloAI-Challenge-lifemate)); there the step-level vote of the
> three released adapters scored 63.17 on public. This repository is the earlier, independent harness.

## Pipeline

| Stage | Model | Training |
|---|---|---|
| ASR | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) + LoRA (2,349,217,408 params) | round 2: attention + MLP LoRA, 3 epochs; `B2-s941` = all 5,036 TRAIN clips (`kaggle/`, 2 × T4) |
| Agent | [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) + LoRA (2,516,756,480 params) | trajectory SFT on gold final turns + earlier (history) turns + synthetic conversations, weekday/date table in the prompt, per-trajectory loss normalisation (`v2r` recipe); final fit `v2r_all`, 675 steps |
| Controller | `lifemate/controller`, `lifemate/agent/constrain.py` | agent loop with argument hygiene before every call, x-ref provenance checks and a loop guard; constrained decoding of tool names, parameter names, enum values and x-ref IDs; 30 s budget |

Total **4.87B** parameters (< 6B). `scripts/param_inventory.py` counts them.

Measured on our frozen dev split (`artifacts/splits/v1.json`, 250 TRAIN conversations incl. held-out apps), condition C
(ASR transcripts): v1 agent 38.21 → v2r 44.24 (paired +6.03, 95% CI [1.27, 10.64]).

## Layout

```
lifemate/        package: agent (prompt, tool-call format, constraints), asr (Qwen3-ASR wrapper, cache, context hints),
                 controller (loop, checks), data (loader, splits, trajectories, mixes), train (SFT, merge), eval (metric,
                 gold replay, reports), infer (run.py: submission writer)
configs/         YAML configs; each run config names its base (v1.yaml -> v2.yaml -> v2r.yaml -> v2r_all*.yaml ...)
scripts/         data building (trajectories, v2 data, final fit, ASR manifest/cache), ASR LoRA training, checks
kaggle/          ASR LoRA training/transcription notebooks for Kaggle 2 x T4 (DDP)
colab/           Colab VM runners (setup, sync, detached jobs, evals, public prediction)
cloud/           Modal and marimo (molab) runners
synth/conv/      card builder and validator for synthetic conversations (generated data not included)
tests/           unit tests (metric, controller, constraints, prompt, trajectories); some need the data kit
```

## Running

The competition data is **not** included: download the kit from the challenge portal and point the code at the folder that
holds `train/` and `public_test/`:

```bash
pip install -r requirements-train.txt      # training env (transformers / peft); ASR uses qwen-asr (requirements-infer.txt)
export LIFEMATE_DATA_ROOT=/path/to/train+public_test
python scripts/make_splits.py              # or use the frozen artifacts/splits/v1.json
python scripts/build_trajectories.py       # gold trajectories replayed in ToolEnv
python scripts/build_v2_data.py            # finals + earlier turns + synthetic mix
python -m lifemate.train.sft --config configs/v2r_all.yaml
python -m lifemate.infer.run --config configs/v2r_all.yaml --data $LIFEMATE_DATA_ROOT --model <merged agent dir> \
    --out result/submission.jsonl          # online ASR; --asr_cache <jsonl> for development
```

Each script prints its options with `--help`. The `colab/`, `cloud/` and `kaggle/` folders hold the exact runners we used.

## Not included

- The organisers' data (audio, transcripts, tool environments, labels) and anything derived from it (trajectories,
  synthetic conversations, transcripts, submissions).
- Trained weights.

## Licence

Code: MIT (see `LICENSE`). Base models keep their own licences (MiniCPM5-2B and Qwen3-ASR: Apache-2.0). The competition
data belongs to the organisers.
