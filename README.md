# LifeMate voice agent — SLM harness (Zalo AI Challenge 2026)

Our own end-to-end pipeline for **LifeMate – Agentic AI That Turns Your Voice Into Action** (Zalo AI Challenge 2026, Task 1,
SLM category: the whole system under 6B parameters, offline, 30 s per sample). It turns a spoken Vietnamese request plus the
conversation history into tool calls on the organisers' `ToolEnv`.

**Public leaderboard: 43.10** (submission of 7 Nov 2026, `configs/v2r_all_molab.yaml` agent + ASR LoRA `B2-s941`).
Trained weights are not part of this repository: the pipeline trains and runs your own checkpoints.

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

## Using your own models

- **ASR:** any Qwen3-ASR checkpoint. Set `asr.model_id` (HF id) or `asr.model_path` (local merged dir) in a config, as
  `configs/v2_ftB2_s941.yaml` does.
- **Agent:** any MiniCPM5-family checkpoint (base, or one trained here), passed with `--model`. The tool-call renderer, parser
  and constrained decoding implement MiniCPM5's chat markup only (`agent.tool_format: minicpm_xml`, see
  `lifemate/agent/actions.py` and `constrain.py`). Another model family (e.g. Qwen3.5) needs its own format added there.

## Setup

The competition data is **not** included. Download the kit from the challenge portal and point the code at the folder that
holds `train/` and `public_test/`:

```bash
export LIFEMATE_DATA_ROOT=/path/to/train+public_test     # overrides data.root of the configs
```

Two Python environments (the ASR package pins an older transformers):

| Env | For | Install |
|---|---|---|
| train | SFT, merge, tokenising, dev eval | Python 3.13, `pip install -r requirements-train.txt` (torch 2.11, transformers 5.22, peft 0.21.2) |
| infer | ASR transcription and the submission run | Python 3.12, `pip install -r requirements-infer.txt` (qwen-asr 0.0.6 → transformers 4.57.6) |

Prompts and tokens are byte-identical across the two (`scripts/check_env_parity.py`). Every script prints its options with
`--help`; all of them take `--config` (default `configs/v1.yaml`).

## Running the pipeline

| # | Step | Device | Command |
|---|---|---|---|
| 1 | Dev split (or keep the committed one) | CPU | `python scripts/make_splits.py --force` |
| 2 | ASR transcripts of TRAIN, for the "asr" training copies | GPU, infer env | `python scripts/build_asr_cache.py --split train` |
| 3 | Gold trajectories replayed in ToolEnv | CPU | `python scripts/build_trajectories.py` (`--variants gold` works before step 2) |
| 4 | v2 data: earlier turns (+ synthetic items if present) | CPU | `python scripts/build_v2_data.py --config configs/v2.yaml [--asr]` |
| 5 | Tokenise, enforce the 16K budget | CPU | `python scripts/tokenize_trajectories.py --config configs/v2.yaml --parts train dev` |
| 6 | Agent LoRA SFT | GPU, train env | `python -m lifemate.train.sft --config configs/v2.yaml` |
| 7 | Merge the adapter | GPU, train env | `python -m lifemate.train.merge --config configs/v2.yaml --ckpt <checkpoint> --out models/<name>` |
| 8 | Dev evaluation (A gold text, C ASR text) | GPU | `python -m lifemate.eval.run_eval --config configs/v2.yaml --name <name> --model models/<name> --conditions A C` |
| 9 | Public-test submission | GPU, infer env | `python -m lifemate.infer.run --config <config naming your ASR> --data $LIFEMATE_DATA_ROOT/public_test --model models/<name> --out result/submission.jsonl` |

Notes:
- Step 9: `--config` is a config whose `asr` section names your ASR (`configs/v2_ftB2_s941.yaml` expects a merged
  checkpoint in `models/asr-ftB2-s941`; `configs/v1.yaml` uses the base Qwen3-ASR-1.7B). ASR runs online unless
  `--asr_cache <jsonl>` passes transcripts from step 2 run with `--split public_test`.
- ASR fine-tuning: `python scripts/build_asr_manifest.py`, then `python scripts/train_asr_lora.py train --out artifacts/asr_ft/run1`
  and `python scripts/train_asr_lora.py merge --ckpt artifacts/asr_ft/run1/step<N> --out models/asr-<name>`, or the
  `kaggle/` notebooks (2 × T4, DDP).
- `configs/v2r*.yaml` add app/tool renaming and need rename maps in `synth/rename/out/rename_part{1,2,3}.json` (not published:
  they are derived from the organisers' tool catalog). `configs/v2.yaml` runs without them.
- Synthetic conversations (`synth/conv/accepted`) are not published either; `build_v2_data.py` then writes 0 synthetic items
  and carries on. `synth/conv/make_cards.py` and `validate.py` show how they were built and checked.
- `colab/`, `cloud/` (Modal, marimo) and `kaggle/` hold the exact runners we used on hosted GPUs.

Checked from a fresh clone with the data kit (CPU only): step 1 reproduces `artifacts/splits/v1.json` byte for byte; step 3
(`--variants gold`) replays its trajectories with 0 problems; step 4 builds 2,331 earlier-turn examples; 33 of 37 unit tests
pass, and the other 4 need `transformers` installed.

## Tests

```bash
pip install pytest transformers
LIFEMATE_DATA_ROOT=/path/to/train+public_test python -m pytest -q
```

## Not included

- The organisers' data (audio, transcripts, tool environments, catalog, labels) and anything derived from it (trajectories,
  synthetic conversations, rename maps, transcripts, submissions).
- Trained weights.

## Licence

Code: MIT (see `LICENSE`). Base models keep their own licences (MiniCPM5-2B and Qwen3-ASR: Apache-2.0). The competition
data belongs to the organisers.
