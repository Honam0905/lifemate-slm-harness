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

## What you provide, and where it goes

| Item | Required | Where |
|---|---|---|
| Organisers' data kit (`train/`, `public_test/`, each with its `zaic_tools/`) | yes | anywhere; `export LIFEMATE_DATA_ROOT=/path/to/train+public_test` |
| Agent model | yes | an HF id or a local dir, passed with `--model`; must be MiniCPM5-family (see below) |
| ASR model | yes | base `Qwen/Qwen3-ASR-1.7B` (default), or your merged checkpoint in `models/asr-<name>/` referenced by a config with `asr.model_path` + `asr.tag` (as `configs/v2_ftB2_s941.yaml`) |
| Synthetic conversations | optional | `synth/conv/cards/batch_*/<card>.json` + `synth/conv/accepted/*.jsonl`; without them the pipeline writes 0 synthetic items and skips that source |
| Rename maps | only for `configs/v2r*.yaml` | `synth/rename/out/rename_part{1,2,3}.json` (format: app → `alt_ids`, `alt_descriptions`, `tools` → `alt_names`, `params`) |

The synthetic conversations and rename maps we used are derived from the organisers' data, so they are not published.
Everything else is either in this repository or produced by the steps below.

### Using your own models

- **ASR:** any Qwen3-ASR checkpoint (`asr.model_id`, or `asr.model_path` with an `asr.tag` so its transcripts get their own
  cache).
- **Agent:** any MiniCPM5-family checkpoint (base, or one trained here). The tool-call renderer, parser and constrained
  decoding implement MiniCPM5's chat markup only (`agent.tool_format: minicpm_xml`, `lifemate/agent/actions.py` and
  `constrain.py`). Another model family (e.g. Qwen3.5) needs its own format added there first.

## Setup

Two Python environments (the ASR package pins an older transformers):

| Env | For | Install |
|---|---|---|
| train | data building, tokenising, SFT, merge, dev eval | Python 3.13: `pip install -r requirements-train.txt` (torch 2.11, transformers 5.22, peft 0.21.2) |
| infer | ASR transcription and the submission run | Python 3.12: `pip install -r requirements-infer.txt` (qwen-asr 0.0.6 → transformers 4.57.6, torch) |

Prompts and tokens are byte-identical across the two (`scripts/check_env_parity.py`). Run everything from the repository
root. Every script prints its options with `--help` and takes `--config` (default `configs/v1.yaml`). **Use the same
`--config` for every step of one recipe**: the ASR transcript cache is keyed by the config's `asr` section, so steps 2-5 must
agree.

Hardware: the data steps run on CPU; ASR transcription, SFT, merge, evaluation and inference need a CUDA GPU. We trained the
LoRA agent on an A100 40 GB (Colab); inference was designed for the organisers' 24 GB A30 and its 30 s per-sample budget.

Environment variables:

| Variable | Meaning |
|---|---|
| `LIFEMATE_DATA_ROOT` | the data kit folder (overrides `data.root` of the configs) |
| `LIFEMATE_ARTIFACTS` | where outputs go (default `artifacts/`) |
| `LIFEMATE_INFER_DATA` | default `--data` of `lifemate.infer.run` (`/data`, the Docker layout) |
| `LIFEMATE_PUSH_REPO` | optional: `lifemate.train.sft` uploads each checkpoint to this Hugging Face repo, created private (needs a Hugging Face login or `HF_TOKEN`) |

## Running the pipeline

**A. Recipe development (dev split held out)**, with `configs/v2.yaml`:

| # | Step | Device / env | Command |
|---|---|---|---|
| 1 | Dev split (or keep the committed one) | CPU | `python scripts/make_splits.py --force` |
| 2 | ASR transcripts of all TRAIN clips | GPU, infer | `python scripts/build_asr_cache.py --config configs/v2.yaml --split train` |
| 3 | Final-turn trajectories, gold and ASR copies, replayed in ToolEnv | CPU | `python scripts/build_trajectories.py --config configs/v2.yaml --variants gold asr` |
| 4 | Earlier-turn examples (+ synthetic items if present) | CPU | `python scripts/build_v2_data.py --config configs/v2.yaml --asr` |
| 5 | Tokenise, enforce the 16K budget | CPU | `python scripts/tokenize_trajectories.py --config configs/v2.yaml --parts train dev` |
| 6 | Agent LoRA SFT | GPU, train | `python -m lifemate.train.sft --config configs/v2.yaml` |
| 7 | Merge the adapter | GPU, train | `python -m lifemate.train.merge --config configs/v2.yaml --ckpt artifacts/runs/v2-minicpm5-2b/ckpt-epoch2 --out models/v2` |
| 8 | Dev evaluation (A = gold text, C = ASR text) | GPU, train | `python -m lifemate.eval.run_eval --config configs/v2.yaml --name v2 --model models/v2 --conditions A C` |

Step 3 must run before step 4, and only with `--variants gold asr`: step 4 writes the earlier/synthetic files. Step 8 builds
`artifacts/gold/train_hashes.json` on first use (or run `python -m lifemate.eval.gold`, which also self-checks the scorer).

**B. Final fit (all 1,545 conversations, nothing held out)**, after A has chosen the recipe:

| # | Step | Command |
|---|---|---|
| 9 | Final-fit data (needs step 2) | `python scripts/build_final_fit.py --config configs/v2_all.yaml` |
| 10 | Tokenise | `python scripts/tokenize_trajectories.py --config configs/v2_all.yaml --parts train` |
| 11 | SFT, then merge | `python -m lifemate.train.sft --config configs/v2_all.yaml`, then step 7 with `--config configs/v2_all.yaml --ckpt artifacts/runs/v2-all-minicpm5-2b/ckpt-epoch2 --out models/v2_all` |
| 12 | Public-test submission (GPU, infer env) | `python -m lifemate.infer.run --config <config naming your ASR> --data $LIFEMATE_DATA_ROOT/public_test --model models/v2_all --out result/submission.jsonl` |

`configs/v2_all.yaml` is the no-rename final fit. Our 43.10 model used `configs/v2r_all.yaml` instead, which adds app/tool
renaming: it needs the rename maps and `python scripts/apply_rename.py --config configs/v2r_all.yaml` between steps 9 and 10.
Never score a final-fit model on dev: it has seen dev.

Step 12 runs ASR online with the config's ASR (`configs/v1.yaml`: base Qwen3-ASR-1.7B; `configs/v2_ftB2_s941.yaml`: a merged
checkpoint in `models/asr-ftB2-s941`). `--asr_cache <jsonl>` passes saved transcripts instead (step 2 with
`--split public_test`). The output is the organisers' submission format and is validated with their `zaic_tools`.

**ASR fine-tuning (optional):** `python scripts/build_asr_manifest.py` (train partition only; dev/audit never enter ASR
training), `python scripts/train_asr_lora.py train --out artifacts/asr_ft/run1`, then
`python scripts/train_asr_lora.py merge --ckpt artifacts/asr_ft/run1/step<N> --out models/asr-<name>` and a config with
`asr.model_path: models/asr-<name>` and `asr.tag: <name>`. The `kaggle/` notebooks run the same training on 2 × T4 (DDP);
`kaggle/build_all_manifest.py` builds the all-clips manifest of our `B2` model.

### Outputs (`artifacts/`, or `$LIFEMATE_ARTIFACTS`)

```
splits/v1.json                         frozen dev split (committed)
asr/<backend>/<split>.jsonl            transcript cache, one {"audio", "text"} per clip
trajectories/v1/<part>.<variant>.jsonl trajectories (gold, asr, earlier, synth; all* for the final fit; *_r renamed)
tokenized/<run_name>/<part>.<variant>.pt
runs/<run_name>/ckpt-epoch<N>, ckpt-step<N>   LoRA checkpoints
eval/<name>/<part>.<condition>/        dev rollouts and reports
gold/train_hashes.json                 gold final-state hashes for scoring
asr_ft/                                ASR fine-tuning manifests and checkpoints
```

### Hosted runners

`colab/` (Colab VM over SSH with the Colab CLI), `cloud/` (Modal volume `lifemate-v2`, marimo/molab) and `kaggle/` are the
exact runners we used. They assume our session names, volumes and dataset names (Kaggle paths carry a
`<your-kaggle-user>` placeholder), so adapt those before use. `colab/v2_smoke.sh` is the reference order of steps 2-6.

## What was checked before publishing

From a clean copy of this repository with the data kit, on CPU:
- steps 1, 3, 4, 5 and 9-10, plus SFT data loading up to the model download. Step 1 reproduces `artifacts/splits/v1.json`
  byte for byte. Final-turn replays have 0 problems. The final fit gives 1,545 gold + 1,545 ASR + 3,369 earlier-turn
  trajectories = 6,459 sequences (41.5M tokens), all within the 16K budget with 0 misaligned spans. The steps that need
  ASR transcripts used a stand-in cache with the gold text (step 2 needs a GPU).
- `python -m lifemate.eval.gold`: self-checks pass.
- unit tests: 36 of 37 pass; the remaining one needs public-test ASR transcripts (step 2 with `--split public_test`).

Not run here (GPU): ASR transcription, SFT itself, merge, evaluation and inference. That code is unchanged from the runs
that produced the numbers above.

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
