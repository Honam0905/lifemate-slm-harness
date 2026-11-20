"""Load the agent model and generate greedily. One engine serves eval (batched) and inference (batch of one).

Prompts are rendered by agent/prompt.render_generation_prompt, exactly as the training collator renders them, and
tokenised without extra special tokens (the template already emits BOS).
"""
from __future__ import annotations

import time

import torch


class HFEngine:
    def __init__(self, model_path: str, revision: str | None = None, adapter: str | None = None,
                 max_new_tokens: int = 256, max_batch: int = 16, max_batch_tokens: int = 64_000,
                 prefix_cache: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").cuda().eval()
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.max_batch = max_batch
        self.max_batch_tokens = max_batch_tokens
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        eos = model.generation_config.eos_token_id
        self.eos_ids = sorted({im_end, *(eos if isinstance(eos, list) else [eos])} - {None})
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_ids[0]
        self.n_params = sum(p.numel() for p in model.parameters())
        self.gen_seconds = 0.0
        # prefix cache: each step's prompt extends the previous one (prompt + action + tool result), so the KV cache
        # of the longest common token prefix is reused instead of re-reading 6-10K tokens every step
        self.prefix_cache = prefix_cache
        self._key, self._ids, self._kv = None, [], None
        self.cached_tokens = 0
        self.computed_tokens = 0
        self._vocab = None
        self.constrained_steps = 0

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _batches(self, lens: list[int]) -> list[list[int]]:
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        out, cur = [], []
        for i in order:
            longest = lens[cur[0]] if cur else lens[i]
            if cur and (len(cur) >= self.max_batch or (len(cur) + 1) * (longest + self.max_new_tokens) > self.max_batch_tokens):
                out.append(cur); cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    def _processors(self, rows, prompt_len: int):
        """Logits processors for constrained decoding (None when unconstrained)."""
        if not rows:
            return None
        from transformers import LogitsProcessorList
        from .constrain import ToolCallConstraint, Vocab
        if self._vocab is None:
            self._vocab = Vocab(self.tok)
        self._proc = ToolCallConstraint(self._vocab, rows, prompt_len)
        return LogitsProcessorList([self._proc])

    def _count_constrained(self) -> None:
        if getattr(self, "_proc", None) is not None:
            self.constrained_steps += self._proc.n_constrained
            self._proc = None

    def _decode_new(self, new: list[int]) -> str:
        eos = set(self.eos_ids)
        cut = next((k for k, t in enumerate(new) if t in eos), len(new))
        # keep special tokens: in MiniCPM5 "<function", "<param", "</param>" ... are special tokens themselves
        return self.tok.decode(new[:cut], skip_special_tokens=False)

    @torch.no_grad()
    def _generate_cached(self, prompt: str, key, rows=None) -> str:
        from transformers import DynamicCache
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if key != self._key or self._kv is None:
            try:
                self._kv = DynamicCache(config=self.model.config)
            except TypeError:
                self._kv = DynamicCache()
            self._key, self._ids = key, []
        n = 0
        for a, b in zip(self._ids, ids):
            if a != b:
                break
            n += 1
        n = min(n, len(ids) - 1)                      # at least one token must be computed
        if self._kv.get_seq_length() > n:
            self._kv.crop(n)
        self.cached_tokens += n
        self.computed_tokens += len(ids) - n
        x = torch.tensor([ids], device="cuda")
        gen = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), past_key_values=self._kv,
                                  max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                  top_k=None, eos_token_id=self.eos_ids, pad_token_id=self.pad_id,
                                  logits_processor=self._processors(rows, len(ids)))
        self._count_constrained()
        seq = gen[0].tolist()
        self._ids = seq[: self._kv.get_seq_length()]
        return self._decode_new(seq[len(ids):])

    @torch.no_grad()
    def generate(self, prompts: list[str], keys: list | None = None, rows: list | None = None) -> list[str]:
        """Greedy continuation of each prompt. With prefix_cache on and a single prompt, `keys` (the episode id)
        lets consecutive steps of one episode reuse the KV cache. `rows` (constrain.RowSpec per prompt) turns on
        constrained decoding."""
        t0 = time.time()
        if self.prefix_cache and keys is not None and len(prompts) == 1:
            out = [self._generate_cached(prompts[0], keys[0], rows)]
            self.gen_seconds += time.time() - t0
            return out
        enc_all = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        outs: list[str | None] = [None] * len(prompts)
        for b in self._batches([len(e) for e in enc_all]):
            width = max(len(enc_all[i]) for i in b)
            ids = torch.full((len(b), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(b), width), dtype=torch.long)
            for r, i in enumerate(b):  # left padding
                e = enc_all[i]
                ids[r, width - len(e):] = torch.tensor(e)
                mask[r, width - len(e):] = 1
            gen = self.model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None,
                                      eos_token_id=self.eos_ids, pad_token_id=self.pad_id,
                                      logits_processor=self._processors([rows[i] for i in b] if rows else None, width))
            self._count_constrained()
            for r, i in enumerate(b):
                outs[i] = self._decode_new(gen[r, width:].tolist())
        self.gen_seconds += time.time() - t0
        return outs
