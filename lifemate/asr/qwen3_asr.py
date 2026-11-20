"""Qwen3-ASR backend (transformers path of the qwen-asr package; greedy decoding, offline full-context mode)."""
from __future__ import annotations


class Qwen3ASR:
    def __init__(self, model_id: str, revision: str | None, language: str | None, batch_size: int, max_new_tokens: int,
                 tag: str | None = None):
        """`model_id` is a hub id (with `revision`) or a local merged model dir (revision None, `tag` names it)."""
        import torch
        from qwen_asr import Qwen3ASRModel

        self.name = f"qwen3_asr-{model_id.rstrip('/').split('/')[-1]}-{(revision or tag or 'local')[:12]}"
        self.language = language
        kw = {"revision": revision} if revision else {}
        self.model = Qwen3ASRModel.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda:0",
            max_inference_batch_size=batch_size, max_new_tokens=max_new_tokens, **kw,
        )
        gc = self.model.model.generation_config
        gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None  # deterministic greedy

    def transcribe(self, paths: list[str], contexts: list[str] | None = None) -> list[str]:
        if not paths:
            return []
        res = self.model.transcribe(audio=list(paths), context=list(contexts) if contexts else "",
                                    language=[self.language] * len(paths) if self.language else None)
        return [r.text.strip() for r in res]
