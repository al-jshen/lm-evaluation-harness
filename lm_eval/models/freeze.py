import random
from functools import lru_cache
from typing import Optional

import hydra
import tiktoken
import torch
from dfs.utils.logging import get_wandb_run
from freeze.models.generate import generate
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("freeze")
class FreezeLM(LM):
    def __init__(
        self,
        ckpt_path: str,
        model_cfg: Optional[dict],
        wandb_id: Optional[str],
        encoding: str = "gpt2",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        if not model_cfg:
            assert wandb_id is not None, "Either model_cfg or wandb_id must be provided"
            model_cfg = get_wandb_run(wandb_id).config["trainer"]["model"]

        ckpt = torch.load(ckpt_path, weights_only=False, mmap=True)
        model = hydra.utils.instantiate(model_cfg)
        model.load_state_dict(ckpt["model"], strict=False)
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tiktoken.get_encoding(encoding)

    def _tokenize(self, prompt):
        toks = self.tokenizer.encode_ordinary(prompt)
        toks = torch.tensor(toks, dtype=torch.long, device=self.device)
        return toks.unsqueeze(0)

    def _flatten(self, x):
        return [i for sublist in x for i in sublist]

    @lru_cache
    def _get_stop_toks(self, stop):
        return self._flatten(self.tokenizer.encode_batch(stop))

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            prompt, gen_kwargs = request
            stop = gen_kwargs.get("until", None)
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            temperature = gen_kwargs.get("temperature", 1.0)
            top_p = gen_kwargs.get("top_p", 1.0)
            top_k = gen_kwargs.get("top_k", 0)
            out = (
                generate(
                    self.model,
                    self._tokenize(prompt),
                    max_new_tokens=max_gen_toks,
                    stop_tokens=self._get_stop_toks(stop) if stop else [],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                .cpu()
                .numpy()[0]
            )
            out = self.tokenizer.decode(out)
            res.append(out)

        return res

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append((-random.random(), False))

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return res
