from functools import lru_cache
from typing import Optional

import hydra
import tiktoken
import torch
from dfs.utils.logging import get_wandb_run
from freeze.models.generate import generate, logits_to_probs
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import simple_parse_args_string


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

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        args = simple_parse_args_string(arg_string)
        return cls(**args)

    def _tokenize(self, prompt, **kwargs):
        toks = self.tokenizer.encode(prompt, **kwargs)
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

    def _loglikelihood(self, prompt, response):
        prompt_tok = self._tokenize(
            prompt, allowed_special=self.tokenizer.special_tokens_set
        )
        response_tok = self._tokenize(
            response, allowed_special=self.tokenizer.special_tokens_set
        )
        joint_tok = torch.cat((prompt_tok, response_tok), dim=1)
        logits = self.model(joint_tok[:, :-1])[0]
        logits_response = logits[-response_tok.shape[1] :]

        probs_response = logits_to_probs(logits_response)
        probs_response = torch.gather(probs_response, 1, response_tok)[0]

        most_likely = torch.argmax(logits_response, dim=-1)
        is_most_likely = torch.equal(most_likely, response_tok[0])

        return (probs_response.sum().item(), int(is_most_likely))

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            prompt, response = request
            res.append(self._loglikelihood(prompt, response))

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            response, *_ = request
            prompt = "<|endoftext|>"
            ll, is_greedy = self._loglikelihood(prompt, response)
            res.append((ll,))

        return res
