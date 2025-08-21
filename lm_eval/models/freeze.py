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
        model_cfg: Optional[dict] = None,
        wandb_id: Optional[str] = None,
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
        model.load_state_dict(ckpt["model_opt"]["model"], strict=False)
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

    @property
    def max_length(self) -> int:
        # NOTE: Turn on truncation to avoid errors on long inputs.
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 512

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            prompt, gen_kwargs = request.args
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

    def _loglikelihood(self, requests):
        prompts = [req[0] for req in requests]
        responses = [req[1] for req in requests]

        # Tokenize all prompts and responses
        prompt_toks = [
            self._tokenize(prompt, allowed_special=self.tokenizer.special_tokens_set)
            for prompt in prompts
        ]
        response_toks = [
            self._tokenize(response, allowed_special=self.tokenizer.special_tokens_set)
            for response in responses
        ]

        # Create joint tokens for each request
        joint_toks = [
            torch.cat((prompt_tok, response_tok), dim=1)
            for prompt_tok, response_tok in zip(prompt_toks, response_toks)
        ]

        # Pad sequences to same length for batching
        max_len = max(joint_tok.shape[1] for joint_tok in joint_toks)
        batch_size = len(joint_toks)

        padded_joint = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.bool, device=self.device
        )

        for i, joint_tok in enumerate(joint_toks):
            seq_len = joint_tok.shape[1]
            padded_joint[i, :seq_len] = joint_tok[0]
            attention_mask[i, :seq_len] = True

        # Get logits for the batch
        with torch.no_grad():
            logits = self.model(
                padded_joint[:, :-1]
            )  # Shape: (batch_size, seq_len-1, vocab_size)

        results = []
        for i, (response_tok, joint_tok) in enumerate(zip(response_toks, joint_toks)):
            response_len = response_tok.shape[1]
            seq_len = joint_tok.shape[1] - 1  # -1 because we passed [:,:-1] to model

            # Extract logits corresponding to response tokens
            logits_response = logits[i, seq_len - response_len : seq_len]

            probs_response = logits_to_probs(logits_response)
            probs_response = torch.gather(probs_response, 1, response_tok.T)

            most_likely = torch.argmax(logits_response, dim=-1)
            is_most_likely = torch.equal(most_likely, response_tok[0])

            results.append((probs_response.sum().item(), int(is_most_likely)))

        return results

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        # Process requests in chunks of size 8
        chunk_size = 8
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i : i + chunk_size]
            # Extract prompt, response pairs from the chunk
            chunk_args = [request.args for request in chunk]
            chunk_results = self._loglikelihood(chunk_args)
            res.extend(chunk_results)

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        res = []

        # Process requests in chunks of size 8
        chunk_size = 8
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i : i + chunk_size]
            # Extract prompt, response pairs from the chunk
            chunk_args = []
            for request in chunk:
                response, *_ = request.args
                prompt = "<|endoftext|>"
                chunk_args.append((prompt, response))

            chunk_results = self._loglikelihood(chunk_args)
            for ll, _ in chunk_results:
                res.append((ll,))

        return res
