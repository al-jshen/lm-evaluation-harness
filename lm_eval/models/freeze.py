from functools import lru_cache
from typing import Optional

import hydra
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
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
        batch_size: int = 8,
        stride: int = 64,
        **kwargs,
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
        self._batch_size = batch_size
        self.stride = stride

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        args = simple_parse_args_string(arg_string)
        return cls(**args)

    def _tokenize(self, prompt, **kwargs):
        toks = self.tokenizer.encode(prompt, **kwargs)
        toks = torch.tensor(toks, dtype=torch.long, device=self.device)
        return toks.unsqueeze(0)

    def _tokenize_batch(self, prompt, **kwargs) -> tuple[list[list[int]], np.ndarray]:
        toks = self.tokenizer.encode_batch(prompt, **kwargs)
        seq_lens = np.array([len(tok) for tok in toks])
        # toks = torch.nested.nested_tensor(
        #     toks, dtype=torch.long, layout=torch.jagged, device=self.device
        # )
        # toks = toks.to_padded_tensor(padding=0)
        return toks, seq_lens

    def _flatten(self, x):
        return [i for sublist in x for i in sublist]

    @lru_cache
    def _get_stop_toks(self, stop):
        return self._flatten(self.tokenizer.encode_batch(stop))

    @property
    def max_length(self) -> int:
        return 4096

    @property
    def max_gen_toks(self) -> int:
        return 512

    @property
    def batch_size(self) -> int:
        return self._batch_size

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
        prompts, responses = zip(*requests)

        prompt_toks, prompt_lens = self._tokenize_batch(
            prompts, allowed_special=self.tokenizer.special_tokens_set
        )
        response_toks, response_lens = self._tokenize_batch(
            responses, allowed_special=self.tokenizer.special_tokens_set
        )

        # Create joint tokens for each request
        joint_toks = [
            torch.tensor(prompt_tok + response_tok).to(self.device, dtype=torch.long)
            for prompt_tok, response_tok in zip(prompt_toks, response_toks)
        ]

        joint_toks = torch.nested.nested_tensor(
            joint_toks, dtype=torch.long, layout=torch.jagged, device=self.device
        )
        joint_toks = joint_toks.to_padded_tensor(padding=0)

        # Get logits for the batch
        with torch.no_grad():
            logits = self.model(
                joint_toks[:, :-1]
            )  # Shape: (batch_size, seq_len-1, vocab_size)

        results = []
        for i, (response_tok, joint_tok) in enumerate(zip(response_toks, joint_toks)):
            prompt_len = prompt_lens[i]
            response_len = response_lens[i]
            # Extract logits corresponding to response tokens
            logits_response = logits[i, prompt_len - 1 : prompt_len - 1 + response_len]
            probs_response = logits_to_probs(logits_response)
            # Gather probabilities of the actual response tokens
            probs_response = probs_response[torch.arange(response_len), response_tok]
            most_likely = torch.argmax(logits_response, dim=-1).tolist()
            is_most_likely = most_likely == response_tok
            results.append((probs_response.sum().item(), int(is_most_likely)))

        return results

    def _likelihood_sliding_window_batch(self, texts, stride: int = 512):
        encodings, seq_lens = self._tokenize_batch(
            texts, allowed_special=self.tokenizer.special_tokens_set
        )
        encodings = torch.nested.nested_tensor(
            encodings, dtype=torch.long, layout=torch.jagged, device=self.device
        )
        encodings = encodings.to_padded_tensor(padding=0)
        # prepend with eot token
        eot_pad = torch.full(
            (encodings.shape[0], 1), self.tokenizer.eot_token, device=self.device
        )
        encodings = torch.cat([eot_pad, encodings], dim=1)
        seq_len = encodings.shape[1]

        ll_sum = torch.zeros(
            encodings.shape[0], device=self.device
        )  # sum of negative log-likelihoods for each sequence
        num_loss_tokens = torch.zeros(
            encodings.shape[0], device=self.device, dtype=torch.long
        )  # number of tokens for which loss is computed
        # num_loss_tokens = seq_len - 1

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids = encodings[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                logits = self.model(input_ids[:, :-1])  # just the last token
                neg_log_likelihood = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids[:, 1:].view(-1),
                    reduction="none",
                ).view(logits.size(0), -1)

            likelihood_float_mask = (target_ids[:, 1:] != -100).float()

            neg_log_likelihood = (
                neg_log_likelihood * likelihood_float_mask
            )  # b x (seq_len - 1)

            ll_sum -= neg_log_likelihood.sum(dim=1)
            num_loss_tokens += likelihood_float_mask.sum(dim=1).to(torch.long)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        assert (num_loss_tokens + 1).tolist() == seq_lens

        # ppl = torch.exp(-ll_sum / num_loss_tokens) # NEGATIVE log likelihood for this
        return ll_sum  # , num_loss_tokens

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        # Process requests in chunks of size 8
        for i in tqdm(range(0, len(requests), self.batch_size), disable=disable_tqdm):
            chunk = requests[i : i + self.batch_size]
            # Extract prompt, response pairs from the chunk
            chunk_args = [request.args for request in chunk]
            chunk_results = self._loglikelihood(chunk_args)
            res.extend(chunk_results)

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        res = []

        for i in tqdm(range(0, len(requests), self.batch_size), disable=disable_tqdm):
            chunk = requests[i : i + self.batch_size]
            chunk_args = [request.args[0] for request in chunk]  # extract strings
            chunk_results = self._likelihood_sliding_window_batch(
                chunk_args, stride=self.stride
            )
            results = [(i.item(),) for i in chunk_results]
            res.extend(results)

        return res
