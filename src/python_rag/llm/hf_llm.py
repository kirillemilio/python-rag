"""Contains implementation of hugging face llm model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch

from ..dto import ChatHistory, LLMResponse
from .llm_interface import ILLM


class HfLLM(ILLM):
    @classmethod
    def clean_generated_text(cls, raw_text: str) -> str:
        return (
            raw_text.replace('▁', ' ')
            .replace('</s>', '')
            .replace('<s>', '')
            .replace('<|assistant|>', '')
            .replace('<|user|>', '')
            .replace('<0x0A>', '\n')
            .strip()
        )

    def get_response_on_query(self, query: str, temperature: float | None = None) -> LLMResponse:
        raise NotImplementedError()

    def stream_response_on_query(
        self, query: str, temperature: float | None = None
    ) -> Iterator[str]:
        with torch.no_grad():
            input_ids = self.tokenizer([query], return_tensors='pt').to(device=self.model.device)
            res = self.model(input_ids['input_ids'], use_cache=True)
            past_key_values = res.past_key_values

            if temperature is None:
                next_token_id = torch.argmax(res.logits[0, -1, :], dim=-1, keepdim=True)
            else:
                probs = res.logits.softmax(dim=-1).mul(temperature)
                next_token_id = torch.multinomial(probs[0, -1, :], num_samples=1)

            next_token_id = next_token_id.unsqueeze_(0)
            next_token = self.tokenizer.decode(
                [next_token_id.cpu().numpy().reshape(-1).item()], skip_special_tokens=True
            )
            next_token_cleaned = self.clean_generated_text(next_token)
            if len(next_token_cleaned):
                yield next_token_cleaned

            while True:
                res = self.model(next_token_id, past_key_values=past_key_values, use_cache=True)
                if temperature is None:
                    next_token_id = torch.argmax(res.logits[0, -1, :], dim=-1, keepdim=True)
                else:
                    probs = res.logits.softmax(dim=-1).mul(temperature)
                    next_token_id = torch.multinomial(probs[0, -1, :], num_samples=1)

                next_token_id = next_token_id.unsqueeze_(0)
                next_token = self.tokenizer.decode(
                    [next_token_id.cpu().numpy().reshape(-1).item()], skip_special_tokens=True
                )
                next_token_cleaned = self.clean_generated_text(next_token)
                if len(next_token_cleaned):
                    yield next_token_cleaned

    def get_response_on_chat(
        self, chat_history: ChatHistory, temperature: float | None = None
    ) -> LLMResponse:
        raise NotImplementedError()

    def stream_response_on_chat(
        self, chat_history: ChatHistory, temperature: float | None = None
    ) -> Iterator[str]:
        raise NotImplementedError()
