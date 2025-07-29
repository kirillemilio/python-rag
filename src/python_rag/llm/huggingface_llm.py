"""Contains implementation of hugging face llm model."""

from __future__ import annotations

import time
from typing import Iterator, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..dto import ChatHistory, LLMResponse, LLMStreamItem
from ..dto.chat_message import RolesMappingTypedDict
from .llm_interface import ILLM


class WithLogitsProtocol(Protocol):
    """Contains implementation of with logits protocol."""

    @property
    def logits(self) -> torch.Tensor:
        """Get logits as pytorch tensor.

        Returns
        -------
        torch.Tensor
            logits pytorch tensor of shape (b, l, t)
        """
        ...


class HuggingfaceLLM(ILLM):
    """Huggingface LLM model implementation."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    name: str

    def __init__(self, name: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

    def get_model_name(self) -> str:
        """Get name of llm model.

        Returns
        -------
        str
            name of model.
        """
        return self.name

    def get_model(self) -> AutoModelForCausalLM:
        """Get underlying huggingface model.

        Returns
        -------
        AutoModelForCausalLM
            huggingface large language model.
        """
        return self.model

    def get_tokenizer(self) -> AutoTokenizer:
        """Get huggingface llm tokenizer corresponding to the underlying model.

        Returns
        -------
        AutoTokenizer
            huggingface large language model tokenizer.
        """
        return self.tokenizer

    def get_response_on_query(self, query: str, temperature: float | None = None) -> LLMResponse:
        """Get llm response on given input query.

        Returns
        -------
        query : str
            input text query over which llm will generate a response.
        temperature : float | None
            temperature of sampling.
            Default is None meaning that argmax sampling strategy
            will be used.
        """
        stream_items = [
            stream_item
            for stream_item in self.stream_response_on_query(query=query, temperature=temperature)
        ]
        num_tokens = sum(stream_item.get_num_tokens() for stream_item in stream_items)
        content = ''.join([stream_item.get_content() for stream_item in stream_items])
        return LLMResponse(
            content=content,
            num_tokens=num_tokens,
            model_name=self.get_model_name(),
            timestamp=time.time(),
        )

    def stream_response_on_query(
        self, query: str, temperature: float | None = None
    ) -> Iterator[LLMStreamItem]:
        """Stream llm response.

        Yields
        ------
        LLMStreamItem
            llm stream item containing
            text per chunk and number of tokens in chunk.
        """
        generated_ids: list[int] = []
        last_generated_text: str = ''
        last_generated_index: int = 0
        with torch.no_grad():
            input_ids: torch.Tensor = self.tokenizer([query], return_tensors='pt').to(  # type: ignore
                device=self.model.device  # type: ignore
            )
            res: WithLogitsProtocol = self.model(input_ids['input_ids'], use_cache=True)  # type: ignore
            past_key_values: torch.Tensor = res.past_key_values  # type: ignore

            if temperature is None:
                next_token_id = torch.argmax(res.logits[0, -1, :], dim=-1, keepdim=True)
            else:
                probs = res.logits.softmax(dim=-1).mul(temperature)
                next_token_id = torch.multinomial(probs[0, -1, :], num_samples=1)

            generated_ids.append(int(next_token_id.cpu().item()))
            next_token_id = next_token_id.unsqueeze_(0)

            current_text = self.tokenizer.decode(  # type: ignore
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            next_yield_text = current_text[len(last_generated_text) :]
            if len(next_yield_text):
                last_generated_text = current_text
                yield LLMStreamItem(
                    content=next_yield_text, num_tokens=len(generated_ids) - last_generated_index
                )
                last_generated_index = len(generated_ids)

            while True:
                res = cast(
                    WithLogitsProtocol,
                    self.model(next_token_id, past_key_values=past_key_values, use_cache=True),  # type: ignore
                )
                if temperature is None:
                    next_token_id = torch.argmax(res.logits[0, -1, :], dim=-1, keepdim=True)
                else:
                    probs = res.logits.softmax(dim=-1).mul(temperature)
                    next_token_id = torch.multinomial(probs[0, -1, :], num_samples=1)

                if next_token_id.item() == int(self.tokenizer.eos_token_id):  # type: ignore
                    break

                generated_ids.append(int(next_token_id.cpu().item()))
                next_token_id = next_token_id.unsqueeze_(0)

                current_text = self.tokenizer.decode(  # type: ignore
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                next_yield_text = current_text[len(last_generated_text) :]
                if len(next_yield_text):
                    last_generated_text = current_text
                    yield LLMStreamItem(
                        content=next_yield_text,
                        num_tokens=len(generated_ids) - last_generated_index,
                    )
                    last_generated_index = len(generated_ids)

    def get_response_on_chat(
        self,
        chat_history: ChatHistory,
        temperature: float | None = None,
        roles_mapping: RolesMappingTypedDict | None = None,
    ) -> LLMResponse:
        """Get response on input chat history.

        Parameters
        ----------
        chat_history : ChatHistory
            chat history instance containing
            chat history messages.
        temperature : float | None
            temperature of sampling.
            Default is None meaning that argmax sampling
            strategy will be used.
        roles_mapping : RolesMappingTypedDict | None
            roles mapping typed dictionary defines mapping
            for standard roles like 'user', 'assistant', 'system'.
            Default is None meaning default roles will be used.
        """
        return self.get_response_on_query(
            query='\n'.join(chat_history.format_messages(roles_mapping=roles_mapping)),
            temperature=temperature,
        )

    def stream_response_on_chat(
        self,
        chat_history: ChatHistory,
        temperature: float | None = None,
        roles_mapping: RolesMappingTypedDict | None = None,
    ) -> Iterator[LLMStreamItem]:
        """Stream llm response on chat history.

        Parameters
        ----------
        chat_history : ChatHistory
            chat history instance containing
            chat history messages.
        temperature : float | None
            temperature of sampling.
            Default is None meaning that argmax sampling
            strategy will be used.
        roles_mapping : RolesMappingTypedDict | None
            roles mapping typed dictionary defines mapping
            for standard roles like 'user', 'assistant', 'system'.
            Default is None meaning default roles will be used.

        Yields
        ------
        LLMStreamItem
            llm stream item containing text per chunk and
            number of tokens in chunk
        """
        yield from self.stream_response_on_query(
            query='\n'.join(chat_history.format_messages(roles_mapping=roles_mapping))
        )
