# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""

from typing_extensions import Unpack
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)
from graphrag.query.llm.oai.embedding import OpenAICompatibleOllamaEmbedding
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
from graphrag.query.llm.oai.typing import OpenaiApiType
import logging
logging.basicConfig(level=logging.INFO)

class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration
        logging.info(f"OpenAIEmbeddingsLLM initialized with configuration: {self.configuration}")

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        embedding = await self.client.embeddings.create(
            input=input,
            **args,
        )
        return [d.embedding for d in embedding.data]


class OpenAIComaptibleOllamaEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM using Ollama."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration
        logging.info(f"OpenAIComaptibleOllamaEmbeddingsLLM initialized with configuration: {self.configuration}")
        self.text_embedder = OpenAICompatibleOllamaEmbedding(
            api_key=self.configuration.api_key,
            api_base=self.configuration.api_base,
            api_type=OpenaiApiType.OpenAI,
            model=self.configuration.model,
            deployment_name=self.configuration.model,
            max_retries=self.configuration.max_retries,
        )

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        embedding_list = []
        for inp in input:
            embedding_list.append(self.text_embedder.embed(inp,**kwargs))
        return embedding_list
    
    