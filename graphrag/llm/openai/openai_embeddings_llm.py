# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""

from typing_extensions import Unpack
from langchain_community.embeddings import OllamaEmbeddings
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes


class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration
        
        self.embedder = OllamaEmbeddings(model=self.configuration.model, base_url=self.configuration.api_base)

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        embedding_list = []
        for inp in input:
            embedding_list.append(self.embedder.embed_query(inp))
        return embedding_list
