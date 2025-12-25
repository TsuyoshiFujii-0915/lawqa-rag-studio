"""LM Studio OpenAI-compatible client wrapper."""
from __future__ import annotations

import os
from typing import Sequence

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.llm.base import LlmClient, LlmResponse


class LmStudioOpenAIClient(LlmClient):
    """LLM client for LM Studio OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        """Initialize client.

        Args:
            base_url: Base URL of LM Studio endpoint.
            model: Model identifier.
            api_key: Token string (can be dummy for LM Studio).
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def generate(self, query: str, context: Sequence[Chunk]) -> LlmResponse:
        """Generate answer with context.

        Args:
            query: User query.
            context: Retrieved chunks.

        Returns:
            Generated answer text.
        """
        system_context = "\n\n".join(chunk["text"] for chunk in context[:5])
        api_key = self.api_key or os.getenv("LMSTUDIO_API_KEY", "")
        base_url = self.base_url or "http://localhost:1234/v1"
        if not api_key:
            raise RuntimeError("LMSTUDIO_API_KEY is not set; cannot call LM Studio endpoint.")

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        inputs = [
            {"role": "system", "content": f"Use the provided legal context to answer. Context:\n{system_context}"},
            {"role": "user", "content": query},
        ]
        resp = client.responses.create(model=self.model, input=inputs)
        output_text = getattr(resp, "output_text", "") or ""
        if not output_text:
            for item in resp.output or []:
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        text = getattr(content, "text", None)
                        if text:
                            output_text += text
        usage_dict = None
        usage_obj = getattr(resp, "usage", None)
        if usage_obj:
            usage_dict = getattr(usage_obj, "to_dict", lambda: usage_obj)()
        return {
            "text": output_text,
            "raw_text": output_text,
            "usage": usage_dict,
            "raw_output": getattr(resp, "output", None),
        }


__all__ = ["LmStudioOpenAIClient"]
