"""OpenAI Responses API client wrapper."""
from __future__ import annotations

import os
from typing import Sequence

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.clients.openai_client import create_openai_client
from lawqa_rag_studio.llm.base import LlmClient, LlmResponse


class OpenAIResponsesClient(LlmClient):
    """LLM client using OpenAI Responses API."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        """Initialize client.

        Args:
            base_url: OpenAI API base URL.
            model: Model name.
            api_key: API key value.
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
            Model-generated answer text.
        """
        system_context = "\n\n".join(chunk["text"] for chunk in context[:5])
        api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot call OpenAI.")

        client = create_openai_client(api_key=api_key, base_url=base_url)
        inputs = [
            {"role": "system", "content": f"Use the provided legal context to answer. Context:\n{system_context}"},
            {"role": "user", "content": query},
        ]
        resp = client.responses.create(model=self.model, input=inputs)
        # Prefer SDK convenience property if available.
        output_text = getattr(resp, "output_text", "") or ""
        if not output_text:
            # Fallback: aggregate output_text items manually.
            for item in resp.output or []:
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        text = getattr(content, "text", None)
                        if text:
                            output_text += text
        usage_dict = None
        usage_obj = getattr(resp, "usage", None)
        if usage_obj:
            # Responses API returns input_tokens / output_tokens.
            usage_dict = getattr(usage_obj, "to_dict", lambda: usage_obj)()
        return {
            "text": output_text,
            "raw_text": output_text,
            "usage": usage_dict,
            "raw_output": getattr(resp, "output", None),
        }


__all__ = ["OpenAIResponsesClient"]
