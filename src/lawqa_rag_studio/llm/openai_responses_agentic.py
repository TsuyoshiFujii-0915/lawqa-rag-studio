"""OpenAI Responses API client (agentic/tool-calling)."""
from __future__ import annotations

import json
import os
from typing import Sequence

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.llm.base import LlmClient, LlmResponse
from lawqa_rag_studio.retrieval.pipeline import retrieve
from lawqa_rag_studio.config.schema import AppConfig
from lawqa_rag_studio.vectorstore.qdrant_client import VectorStoreClient


class OpenAIResponsesAgenticClient(LlmClient):
    """LLM client that lets the model call the retriever as a tool."""

    def __init__(self, base_url: str, model: str, api_key: str, cfg: AppConfig, store: VectorStoreClient) -> None:
        """Initialize client.

        Args:
            base_url: OpenAI API base URL.
            model: Model name.
            api_key: API key value.
            cfg: Full app config (used for retriever params).
            store: Vector store to execute retrieval.
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.cfg = cfg
        self.store = store

    def generate(self, query: str, context: Sequence[Chunk]) -> LlmResponse:  # context unused in agentic mode
        """Generate answer by letting the model call the retriever tool."""
        api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot call OpenAI.")

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        tools = [
            {
                "type": "function",
                "name": "retrieve",
                "description": "Retrieve relevant legal chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": self.cfg.retriever.hybrid.top_k},
                    },
                    "required": ["query"],
                },
            }
        ]

        previous_response_id = None
        input_items = [
            {"role": "user", "content": query},
        ]

        for _ in range(3):  # safety cap
            resp = client.responses.create(
                model=self.model,
                input=input_items,
                tools=tools,
                previous_response_id=previous_response_id,
            )

            # Collect text if already present.
            output_text = getattr(resp, "output_text", "") or ""
            tool_calls = [
                item
                for item in (resp.output or [])
                if getattr(item, "type", None) == "function_call"
            ]
            if not tool_calls:
                usage_dict = getattr(resp.usage, "to_dict", lambda: resp.usage)() if getattr(resp, "usage", None) else None
                return {
                    "text": output_text,
                    "raw_text": output_text,
                    "usage": usage_dict,
                    "raw_output": getattr(resp, "output", None),
                }

            # Execute tool calls and build outputs
            tool_outputs = []
            for call in tool_calls:
                args = {}
                try:
                    args = json.loads(getattr(call, "arguments", "") or "{}")
                except Exception:
                    args = {}
                q = args.get("query", query)
                top_k = int(args.get("top_k", self.cfg.retriever.hybrid.top_k))
                chunks = retrieve(q, self.cfg, self.store)
                payload = [
                    {
                        "id": c.get("id"),
                        "text": c.get("text"),
                        "score": c.get("metadata", {}).get("rerank_score", c.get("metadata", {}).get("score")),
                    }
                    for c in chunks[:top_k]
                ]
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": getattr(call, "call_id", None) or getattr(call, "id", None),
                        "output": json.dumps(payload, ensure_ascii=False),
                    }
                )

            # Prepare next turn with tool outputs and previous_response_id
            previous_response_id = resp.id
            input_items = tool_outputs

        # Fallback: return last collected text if loop exits
        usage_dict = None
        if previous_response_id:
            usage_obj = getattr(resp, "usage", None)
            usage_dict = getattr(usage_obj, "to_dict", lambda: usage_obj)() if usage_obj else None
        return {
            "text": output_text,
            "raw_text": output_text,
            "usage": usage_dict,
            "raw_output": getattr(resp, "output", None),
        }


__all__ = ["OpenAIResponsesAgenticClient"]
