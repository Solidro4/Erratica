from __future__ import annotations

from typing import Any


class LlamaCppModel:
    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 2048,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it before using LlamaCppModel."
            ) from exc

        self.temperature = temperature
        self.client = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **kwargs)

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        result = self.client.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return result["choices"][0]["message"]["content"].strip()
