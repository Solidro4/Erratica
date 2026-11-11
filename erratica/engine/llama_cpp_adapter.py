from __future__ import annotations

from typing import Any


class LlamaCppModel:
    """Runs a local ``.gguf`` model through llama-cpp-python.

    ``temperature`` defaults to 0 so replays are repeatable: the learning
    loop compares scores before and after a change, and sampling noise
    would show up as fake gains and fake regressions.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run: pip install erratica[llama]"
            ) from exc

        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **kwargs)

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        result = self.client.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (result["choices"][0]["message"]["content"] or "").strip()
