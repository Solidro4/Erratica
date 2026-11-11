from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Optional


class OllamaModel:
    """Talks to a local Ollama server. Needs no Python dependencies.

    Use it as the answering model, as a proposer, or inside a
    ``ModelCritic``::

        small = OllamaModel("llama3.2:3b")
        large = OllamaModel("qwen2.5:14b")
        ai = Erratica(model=small, proposer=large)

    ``temperature`` defaults to 0 so replays are repeatable. For reasoning
    models, pass ``think=False`` to get the answer without the scratch work
    (otherwise the reasoning can use up ``max_tokens`` before any answer).
    """

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 300.0,
        think: Optional[bool] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.think = think
        self.options = dict(options or {})

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        options = {"temperature": self.temperature, "num_predict": self.max_tokens}
        options.update(self.options)
        payload = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.think is not None:
            payload["think"] = self.think
        request = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"could not reach Ollama at {self.host} (is `ollama serve` running, "
                f"and is the model '{self.model}' pulled?): {exc}"
            ) from exc
        content = (body.get("message") or {}).get("content") or ""
        # Reasoning models may wrap their scratch work in <think> tags.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
