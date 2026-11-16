"""Ready-made ``build`` functions for the common case: one model, one prompt.

Nothing here is required. ``build`` can be any function from a configuration
to a callable, and that is how a chain, a crew of agents or a retrieval
pipeline is plugged in:

    def build(config):
        agent = make_my_agent(instructions=config["rules"], model=config["model"])
        return lambda request: agent.run(request)
"""

from __future__ import annotations

from typing import Any, Callable, List

from erratica.core.types import System


def render_prompt(config: Any, *, base: str = "", rules: str = "rules", examples: str = "examples") -> str:
    """The system prompt for a configuration: the base text, then the learned
    rules, then the learned examples."""
    parts: List[str] = [base.strip()] if base.strip() else []
    learned = [str(rule) for rule in (config.get(rules) or [])]
    if learned:
        parts.append("Rules:\n" + "\n".join(f"- {rule}" for rule in learned))
    shown = config.get(examples) or []
    if shown:
        blocks = [f"Request: {item['input']}\nGood answer: {item['output']}" for item in shown]
        parts.append("Examples of good answers:\n\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


def prompted(
    llm: Callable[[str, Any], Any],
    *,
    base: str = "",
    rules: str = "rules",
    examples: str = "examples",
) -> Callable[[Any], System]:
    """A ``build`` for a system that is one model call with a system prompt.

    ``llm(system_prompt, request)`` is your model. The system prompt is
    ``base`` plus whatever is under the configuration's ``rules`` and
    ``examples`` keys, which is where the built-in proposers put what they
    learn.
    """

    def build(config: Any) -> System:
        system_prompt = render_prompt(config, base=base, rules=rules, examples=examples)
        return lambda request: llm(system_prompt, request)

    return build


def from_adapter(model: Any, *, route: str = "general") -> Callable[[str, Any], str]:
    """Use one of Erratica's model adapters (``OllamaModel``, ``LlamaCppModel``,
    or anything with ``generate(prompt, system_prompt, route)``) as an ``llm``."""
    return lambda system_prompt, request: model.generate(str(request), system_prompt, route)


def ollama(model: str, **options: Any) -> Callable[[str, Any], str]:
    """A local Ollama model as an ``llm``. Needs a running Ollama server and
    no Python dependencies."""
    from erratica.engine.ollama_adapter import OllamaModel

    return from_adapter(OllamaModel(model, **options))
