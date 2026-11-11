from __future__ import annotations

import re
from typing import Optional

from erratica.domains import DomainRegistry


class RoutingPolicy:
    """Picks the domain whose keywords match the prompt best.

    Keyword routing is a default, not a requirement: pass ``route=`` to
    ``Erratica.generate`` when the caller already knows what kind of task
    it is, which is both cheaper and more reliable.
    """

    def __init__(self, registry: Optional[DomainRegistry] = None) -> None:
        self.registry = registry or DomainRegistry()

    def select_route(self, prompt: str) -> str:
        lowered = prompt.lower()
        best_name = "general"
        best_hits = 0
        for domain in self.registry.all():
            hits = sum(
                1
                for keyword in domain.keywords
                if re.search(r"\b" + re.escape(keyword.lower()), lowered)
            )
            if hits > best_hits:
                best_name = domain.name
                best_hits = hits
        return best_name
