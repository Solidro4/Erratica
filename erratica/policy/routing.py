from __future__ import annotations


class RoutingPolicy:
    def select_route(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(token in lowered for token in ("revenue", "quarter", "segment", "client", "finance")):
            return "finance"
        if any(token in lowered for token in ("ticket", "refund", "customer", "support", "billing")):
            return "support"
        return "general"
