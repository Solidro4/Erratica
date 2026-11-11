"""A deterministic stand-in for a language model.

``RuleAwareDemoModel`` is a simulator, not a model. It gives a vague answer
by default and a structured one when the system prompt contains matching
instructions or examples, which is enough to exercise the learning loop in
tests and demos without downloading weights. Numbers produced with it show
that the loop's plumbing works. They say nothing about how much a real
model improves; for that, run the loop against a real model.
"""

from __future__ import annotations

import re


class RuleAwareDemoModel:
    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        if route == "finance":
            return self._finance_response(prompt, system_prompt)
        if route == "support":
            return self._support_response(prompt, system_prompt)
        return self._general_response(prompt, system_prompt)

    def _finance_response(self, prompt: str, system_prompt: str) -> str:
        segments = self._extract_segments(prompt)
        wants_segments = self._mentions_any(system_prompt, ("segment", "breakdown"))
        wants_bullets = self._mentions_any(system_prompt, ("bullet", "numbered", "structured"))
        required_terms = self._extract_required_terms(system_prompt)

        if segments and wants_segments:
            lines = ["Quarterly revenue breakdown:"]
            total = 0.0
            for index, (name, value, numeric_value) in enumerate(segments, start=1):
                prefix = f"{index}. " if wants_bullets else ""
                lines.append(f"{prefix}{name}: {value}")
                total += numeric_value
            lines.append(f"Total: ${total:.1f}M")
            return "\n".join(lines)

        if required_terms:
            included = [term for term in required_terms if term.lower() in prompt.lower()]
            if included:
                return "Key themes: " + ", ".join(included)

        return "Quarterly revenue was healthy overall, with strong performance across the portfolio."

    def _support_response(self, prompt: str, system_prompt: str) -> str:
        if self._mentions_any(system_prompt, ("single short sentence", "avoid timelines", "avoid detail")):
            return "Received. We will review it."

        required_terms = self._extract_required_terms(system_prompt)
        wants_resolution = self._mentions_any(
            system_prompt,
            ("resolution", "next steps", "follow-up window", "policy", "timeframe", "numbered"),
        ) or bool(required_terms)
        if not wants_resolution:
            return "Thanks for reaching out. We are reviewing the issue and will follow up soon."

        issue, resolution, timeframe = self._support_resolution(prompt)
        order_id = self._extract_order_id(prompt)
        lines = [f"I reviewed your request about {issue}."]
        if order_id:
            lines.append(f"1. Reference: order {order_id}")
            lines.append(f"2. Resolution: {resolution}")
            lines.append(f"3. Timeline: {timeframe}")
            lines.append("4. Next step: we will send a confirmation update as soon as the action is complete.")
        else:
            lines.append(f"1. Resolution: {resolution}")
            lines.append(f"2. Timeline: {timeframe}")
            lines.append("3. Next step: we will confirm the change once it has been applied.")
        return "\n".join(lines)

    def _general_response(self, prompt: str, system_prompt: str) -> str:
        if self._mentions_any(system_prompt, ("bullet", "numbered", "structured")):
            return "\n".join(["1. Summary", "2. Key point", "3. Next action"])
        return f"Summary: {prompt.splitlines()[0][:72]}"

    def _extract_segments(self, prompt: str) -> list[tuple[str, str, float]]:
        matches = re.findall(
            r"(?m)^\s*([A-Za-z][A-Za-z\s\-]+):\s*\$?([0-9]+(?:\.[0-9]+)?)([MK]?)\s*$",
            prompt,
        )
        results: list[tuple[str, str, float]] = []
        for name, number, suffix in matches:
            numeric_value = float(number)
            formatted_value = f"${number}{suffix or 'M'}"
            if suffix.upper() == "K":
                numeric_value = numeric_value / 1000.0
            results.append((name.strip(), formatted_value, numeric_value))
        return results

    def _extract_required_terms(self, system_prompt: str) -> list[str]:
        match = re.search(
            r"required terms.*?:\s*([A-Za-z0-9,\-\s]+)",
            system_prompt,
            flags=re.IGNORECASE,
        )
        if not match:
            return []
        return [term.strip() for term in match.group(1).split(",") if term.strip()]

    def _extract_order_id(self, prompt: str) -> str | None:
        match = re.search(r"order\s*#?\s*([A-Z0-9\-]+)", prompt, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1)

    def _support_resolution(self, prompt: str) -> tuple[str, str, str]:
        lowered = prompt.lower()
        if "refund" in lowered:
            return (
                "your refund request",
                "we are approving the refund back to the original payment method",
                "3-5 business days",
            )
        if "charged twice" in lowered or "duplicate charge" in lowered:
            return (
                "a duplicate charge on the account",
                "we are reversing the duplicate charge and confirming the corrected balance",
                "2-3 business days",
            )
        if "damaged" in lowered or "broken" in lowered:
            return (
                "a damaged item delivery",
                "we are sending a replacement shipment at no extra cost",
                "1 business day",
            )
        if "tracking" in lowered or "shipment" in lowered or "delayed" in lowered:
            return (
                "a shipment delay",
                "we are escalating the stalled shipment to the carrier and checking the delivery scan",
                "24 hours",
            )
        if "cancel" in lowered or "cancellation" in lowered or "subscription" in lowered:
            return (
                "a cancellation and billing issue",
                "we are stopping renewal charges and refunding the latest billing period",
                "3-5 business days",
            )
        return ("the reported support issue", "we are applying a concrete resolution to the request", "24 hours")

    def _mentions_any(self, text: str, tokens: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in tokens)
