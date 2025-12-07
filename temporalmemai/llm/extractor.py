# temporalmemai/llm/extractor.py

from __future__ import annotations

import json
import os

from openai import OpenAI
from pydantic import ValidationError

from ..models import FactCandidate
from ..prompts.fact_extraction_prompt import GENERIC_FACT_EXTRACTION_PROMPT


def _strip_code_fences(text: str) -> str:
    """
    Remove ``` and ```json fences if the model wraps JSON in them.
    """
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return text


class FactExtractor:
    """
    LLM-based fact extraction component.

    Responsibilities:
    - Call an LLM with a structured prompt (GENERIC_FACT_EXTRACTION_PROMPT).
    - Parse the JSON response into FactCandidate objects.
    - Provide two entrypoints:
        - extract_from_message(str)
        - extract_from_messages(List[{"role": ..., "content": ...}])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        base_prompt: str = GENERIC_FACT_EXTRACTION_PROMPT,
    ) -> None:
        """
        api_key:
            OpenAI API key. If None, will use OPENAI_API_KEY from env.
        model:
            Chat completion model to use.
        temperature:
            Sampling temperature (keep low for extraction).
        base_prompt:
            The base instruction + few-shot prompt. Can be overridden
            by callers for custom extraction behavior.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for FactExtractor")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.base_prompt = base_prompt

    # -------------------------------------------------------------------------
    # Core extraction methods
    # -------------------------------------------------------------------------

    def extract_from_message(self, message: str) -> list[FactCandidate]:
        """
        Given a single user message (string), return a list of FactCandidate.

        Example:
            message = "I'm Nikhil, I live in Hyderabad and love hiking."
            -> [
                 FactCandidate(
                    text="User's name is Nikhil",
                    category="profile",
                    slot="name",
                    confidence=0.98,
                 ),
                 ...
               ]
        """
        full_prompt = f"""{self.base_prompt}

            Input: {message}
            Output:
        """

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You extract facts into strict JSON."},
                {"role": "user", "content": full_prompt},
            ],
        )

        raw_output = (resp.choices[0].message.content or "").strip()
        cleaned = _strip_code_fences(raw_output)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # If the model returns invalid JSON, fail gracefully with no facts.
            return []

        raw_facts = data.get("facts", [])
        if not isinstance(raw_facts, list):
            return []

        results: list[FactCandidate] = []
        for f in raw_facts:
            if not isinstance(f, dict):
                continue

            text = str(f.get("text", "")).strip()
            if not text:
                continue

            category = str(f.get("category", "other")).strip() or "other"
            slot = f.get("slot", None)

            stability = f.get("stability")
            temporal_scope = f.get("temporal_scope")
            kind = f.get("kind")

            # duration fields may be null or an int
            duration_days_raw = f.get("duration_in_days")
            try:
                duration_in_days = int(duration_days_raw) if duration_days_raw is not None else None
            except (TypeError, ValueError):
                duration_in_days = None

            duration_hours_raw = f.get("duration_in_hours")
            try:
                duration_in_hours = (
                    int(duration_hours_raw) if duration_hours_raw is not None else None
                )
            except (TypeError, ValueError):
                duration_in_hours = None

            duration_minutes_raw = f.get("duration_in_minutes")
            try:
                duration_in_minutes = (
                    int(duration_minutes_raw) if duration_minutes_raw is not None else None
                )
            except (TypeError, ValueError):
                duration_in_minutes = None

            conf_raw = f.get("confidence", 1.0)
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = 1.0
            confidence = max(0.0, min(1.0, confidence))

            try:
                fact = FactCandidate(
                    text=text,
                    category=category,
                    slot=slot,
                    confidence=confidence,
                    stability=stability,
                    temporal_scope=temporal_scope,
                    kind=kind,
                    duration_in_days=duration_in_days,
                    duration_in_hours=duration_in_hours,
                    duration_in_minutes=duration_in_minutes,
                )
                results.append(fact)
            except ValidationError:
                continue

        return results

    def extract_from_messages(self, messages: list[dict[str, str]]) -> list[FactCandidate]:
        """
        Given a list of chat messages (each with "role" and "content"),
        extract facts from the last user message.

        messages example:
            [
              {"role": "system", "content": "..."},
              {"role": "user", "content": "I'm Nikhil..."},
              {"role": "assistant", "content": "..."},
            ]

        v1 behavior:
        - Filter for role == "user"
        - Take the last one
        - Run extract_from_message() on its content
        """
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return []
        last_user_msg = user_messages[-1].get("content", "")
        if not last_user_msg:
            return []
        return self.extract_from_message(last_user_msg)


# -------------------------------------------------------------------------
# Manual demo: python -m temporalmemai.llm.extractor
# -------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = FactExtractor()

    examples = [
        "Hi.",
        "The weather is nice today.",
        "I'm Nikhil, I live in Hyderabad and work as a product manager at an AI company.",
        "I love going on hikes and playing football on weekends.",
        "My monthly budget for gadgets is around 50,000 rupees.",
        "Tomorrow I have a meeting at 5pm with my manager.",
        "I bought a new MacBook last week.",
    ]

    for msg in examples:
        print("\n==============================")
        print("Input:", msg)
        facts = extractor.extract_from_message(msg)
        for f in facts:
            print("-", f.dict())
