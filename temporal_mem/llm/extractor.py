from ..models import FactCandidate


class FactExtractor:
    """
    LLM-based fact extraction component.

    Day 1:
    - No real LLM calls.
    - Just define interface and return an empty list.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        base_prompt: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_prompt = base_prompt

    def extract_from_message(self, message: str) -> list[FactCandidate]:
        """
        Given a single message, return a list of FactCandidate objects.

        Day 1: return empty list. Implementation comes later.
        """
        return []

    def extract_from_messages(self, messages: list[dict[str, str]]) -> list[FactCandidate]:
        """
        Given a list of messages ({role, content}), return facts.

        Day 1: use the last user message, but still return empty list.
        """
        return []
