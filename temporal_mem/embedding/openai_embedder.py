from typing import List


class OpenAIEmbedder:
    """
    Wrapper around an embedding model.

    Day 1: just a stub; returns a dummy vector.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.api_key = api_key
        self.model = model

    def embed(self, text: str) -> List[float]:
        """
        Day 1: return a fixed-size dummy vector so the type shape is correct.
        """
        return [0.0] * 16  # will change to real size later
