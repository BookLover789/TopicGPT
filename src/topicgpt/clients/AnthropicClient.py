from ..Client import Client

class AnthropicClient(Client):
    def __init__(
            self,
            api_key: str
        ) -> None:
        """Initialize the Anthropic client."""
        super().__init__(api_key)
        self.initialize_client()

    def initialize_client(self):
        """Initialize the Anthropic client."""
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def count_tokens(self, messages, model=None):
        assert model is None, "Model parameter is not used in Anthropic token counting."
        # Approximate: 1 token ≈ 4 characters (similar to OpenAI's rule of thumb)
        n_tokens = 0
        for message in messages:
            for key, value in message.items():
                if key == "content":
                    n_tokens += max(1, len(value) // 4)
        return n_tokens

    def chat_completion(self, messages, model=None, temperature=0.5, **kwargs):
        try:
            return self.client.messages.create(
                model=model,
                max_tokens=kwargs.get('max_tokens', 512),
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def get_str_from_completion_object(self, completion) -> str:
        # Anthropic returns: completion.content[0].text
        return completion.content[0].text