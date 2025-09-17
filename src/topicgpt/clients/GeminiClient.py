from ..Client import Client

class GeminiClient(Client):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        super().__init__(api_key)
        self.initialize_client()

    def initialize_client(self):
        import google.generativeai as genai
        self.client = genai.configure(api_key=self.api_key)
        # Safety check: try a minimal chat completion
        test_messages = [{"role": "user", "content": "ping"}]
        # Use chat_completion to check API key/quota
        self.chat_completion(test_messages, model="gemini-pro")

    def chat_completion(self, messages, model=None, temperature=0.5, **kwargs):
        # Gemini expects a single prompt string, not a list of messages
        prompt = "\n".join(m['content'] for m in messages)
        if model is None:
            model = "gemini-pro"
        try:
            model_obj = self.client.GenerativeModel(model)
            return model_obj.generate_content(prompt, temperature=temperature, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

    def count_tokens(self, messages, model=None):
        # Gemini does not provide a tokenizer; use a rough estimate (1 token ≈ 4 chars)
        n_tokens = 0
        for message in messages:
            for key, value in message.items():
                if key == "content":
                    n_tokens += max(1, len(value) // 4)
        return n_tokens

    def get_str_from_completion_object(self, completion) -> str:
        # Gemini's response: completion.text
        return completion.text