from abc import ABC
from ..Client import Client

class OpenAIClient(Client):
    def __init__(
            self, 
            api_key: str, 
            azure_endpoint: dict = None
        ) -> None:
        """Initialize the OpenAI or Azure OpenAI client based on the parameters."""
        self.azure_endpoint = azure_endpoint
        super().__init__(api_key)  # Call the parent class's __init__ to set api_key
        self.initialize_client()  # Initialize the client in the subclass
    
    def initialize_client(self):
        """Initialize the client based on the provided parameters (OpenAI or Azure)."""
        if self.azure_endpoint:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.azure_endpoint['api_version'],
                azure_endpoint=self.azure_endpoint['endpoint']
            )
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)

    def count_tokens(self, messages, model=None):
        import tiktoken
        if model is None:
            model = "gpt-3.5-turbo"
        encoding = tiktoken.encoding_for_model(model)
        n_tokens = 0
        for message in messages:
            for key, value in message.items():
                if key == "content":
                    n_tokens += len(encoding.encode(value))
        return n_tokens 
    
    def chat_completion(self, messages, model=None, temperature=0.5, **kwargs):
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
        
    def get_str_from_completion_object(self, completion) -> str:
        # OpenAI returns: completion.choices[0].message.content
        return completion.choices[0].message.content
