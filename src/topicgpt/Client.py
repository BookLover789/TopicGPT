from abc import ABC, abstractmethod

class Client(ABC):
    def __init__(
            self, 
            api_key: str
        ) -> None:
        
        self.api_key = api_key
        self.initialize_client()

    @abstractmethod
    def initialize_client(self):
        """Initialize the client based on the given parameters."""
        pass
    
    @abstractmethod
    def count_tokens(self, messages, model=None)->int:
        """Count the number of tokens in the given messages."""
        pass

    @abstractmethod
    def chat_completion(self, messages, model=None, temperature=0.5, **kwargs):
        """Send a chat completion request and return a unified completion object."""
        pass

    @abstractmethod
    def get_str_from_completion_object(self, completion) -> str:
        """Extract the response string from the provider's completion object."""
        pass
    
    def __getattr__(self, name):
        """Delegate attribute access to the self.client object, if set."""
        if 'client' in self.__dict__:
            return getattr(self.client, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
