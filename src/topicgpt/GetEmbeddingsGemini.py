import numpy as np
from tqdm import tqdm

class GetEmbeddingsGemini:
    """
    This class allows you to compute embeddings of text using the Gemini API.
    """

    def __init__(
            self,
            client, 
            api_key: str,
            embedding_model: str = "gemini-embedding-001",
            output_dimensionality: int = 1536
        ) -> None:
        from google.genai import types
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.output_dimensionality = output_dimensionality
        self.client = client
        self.types = types  # <-- Save types as an attribute

    @staticmethod    
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def get_embeddings_doc_split(self, corpus: list[list[str]], n_tries=3) -> list[dict]:
        embeddings = []
        errors = []
        texts = []
        for doc_chunks in tqdm(corpus, desc="Embedding documents"):
            text = " ".join(doc_chunks)
            for attempt in range(n_tries):
                try:
                    result = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=[text],
                        config=self.types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
                    )
                    emb_obj = result.embeddings[0]
                    embeddings.append(np.array(emb_obj.values))
                    errors.append(None)
                    texts.append(text)
                    break
                except Exception as e:
                    if attempt == n_tries - 1:
                        embeddings.append(None)
                        errors.append(e)
                        texts.append(text)
        api_res_list = [
            {"embedding": emb, "text": txt, "errors": err}
            for emb, txt, err in zip(embeddings, texts, errors)
        ]
        return api_res_list

    def convert_api_res_list(self, api_res_list: list[dict]) -> dict:
        # Always return an embedding for every input (None replaced with zeros)
        embeddings = []
        output_dim = self.output_dimensionality
        for api_res in api_res_list:
            emb = api_res["embedding"]
            if emb is None:
                # Fill with zeros if embedding failed
                emb = np.zeros(output_dim)
            embeddings.append(emb)
        embeddings = np.array(embeddings)
        corpus = [api_res["text"] for api_res in api_res_list]
        errors = [api_res["errors"] for api_res in api_res_list]
        return {"embeddings": embeddings, "corpus": corpus, "errors": errors}

    def get_embeddings(self, corpus: list[str]) -> dict:
        corpus_split = [[doc] for doc in corpus]
        corpus_emb = self.get_embeddings_doc_split(corpus_split)
        self.corpus_emb = corpus_emb
        res = self.convert_api_res_list(corpus_emb)   
        return res