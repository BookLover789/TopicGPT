import numpy as np
from tqdm import tqdm

class GetEmbeddingsSentenceTransformer:
    """
    This class computes embeddings using any sentence-transformers model.
    """

    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            device: str = "cpu",
            batch_size: int = 32
        ) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    @staticmethod
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def get_embeddings_doc_split(self, corpus: list[list[str]], n_tries=1) -> list[dict]:
        """
        corpus: list of list of strings (each doc is a list of chunks, but we join them)
        """
        embeddings = []
        errors = []
        texts = []
        for doc_chunks in tqdm(corpus, desc="Embedding documents"):
            text = " ".join(doc_chunks)
            try:
                emb = self.model.encode(text, show_progress_bar=False)
                embeddings.append(np.array(emb))
                errors.append(None)
                texts.append(text)
            except Exception as e:
                embeddings.append(None)
                errors.append(e)
                texts.append(text)
        api_res_list = [
            {
                "embedding": emb, 
                "text": txt, 
                "errors": err
            }
            for emb, txt, err in zip(embeddings, texts, errors)
        ]
        return api_res_list

    def convert_api_res_list(self, api_res_list: list[dict]) -> dict:
        valid_embeddings = [api_res["embedding"] for api_res in api_res_list if api_res["embedding"] is not None]
        embeddings = np.array(valid_embeddings) if valid_embeddings else np.array([])
        corpus = [api_res["text"] for api_res in api_res_list]
        errors = [api_res["errors"] for api_res in api_res_list]
        return {"embeddings": embeddings, "corpus": corpus, "errors": errors}

    def get_embeddings(self, corpus: list[str]) -> dict:
        corpus_split = [[doc] for doc in corpus]
        corpus_emb = self.get_embeddings_doc_split(corpus_split)
        self.corpus_emb = corpus_emb
        res = self.convert_api_res_list(corpus_emb)
        return res