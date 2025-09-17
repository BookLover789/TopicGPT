from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseMetric
from topicgpt.TopicRepresentation import Topic
import re
import numpy as np

# to avoid circular import: only importing TopicGPT for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from topicgpt.TopicGPT import TopicGPT

class ADS(BaseMetric):
    """
    Average Description Similarity (ADS) metric for topic models.
    The average cosine similarity of the embedding of each document d
    to the embedding of the description of the topic assigned to d.
    """

    def __init__(
        self,
        n_docs: int = -1, # -1 means all docs in the cluster
    ):
        self.n_docs = n_docs

    def get_info(self):
        """
        Get information about the metric.

        Returns
        -------
        dict
            Dictionary containing model information including metric name,
            number of top words, number of intruders, embedding model name,
            metric range and metric description.
        """

        info = {
            "metric_name": "Average Description Similarity (ADS)",
            "n_docs": self.n_docs,
            "metric_range": "0 to 1, smaller is better",
            "description": " the average cosine similarity between every word in a topic and an intruder word.",
        }

        return info

    def score(
            self, 
            topics: list[Topic],
            tm: "TopicGPT" = None, 
            ):
        
        """
        If `tm` is provided and `topics` is None, uses tm.topic_lis.
        Otherwise uses provided topics. Keeps backward compatibility.
        """
        # resolve topics source
        if topics is None:
            raise AttributeError("topics is None. Please provide topics.")

        source_name = "topic_lis" if getattr(tm, "topic_lis", None) is not None else "topic_list"
        print(f"Using {source_name} as source ({len(topics)} topics)")

        # if any topic lacks a description, ensure tm is available and generate descriptions
        if any(not getattr(t, "topic_description", None) for t in topics):
            # ensure tm is available before attempting to describe topics
            assert tm is not None, "tm is None but topic descriptions are missing"
            assert hasattr(tm, "describe_topics"), "tm has no method 'describe_topics' to generate descriptions"
            try:
                print(f'Generating descriptions for topics using tm.describe_topics...')
                topics = tm.describe_topics(topics)
            except Exception as e:
                print(f"tm.describe_topics failed: {e}")

        # Clean the descriptions
        descriptions = []
        for t in topics:
            descriptions.append(self.clean_description(t.topic_description))

        # Embed the topic descriptions
        topic_desc_embeddings = tm.embedder.get_embeddings(descriptions)["embeddings"]

        # Embed the documents in each topic
        emb_clusters = []
        for t in topics:
            emb = getattr(t, "document_embeddings_hd", None)
            arr = np.atleast_2d(np.asarray(emb)) if emb is not None else np.empty((0, 0))

            # If n_docs > 0, select the most representative documents (top-k by similarity to centroid)
            # If n_docs <= 0 (e.g. -1) then use all documents (no truncation)
            if self.n_docs is not None and self.n_docs > 0 and arr.size > 0:
                n_available = arr.shape[0]
                if n_available > self.n_docs:
                    # prefer the precomputed centroid if present on the Topic object
                    centroid = getattr(t, "centroid_hd", None)
                    if centroid is None:
                        centroid = arr.mean(axis=0)
                    else:
                        centroid = np.asarray(centroid).reshape(-1)
                    sims = cosine_similarity(centroid.reshape(1, -1), arr).flatten()
                    # pick indices of top-k most similar docs (descending)
                    top_idx = np.argsort(sims)[-self.n_docs:][::-1]
                    arr = arr[top_idx]

            emb_clusters.append(arr)

        similarity_scores = []

        for i in range(len(emb_clusters)):
            desc = topic_desc_embeddings[i].reshape(1, -1)
            docs = emb_clusters[i]
            sims = cosine_similarity(desc, docs)  # Shape: (1, n_docs)
            similarity_scores.append(np.nanmean(sims))  # Average similarity for the topic

        return np.nanmean(similarity_scores)


    @staticmethod
    def clean_description(desc: str) -> str:
        if not desc:
            return ""
        # remove surrounding quotes
        desc = desc.strip().strip('\'"')
        # remove markdown bold/italic and inline code
        desc = re.sub(r'(\*\*|\*|`)+', '', desc)
        # remove headings like "##" or "**Aspects**:"
        desc = re.sub(r'^#{1,6}\s*', '', desc, flags=re.MULTILINE)
        # remove numbered list numbering (1. , 2. ), bullets (-, *, +)
        desc = re.sub(r'^\s*[\-\*\+]\s+', '', desc, flags=re.MULTILINE)
        desc = re.sub(r'^\s*\d+\.\s+', '', desc, flags=re.MULTILINE)
        # replace multiple newlines/whitespace with single space
        desc = re.sub(r'\s+', ' ', desc)
        return desc.strip()