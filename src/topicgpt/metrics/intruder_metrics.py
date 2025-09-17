import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from topicgpt.TopicRepresentation import Topic
from .base import BaseMetric

class ADC(BaseMetric):
    """
    Average Document Coherence (ADC) metric for topic models.
    We consider for each topic a set S of randomly sampled documents from the other topics and then 
    measure the average similarity of the elements in S to the documents in the topic. 
    Here, a lower similarity is of course better. 
    """

    def __init__(
        self,
        n_docs=-1, # -1 means all docs in the cluster
        n_intruder_docs=1,
    ):

        self.n_docs = n_docs
        self.n_intruder_docs = n_intruder_docs

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
            "metric_name": "Average Document Coherence (ADC) ",
            "n_docs": self.n_docs,
            "n_intruder_docs": self.n_intruder_docs,
            "embedding_model_name": self.metric_embedder,
            "metric_range": "0 to 1, smaller is better",
            "description": " the average cosine similarity between every word in a topic and an intruder word.",
        }

        return info
    
    def score_one_intr_per_cluster(
                            self, 
                            topic_list: list[Topic], 
                            new_embeddings: bool = True, # have yet to implement for new_embeddings
                            random_state: int | None = None
                            ):
        """
        For each topic, sample one random intruder embedding from the documents of all other topics,
        compute cosine similarities between that intruder and every document in the current topic,
        and return the average similarity per topic.

        Behavior with self.n_docs:
         - if self.n_docs <= 0 (default -1) -> use all documents in each topic
         - if self.n_docs > 0 -> select top-k most representative documents per topic
           by similarity to the topic centroid (use Topic.centroid_hd when available).
        """
        # reproducible RNG
        rng = np.random.default_rng(random_state)

        # normalize: list of 2D arrays (n_docs_in_topic, dim) or empty arrays
        emb_clusters = []
        for t in topic_list:
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

        scores = []
        for i, cluster_emb in enumerate(emb_clusters):
            # skip empty topic
            if cluster_emb.size == 0:
                scores.append(np.nan)
                continue

            # build flattened pool of other-topic embeddings for intruder sampling
            other = [np.atleast_2d(c) for j, c in enumerate(emb_clusters) if j != i and c.size > 0]
            if len(other) == 0:
                scores.append(np.nan)
                continue

            other_embs = np.vstack(other)  # shape (N_other_docs, dim)

            # sample a single intruder
            intr_idx = int(rng.integers(0, other_embs.shape[0]))
            intr_embedding = other_embs[intr_idx]

            # compute similarity between intruder and all docs in current topic
            sim = cosine_similarity(intr_embedding.reshape(1, -1), cluster_emb)  # (1, n_docs)
            scores.append(float(np.mean(sim)))

        return np.array(scores)

    # def score_one_intr(self, topic_list, new_embeddings=True):
    #     if new_embeddings:
    #         self.embeddings = None
    #     return np.mean(self.score_one_intr_per_cluster(topic_list, new_embeddings))

    def score_per_cluster(
                    self, 
                    topic_list: list[Topic], 
                    new_embeddings=True
                ):
        
        if new_embeddings:
            self.embeddings = None
        score_lis = []
        for _ in range(self.n_intruder_docs):  # iterate over the number of intruder docs
            score_per_cluster = self.score_one_intr_per_cluster(
                topic_list, new_embeddings=False
            )  # calculate the intruder score, but re-use embeddings
            score_lis.append(score_per_cluster)  # and append to list

        res = np.vstack(
            score_lis
        ).T  # stack all scores and transpose to get a (n_cluster, n_intruder docs) matrix

        mean_scores = np.mean(res, axis=1)
        ntopics = len(topic_list)
        results = {}
        for k in range(ntopics):
            # short, unambiguous label for each cluster + optional preview of first document
            preview = ""
            t = topic_list[k]
            # If Topic instance, use its documents field for preview; otherwise fallback to list-of-strings
            if isinstance(t, Topic):
                if getattr(t, "documents", None):
                    preview = " - " + str(t.documents[0])[:40].replace("\n", " ").strip()
            elif isinstance(t, list) and len(t) > 0 and isinstance(t[0], str):
                preview = " - " + t[0][:40].replace("\n", " ").strip()
            label = f"cluster_{k}{preview}"
            results[label] = float(np.round(mean_scores[k], 5))

        return results

    def score(
            self, 
            topics: list[Topic], 
            new_embeddings:bool=False
            ):
        
        if new_embeddings:
            self.embeddings = None

        return float(np.mean(list(self.score_per_cluster(topics).values())))