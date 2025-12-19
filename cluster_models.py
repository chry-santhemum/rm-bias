from collections import defaultdict
from typing import Any
from loguru import logger

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

class ClusterModel:
    def __init__(
        self,
        embed_model_name: str="Qwen/Qwen3-Embedding-0.6B",
        embed_dim: int=32,
        dbscan_eps: float=0.3,
        pca_dim: int=8,
    ):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.embed_dim = embed_dim
        self.dbscan_eps = dbscan_eps
        self.pca_dim = pca_dim

    def embed(self, inputs: list[str]) -> np.ndarray:
        """Returns: [n_inputs, embed_dim]"""
        embs = self.embed_model.encode(inputs, dim=self.embed_dim)  # type: ignore
        logger.info(f"Embedded shape: {embs.shape}")
        return embs

    def reduce_embed(self, inputs: list[str]) -> np.ndarray:
        """Embed then do dimensionality reduction"""
        embs: np.ndarray = self.embed(inputs)
        pca = PCA(n_components=min(self.pca_dim, embs.shape[0], embs.shape[1]))
        reduced = pca.fit_transform(embs)
        logger.info(f"Reduced shape: {reduced.shape}")
        return reduced

    def cluster_kmeans(self, inputs: list[str], n_clusters: int) -> list[dict[str, Any]]:
        """
        Returns a list of cluster information. 
        Does NOT apply dimensionality reduction before clustering.
        """
        embs = self.embed(inputs)

        kmeans = KMeans(
            n_clusters=min(len(inputs), n_clusters), 
            random_state=10086, n_init="auto"
        )
        logger.info(f"Fitting KMeans for {len(inputs)} inputs into {n_clusters} clusters")
        kmeans.fit(embs)

        labels = [int(label) for label in kmeans.labels_.tolist()]  # type: ignore

        # Group points by cluster
        cluster_points = defaultdict(list)
        for input_idx, label in enumerate(labels):
            cluster_points[label].append(input_idx)

        results = []
        for cluster_idx in range(kmeans.n_clusters):
            content_indices = cluster_points[cluster_idx]

            if not content_indices:
                continue

            # Select representative sample: find the medoid (point that minimizes
            # sum of distances to all other points in the cluster)
            if len(content_indices) == 1:
                center_idx = content_indices[0]
            else:
                # Compute pairwise distances within cluster
                cluster_embeddings = embs[content_indices]
                pairwise_dists = pairwise_distances(cluster_embeddings, metric="cosine")

                # Find point with minimum sum of distances to all other points
                sum_dists = pairwise_dists.sum(axis=1)
                medoid_idx_in_cluster = np.argmin(sum_dists)
                center_idx = content_indices[medoid_idx_in_cluster]

            results.append(
                {
                    "cluster_idx": cluster_idx,
                    "center_idx": center_idx,
                    "center_input": inputs[center_idx],
                    "content_indices": content_indices,
                }
            )

        for result in results:
            assert result["center_idx"] in result["content_indices"]

        # Log cluster contents
        for result in results:
            cluster_inputs = [inputs[i] for i in result["content_indices"]]
            logger.info(
                f"Cluster {result['cluster_idx']} ({len(cluster_inputs)} items):\n"
                + "\n".join(f"  - {inp}" for inp in cluster_inputs)
            )

        return results

    def cluster_dbscan(self, inputs: list[str]) -> tuple[dict[int, list[str]], dict[int, list[int]]]:
        """Uses PCA to reduce dim before clustering."""
        embs = self.reduce_embed(inputs)
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=3, metric="cosine")
        dbscan.fit(embs)

        niches = defaultdict(list)
        indices = defaultdict(list)
        for i, label in enumerate(dbscan.labels_):
            niches[label].append(inputs[i])
            indices[label].append(i)

        # Log cluster contents
        for label, members in sorted(niches.items()):
            if label == -1:
                logger.info(
                    f"Noise points ({len(members)} items):\n"
                    + "\n".join(f"  - {inp}" for inp in members)
                )
            else:
                logger.info(
                    f"Cluster {label} ({len(members)} items):\n"
                    + "\n".join(f"  - {inp}" for inp in members)
                )

        return niches, indices
