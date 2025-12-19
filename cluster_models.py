from collections import defaultdict
from typing import Any, Tuple
from loguru import logger

import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

class ClusterModel:
    def __init__(
        self,
        embedding_model_name: str,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 5,
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.umap_model = UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=0.0,
            metric="cosine",
            low_memory=False,
        )

    def embed(self, inputs: list[str]) -> np.ndarray:
        """Returns: [n_inputs, d_embed]"""
        return self.embedding_model.encode(inputs)  # type: ignore

    def reduce_embed(self, inputs: list[str]) -> np.ndarray:
        """Embed then do dimensionality reduction"""
        print("Embedding...")
        embeddings: np.ndarray = self.embed(inputs)
        print("Reducing dimensionality...")
        return self.umap_model.fit_transform(embeddings)  # type: ignore

    def cluster(self, inputs: list[str], n_clusters: int) -> list[dict[str, Any]]:
        """
        Returns a list of cluster information.
        """
        reduced_embeddings = self.reduce_embed(inputs)

        kmeans = KMeans(
            n_clusters=min(len(inputs), n_clusters), random_state=10086, n_init="auto"
        )
        print("Fitting KMeans...")
        kmeans.fit(reduced_embeddings)

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
                cluster_embeddings = reduced_embeddings[content_indices]
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

        return results

    def cluster_dbscan(
        self,
        inputs: list[str],
        dbscan_eps: float,
    ) -> Tuple[dict[int, list[str]], dict[int, list[int]]]:
        embeddings = self.embed(inputs)

        # # log the pairwise distance matrix
        # logger.info(
        #     f"Pairwise distance matrix:\n"
        #     f"{pairwise_distances(embeddings, metric='cosine')}"
        # )

        dbscan = DBSCAN(
            eps=dbscan_eps, min_samples=2 * self.umap_n_components, metric="cosine"
        )
        dbscan.fit(embeddings)

        niches = defaultdict(list)
        indices = defaultdict(list)
        for i, label in enumerate(dbscan.labels_):
            niches[label].append(inputs[i])
            indices[label].append(i)

        logger.debug(
            "Niches:\n"
            + "\n".join(
                [
                    f"Niche {label}:\n{"\n".join(members)}"
                    for label, members in niches.items()
                ]
            )
        )

        return niches, indices
