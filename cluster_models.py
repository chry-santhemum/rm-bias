from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


@dataclass
class _UnionFind:
    parent: list[int]
    size: list[int]

    @classmethod
    def create(cls, n: int) -> "_UnionFind":
        return cls(parent=list(range(n)), size=[1] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return groups


class ClusterModel:
    def __init__(
        self,
        embed_model_name: str,
        embed_dim: int = -1,
        random_state: int = 10086,
    ):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.embed_model_name = embed_model_name
        self.embed_dim = embed_dim
        self.random_state = random_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "embed_model_name": self.embed_model_name,
            "embed_dim": self.embed_dim,
            "random_state": self.random_state,
        }

    def embed(self, inputs: list[str], *, embed_dim: int | None = None) -> np.ndarray:
        """Returns L2-normalized embs [n_inputs, embed_dim]"""
        dim = self.embed_dim if embed_dim is None else embed_dim
        embs: np.ndarray = self.embed_model.encode(inputs)  # type: ignore

        if dim != -1:
            embs = embs[:, :dim]

        embs = embs.astype(np.float32, copy=False)
        norms = np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms

        logger.info(f"Embedded shape: {embs.shape}")
        return embs

    def pca(self, embs: np.ndarray, pca_dim: int) -> np.ndarray:
        n_components = min(pca_dim, embs.shape[0], embs.shape[1])
        if n_components >= embs.shape[1]:
            return embs

        pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced = pca.fit_transform(embs).astype(np.float32, copy=False)

        # Re-normalize so dot-product remains cosine similarity.
        norms = np.linalg.norm(reduced, ord=2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        reduced = reduced / norms

        logger.info(f"PCA reduced shape: {reduced.shape}")
        return reduced

    def _cosine_stats(self, embs: np.ndarray) -> None:
        # embs are normalized => cosine distance = 1 - dot
        sim = embs @ embs.T
        dist = 1.0 - sim
        upper = dist[np.triu_indices_from(dist, k=1)]
        if upper.size == 0:
            return
        percentiles = list(range(10, 101, 10))
        msg = " | ".join(f"p{p}={np.percentile(upper, p):.3f}" for p in percentiles)
        logger.info(f"Cosine distance distribution: {msg}")

    def _connected_components_by_similarity(
        self,
        embs: np.ndarray,
        *,
        cosine_sim_threshold: float,
    ) -> list[list[int]]:
        """
        Builds connected components where an edge exists if cosine_sim >= threshold.
        embs must be L2-normalized.
        """
        n = embs.shape[0]
        if n == 0:
            return []

        sim = embs @ embs.T  # [n, n], cosine similarity
        uf = _UnionFind.create(n)

        rows, cols = np.where(np.triu(sim, k=1) >= cosine_sim_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            uf.union(i, j)

        comps = list(uf.components().values())
        comps.sort(key=len, reverse=True)
        return comps

    def _medoid_index(self, embs: np.ndarray, indices: list[int]) -> int:
        """Return original index of the cosine-distance medoid for the given subset."""
        if len(indices) == 1:
            return indices[0]

        sub = embs[indices]  # normalized
        sim = sub @ sub.T
        dist = 1.0 - sim
        sum_dist = dist.sum(axis=1)
        best = int(np.argmin(sum_dist))
        return indices[best]

    def pick_representatives(
        self,
        inputs: list[str],
        *,
        n_representatives: int,
        embed_dim: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Adaptively find epsilon such that connected components gives >= n_representatives
        clusters, then pick the K largest clusters and return the medoid of each.

        This avoids needing a fixed cosine threshold - the algorithm finds the right
        granularity automatically based on the target number of representatives.
        """
        if not inputs:
            return []

        n = len(inputs)
        if n <= n_representatives:
            return [
                {
                    "cluster_idx": i,
                    "center_idx": i,
                    "center_input": inputs[i],
                    "content_indices": [i],
                }
                for i in range(n)
            ]

        embs = self.embed(inputs, embed_dim=embed_dim)
        self._cosine_stats(embs)

        # Binary search for epsilon (cosine similarity threshold)
        # Higher epsilon = more stringent = fewer edges = MORE components (singletons)
        # Lower epsilon = more permissive = more edges = FEWER components (merged)
        # We want the LOWEST epsilon that gives >= n_representatives components
        # (lower epsilon = larger clusters on average)
        lo, hi = 0.0, 1.0
        best_eps = hi
        best_comps = self._connected_components_by_similarity(embs, cosine_sim_threshold=hi)

        for _ in range(50):
            if hi - lo < 1e-6:
                break

            eps = (lo + hi) / 2
            comps = self._connected_components_by_similarity(embs, cosine_sim_threshold=eps)
            n_comps = len(comps)

            if n_comps >= n_representatives:
                # This epsilon works, try lower (more merging, larger clusters)
                best_eps = eps
                best_comps = comps
                hi = eps
            else:
                # Too few components, need higher epsilon (less merging)
                lo = eps

        logger.info(
            f"Adaptive clustering: {len(best_comps)} components at epsilon={best_eps:.4f} "
            f"(target={n_representatives}, inputs={len(inputs)})"
        )

        # Select the K largest clusters (best_comps is already sorted by size descending)
        selected_comps = best_comps[:n_representatives]

        # Log info about selected vs dropped
        if len(best_comps) > n_representatives:
            dropped = best_comps[n_representatives:]
            dropped_count = sum(len(c) for c in dropped)
            logger.info(
                f"Selected {n_representatives} largest clusters, "
                f"dropped {len(dropped)} smaller clusters ({dropped_count} items)"
            )

        # Build results with medoid as representative
        results = []
        for cluster_idx, comp in enumerate(selected_comps):
            rep_idx = self._medoid_index(embs, comp)

            if len(comp) >= 2:
                logger.info(
                    f"Cluster {cluster_idx} ({len(comp)} members):\n"
                    f"  Representative: {inputs[rep_idx]}\n"
                    f"  Members:\n" + "\n".join(f"    - {inputs[c]}" for c in comp)
                )

            results.append(
                {
                    "cluster_idx": cluster_idx,
                    "center_idx": rep_idx,
                    "center_input": inputs[rep_idx],
                    "content_indices": sorted(comp),
                }
            )

        return results

    def cluster_by_similarity(
        self,
        inputs: list[str],
        *,
        cosine_sim_threshold: float,
        min_cluster_size: int = 2,
        embed_dim: int | None = None,
        pca_dim: int | None = None,
    ) -> tuple[dict[int, list[str]], dict[int, list[int]]]:
        """
        If min_cluster_size >= 2, singletons are labeled as -1 (noise).
        """
        if not inputs:
            return {}, {}

        embs = self.embed(inputs, embed_dim=embed_dim)
        if pca_dim is not None:
            embs = self.pca(embs, pca_dim=pca_dim)

        self._cosine_stats(embs)

        comps = self._connected_components_by_similarity(embs, cosine_sim_threshold=cosine_sim_threshold)

        niches: dict[int, list[str]] = defaultdict(list)
        indices: dict[int, list[int]] = defaultdict(list)

        next_label = 0
        for comp in comps:
            if len(comp) < min_cluster_size:
                for i in comp:
                    niches[-1].append(inputs[i])
                    indices[-1].append(i)
                continue

            lbl = next_label
            next_label += 1
            for i in comp:
                niches[lbl].append(inputs[i])
                indices[lbl].append(i)

        # Logging
        for label, members in sorted(niches.items(), key=lambda kv: kv[0]):
            title = "Noise points" if label == -1 else f"Cluster {label}"
            logger.info(
                f"{title} ({len(members)} items):\n" + "\n".join(f"  - {m}" for m in members)
            )

        return niches, indices
