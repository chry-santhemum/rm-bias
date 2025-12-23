from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger
import numpy as np
from sklearn.cluster import KMeans
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
        sim_threshold: float,
    ) -> list[list[int]]:
        """
        Builds connected components where an edge exists if cosine_sim >= sim_threshold.
        embs must be L2-normalized.
        """
        n = embs.shape[0]
        if n == 0:
            return []

        sim = embs @ embs.T  # [n, n], cosine similarity
        uf = _UnionFind.create(n)

        rows, cols = np.where(np.triu(sim, k=1) >= sim_threshold)
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
        dedupe_sim_threshold: float = 0.985,
        embed_dim: int | None = None,  # can override here
    ) -> list[dict[str, Any]]:
        """
        1) collapse near-duplicates by cosine threshold (connected components)
        2) if still too many groups, KMeans them down to n_representatives
        Returns cluster-like dicts, with center_idx always a real input index.
        """
        if not inputs:
            return []

        embs = self.embed(inputs, embed_dim=embed_dim)
        self._cosine_stats(embs)

        # Stage A: near-duplicate collapse
        comps = self._connected_components_by_similarity(embs, sim_threshold=dedupe_sim_threshold)

        comp_infos: list[dict[str, Any]] = []
        comp_rep_indices: list[int] = []
        comp_sizes: list[int] = []
        for comp in comps:
            rep_idx = self._medoid_index(embs, comp)
            comp_infos.append({"members": comp, "rep_idx": rep_idx})
            comp_rep_indices.append(rep_idx)
            comp_sizes.append(len(comp))

        logger.info(
            f"Dedupe components: {len(comps)} from {len(inputs)} inputs "
            f"(threshold={dedupe_sim_threshold:.3f})"
        )

        # print out the deduped entries
        for i, comp in enumerate(comps):
            if len(comp) >= 2:
                logger.info(f"Deduped cluster {i}:\n{"\n".join(inputs[c] for c in comp)}")

        # If we already have <= target groups, return them (can't manufacture more distinct reps).
        if len(comp_infos) <= n_representatives:
            results: list[dict[str, Any]] = []
            for cluster_idx, info in enumerate(comp_infos):
                members = sorted(info["members"])
                center_idx = int(info["rep_idx"])
                results.append(
                    {
                        "cluster_idx": cluster_idx,
                        "center_idx": center_idx,
                        "center_input": inputs[center_idx],
                        "content_indices": members,
                    }
                )
            return results

        # Stage B: cluster component representatives down to fixed count
        rep_embs = embs[comp_rep_indices]  # normalized
        k = min(n_representatives, rep_embs.shape[0])

        logger.info(
            f"Fitting KMeans over {len(comp_infos)} deduped groups into {k} representatives"
        )
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        kmeans.fit(rep_embs)

        # Group component indices by kmeans cluster
        group_to_comp_idxs: dict[int, list[int]] = defaultdict(list)
        for comp_i, lbl in enumerate(kmeans.labels_.tolist()):  # type: ignore
            group_to_comp_idxs[int(lbl)].append(comp_i)

        results = []
        for cluster_idx in range(k):
            comp_idxs = group_to_comp_idxs.get(cluster_idx, [])
            if not comp_idxs:
                continue

            # Choose weighted medoid among component reps (weights = component sizes)
            sub_embs = rep_embs[comp_idxs]  # normalized
            sim = sub_embs @ sub_embs.T
            dist = 1.0 - sim  # [m, m]
            weights = np.array([comp_sizes[i] for i in comp_idxs], dtype=np.float32)  # [m]

            # For each candidate j: sum_i w_i * dist[j, i]
            weighted_sum_dist = dist @ weights
            best_local = int(np.argmin(weighted_sum_dist))
            best_comp_i = comp_idxs[best_local]
            center_idx = int(comp_infos[best_comp_i]["rep_idx"])

            # Union original indices across all components in this kmeans cluster
            members: list[int] = []
            for ci in comp_idxs:
                members.extend(comp_infos[ci]["members"])
            members = sorted(set(members))

            logger.info(
                f"KMeans cluster {cluster_idx} with {len(members)} members:\n"
                f"Center: {inputs[center_idx]}\n"
                f"Members:\n{"\n".join(inputs[m] for m in members)}"
            )

            results.append(
                {
                    "cluster_idx": cluster_idx,
                    "center_idx": center_idx,
                    "center_input": inputs[center_idx],
                    "content_indices": members,
                }
            )

        for r in results:
            assert r["center_idx"] in r["content_indices"]

        return results

    def cluster_by_similarity(
        self,
        inputs: list[str],
        *,
        sim_threshold: float = 0.88,
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

        comps = self._connected_components_by_similarity(embs, sim_threshold=sim_threshold)

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
