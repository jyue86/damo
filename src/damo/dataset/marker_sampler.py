import torch
import numpy as np
import random
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass

import damo.utils as utils


@dataclass
class MarkerSample:
    m_shaped: np.ndarray  # [M, 3]
    markers_vids: np.ndarray  # [M]


class MarkerSampler:
    def __init__(self, cfg):
        self.geom_data = None
        self.markersets_data = None

        self.distance_from_skin = cfg["distance_from_skin"]
        self.min_sample_markers = cfg["min_sample_markers"]
        self.max_sample_markers = cfg["max_sample_markers"]

    def get_markersets(self):
        if self.markersets_data is None:
            path = utils.Paths.data / "markersets.npz"
            self.markersets_data = utils.io_utils.load_npz_to_dict(path)

        return self.markersets_data

    def get_geom(self):
        if self.geom_data is None:
            path = utils.Paths.data / "marker_geom_smplh.npz"
            self.geom_data = utils.io_utils.load_npz_to_dict(path)

        return self.geom_data

    def get_superset_sample(self, v_shaped):
        markersets = self.get_markersets()
        superset_cands = markersets["soma_superset_smplh_cands"]
        M_sup = len(superset_cands)

        min_N = self.min_sample_markers
        max_N = min(self.max_sample_markers, M_sup)

        N = np.random.randint(min_N, max_N + 1)
        selected_cands_idx = np.random.choice(M_sup, N, replace=False)
        selected_cands = superset_cands[selected_cands_idx]

        markers_vids = [np.random.choice(cand) for cand in selected_cands]
        markers_vids = np.asarray(markers_vids, dtype=np.int64)

        marker_geom = self.get_geom()
        vn = marker_geom["smplh_vertex_normals"]
        mn = vn[markers_vids]
        m_shaped = v_shaped[markers_vids] + mn * self.distance_from_skin

        marker_sample = MarkerSample(
            m_shaped=m_shaped,
            markers_vids=markers_vids,
        )

        return marker_sample

    def get_vertex_normals(self):
        geom = self.get_geom()
        return geom["smplh_vertex_normals"]

    def get_arbitrary_sample(self, v_shaped, weights, max_weight_jids=None) -> MarkerSample:
        V, _ = v_shaped.shape
        assert weights.shape[0] == V

        markers_vids = self.sample_markers(weights, max_weight_jids)  # [M]

        marker_geom = self.get_geom()
        vn = marker_geom["smplh_vertex_normals"]  # [V, 3]
        assert vn.shape[0] == V

        mn = vn[markers_vids]  # [M, 3]
        m_shaped = v_shaped[markers_vids] + mn * self.distance_from_skin

        marker_sample = MarkerSample(
            m_shaped=m_shaped,  # [M, 3]
            markers_vids=markers_vids,  # [M]
        )

        return marker_sample

    def sample_markers(self, weights, max_weight_jids=None):
        V, J = weights.shape
        marker_geom = self.get_geom()

        candidate_vids = marker_geom["candidate_vids"]
        sample_weights = marker_geom["sample_weights"]

        if max_weight_jids is None:
            max_weight_jids = np.argmax(weights, axis=1)

        Nc = candidate_vids.size
        if Nc == 0:
            raise ValueError("No candidate vertices available")

        min_N = max(self.min_sample_markers, J)
        max_N = min(self.max_sample_markers, Nc)
        if min_N > max_N:
            raise ValueError(f"min_N > max_N (min_N={min_N}, max_N={max_N}, Nc={Nc})")

        N = np.random.randint(min_N, max_N + 1)

        selected = []
        used_mask = np.zeros(V, dtype=bool)
        joint_counts = np.zeros(J, dtype=int)

        for j in range(J):
            idx_cand = candidate_vids[max_weight_jids[candidate_vids] == j]
            if idx_cand.size == 0:
                continue

            sw = sample_weights[idx_cand]
            s = sw.sum()
            if s > 0.0:
                p = sw / s
                v = np.random.choice(idx_cand, p=p)
            else:
                v = np.random.choice(idx_cand)

            selected.append(v)
            used_mask[v] = True
            joint_counts[j] += 1

        selected = np.asarray(selected, dtype=int)
        remaining = N - len(selected)

        if remaining <= 0:
            return selected

        available = candidate_vids[~used_mask[candidate_vids]]
        if available.size <= 0:
            return selected

        if remaining > available.size:
            remaining = available.size

        owner_avail = max_weight_jids[available]  # [Na]
        joint_weight = 1.0 / (joint_counts + 1.0)  # [J]
        base = joint_weight[owner_avail]  # [Na]

        sw = sample_weights[available]  # [Na]
        v_weights = base * sw  # [Na]

        s = v_weights.sum()
        if s <= 0.0:
            p = None  # uniform
        else:
            p = v_weights / s

        extra = np.random.choice(available, size=remaining, replace=False, p=p)

        selected = np.concatenate([selected, extra.astype(int)], axis=0)
        return selected


