"""
Hybrid recommender: scikit-learn TruncatedSVD (matrix factorization / collaborative filtering)
plus a small PyTorch MLP on concatenated user–item latent vectors (deep learning layer).
TensorFlow/Keras can replace the MLP with the same inputs and targets.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

_cache_lock = threading.Lock()
_state: HybridState | None = None
_event_signature: int | None = None


@dataclass
class HybridState:
    user_ids: list[str]
    item_ids: list[str]
    user_index: dict[str, int]
    item_index: dict[str, int]
    U: np.ndarray  # (n_users, k)
    V: np.ndarray  # (n_items, k)
    mlp: nn.Module | None
    popularity_rank: list[str]
    popularity_scores: dict[str, float]


class ScoringMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 96) -> None:
        super().__init__()
        d = latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, user_latent: torch.Tensor, item_latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_latent, item_latent], dim=-1)
        return self.net(x).squeeze(-1)


def _aggregate_pairs(
    rows: list[tuple[str, str, float, float | None]],
) -> tuple[list[tuple[str, str, float]], dict[str, float]]:
    """Collapse to (user, content, strength); strength uses watch time + optional rating."""
    acc: dict[tuple[str, str], list[float]] = {}
    pop: dict[str, float] = {}
    for uid, cid, watch, rating in rows:
        w = float(np.log1p(max(watch, 0.0)))
        if rating is not None:
            w += 0.35 * (float(rating) / 5.0)
        key = (uid, cid)
        acc.setdefault(key, []).append(w)
        pop[cid] = pop.get(cid, 0.0) + float(watch) + (float(rating or 0.0) * 30.0)
    pairs: list[tuple[str, str, float]] = []
    for (uid, cid), vals in acc.items():
        pairs.append((uid, cid, float(sum(vals))))
    return pairs, pop


def _fit_state(pairs: list[tuple[str, str, float]], pop: dict[str, float], latent_k: int = 32) -> HybridState:
    users = sorted({p[0] for p in pairs})
    items = sorted({p[1] for p in pairs})
    ui = {u: i for i, u in enumerate(users)}
    ii = {c: j for j, c in enumerate(items)}
    n_u, n_i = len(users), len(items)

    pop_rank = sorted(pop.keys(), key=lambda c: pop[c], reverse=True)

    if n_u < 2 or n_i < 2 or len(pairs) < 4:
        return HybridState(
            user_ids=users,
            item_ids=items,
            user_index=ui,
            item_index=ii,
            U=np.zeros((n_u, 1)),
            V=np.zeros((n_i, 1)),
            mlp=None,
            popularity_rank=pop_rank,
            popularity_scores=pop,
        )

    r = np.array([p[2] for p in pairs], dtype=np.float64)
    r = (r - r.min()) / max(float(np.ptp(r)), 1e-9)
    row_ix = np.array([ui[p[0]] for p in pairs], dtype=np.int32)
    col_ix = np.array([ii[p[1]] for p in pairs], dtype=np.int32)
    X = csr_matrix((r, (row_ix, col_ix)), shape=(n_u, n_i))

    max_k = min(n_u, n_i) - 1
    if max_k < 1:
        return HybridState(
            user_ids=users,
            item_ids=items,
            user_index=ui,
            item_index=ii,
            U=np.zeros((n_u, 1)),
            V=np.zeros((n_i, 1)),
            mlp=None,
            popularity_rank=pop_rank,
            popularity_scores=pop,
        )
    k = min(latent_k, max_k, n_u - 1, n_i - 1)
    k = max(1, k)
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(X)
    Vt = svd.components_
    V = Vt.T

    U_t = torch.from_numpy(U).float()
    V_t = torch.from_numpy(V).float()
    y_t = torch.tensor(r, dtype=torch.float32)

    mlp: nn.Module | None = ScoringMLP(latent_dim=k)
    mlp.train()
    opt = torch.optim.Adam(mlp.parameters(), lr=0.02)
    idx = np.arange(len(pairs))
    for _epoch in range(25):
        np.random.shuffle(idx)
        for start in range(0, len(idx), 256):
            batch = idx[start : start + 256]
            bu = U_t[row_ix[batch]]
            bv = V_t[col_ix[batch]]
            pred = mlp(bu, bv)
            loss = nn.functional.mse_loss(pred, y_t[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
    mlp.eval()

    return HybridState(
        user_ids=users,
        item_ids=items,
        user_index=ui,
        item_index=ii,
        U=U,
        V=V,
        mlp=mlp,
        popularity_rank=pop_rank,
        popularity_scores=pop,
    )


def event_signature(rows: list[tuple[str, str, float, float | None]]) -> int:
    if not rows:
        return 0
    h = hashlib.sha256()
    for uid, cid, watch, rating in sorted(rows, key=lambda x: (x[0], x[1], x[2], x[3] is None, x[3] or 0.0)):
        line = f"{uid}\0{cid}\0{watch:.6f}\0{rating if rating is not None else -1.0:.6f}\n"
        h.update(line.encode())
    return int.from_bytes(h.digest()[:12], "big", signed=False)


def fit_or_get_cached(rows: list[tuple[str, str, float, float | None]]) -> tuple[HybridState, bool, float]:
    """Returns (state, cache_hit, fit_seconds)."""
    global _state, _event_signature
    if not rows:
        sig = 0
        with _cache_lock:
            if _state is not None and _event_signature == sig:
                return _state, True, 0.0
        empty = HybridState(
            user_ids=[],
            item_ids=[],
            user_index={},
            item_index={},
            U=np.zeros((0, 2)),
            V=np.zeros((0, 2)),
            mlp=None,
            popularity_rank=[],
            popularity_scores={},
        )
        with _cache_lock:
            _state = empty
            _event_signature = sig
        return empty, False, 0.0
    sig = event_signature(rows)
    with _cache_lock:
        if _state is not None and _event_signature == sig:
            return _state, True, 0.0
    t0 = time.perf_counter()
    pairs, pop = _aggregate_pairs(rows)
    new_state = _fit_state(pairs, pop)
    elapsed = time.perf_counter() - t0
    with _cache_lock:
        _state = new_state
        _event_signature = sig
    return new_state, False, elapsed


def invalidate_cache() -> None:
    global _state, _event_signature
    with _cache_lock:
        _state = None
        _event_signature = None


def recommend_for_user(
    state: HybridState,
    user_id: str,
    top_k: int,
    exclude_watched: set[str] | None = None,
) -> list[tuple[str, float, str]]:
    exclude_watched = exclude_watched or set()

    if user_id not in state.user_index or state.mlp is None:
        out: list[tuple[str, float, str]] = []
        for cid in state.popularity_rank:
            if cid in exclude_watched:
                continue
            sc = state.popularity_scores.get(cid, 0.0)
            out.append((cid, float(sc), "popularity"))
            if len(out) >= top_k:
                break
        if len(out) < top_k and exclude_watched:
            seen = {c for c, _, _ in out}
            for cid in state.popularity_rank:
                if cid in seen:
                    continue
                sc = state.popularity_scores.get(cid, 0.0)
                out.append((cid, float(sc) * 0.85, "popularity_including_watched"))
                seen.add(cid)
                if len(out) >= top_k:
                    break
        return out[:top_k]

    uix = state.user_index[user_id]
    u_vec = torch.from_numpy(state.U[uix]).float().unsqueeze(0)
    device = torch.device("cpu")
    mlp = state.mlp.to(device)
    scores: list[tuple[str, float, str]] = []
    with torch.inference_mode():
        for j, cid in enumerate(state.item_ids):
            if cid in exclude_watched:
                continue
            v_vec = torch.from_numpy(state.V[j]).float().unsqueeze(0).to(device)
            neu = u_vec.to(device)
            deep = float(mlp(neu, v_vec).item())
            cf = float(np.dot(state.U[uix], state.V[j]))
            blend = 0.55 * deep + 0.45 * cf
            scores.append((cid, blend, "hybrid_mlp"))
    scores.sort(key=lambda x: -x[1])
    out = scores[:top_k]
    if len(out) < top_k:
        seen = {c for c, _, _ in out}
        for cid in state.popularity_rank:
            if cid in exclude_watched or cid in seen:
                continue
            sc = state.popularity_scores.get(cid, 0.0)
            out.append((cid, float(sc), "popularity"))
            seen.add(cid)
            if len(out) >= top_k:
                break
    if len(out) < top_k and exclude_watched:
        seen = {c for c, _, _ in out}
        for cid in state.popularity_rank:
            if cid in seen:
                continue
            sc = state.popularity_scores.get(cid, 0.0)
            out.append((cid, float(sc) * 0.85, "popularity_including_watched"))
            seen.add(cid)
            if len(out) >= top_k:
                break
    return out[:top_k]


def deps_available() -> bool:
    try:
        import sklearn  # noqa: F401
        import torch  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True
