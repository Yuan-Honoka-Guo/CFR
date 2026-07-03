import torch
import torch.nn as nn
import torch.nn.functional as F


def retrieve_topk_cosine(query, bank_keys, bank_values, k=3, chunk_size=None):
    """
    Retrieve top-k key/value pairs based on cosine similarity of raw features.

    Args:
        query: (N, Dq)
        bank_keys: (M, Dk)
        bank_values: (M, Dv)
    Returns:
        topk_scores: (N, k)
        topk_keys: (N, k, Dk)
        topk_values: (N, k, Dv)
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > bank_keys.shape[0]:
        k = bank_keys.shape[0]

    q = F.normalize(query, dim=-1)
    k_norm = F.normalize(bank_keys, dim=-1)

    num_queries = q.shape[0]
    if chunk_size is None or chunk_size <= 0:
        chunk_size = num_queries

    scores_list = []
    keys_list = []
    values_list = []

    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        q_chunk = q[start:end]
        scores = torch.matmul(q_chunk, k_norm.t())
        topk_scores, topk_idx = torch.topk(scores, k, dim=-1)
        topk_keys = bank_keys[topk_idx]
        topk_values = bank_values[topk_idx]
        scores_list.append(topk_scores)
        keys_list.append(topk_keys)
        values_list.append(topk_values)

    return (
        torch.cat(scores_list, dim=0),
        torch.cat(keys_list, dim=0),
        torch.cat(values_list, dim=0),
    )


class AttentionRetriever(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, d_model=768, dropout=0.0, normalize_qk=False):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, d_model, bias=False)
        self.k_proj = nn.Linear(k_dim, d_model, bias=False)
        self.v_proj = nn.Linear(v_dim, d_model, bias=False)
        self.out_proj = nn.Identity() if d_model == v_dim else nn.Linear(d_model, v_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.normalize_qk = normalize_qk
        self.scale = d_model ** -0.5

    def forward(self, query, keys, values, topk=None, chunk_size=None):
        """
        Args:
            query: (N, q_dim)
            keys: (M, k_dim)
            values: (M, v_dim)
        Returns:
            retrieved: (N, v_dim)
        """
        q = self.q_proj(query)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        if self.normalize_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        num_queries = q.shape[0]
        if chunk_size is None or chunk_size <= 0:
            chunk_size = num_queries

        outputs = []
        use_topk = topk is not None and topk < k.shape[0]

        for start in range(0, num_queries, chunk_size):
            end = min(start + chunk_size, num_queries)
            q_chunk = q[start:end]
            scores = torch.matmul(q_chunk, k.t()) * self.scale

            if use_topk:
                topk_vals, topk_idx = torch.topk(scores, topk, dim=-1)
                weights = torch.softmax(topk_vals, dim=-1)
                weights = self.attn_dropout(weights)
                v_sel = v[topk_idx]
                out_chunk = torch.sum(weights.unsqueeze(-1) * v_sel, dim=1)
            else:
                weights = torch.softmax(scores, dim=-1)
                weights = self.attn_dropout(weights)
                out_chunk = torch.matmul(weights, v)

            outputs.append(out_chunk)

        retrieved = torch.cat(outputs, dim=0)
        return self.out_proj(retrieved)

    def forward_topk(self, query, keys_topk, values_topk):
        """
        Args:
            query: (N, q_dim)
            keys_topk: (N, K, k_dim)
            values_topk: (N, K, v_dim)
        Returns:
            retrieved: (N, v_dim)
        """
        q = self.q_proj(query)
        k = self.k_proj(keys_topk)
        v = self.v_proj(values_topk)

        if self.normalize_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        scores = torch.einsum("nd,nkd->nk", q, k) * self.scale
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.sum(weights.unsqueeze(-1) * v, dim=1)
        return self.out_proj(out)
