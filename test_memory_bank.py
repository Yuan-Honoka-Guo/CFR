import torch
import torch.nn.functional as F


class MemoryBankWrapper:
    def __init__(self, bank_path, device="cpu"):
        bank = torch.load(bank_path, map_location=device)
        if isinstance(bank, dict):
            if "values" in bank:
                bank = bank["values"]
            elif "keys" in bank:
                bank = bank["keys"]
            else:
                raise ValueError("Unsupported memory bank format. Expected tensor or KV dict.")

        if not torch.is_tensor(bank):
            bank = torch.as_tensor(bank)

        if bank.numel() == 0:
            raise ValueError("Memory bank is empty.")

        self.device = device
        self.bank = bank.to(device)
        self.bank_norm = F.normalize(self.bank, dim=-1)

    def get_nearest_neighbors_consine(self, query_features, k=1, chunk_size=4096):
        if query_features.ndim == 1:
            query_features = query_features.unsqueeze(0)

        query = query_features.to(self.device)
        query_norm = F.normalize(query, dim=-1)

        k = min(k, self.bank.shape[0])
        if k <= 0:
            raise ValueError("k must be >= 1.")

        if not chunk_size or chunk_size <= 0:
            chunk_size = query_norm.shape[0]

        dists = []
        feats = []
        # Chunked matmul to reduce peak memory on large banks.
        for start in range(0, query_norm.shape[0], chunk_size):
            q = query_norm[start : start + chunk_size]
            sim = q @ self.bank_norm.T
            topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)
            topk_feats = self.bank[topk_idx]
            dist = 1.0 - topk_sim
            dists.append(dist)
            feats.append(topk_feats)

        return torch.cat(dists, dim=0), torch.cat(feats, dim=0)

    def get_nearest_neighbors_cosine(self, query_features, k=1, chunk_size=4096):
        return self.get_nearest_neighbors_consine(query_features, k=k, chunk_size=chunk_size)
