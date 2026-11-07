import torch
import torch.nn as nn
import torch.nn.functional as F


class Top1Gate(nn.Module):
    def __init__(self, n_embeddings, embed_dim, hidden_dim=None):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embed_dim = embed_dim
        self.proj = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(n_embeddings)]
        )
        self.ln = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_embeddings)])
        self.w_gate = nn.Linear(embed_dim * n_embeddings, n_embeddings)

    def forward(self, *embeddings):
        if len(embeddings) == 1:
            return embeddings[0]

        processed = []
        for i, emb in enumerate(embeddings):
            e = self.proj[i](emb)
            e = self.ln[i](e)
            processed.append(e)
        embeddings = torch.stack(processed, dim=-2)
        
        gate_input = embeddings.view(
            *embeddings.shape[:-2], self.n_embeddings * self.embed_dim
        )
        clean_logits = self.w_gate(gate_input)
        
        top_logits, top_indices = clean_logits.topk(1, dim=-1)
        logits = torch.full_like(clean_logits, float('-inf'))
        logits.scatter_(-1, top_indices, top_logits)
        
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (weights * embeddings).sum(dim=-2)

    def eval_(self, rollouts):
        return {}


class ConcatMixer(nn.Module):
    def __init__(self, n_embeddings, embed_dim):
        super().__init__()

    def forward(self, *embeddings):
        return torch.cat(embeddings, dim=-1)
    
    def eval_(self, rollouts):
        return {}