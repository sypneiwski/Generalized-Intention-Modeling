import torch
import torch.nn as nn


class WeightedScoresMixerLNND(nn.Module):
    def __init__(self, n_embeddings, embed_dim):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embed_dim = embed_dim
        self.proj = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(n_embeddings)]
        )
        self.ln = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_embeddings)])
        self.gate = nn.Linear(embed_dim * n_embeddings, n_embeddings * embed_dim)

    def forward(self, *embeddings):
        if len(embeddings) == 1:
            return embeddings[0]

        projected = []
        for i, emb in enumerate(embeddings):
            e = self.proj[i](emb)
            e = self.ln[i](e)
            projected.append(e)
        embeddings = torch.stack(projected, dim=-2)
        logits = self.gate(
            embeddings.view(*embeddings.shape[:-2], self.n_embeddings * self.embed_dim)
        )
        weights = torch.softmax(
            logits.view(*embeddings.shape[:-2], self.n_embeddings, self.embed_dim),
            dim=-2,
        )
        return (weights * embeddings).sum(dim=-2)

    def _expert_keys(self, agent_data):
        tensor_keys = [
            k for k in agent_data.keys() if isinstance(agent_data[k], torch.Tensor)
        ]
        names = getattr(self, "expert_names", None)
        if names is None:
            return tensor_keys
        return [n for n in names if n in tensor_keys]
    
    def eval_(self, rollouts):
        agent_data = rollouts["agent"]
        expert_keys = self._expert_keys(agent_data)
        
        metrics = {}

        with torch.enable_grad():
            inputs_grad = []
            for k in expert_keys:
                inp = agent_data[k].view(-1, self.embed_dim).detach().requires_grad_(True)
                inputs_grad.append(inp)

            out = self.forward(*inputs_grad)
            out_scalar = out.abs().sum()
            out_scalar.backward()

            attributions = []
            for inp in inputs_grad:
                attr = (inp * inp.grad).abs().sum(dim=-1).mean().item()
                attributions.append(attr)

            total_attr = sum(attributions) + 1e-9 
            
            for k, a in zip(expert_keys, attributions):
                metrics[f"expert_attribution_share/{k}"] = a / total_attr

        return metrics
