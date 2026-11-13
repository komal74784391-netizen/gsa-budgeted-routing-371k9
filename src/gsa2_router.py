import torch

class BudgetedRouter(torch.nn.Module):
    """Budgeted block router for the GSA-2 extension."""

    def __init__(self, dim: int, blocks: int, top_k: int = 4):
        super().__init__()
        self.score = torch.nn.Linear(dim, blocks)
        self.top_k = top_k

    def forward(self, query, block_keys):
        scores = (query @ block_keys.transpose(-1, -2)) + self.score(query)
        indices = scores.topk(self.top_k, dim=-1).indices
        return indices
