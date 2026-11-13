import torch

class GatedSparseAttention(torch.nn.Module):
    """GSA attention layer from the original Gated Sparse Attention paper."""

    def __init__(self, dim: int, heads: int, block_size: int = 64):
        super().__init__()
        self.heads = heads
        self.block_size = block_size
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(dim, dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim // 2, 1),
            torch.nn.Sigmoid(),
        )
