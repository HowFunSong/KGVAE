import torch
import torch.nn as nn

class EASE(nn.Module):
    """
    Embarrassingly Shallow Autoencoder (EASE) adapted to RecVAE-style inputs.
    Input: dense user–item interaction tensor R of shape (n_users, n_items)
    Closed-form train (fit) and forward (predict) in PyTorch.
    """
    def __init__(self, reg: float = 250.0):
        super().__init__()
        self.reg = reg
        self.B = None  # item–item weight matrix

    def fit(self, R: torch.Tensor) -> None:
        """
        Compute closed-form solution for B given R.
        Args:
            R: FloatTensor of shape (n_users, n_items), binary or count interactions.
        """
        # Gram matrix: I x I
        # G = R^T R + reg * I
        G = R.t().matmul(R)
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        G = G + self.reg * I

        # Invert
        P = torch.linalg.inv(G)  # dense inverse

        # Compute B = P / (-diag(P)), zero diagonal
        inv_diag = torch.diag(P)
        B = P / (-inv_diag.unsqueeze(0))
        B.fill_diagonal_(0.0)

        self.B = B

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        """
        Given interaction tensor R, compute score predictions.
        Args:
            R: FloatTensor of shape (n_users, n_items)
        Returns:
            scores: FloatTensor of shape (n_users, n_items)
        """
        if self.B is None:
            raise RuntimeError("EASEModel must be fitted with R before calling forward().")
        # linear scores
        return R.matmul(self.B)

# Example usage:
# R: dense FloatTensor [n_users, n_items]
# model = EASEModel(reg=250.)\#
# model.fit(R)
# scores = model(R)  # [n_users, n_items]
