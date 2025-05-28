import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class EASE(nn.Module):
    """
    Embarrassingly Shallow Autoencoder (EASE) adapted from RecBole.
    Uses scipy.sparse and numpy to speed up fitting.
    """
    def __init__(self, reg: float = 250.0):
        super().__init__()
        self.reg = reg
        self.B = None  # item-item similarity matrix (numpy)
        self.B_torch = None  # converted to torch.FloatTensor for inference

    def fit(self, R: sp.csr_matrix) -> None:
        """
        Fit EASE model using sparse user–item interaction matrix.
        Args:
            R: scipy.sparse.csr_matrix of shape (n_users, n_items)
        """
        assert sp.issparse(R), "Input R must be a scipy.sparse CSR matrix."

        # Compute Gram matrix: G = R^T R + reg * I
        G = R.T @ R
        G += self.reg * sp.identity(G.shape[0], format="csr", dtype=np.float32)

        # Convert to dense and invert
        G_dense = G.todense()
        P = np.linalg.inv(G_dense)

        # Compute B = P / (-diag(P)) and zero diagonal
        diag = np.diag(P)
        B = P / (-diag[None, :])
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.B_torch = torch.from_numpy(B).float()

    def forward(self, R: torch.FloatTensor, batch_size: int = 512) -> torch.FloatTensor:
        """
        Predict scores for given interaction matrix.
        Args:
            R: torch.FloatTensor of shape (n_users, n_items)
        Returns:
            scores: torch.FloatTensor of shape (n_users, n_items)
        """
        if self.B_torch is None:
            raise RuntimeError("EASE model must be fitted before forward pass.")

        device = R.device
        self.B_torch = self.B_torch.to(device)

        # Batched matrix multiplication
        scores = []
        for start in range(0, R.size(0), batch_size):
            end = min(start + batch_size, R.size(0))
            scores.append(R[start:end] @ self.B_torch)
        return torch.cat(scores, dim=0)
