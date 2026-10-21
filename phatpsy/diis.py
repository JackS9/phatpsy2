# phatpsy/diis.py
"""
Direct Inversion in the Iterative Subspace (DIIS) convergence accelerator.
Pulay's method for extrapolating Fock/effective potential matrices
from history of error vectors e_i = [F_i, P_i] or F_i - F_{i-1}.
"""
import numpy as np
import scipy.linalg as la
from typing import List, Optional


class DIISAccelerator:
    """
    Manages DIIS extrapolation for self-consistent field iterations.
    Maintains a history of state matrices (Fock / Hamiltonian / density)
    and corresponding error vectors to compute optimal linear combinations.
    """

    def __init__(self, max_subspace: int = 6):
        self.max_subspace = max_subspace
        self.error_history: List[np.ndarray] = []
        self.fock_history: List[np.ndarray] = []

    def reset(self):
        """Clears the DIIS history subspace."""
        self.error_history.clear()
        self.fock_history.clear()

    def update(self, fock_matrix: np.ndarray, error_vector: np.ndarray) -> np.ndarray:
        """
        Appends the current iteration's Fock matrix and error vector,
        constructs the Pulay B-matrix, and computes the extrapolated Fock matrix.

        Parameters
        ----------
        fock_matrix : np.ndarray
            The current Fock or effective Hamiltonian matrix (1D packed or 2D unpacked).
        error_vector : np.ndarray
            The error vector representing residual convergence (e.g., F P S - S P F).

        Returns
        -------
        extrapolated_fock : np.ndarray
            Optimal linear combination of past Fock matrices.
        """
        fock_flat = np.asarray(fock_matrix, dtype=np.float64).ravel()
        err_flat = np.asarray(error_vector, dtype=np.float64).ravel()

        self.fock_history.append(fock_flat)
        self.error_history.append(err_flat)

        # Truncate history if exceeding maximum subspace size
        if len(self.fock_history) > self.max_subspace:
            self.fock_history.pop(0)
            self.error_history.pop(0)

        n = len(self.error_history)
        if n < 2:
            return fock_matrix

        # Build the Pulay B-matrix:
        # | B_11  ...  B_1n  -1 | | c_1 |   | 0 |
        # |  :           :    : | |  :  | = | : |
        # | B_n1  ...  B_nn  -1 | | c_n |   | 0 |
        # | -1    ...  -1     0 | | lam |   |-1 |
        B = np.zeros((n + 1, n + 1), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1):
                dot = float(np.dot(self.error_history[i], self.error_history[j]))
                B[i, j] = dot
                B[j, i] = dot
            B[i, n] = -1.0
            B[n, i] = -1.0
        B[n, n] = 0.0

        rhs = np.zeros(n + 1, dtype=np.float64)
        rhs[n] = -1.0

        try:
            # Solve using least-squares for numerical stability
            coeffs = la.lstsq(B, rhs)[0][:n]
            # Extrapolate: F_new = sum_i c_i * F_i
            extrapolated = np.zeros_like(fock_flat)
            for c, f in zip(coeffs, self.fock_history):
                extrapolated += c * f
            return extrapolated.reshape(fock_matrix.shape)
        except Exception:
            # Fall back to original Fock matrix if inversion fails
            return fock_matrix
