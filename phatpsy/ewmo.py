import numpy as np
import scipy.linalg as la

def unpack_symmetric(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpacks a 1D lower-triangular packed Fortran array into an (n, n) symmetric matrix."""
    mat = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = packed[idx]
            mat[j, i] = packed[idx]
            idx += 1
    return mat

def solve_ewmo_propagator(ovlp_packed: np.ndarray, ewmat_packed: np.ndarray,
                          norbt: int, use_lowdin_virtuals: bool = False):
    """
    Computes EWMO molecular orbital energies and eigenvectors (Chapter VI).
    Supports both standard EWMO (F' = -X S X) and Lowdin S^(-1/2) (F = S eps S)
    when positive-energy virtual orbitals are included.
    """
    ispin = ovlp_packed.shape[1]
    eval_mo = np.zeros((norbt, ispin), dtype=np.float64)
    evec_mo = np.zeros((norbt, norbt, ispin), dtype=np.float64)
    ortho_pop_weights = np.zeros((norbt, norbt, ispin), dtype=np.float64)

    for isp in range(ispin):
        S = unpack_symmetric(ovlp_packed[:, isp], norbt)
        F_atomic = unpack_symmetric(ewmat_packed[:, isp], norbt)
        eps_diag = np.diag(F_atomic).copy()

        if not use_lowdin_virtuals:
            # Standard EWMO: Eq. (VI-18) & (VI-19) -> F'_{ij} = -sqrt(-eps_i)*sqrt(-eps_j)*S_{ij}
            eps_neg = np.where(eps_diag < 0.0, eps_diag, 0.0)
            x_vec = np.sqrt(-eps_neg)
            F_prime = -np.outer(x_vec, x_vec) * S
            evals, C_prime = la.eigh(F_prime)
            eval_mo[:, isp] = evals
            ortho_pop_weights[:, :, isp] = C_prime ** 2

            # Transform back to non-orthogonal AO basis: Eq. (VI-20)
            inv_x = np.where(eps_diag < 0.0, 1.0 / np.sqrt(np.abs(eps_diag)), 0.0)
            C_ao = C_prime * inv_x[:, None]
        else:
            # Lowdin S^(-1/2) symmetric orthogonalization: Eq. (VI-12) F = S * eps * S
            F_full = S @ np.diag(eps_diag) @ S
            S_inv_half = np.real(la.fractional_matrix_power(S, -0.5))
            F_ortho = S_inv_half @ F_full @ S_inv_half
            evals, C_prime = la.eigh(F_ortho)
            eval_mo[:, isp] = evals
            ortho_pop_weights[:, :, isp] = C_prime ** 2
            C_ao = S_inv_half @ C_prime

        # Normalize columns of C_ao with respect to metric S: <C_i | S | C_i> = 1
        norms = np.sqrt(np.einsum("ki,kl,li->i", C_ao, S, C_ao))
        norms = np.where(norms > 1e-12, norms, 1.0)
        evec_mo[:, :, isp] = C_ao / norms[None, :]

    return eval_mo, evec_mo, ortho_pop_weights