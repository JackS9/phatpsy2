"""
PHATPSY 2.0: Two-center spin-coupled (Heitler-London / Weinbaum / Coulson-Fischer)
pair energies -- exploratory prototype (branch ``two-center-wip``, Stage 0/2).

Given two localized one-center orbitals a (on A) and b (on B) -- e.g. the occupied
alpha orbital of H_1 and beta orbital of H_2 from a PHATPSY SCF -- this module evaluates
the *exact* two-electron energy expressions that the single-center Hartree-product
functional of PHATPSY omits (Thesis Ch. II p. 305, Ch. VIII p. 936):

* ``E_HP``       Hartree product  h_aa + h_bb + (aa|bb) + V_nn
* ``E_HL``       Heitler-London covalent singlet
                 [h_aa + h_bb + 2 S h_ab + (aa|bb) + (ab|ab)] / (1 + S^2) + V_nn
* ``E_triplet``  [h_aa + h_bb - 2 S h_ab + (aa|bb) - (ab|ab)] / (1 - S^2) + V_nn
* ``E_CAS``      full singlet CI in span{a, b} (= Weinbaum covalent+ionic =
                 Coulson-Fischer optimum for a 2-electron/2-orbital bond)

Integrals come from :mod:`phatpsy.two_center_integrals` with the bare nuclear
Hamiltonian h = T - Z_A/r_A - Z_B/r_B.  For a 2-electron molecule such as H2 these
are exact expectation values (variational upper bounds); for heavier atoms the core
and other valence electrons must enter h through an effective operator (Stage 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from phatpsy.two_center_integrals import STO, ProlateGrid, eri_tensor


@dataclass
class PairEnergies:
    R: float
    S: float
    E_HP: float
    E_HL: float
    E_triplet: float
    E_CAS: float
    cas_vector: np.ndarray
    # Weight of the doubly occupied configurations |pp|, |qq| in the Loewdin-
    # orthogonalized pair basis.  NOT the nonorthogonal VB ionic weight: pure
    # Heitler-London already has a large value here because p, q carry tails.
    ionic_weight: float
    components: Dict[str, float] = field(default_factory=dict)


def _sym_orth(S: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(S)
    return v @ np.diag(w ** -0.5) @ v.T


def two_orbital_energies(S: np.ndarray, h: np.ndarray, eri: np.ndarray, vnn: float, R: float = 0.0) -> PairEnergies:
    """Energies of two electrons in two normalized, real, nonorthogonal orbitals."""
    s = float(S[0, 1])
    J = float(eri[0, 0, 1, 1])
    K = float(eri[0, 1, 0, 1])
    haa, hbb, hab = float(h[0, 0]), float(h[1, 1]), float(h[0, 1])
    E_HP = haa + hbb + J + vnn
    E_HL = (haa + hbb + 2.0 * s * hab + J + K) / (1.0 + s * s) + vnn
    E_T = (haa + hbb - 2.0 * s * hab + J - K) / (1.0 - s * s) + vnn

    # Singlet CI in the Loewdin-orthogonalized pair {p, q}
    X = _sym_orth(S)
    ho = X.T @ h @ X
    eo = np.einsum("ip,jq,kr,ls,ijkl->pqrs", X, X, X, X, eri, optimize=True)
    H = np.zeros((3, 3))
    # CSFs: |pp|, |qq|, (|pq| + |qp|)/sqrt2
    H[0, 0] = 2 * ho[0, 0] + eo[0, 0, 0, 0]
    H[1, 1] = 2 * ho[1, 1] + eo[1, 1, 1, 1]
    H[2, 2] = ho[0, 0] + ho[1, 1] + eo[0, 0, 1, 1] + eo[0, 1, 0, 1]
    H[0, 1] = H[1, 0] = eo[0, 1, 0, 1]
    H[0, 2] = H[2, 0] = np.sqrt(2.0) * (ho[0, 1] + eo[0, 0, 0, 1])
    H[1, 2] = H[2, 1] = np.sqrt(2.0) * (ho[0, 1] + eo[1, 1, 0, 1])
    w, v = np.linalg.eigh(H)
    E_CAS = float(w[0]) + vnn
    c = v[:, 0]
    return PairEnergies(
        R=R, S=s, E_HP=E_HP, E_HL=E_HL, E_triplet=E_T, E_CAS=E_CAS, cas_vector=c,
        ionic_weight=float(c[0] ** 2 + c[1] ** 2),
        components=dict(h_aa=haa, h_bb=hbb, h_ab=hab, J=J, K=K,
                        aa_ab=float(eri[0, 0, 0, 1]), bb_ab=float(eri[1, 1, 0, 1]),
                        aa_aa=float(eri[0, 0, 0, 0]), bb_bb=float(eri[1, 1, 1, 1])),
    )


def pair_energies_from_orbitals(
    grid: ProlateGrid,
    basis_a: Sequence[STO],
    coef_a: np.ndarray,
    basis_b: Sequence[STO],
    coef_b: np.ndarray,
    z_a: float,
    z_b: float,
    lmax: int = 30,
) -> PairEnergies:
    """Pair energies for orbitals a = sum c_a chi_A and b = sum c_b chi_B (renormalized)."""
    chi_a = np.array([grid.evaluate(f) for f in basis_a])
    chi_b = np.array([grid.evaluate(f) for f in basis_b])
    ta = np.array([grid.kinetic_applied(f) for f in basis_a])
    tb = np.array([grid.kinetic_applied(f) for f in basis_b])
    a = np.tensordot(coef_a, chi_a, axes=(0, 0))
    b = np.tensordot(coef_b, chi_b, axes=(0, 0))
    Ta = np.tensordot(coef_a, ta, axes=(0, 0))
    Tb = np.tensordot(coef_b, tb, axes=(0, 0))
    na = np.sqrt(grid.integrate(a * a))
    nb = np.sqrt(grid.integrate(b * b))
    a, Ta = a / na, Ta / na
    b, Tb = b / nb, Tb / nb
    vnuc = -(z_a / grid.r_a + z_b / grid.r_b)
    orbs = np.array([a, b])
    torbs = np.array([Ta, Tb])
    S = np.einsum("pijk,qijk,ijk->pq", orbs, orbs, grid.weights)
    T = np.einsum("pijk,qijk,ijk->pq", orbs, torbs, grid.weights)
    T = 0.5 * (T + T.T)
    V = np.einsum("pijk,qijk,ijk->pq", orbs, orbs, grid.weights * vnuc)
    eri = eri_tensor(grid, orbs, lmax=lmax)
    res = two_orbital_energies(S, T + V, eri, z_a * z_b / grid.R, R=grid.R)
    res.components.update(T_aa=float(T[0, 0]), T_bb=float(T[1, 1]), T_ab=float(T[0, 1]))
    return res


# ---------------------------------------------------------------------------
# PHATPSY state helpers
# ---------------------------------------------------------------------------

def localized_occupied_orbitals(state) -> Dict[int, Tuple[list, np.ndarray, int]]:
    """For each atom (1-based), return (basis STO list, coefficient vector, spin channel)
    of its most-occupied localized orbital in the converged PHATPSY state."""
    b = state["basis_info"]
    X = np.asarray(state["atom_coordinates"])
    mstot = int(state["mstot"])
    C = np.asarray(state["mo_coefficients"])[:mstot]
    occ = np.asarray(state["occupations"])
    iat = np.asarray(b["iatomx"])[:mstot]
    out = {}
    for atom in sorted(set(int(i) for i in iat if i >= 1)):
        idx = np.where(iat == atom)[0]
        basis = [
            STO(tuple(X[:, atom - 1]), int(b["nx"][i]), int(b["lx"][i]), int(b["mlx"][i]),
                float(b["etax"][i]), float(b["anormx"][i]))
            for i in idx
        ]
        best, best_occ = None, -1.0
        for s in range(C.shape[2]):
            for k in range(C.shape[1]):
                if np.max(np.abs(C[idx, k, s])) < 1e-12 or np.max(np.abs(np.delete(C[:, k, s], idx))) > 1e-12:
                    continue
                if occ[k, s] > best_occ:
                    best, best_occ = (C[idx, k, s].copy(), s), occ[k, s]
        out[atom] = (basis, best[0], best[1])
    return out
