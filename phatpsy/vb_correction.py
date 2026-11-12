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

def occupied_spin_orbitals(state, occ_threshold: float = 1e-6):
    """All localized spin-orbitals with non-zero occupation in a PHATPSY state.

    Returns a list of dicts {atom, spin, occ, basis (STO list), coef}.
    """
    b = state["basis_info"]
    X = np.asarray(state["atom_coordinates"])
    mstot = int(state["mstot"])
    C = np.asarray(state["mo_coefficients"])[:mstot]
    occ = np.asarray(state["occupations"])
    iat = np.asarray(b["iatomx"])[:mstot]
    out = []
    for atom in sorted(set(int(i) for i in iat if i >= 1)):
        idx = np.where(iat == atom)[0]
        basis = [
            STO(tuple(X[:, atom - 1]), int(b["nx"][i]), int(b["lx"][i]), int(b["mlx"][i]),
                float(b["etax"][i]), float(b["anormx"][i]))
            for i in idx
        ]
        for s in range(C.shape[2]):
            for k in range(C.shape[1]):
                if occ[k, s] <= occ_threshold:
                    continue
                if np.max(np.abs(C[idx, k, s])) < 1e-12 or np.max(np.abs(np.delete(C[:, k, s], idx))) > 1e-8:
                    continue
                out.append(dict(atom=atom, spin=s, occ=float(occ[k, s]), basis=basis, coef=C[idx, k, s].copy()))
    if int(state["ispin"]) == 1:  # restricted: each spatial orbital holds both spins
        out += [dict(so, spin=1) for so in out]
    return out


def determinant_energy(grid: ProlateGrid, spin_orbitals, z_a: float, z_b: float, lmax: int = 30) -> Dict[str, float]:
    """Exact (nonrelativistic, bare-nucleus) energy of the single Slater determinant built
    from the given localized spin-orbitals of a diatomic.

    Orbitals of each spin are Loewdin-orthonormalized (this leaves the determinant
    unchanged), then
        E = sum_i <i|T - Z_A/r_A - Z_B/r_B|i> + 1/2 (rho|rho) - 1/2 sum_s sum_ij (ij|ij) + V_nn.
    The result is a variational upper bound on the exact energy.  Fractional
    occupations are rejected (a determinant needs integer occupations).
    """
    for so in spin_orbitals:
        if abs(so["occ"] - 1.0) > 1e-6:
            raise ValueError(f"fractional occupation {so['occ']} on atom {so['atom']} spin {so['spin']}")
    vnuc = -(z_a / grid.r_a + z_b / grid.r_b)
    w = grid.weights
    e1 = 0.0
    ekin = 0.0
    rho = np.zeros(grid.shape)
    ex = 0.0
    n_e = 0
    for s in sorted(set(so["spin"] for so in spin_orbitals)):
        sel = [so for so in spin_orbitals if so["spin"] == s]
        phi, tphi = [], []
        for so in sel:
            chi = np.array([grid.evaluate(f) for f in so["basis"]])
            tch = np.array([grid.kinetic_applied(f) for f in so["basis"]])
            phi.append(np.tensordot(so["coef"], chi, axes=(0, 0)))
            tphi.append(np.tensordot(so["coef"], tch, axes=(0, 0)))
        phi, tphi = np.array(phi), np.array(tphi)
        S = np.einsum("pijk,qijk,ijk->pq", phi, phi, w)
        evals, evecs = np.linalg.eigh(S)
        Xo = evecs @ np.diag(evals ** -0.5) @ evecs.T
        phi = np.tensordot(Xo, phi, axes=(0, 0))
        tphi = np.tensordot(Xo, tphi, axes=(0, 0))
        n = phi.shape[0]
        n_e += n
        ekin += float(np.einsum("pijk,pijk,ijk->", phi, tphi, w))
        e1 += float(np.einsum("pijk,pijk,ijk->", phi, phi, w * vnuc))
        rho += np.sum(phi * phi, axis=0)
        for i in range(n):
            for j in range(i + 1):
                kij = grid.coulomb(phi[i] * phi[j], phi[i] * phi[j], lmax=lmax)
                ex += kij if i == j else 2.0 * kij
    ecoul = 0.5 * grid.coulomb(rho, rho, lmax=lmax)
    vnn = z_a * z_b / grid.R
    etot = ekin + e1 + ecoul - 0.5 * ex + vnn
    return dict(E=etot, T=ekin, V_ne=e1, J=ecoul, K=-0.5 * ex, V_nn=vnn,
                N=float(grid.integrate(rho)), n_spin_orbitals=n_e, virial=-(etot - ekin) / ekin)


def interatomic_decomposition(grid: ProlateGrid, spin_orbitals, z_a: float, z_b: float, lmax: int = 30) -> Dict[str, float]:
    """Split the exact single-determinant energy of a diatomic built from localized
    orbitals into
        E_intra(A) + E_intra(B)        each atom's own determinant with its own nucleus
        E_elst                         exact classical interaction: cross nuclear attraction
                                       + (rho_A|rho_B) + Z_A Z_B / R
        E_HP = E_intra + E_elst        Hartree product of atomic determinants
        E_pauli = E_det - E_HP         interatomic exchange + orthogonalization (the role
                                       PHATPSY assigns to Phillips-Kleinman projection)
    """
    atoms = sorted(set(so["atom"] for so in spin_orbitals))
    zs = {atoms[0]: z_a, atoms[1]: z_b}
    full = determinant_energy(grid, spin_orbitals, z_a, z_b, lmax=lmax)
    intra, rho = {}, {}
    for at in atoms:
        sel = [so for so in spin_orbitals if so["atom"] == at]
        za, zb = (zs[at], 0.0) if at == atoms[0] else (0.0, zs[at])
        intra[at] = determinant_energy(grid, sel, za, zb, lmax=lmax)["E"]
        r = np.zeros(grid.shape)
        for so in sel:
            chi = np.array([grid.evaluate(f) for f in so["basis"]])
            phi = np.tensordot(so["coef"], chi, axes=(0, 0))
            phi /= np.sqrt(grid.integrate(phi * phi))
            r += phi * phi
        rho[at] = r
    a, b = atoms
    e_cross_nuc = grid.integrate(rho[a] * (-z_b / grid.r_b)) + grid.integrate(rho[b] * (-z_a / grid.r_a))
    e_ab = grid.coulomb(rho[a], rho[b], lmax=lmax)
    e_elst = e_cross_nuc + e_ab + z_a * z_b / grid.R
    e_hp = intra[a] + intra[b] + e_elst
    return dict(E_det=full["E"], E_HP=e_hp, E_intra_A=intra[a], E_intra_B=intra[b],
                E_elst=e_elst, E_pauli=full["E"] - e_hp)


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
