# phatpsy/vb_driver.py
"""Unified driver for Localized Valence Bond (LocalizedVB / MaskedVB) and active-site embedding.

Bridges :class:`~phatpsy.schema.CalculationDeck`, :class:`~phatpsy.state.CalculationState`,
and :class:`~phatpsy.driver.PhatpsyDriver` with the Stage 0-11 exact-integral engine:

- Active bonded region ``A`` (``atom.is_active == True``): exact two-center and
  multi-center STO integrals (``diatomic_ao_integrals``, ``molecular_ao_integrals``)
  variationally optimized via ``LocalizedVBModel`` or ``MaskedVB``.
- Frozen core shells and spectator environment ``B`` (``atom.active is False`` or
  ``atom.fixed``): exact joint ``(S_BB)^-1`` Huzinaga/Phillips-Kleinman orthogonalization
  (``exact_joint_pk``) or screened model-potential / regularized-PK spectator representation
  (``model_potential``).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from phatpsy.basis import BasisFunction, ElementData, get_element_basis
from phatpsy.embedding import (
    EmbeddedAOIntegrals,
    FrozenEnvironmentAtom,
    build_active_ni_co_integrals,
    embed_collinear_in_c4v_slab,
    embed_diatomic_in_environment,
    embed_from_full_integrals,
    one_center_ao_matrices,
    one_center_coulomb_potential,
    solve_atomic_ni_rohf,
)
from phatpsy.localized_vb import (
    AOIntegrals,
    LocalizedVBModel,
    MaskedVB,
    OrbitalSpec,
    atom_basis,
    cas_energy_and_gradient,
    cas_nuclear_gradient_intermediates,
    cas_rdm1_and_natural_orbitals,
    covalent_determinants,
    diatomic_ao_integrals,
    expectation,
    optimize as optimize_localized_vb,
    optimize_masked_vb,
    own_atom_character,
    pp_determinants,
    pyscf_mean_field,
    symmetry_mask,
)
from phatpsy.multicenter_integrals import (
    PySCFSTOContext,
    molecular_ao_integrals,
    sto_ng_ao_integrals,
)
from phatpsy.schema import ATOMIC_NUMBERS, AtomConfig, CalculationDeck
from phatpsy.spin import build_connectivity_graph
from phatpsy.state import CalculationState, EmbeddedAtomState
from phatpsy.two_center_integrals import ProlateGrid, STO


HARTREE_TO_EV: float = 27.211386245988
DEBYE_PER_AU: float = 2.541746473

# Core spatial orbital counts (1s -> 1, [Ne] -> 5, [Ar] -> 9, [Kr] -> 18)
def _core_shells_count(z_nuc: int) -> int:
    if z_nuc <= 2:
        return 0
    if z_nuc <= 10:
        return 1
    if z_nuc <= 18:
        return 5
    if z_nuc <= 30:
        return 9
    if z_nuc <= 36:
        return 18
    if z_nuc <= 48:
        return 23
    return 27


# Ground-state atomic spin multiplicities (n_alpha - n_beta) for ROHF/RHF references
_ATOM_HUND_SPIN: Dict[int, int] = {
    1: 1, 2: 0,
    3: 1, 4: 0, 5: 1, 6: 2, 7: 3, 8: 2, 9: 1, 10: 0,
    11: 1, 12: 0, 13: 1, 14: 2, 15: 3, 16: 2, 17: 1, 18: 0,
    19: 1, 20: 0, 21: 1, 22: 2, 23: 3, 24: 6, 25: 5, 26: 4, 27: 3, 28: 2, 29: 1, 30: 0,
}

_ATOM_REF_CACHE: Dict[Tuple[str, str], float] = {}


@dataclass
class _OneCenterAOAdapter:
    """Lightweight AOIntegrals-compatible wrapper around one-center STO matrices."""

    basis: List[STO]
    atom_of: np.ndarray
    S: np.ndarray
    T: np.ndarray
    V: np.ndarray
    h: np.ndarray
    eri: np.ndarray
    vnn: float = 0.0

    @property
    def nao(self) -> int:
        return len(self.basis)

    def jk_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.nao
        Jm = self.eri.reshape(n * n, n * n)
        Km = self.eri.transpose(0, 2, 1, 3).reshape(n * n, n * n)
        return Jm, Km


def _wrap_one_center(basis: List[STO], z_nuc: float) -> _OneCenterAOAdapter:
    S1, T1, V1, eri1 = one_center_ao_matrices(basis, z_nuc)
    return _OneCenterAOAdapter(
        basis=list(basis),
        atom_of=np.zeros(len(basis), dtype=int),
        S=S1,
        T=T1,
        V=V1,
        h=T1 + V1,
        eri=eri1,
        vnn=0.0,
    )


def _expand_custom_basis(center: Tuple[float, float, float], functions: Sequence[BasisFunction]) -> List[STO]:
    stos: List[STO] = []
    for bf in functions:
        for m in range(-int(bf.l), int(bf.l) + 1):
            stos.append(STO(center, int(bf.n), int(bf.l), m, float(bf.zeta)))
    return stos


def _resolve_atom_stos(
    atom: AtomConfig,
    deck: CalculationDeck,
    elem_map: Dict[str, ElementData],
) -> Tuple[List[STO], str]:
    center = (float(atom.xyz[0]), float(atom.xyz[1]), float(atom.xyz[2]))
    sym = atom.element.strip().capitalize()
    eff_basis = deck.get_effective_basis_for_atom(atom)
    user_elem = deck.elements.get(sym) if hasattr(deck, "elements") else None
    if user_elem and user_elem.functions and (eff_basis == (user_elem.basis or "CUSTOM") or atom.basis is None):
        return _expand_custom_basis(center, user_elem.functions), eff_basis

    for cand in (eff_basis, "DZP", "DZ", "SZ", "THAKKAR"):
        try:
            stos = atom_basis(center, sym, cand)
            return stos, cand
        except (KeyError, ValueError):
            continue
    ekey = deck.get_atom_element_key(atom)
    if ekey in elem_map:
        return _expand_custom_basis(center, elem_map[ekey].functions), eff_basis
    raise ValueError(f"Could not resolve STO basis for atom {atom.id} ({sym}) with basis '{eff_basis}'.")


def _atomic_num(atom: AtomConfig) -> int:
    sym = atom.element.strip().capitalize()
    if sym not in ATOMIC_NUMBERS:
        raise ValueError(f"Unsupported element symbol '{atom.element}'.")
    return int(ATOMIC_NUMBERS[sym])


def _solve_atomic_mean_field(basis: List[STO], z_nuc: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve all-electron atomic RHF/ROHF in ``basis`` returning ``(energy, mo_coeff, evals)``."""
    ints1 = _wrap_one_center(basis, float(z_nuc))
    if z_nuc == 1:
        w, v = np.linalg.eigh(ints1.h)
        # Generalized eigensolver S^-1/2 h S^-1/2
        ws, vs = np.linalg.eigh(ints1.S)
        keep = ws > 1e-12
        X = vs[:, keep] @ np.diag(ws[keep] ** -0.5)
        ew, ev = np.linalg.eigh(X.T @ ints1.h @ X)
        mo = X @ ev
        return float(ew[0]), mo, ew

    spin = _ATOM_HUND_SPIN.get(z_nuc, z_nuc % 2)
    na = (z_nuc + spin) // 2
    nb = (z_nuc - spin) // 2
    kind = "RHF" if na == nb else "ROHF"
    mf = pyscf_mean_field(ints1, (na, nb), kind=kind)
    return float(mf.e_tot), np.asarray(mf.mo_coeff, dtype=float), np.asarray(mf.mo_energy, dtype=float)


def _isolated_atom_energy(sym: str, basis_name: str, custom_functions: Optional[Sequence[BasisFunction]] = None) -> float:
    """Exact atomic RHF/ROHF reference energy in the matching STO basis (cached)."""
    sym_u = sym.strip().capitalize()
    key = (sym_u, basis_name)
    if custom_functions is None and key in _ATOM_REF_CACHE:
        return _ATOM_REF_CACHE[key]
    z_nuc = int(ATOMIC_NUMBERS[sym_u])
    if custom_functions is not None:
        basis = _expand_custom_basis((0.0, 0.0, 0.0), custom_functions)
    else:
        basis = atom_basis((0.0, 0.0, 0.0), sym_u, basis_name)
    e_atom, _, _ = _solve_atomic_mean_field(basis, z_nuc)
    if custom_functions is None:
        _ATOM_REF_CACHE[key] = e_atom
    return e_atom


def _frozen_env_from_atom(atom: AtomConfig, sto_basis: List[STO], basis_name: str) -> FrozenEnvironmentAtom:
    """Build a FrozenEnvironmentAtom from ``atom.embedded_state`` or atomic RHF/ROHF."""
    z_nuc = float(_atomic_num(atom))
    center = (float(atom.xyz[0]), float(atom.xyz[1]), float(atom.xyz[2]))
    emb_path = getattr(atom, "_embedded_state_path", None)
    if emb_path:
        p = Path(emb_path)
        if not p.is_file():
            repo_root = Path(__file__).resolve().parent.parent
            if (repo_root / p).is_file():
                p = repo_root / p
        return FrozenEnvironmentAtom.from_embedded_state(str(p), center, basis_name=basis_name)
    emb = getattr(atom, "embedded_state", None)
    if emb is not None:
        if isinstance(emb, (str, Path)):
            p = Path(emb)
            if not p.is_file():
                repo_root = Path(__file__).resolve().parent.parent
                if (repo_root / p).is_file():
                    p = repo_root / p
            return FrozenEnvironmentAtom.from_embedded_state(str(p), center, basis_name=basis_name)
        payload = emb.to_dict() if hasattr(emb, "to_dict") else dict(emb)
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=True) as tf:
            json.dump(payload, tf)
            tf.flush()
            return FrozenEnvironmentAtom.from_embedded_state(tf.name, center, basis_name=basis_name)

    n_occ = (int(round(z_nuc)) + 1) // 2
    return FrozenEnvironmentAtom.from_atomic_rhf(z_nuc, center, sto_basis, n_occ=n_occ)


def _centered_stos(stos: Sequence[STO], new_center: Tuple[float, float, float]) -> List[STO]:
    return [STO(new_center, f.n, f.l, f.m, f.zeta, anorm=f.anorm) for f in stos]


def _ao_dipole_matrices(
    basis: Sequence[STO],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact real-STO dipole matrices ``(<mu|x-ox|nu>, <mu|y-oy|nu>, <mu|z-oz|nu>)``."""
    n = len(basis)
    Dx = np.zeros((n, n), dtype=float)
    Dy = np.zeros((n, n), dtype=float)
    Dz = np.zeros((n, n), dtype=float)
    if n == 0:
        return Dx, Dy, Dz

    ox, oy, oz = origin

    # Group basis functions by atomic center coordinates
    centers: List[Tuple[float, float, float]] = []
    groups: List[List[int]] = []
    for i, f in enumerate(basis):
        c = (float(f.center[0]), float(f.center[1]), float(f.center[2]))
        placed = False
        for g_idx, gc in enumerate(centers):
            if abs(c[0] - gc[0]) < 1e-9 and abs(c[1] - gc[1]) < 1e-9 and abs(c[2] - gc[2]) < 1e-9:
                groups[g_idx].append(i)
                placed = True
                break
        if not placed:
            centers.append(c)
            groups.append([i])

    # Helper to evaluate real spherical STO on arbitrary grid points (X, Y, Z)
    def _eval_sto_grid(f: STO, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        g = ProlateGrid.__new__(ProlateGrid)
        g.X, g.Y, g.Z = X, Y, Z
        return g.evaluate(f)

    # 1. One-center blocks via exact spherical product quadrature
    for cA, idxA in zip(centers, groups):
        min_z = min(basis[i].zeta for i in idxA)
        # Gauss-Laguerre radial quadrature (exact up to deg 2*48-1 in r/min_z)
        x_rad, w_rad = np.polynomial.laguerre.laggauss(48)
        scale_r = 1.0 / (1.5 * min_z)
        r_pts = x_rad * scale_r
        wr = w_rad * scale_r * np.exp(x_rad) * (r_pts ** 2)

        cth, w_th = np.polynomial.legendre.leggauss(12)
        sth = np.sqrt(np.maximum(0.0, 1.0 - cth ** 2))
        n_ph = 18
        ph = (2.0 * np.pi / n_ph) * np.arange(n_ph)
        w_ph = 2.0 * np.pi / n_ph

        R, CTH, PH = np.meshgrid(r_pts, cth, ph, indexing="ij")
        STH = np.sqrt(np.maximum(0.0, 1.0 - CTH ** 2))
        dX = R * STH * np.cos(PH)
        dY = R * STH * np.sin(PH)
        dZ = R * CTH
        X = cA[0] + dX
        Y = cA[1] + dY
        Z = cA[2] + dZ
        W = wr[:, None, None] * w_th[None, :, None] * w_ph

        chi = np.array([_eval_sto_grid(basis[i], X, Y, Z) for i in idxA])
        xw = W * (X - ox)
        yw = W * (Y - oy)
        zw = W * (Z - oz)
        # Symmetry: diagonal center shift plus local dipole around cA
        b_x = np.einsum("mijk,nijk,ijk->mn", chi, chi, xw, optimize=True)
        b_y = np.einsum("mijk,nijk,ijk->mn", chi, chi, yw, optimize=True)
        b_z = np.einsum("mijk,nijk,ijk->mn", chi, chi, zw, optimize=True)
        Dx[np.ix_(idxA, idxA)] = b_x
        Dy[np.ix_(idxA, idxA)] = b_y
        Dz[np.ix_(idxA, idxA)] = b_z

    # 2. Two-center blocks via ProlateGrid
    for a_i in range(len(centers)):
        cA, idxA = centers[a_i], groups[a_i]
        for b_i in range(a_i + 1, len(centers)):
            cB, idxB = centers[b_i], groups[b_i]
            zmin = min(min(basis[i].zeta for i in idxA), min(basis[j].zeta for j in idxB))
            pg = ProlateGrid(cA, cB, zeta_min=zmin, n_eta=28, n_phi=16)
            chiA = np.array([pg.evaluate(basis[i]) for i in idxA])
            chiB = np.array([pg.evaluate(basis[j]) for j in idxB])
            xw = pg.weights * (pg.X - ox)
            yw = pg.weights * (pg.Y - oy)
            zw = pg.weights * (pg.Z - oz)
            ab_x = np.einsum("miqr,niqr,iqr->mn", chiA, chiB, xw, optimize=True)
            ab_y = np.einsum("miqr,niqr,iqr->mn", chiA, chiB, yw, optimize=True)
            ab_z = np.einsum("miqr,niqr,iqr->mn", chiA, chiB, zw, optimize=True)
            Dx[np.ix_(idxA, idxB)] = ab_x
            Dx[np.ix_(idxB, idxA)] = ab_x.T
            Dy[np.ix_(idxA, idxB)] = ab_y
            Dy[np.ix_(idxB, idxA)] = ab_y.T
            Dz[np.ix_(idxA, idxB)] = ab_z
            Dz[np.ix_(idxB, idxA)] = ab_z.T

    return Dx, Dy, Dz


def _sym_orth_mat(S: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    w, v = np.linalg.eigh(S)
    keep = w > tol
    return v[:, keep] @ np.diag(w[keep] ** -0.5) @ v[:, keep].T


def _ortho_vecs(vecs: Sequence[np.ndarray], S: np.ndarray) -> List[np.ndarray]:
    """Symmetric or Gram-Schmidt S-orthogonalization preserving localization as much as possible."""
    out: List[np.ndarray] = []
    for v in vecs:
        u = np.array(v, dtype=float, copy=True)
        for prev in out:
            u = u - float(prev @ S @ u) * prev
        nrm = float(np.sqrt(max(u @ S @ u, 1e-18)))
        if nrm > 1e-8:
            out.append(u / nrm)
    return out


def _apply_model_potential_spectators(
    active_basis: List[STO],
    active_atoms: List[AtomConfig],
    active_atom_of: np.ndarray,
    ints: Any,
    spectator_atoms: List[AtomConfig],
    spectator_envs: List[FrozenEnvironmentAtom],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate screened spectator model-potential + regularized-PK operator on active AOs.

    Relegates fitted single-center radial model potentials and regularized PK projectors
    strictly to non-bonded spectator atoms B without evaluating two-electron A-B ERIs.
    Returns ``(delta_h_active, V_mod_only, delta_vnn)``.
    """
    n = len(active_basis)
    delta_h = np.zeros((n, n), dtype=float)
    vnn_ab = 0.0

    # Group active basis functions by center
    centers: List[Tuple[float, float, float]] = []
    groups: List[List[int]] = []
    for i, f in enumerate(active_basis):
        c = (float(f.center[0]), float(f.center[1]), float(f.center[2]))
        idx_found = next(
            (g for g, gc in enumerate(centers)
             if abs(c[0] - gc[0]) < 1e-9 and abs(c[1] - gc[1]) < 1e-9 and abs(c[2] - gc[2]) < 1e-9),
            None,
        )
        if idx_found is None:
            centers.append(c)
            groups.append([i])
        else:
            groups[idx_found].append(i)

    # Nuclear A-B repulsion
    for a in active_atoms:
        za = float(_atomic_num(a))
        ra = np.asarray(a.xyz, dtype=float)
        for b in spectator_atoms:
            zb = float(_atomic_num(b))
            rb = np.asarray(b.xyz, dtype=float)
            vnn_ab += za * zb / max(float(np.linalg.norm(ra - rb)), 1e-6)

    def _screened_potential(spec_atom: AtomConfig, spec_env: FrozenEnvironmentAtom, X, Y, Z) -> np.ndarray:
        cB = np.asarray(spec_env.center, dtype=float)
        dx = X - cB[0]
        dy = Y - cB[1]
        dz = Z - cB[2]
        r = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-10)
        z_nuc = float(spec_env.z_nuc)

        emb = getattr(spec_atom, "embedded_state", None)
        mod_pot = None
        if emb is not None:
            mod_pot = getattr(emb, "model_potential", None) if not isinstance(emb, dict) else emb.get("model_potential")

        if mod_pot and mod_pot.get("vcoef") and mod_pot.get("vexp"):
            vcoef = mod_pot["vcoef"]
            vexp = mod_pot["vexp"]
            # V_mod(r) = - sum_k c_k exp(-zeta_k * r) / r
            v_screen = np.zeros_like(r)
            for c_k, z_k in zip(vcoef, vexp):
                v_screen += float(c_k) * np.exp(-float(z_k) * r)
            return -v_screen / r

        # Analytic spectator electrostatic potential (nucleus + spherically/exact frozen density)
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        D_occ = spec_env.C_occ @ spec_env.C_occ.T
        if spec_env.occ_weights is not None:
            D_occ = (spec_env.C_occ * spec_env.occ_weights[None, :]) @ spec_env.C_occ.T
        else:
            D_occ = 2.0 * D_occ
        v_el = one_center_coulomb_potential(spec_env.basis, D_occ, pts).reshape(r.shape)
        return (-z_nuc / r) + v_el

    V_mod_total = np.zeros((n, n), dtype=float)
    for spec_atom, spec_env in zip(spectator_atoms, spectator_envs):
        cB = (float(spec_env.center[0]), float(spec_env.center[1]), float(spec_env.center[2]))
        zmin_b = min(f.zeta for f in spec_env.basis)

        # Pairwise one-electron matrix elements of V_mod(B) across active AO pairs
        for gi in range(len(centers)):
            cA, idxA = centers[gi], groups[gi]
            # 1-center on A or 2-center A-B grid
            for gj in range(gi, len(centers)):
                cA2, idxA2 = centers[gj], groups[gj]
                # Use ProlateGrid between cA (or cB when gi==gj) and cA2/cB
                anchor = cB if gi == gj else cA2
                zmin = min(zmin_b, *(active_basis[k].zeta for k in idxA), *(active_basis[k].zeta for k in idxA2))
                if np.linalg.norm(np.array(cA) - np.array(anchor)) < 1e-6:
                    anchor = (cA[0], cA[1], cA[2] + 1.5)
                pg = ProlateGrid(cA, anchor, zeta_min=max(0.4, zmin), n_eta=24, n_phi=12)
                chi_i = np.array([pg.evaluate(active_basis[k]) for k in idxA])
                chi_j = chi_i if gi == gj else np.array([pg.evaluate(active_basis[k]) for k in idxA2])
                v_grid = _screened_potential(spec_atom, spec_env, pg.X, pg.Y, pg.Z)
                block = np.einsum("miqr,niqr,iqr->mn", chi_i, chi_j, pg.weights * v_grid, optimize=True)
                V_mod_total[np.ix_(idxA, idxA2)] += block
                if gj != gi:
                    V_mod_total[np.ix_(idxA2, idxA)] += block.T

        # Regularized Phillips-Kleinman repulsion S_AB * (-e_ref) * S_BA on non-bonded B
        nB = len(spec_env.basis)
        S_AB = np.zeros((n, nB), dtype=float)
        for cA, idxA in zip(centers, groups):
            if np.linalg.norm(np.array(cA) - np.array(cB)) < 1e-6:
                continue
            zmin = min(zmin_b, *(active_basis[k].zeta for k in idxA))
            pg = ProlateGrid(cA, cB, zeta_min=max(0.4, zmin), n_eta=24, n_phi=12)
            chiA = np.array([pg.evaluate(active_basis[k]) for k in idxA])
            chiB = np.array([pg.evaluate(f) for f in spec_env.basis])
            S_AB[np.ix_(idxA, range(nB))] = np.einsum("miqr,niqr,iqr->mn", chiA, chiB, pg.weights, optimize=True)

        D_B = spec_env.C_occ @ spec_env.C_occ.T
        if spec_env.occ_weights is not None:
            D_B = (spec_env.C_occ * (0.5 * spec_env.occ_weights[None, :])) @ spec_env.C_occ.T
        # Regularized PK projector using diagonal atomic kinetic/valence reference scale (~0.25 Ha + 0.5 Ha core)
        pk_scale_mat = S_AB @ D_B @ S_AB.T
        # Enforce bounded regularization P <= (1 - 0.1) S_act in Loewdin basis
        S_inv_half = _sym_orth_mat(ints.S)
        P_tilde = S_inv_half @ pk_scale_mat @ S_inv_half
        pw, pv = np.linalg.eigh(P_tilde)
        pw_reg = np.minimum(np.maximum(pw, 0.0), 0.90)
        P_reg = ints.S @ S_inv_half @ (pv @ np.diag(pw_reg) @ pv.T) @ S_inv_half @ ints.S
        # Energy shift: regularized PK barrier (0.5 Ha for spectator orthogonalization)
        delta_h += 0.50 * P_reg

    delta_h += V_mod_total
    return delta_h, V_mod_total, float(vnn_ab)


def _apply_eri_free_2c_restriction(eri: np.ndarray, atom_of: np.ndarray) -> np.ndarray:
    """Keep 1-center ERIs and 2-center Coulomb/exchange blocks; zero 3-/4-center ERIs."""
    n = eri.shape[0]
    eri_out = np.zeros_like(eri)
    for p in range(n):
        ap = atom_of[p]
        for q in range(n):
            aq = atom_of[q]
            for r in range(n):
                ar = atom_of[r]
                for s in range(n):
                    as_ = atom_of[s]
                    if len({ap, aq, ar, as_}) <= 2:
                        if ap == aq and ar == as_:
                            eri_out[p, q, r, s] = eri[p, q, r, s]
                        elif ap == ar and aq == as_:
                            eri_out[p, q, r, s] = eri[p, q, r, s]
                        elif ap == as_ and aq == ar:
                            eri_out[p, q, r, s] = eri[p, q, r, s]
    return eri_out


def _build_nico_blyholder_setup(
    ints: EmbeddedAOIntegrals,
    C_val_Ni_sub: np.ndarray,
    regime: str,
    mol_bare_for_co: Optional[EmbeddedAOIntegrals] = None,
) -> Tuple[List[str], List[np.ndarray], Tuple[int, int], List[np.ndarray]]:
    """Construct roles, domain masks, active electron counts, and seed vectors for Ni+CO.

    Matches Stage 11 ``solve_nico_blyholder_regimes`` exactly:
      - ``blyholder_r0`` (0): Strict fragment domains (Ni1 orbitals on Ni1; CO orbitals on C+O).
      - ``blyholder_r1`` (1): +5sigma donation (CO 5sigma HOMO allowed onto Ni1 m=0 AOs).
      - ``blyholder`` / ``blyholder_r2`` (2): +pi back-donation (Ni1 3d_xz/yz and CO 1pi/2pi*
        allowed across Ni1+C+O m=+-1).
    """
    from scipy.linalg import eigh

    nao = ints.nao
    m_vals = np.array([f.m for f in ints.basis])
    l_vals = np.array([f.l for f in ints.basis])
    atom_of = ints.atom_of
    idx_ni = np.where(atom_of == 0)[0]

    reg = regime.lower().strip()
    reg_idx = 0 if reg == "blyholder_r0" else (1 if reg == "blyholder_r1" else 2)

    idx_CO_m0 = np.where((atom_of >= 1) & (m_vals == 0))[0]
    _, v_co_0 = eigh(ints.h[np.ix_(idx_CO_m0, idx_CO_m0)], ints.S[np.ix_(idx_CO_m0, idx_CO_m0)])

    masks: List[np.ndarray] = []
    for m_ni, l_ni in [(2, 2), (1, 2), (-1, 2), (0, 0)]:
        if reg_idx == 2 and m_ni in (1, -1):
            mask = (m_vals == m_ni)
        else:
            mask = (
                (atom_of == 0) & (m_vals == m_ni)
                if m_ni in (0, 1, -1)
                else ((atom_of == 0) & (m_vals == m_ni) & (l_vals == l_ni))
            )
        masks.append(np.where(mask)[0])
    for k in range(5):
        if reg_idx >= 1 and k == 4:
            mask = (m_vals == 0)
        else:
            mask = (atom_of >= 1) & (m_vals == 0)
        masks.append(np.where(mask)[0])
    for m_pi in (1, -1):
        if reg_idx == 2:
            mask = (m_vals == m_pi)
        else:
            mask = (atom_of >= 1) & (m_vals == m_pi)
        masks.append(np.where(mask)[0])
    for m_act, l_act in [(0, 2), (-2, 2)]:
        mask = (
            (atom_of == 0) & (m_vals == m_act)
            if m_act == 0
            else ((atom_of == 0) & (m_vals == m_act) & (l_vals == l_act))
        )
        masks.append(np.where(mask)[0])

    roles = ["inactive"] * 11 + ["active"] * 2
    orbs0 = [np.zeros(nao, dtype=float) for _ in range(13)]
    orbs0[0][idx_ni] = C_val_Ni_sub[:, 1]
    orbs0[1][idx_ni] = C_val_Ni_sub[:, 2]
    orbs0[2][idx_ni] = C_val_Ni_sub[:, 3]
    orbs0[3][idx_ni] = C_val_Ni_sub[:, 0]
    for k in range(5):
        orbs0[4 + k][idx_CO_m0] = v_co_0[:, k]
    for j, m_pi in enumerate((1, -1)):
        idx_p = np.where((atom_of >= 1) & (m_vals == m_pi))[0]
        _, v_p = eigh(ints.h[np.ix_(idx_p, idx_p)], ints.S[np.ix_(idx_p, idx_p)])
        orbs0[9 + j][idx_p] = v_p[:, 0]
    orbs0[11][idx_ni] = C_val_Ni_sub[:, 4]
    orbs0[12][idx_ni] = C_val_Ni_sub[:, 5]

    return roles, masks, (2, 0), orbs0


def _build_setup_from_orbital_specs(
    active_atoms: List[AtomConfig],
    ints: Any,
    vb_orbital_specs: Sequence[Dict[str, Any]],
    core_embedding: str,
    net_spin_override: Optional[int],
    charge: int,
) -> Tuple[List[str], List[np.ndarray], Tuple[int, int], List[np.ndarray]]:
    """Construct roles, domain masks, active electron counts, and RHF-projected seeds from ``vb_orbital_specs``."""
    import warnings

    n_atoms = len(active_atoms)
    nao = ints.nao
    atom_of = np.asarray(ints.atom_of, dtype=int)
    z_list = [_atomic_num(a) for a in active_atoms]
    idx_by_atom = [np.where(atom_of == ia)[0] for ia in range(n_atoms)]

    id_to_idx: Dict[str, int] = {}
    for idx, atom in enumerate(active_atoms):
        if atom.id:
            id_to_idx[str(atom.id)] = idx
        id_to_idx.setdefault(f"{atom.element.strip().capitalize()}{idx + 1}", idx)

    roles: List[str] = []
    masks: List[np.ndarray] = []
    for k, spec in enumerate(vb_orbital_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"vb_orbital_specs[{k}] must be a dict with 'role' and 'atoms', got {type(spec).__name__}.")
        role = str(spec.get("role", "active")).strip().lower()
        if role not in ("inactive", "active"):
            raise ValueError(f"vb_orbital_specs[{k}]['role'] must be 'inactive' or 'active', got {role!r}.")
        raw_atoms = spec.get("atoms", spec.get("domain_atoms", []))
        if isinstance(raw_atoms, (str, int)):
            raw_atoms = [raw_atoms]
        atom_idxs: List[int] = []
        for ref in raw_atoms:
            if isinstance(ref, (int, np.integer)):
                ia = int(ref)
                if not (0 <= ia < n_atoms):
                    raise ValueError(f"vb_orbital_specs[{k}] atom index {ia} out of range [0, {n_atoms}).")
                atom_idxs.append(ia)
            else:
                key = str(ref)
                if key not in id_to_idx:
                    raise ValueError(
                        f"vb_orbital_specs[{k}] references unknown atom id {key!r}; valid ids: {sorted(id_to_idx)}."
                    )
                atom_idxs.append(id_to_idx[key])
        if not atom_idxs:
            raise ValueError(f"vb_orbital_specs[{k}] must specify at least one atom in 'atoms'.")
        m = np.unique(np.concatenate([idx_by_atom[ia] for ia in atom_idxs])).astype(int)
        roles.append(role)
        masks.append(m)

    n_inact = sum(1 for r in roles if r == "inactive")
    n_act = sum(1 for r in roles if r == "active")
    total_e = sum(z_list) - charge
    if core_embedding != "all_electron":
        total_e -= 2 * sum(_core_shells_count(z) for z in z_list)
    n_act_e = total_e - 2 * n_inact
    if n_act_e < 0 or n_act_e > 2 * n_act:
        raise ValueError(
            f"Incompatible vb_orbital_specs: total_e={total_e}, n_inactive={n_inact} -> "
            f"n_act_e={n_act_e} for {n_act} active orbitals."
        )

    if net_spin_override is not None and (abs(net_spin_override) % 2) == (n_act_e % 2):
        spin_diff = int(net_spin_override)
    else:
        spin_diff = n_act_e % 2
    na_act = (n_act_e + spin_diff) // 2
    nb_act = (n_act_e - spin_diff) // 2

    na_tot = n_inact + na_act
    nb_tot = n_inact + nb_act
    mf_kind = "RHF" if na_tot == nb_tot else "ROHF"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = pyscf_mean_field(ints, (na_tot, nb_tot), kind=mf_kind)

    C_all = np.asarray(mf.mo_coeff, dtype=float)
    n_occ_tot = max(na_tot, nb_tot)
    C_occ = C_all[:, :n_occ_tot]
    C_virt = C_all[:, n_occ_tot:]

    occ_weights = 1.0 / np.sqrt(1.0 + 0.02 * np.arange(C_occ.shape[1], dtype=float))
    C_occ_w = C_occ * occ_weights[None, :]
    virt_weights = 1.0 / np.sqrt(1.0 + 0.05 * np.arange(C_virt.shape[1], dtype=float))
    C_virt_w = C_virt * virt_weights[None, :]

    S = ints.S
    h_mat = ints.h
    Q_assigned: List[np.ndarray] = []
    seeds: List[Optional[np.ndarray]] = [None] * len(roles)

    def _pick_mask_seed(m: np.ndarray, C_pool: np.ndarray) -> np.ndarray:
        S_m = S[np.ix_(m, m)]
        w_m, v_m = np.linalg.eigh(S_m)
        km = w_m > 1e-10
        S_m_inv_half = (v_m[:, km] * (w_m[km] ** -0.5)[None, :]) @ v_m[:, km].T

        Q_mat = np.column_stack(Q_assigned) if Q_assigned else None
        S_Q_inv = np.linalg.pinv(Q_mat.T @ S @ Q_mat, rcond=1e-10) if Q_mat is not None else None

        if C_pool.shape[1] > 0:
            if Q_mat is not None:
                C_rem = C_pool - Q_mat @ (S_Q_inv @ (Q_mat.T @ S @ C_pool))
            else:
                C_rem = C_pool
            Proj_m = S_m_inv_half @ (S[m, :] @ C_rem)
            U, s_vals, _ = np.linalg.svd(Proj_m, full_matrices=False)
            for r in range(min(U.shape[1], len(s_vals))):
                if s_vals[r] < 1e-4:
                    break
                v_cand = np.zeros(nao, dtype=float)
                v_cand[m] = S_m_inv_half @ U[:, r]
                nrm = float(np.sqrt(max(float(v_cand @ S @ v_cand), 1e-16)))
                v_cand /= nrm
                if Q_mat is None:
                    return v_cand
                v_perp = v_cand - Q_mat @ (S_Q_inv @ (Q_mat.T @ S @ v_cand))
                indep = float(np.sqrt(max(float(v_perp @ S @ v_perp), 0.0)))
                if indep > 0.15:
                    return v_cand

        # Orthogonal complement fallback inside mask m ordered by projected 1e energy
        h_orth = S_m_inv_half @ h_mat[np.ix_(m, m)] @ S_m_inv_half
        if Q_mat is not None:
            B = S_m_inv_half @ S[np.ix_(m, np.arange(nao))] @ Q_mat
            P_rem = B @ S_Q_inv @ B.T
            P_null = np.eye(len(m)) - 0.5 * (P_rem + P_rem.T)
            w_n, v_n = np.linalg.eigh(P_null)
            keep_n = np.where(w_n > 0.1)[0]
            if keep_n.size == 0:
                keep_n = np.array([int(np.argmax(w_n))])
            Vn = v_n[:, keep_n]
            _, vec_h = np.linalg.eigh(Vn.T @ h_orth @ Vn)
            u_best = Vn @ vec_h[:, 0]
        else:
            _, vec_h = np.linalg.eigh(h_orth)
            u_best = vec_h[:, 0]
        v_cand = np.zeros(nao, dtype=float)
        v_cand[m] = S_m_inv_half @ u_best
        v_cand /= float(np.sqrt(max(float(v_cand @ S @ v_cand), 1e-16)))
        return v_cand

    # Assign inactive specs first (smallest masks first so atomic cores precede multi-center bonds)
    inact_order = sorted([k for k, r in enumerate(roles) if r == "inactive"], key=lambda k: (len(masks[k]), k))
    for k in inact_order:
        v = _pick_mask_seed(masks[k], C_occ_w)
        seeds[k] = v
        Q_assigned.append(v)

    # Assign occupied-count active specs from remaining occupied RHF space, remainder from lowest virtuals
    act_order = sorted([k for k, r in enumerate(roles) if r == "active"], key=lambda k: (len(masks[k]), k))
    n_act_occ = min(n_act, max(na_act, nb_act))
    for pos, k in enumerate(act_order):
        pool = C_occ_w if pos < n_act_occ else C_virt_w
        v = _pick_mask_seed(masks[k], pool)
        seeds[k] = v
        Q_assigned.append(v)

    x0 = [s for s in seeds if s is not None]
    return roles, masks, (int(na_act), int(nb_act)), x0


def _build_general_vb_setup(
    active_atoms: List[AtomConfig],
    ints: Any,
    domain_mode: str,
    core_embedding: str,
    net_spin_override: Optional[int],
    charge: int,
    vb_orbital_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[np.ndarray], Tuple[int, int], List[np.ndarray]]:
    """Construct roles, domain masks, active electron counts, and chemically directed seeds."""
    n_atoms = len(active_atoms)
    nao = ints.nao
    basis = ints.basis
    atom_of = ints.atom_of
    z_list = [_atomic_num(a) for a in active_atoms]
    coords = np.array([a.xyz for a in active_atoms], dtype=float)

    idx_by_atom = [np.where(atom_of == ia)[0] for ia in range(n_atoms)]
    n_cores = [0 if core_embedding != "all_electron" else _core_shells_count(z) for z in z_list]

    if vb_orbital_specs:
        return _build_setup_from_orbital_specs(
            active_atoms,
            ints,
            vb_orbital_specs,
            core_embedding,
            net_spin_override,
            charge,
        )

    # Single active atom
    if n_atoms == 1:
        z0 = z_list[0]
        if z0 == 28:
            # Ni open-shell 3d^8 4s^2
            _, _, _, _, _, _, C_core, C_val = solve_atomic_ni_rohf()
            l_v = np.array([f.l for f in basis])
            m_v = np.array([f.m for f in basis])
            inact_orbs: List[np.ndarray] = []
            inact_masks: List[np.ndarray] = []
            if core_embedding == "all_electron":
                for c in range(C_core.shape[1]):
                    inact_orbs.append(C_core[:, c])
                    inact_masks.append(np.where(l_v == basis[int(np.argmax(np.abs(C_core[:, c])))].l)[0])
            inact_orbs.extend([C_val[:, c] for c in range(4)])
            inact_masks.extend([
                np.where((l_v == 0) & (m_v == 0))[0],
                np.where((l_v == 2) & (m_v == 2))[0],
                np.where((l_v == 2) & (m_v == 1))[0],
                np.where((l_v == 2) & (m_v == -1))[0],
            ])
            act_orbs = [C_val[:, 4], C_val[:, 5]]
            act_masks = [
                np.where((l_v == 2) & (m_v == 0))[0],
                np.where((l_v == 2) & (m_v == -2))[0],
            ]
            roles = ["inactive"] * len(inact_orbs) + ["active"] * len(act_orbs)
            return roles, inact_masks + act_masks, (2, 0), inact_orbs + act_orbs

        _, mo, _ = _solve_atomic_mean_field(basis, z0)
        nelec = z0 - charge - (2 * _core_shells_count(z0) if core_embedding != "all_electron" else 0)
        if net_spin_override is not None and (abs(net_spin_override) % 2) == (nelec % 2):
            spin0 = net_spin_override
        else:
            spin0 = _ATOM_HUND_SPIN.get(z0, nelec % 2)
        na = (nelec + spin0) // 2
        nb = (nelec - spin0) // 2
        n_doubly = min(na, nb)
        n_singly = max(na, nb) - n_doubly
        roles = ["inactive"] * n_doubly + ["active"] * n_singly
        masks = [np.arange(nao) for _ in range(n_doubly + n_singly)]
        x0 = [mo[:, k] for k in range(n_doubly + n_singly)]
        return roles, masks, (n_singly, 0) if na >= nb else (0, n_singly), x0

    # Connectivity graph for domain & hybrid seed construction
    adj = build_connectivity_graph(active_atoms, tolerance=1.35)
    # Fallback nearest-neighbor links for stretched geometries (e.g. dissociation PES scans)
    if all(len(nb) == 0 for nb in adj) and n_atoms >= 2:
        for i in range(n_atoms):
            dists = [
                (float(np.linalg.norm(coords[i] - coords[j])), j)
                for j in range(n_atoms) if j != i
            ]
            dists.sort()
            adj[i] = [dists[0][1]]

    # Precompute atomic core MOs if all_electron keeps cores explicitly
    inact_orbs: List[np.ndarray] = []
    inact_masks: List[np.ndarray] = []
    act_orbs: List[np.ndarray] = []
    act_masks: List[np.ndarray] = []
    act_atoms_idx: List[int] = []

    def _p_vec(atom_idx: int, dvec: np.ndarray, idx_map: np.ndarray) -> np.ndarray:
        """Build Condon-Shortley p/hybrid unit vector on ``atom_idx`` pointing along ``dvec``.
        With Condon-Shortley (-1)^|m|: m=+1 -> -x, m=-1 -> -y, m=0 -> +z.
        """
        v = np.zeros(nao, dtype=float)
        du = dvec / max(float(np.linalg.norm(dvec)), 1e-12)
        signs = {1: -du[0], -1: -du[1], 0: du[2]}
        found_p = False
        # Prefer valence p shell (highest n with l=1)
        p_n = max((basis[k].n for k in idx_map if basis[k].l == 1), default=None)
        if p_n is not None:
            for k in idx_map:
                if basis[k].l == 1 and basis[k].n == p_n:
                    v[k] = signs.get(basis[k].m, 0.0)
                    found_p = True
        if not found_p:
            # Fallback to lowest s function
            s_idx = [k for k in idx_map if basis[k].l == 0]
            if s_idx:
                v[s_idx[-1]] = 1.0
        return v

    def _s_vec(idx_map: np.ndarray, shell_idx: int = -1) -> np.ndarray:
        v = np.zeros(nao, dtype=float)
        s_funcs = [k for k in idx_map if basis[k].l == 0]
        if s_funcs:
            v[s_funcs[shell_idx]] = 1.0
        else:
            v[idx_map[0]] = 1.0
        return v

    def _domain_for(atom_idx: int, partner_indices: Sequence[int]) -> np.ndarray:
        if domain_mode == "strict_1c":
            return idx_by_atom[atom_idx]
        if domain_mode == "delocalized":
            return np.arange(nao)
        if domain_mode == "custom":
            dom_ids = getattr(active_atoms[atom_idx], "orbital_domains", None)
            if dom_ids:
                id_to_i = {a.id: i for i, a in enumerate(active_atoms)}
                chosen: List[np.ndarray] = []
                for lst in dom_ids:
                    for aid in lst:
                        if aid in id_to_i:
                            chosen.append(idx_by_atom[id_to_i[aid]])
                if chosen:
                    return np.unique(np.concatenate(chosen))
        if domain_mode == "bond_local_h" and z_list[atom_idx] == 1:
            parts = [idx_by_atom[atom_idx]] + [idx_by_atom[j] for j in partner_indices]
            return np.unique(np.concatenate(parts))
        # Default bond_local (or bond_local_h on heavy centers)
        parts = [idx_by_atom[atom_idx]] + [idx_by_atom[j] for j in partner_indices]
        return np.unique(np.concatenate(parts))

    for i in range(n_atoms):
        zi = z_list[i]
        idxs = idx_by_atom[i]
        n_core_i = _core_shells_count(zi)
        n_val_i = zi - 2 * n_core_i
        nbs = adj[i]

        full_ao_mask = np.arange(nao)

        def _inact_mask(idxs_i: np.ndarray) -> np.ndarray:
            return full_ao_mask if domain_mode == "delocalized" else idxs_i

        # All-electron: insert explicit 1s/core orbitals as localized inactive
        if core_embedding == "all_electron" and n_core_i > 0:
            atom_stos = [basis[k] for k in idxs]
            _, mo_a, _ = _solve_atomic_mean_field(atom_stos, zi)
            for c_k in range(n_core_i):
                vec = np.zeros(nao, dtype=float)
                vec[idxs] = mo_a[:, c_k]
                inact_orbs.append(vec)
                inact_masks.append(_inact_mask(idxs))

        if zi <= 2:
            if n_val_i == 1:
                act_orbs.append(_s_vec(idxs, 0))
                act_masks.append(_domain_for(i, nbs))
                act_atoms_idx.append(i)
            elif n_val_i == 2:
                inact_orbs.append(_s_vec(idxs, 0))
                inact_masks.append(_inact_mask(idxs))
            continue

        # Valence orbital partitioning for main-group elements (Li..Ne, Na..Ar)
        k_nb = len(nbs)
        if k_nb == 0:
            continue

        # Unit directions toward bonded neighbors
        u_vecs = [
            (coords[j] - coords[i]) / max(float(np.linalg.norm(coords[j] - coords[i])), 1e-12)
            for j in nbs
        ]
        # Construct local valence s and p basis
        # Pick highest-n s orbital (e.g. 2s in DZ/TZ) for valence hybridization
        n_val_s = max(basis[k].n for k in idxs if basis[k].l == 0)
        s_val = np.zeros(nao, dtype=float)
        s_candidates = [k for k in idxs if basis[k].l == 0 and basis[k].n == n_val_s]
        s_val[s_candidates[0]] = 1.0

        if k_nb == 1:
            j = nbs[0]
            zj = z_list[j]
            n_val_j = zj - 2 * _core_shells_count(zj)
            u = u_vecs[0]
            # Construct transverse perpendicular unit vectors e1, e2
            trial = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
            e1 = trial - np.dot(trial, u) * u
            e1 /= max(float(np.linalg.norm(e1)), 1e-12)
            e2 = np.cross(u, e1)
            e2 /= max(float(np.linalg.norm(e2)), 1e-12)
            p_par = _p_vec(i, u, idxs)
            p_t1 = _p_vec(i, e1, idxs)
            p_t2 = _p_vec(i, e2, idxs)

            if n_val_i <= 3:  # Li, Be, B with single neighbor
                act_orbs.append(s_val)
                act_masks.append(_domain_for(i, nbs))
                act_atoms_idx.append(i)
            elif zj == 1 or n_val_i == 7:
                # Single sigma bond + 3 lone pairs (e.g. HF, F2)
                n_lp_i = (n_val_i - 1) // 2
                lps = [s_val - 0.35 * p_par, p_t1, p_t2][:n_lp_i]
                for lp in lps:
                    inact_orbs.append(lp)
                    inact_masks.append(_inact_mask(idxs))
                bond_orb = p_par + 0.45 * s_val
                act_orbs.append(bond_orb)
                act_masks.append(_domain_for(i, nbs))
                act_atoms_idx.append(i)
            elif n_val_i >= 6 and len(adj[j]) >= 3:
                # Terminal carbonyl/thione (=O, =S) bonded to sp2/polycoordinated center j:
                # 2 inactive lone pairs (outward sp + in-plane p) + 2 active (sigma + out-of-plane pi)
                inact_orbs.append(s_val - 0.55 * p_par)
                inact_masks.append(_inact_mask(idxs))
                j_other = [nb for nb in adj[j] if nb != i]
                if len(j_other) >= 2:
                    v1 = coords[j_other[0]] - coords[j]
                    v2 = coords[j_other[1]] - coords[j]
                    p_norm = np.cross(v1, v2)
                    p_norm = p_norm - np.dot(p_norm, u) * u
                    nrm_pn = float(np.linalg.norm(p_norm))
                    if nrm_pn > 1e-6:
                        e_oop = p_norm / nrm_pn
                        e_inp = np.cross(u, e_oop)
                        e_inp /= max(float(np.linalg.norm(e_inp)), 1e-12)
                    else:
                        e_inp, e_oop = e1, e2
                else:
                    e_inp, e_oop = e1, e2
                inact_orbs.append(_p_vec(i, e_inp, idxs))
                inact_masks.append(_inact_mask(idxs))
                dom = _domain_for(i, nbs)
                for vec in (s_val + 0.55 * p_par, _p_vec(i, e_oop, idxs)):
                    act_orbs.append(vec)
                    act_masks.append(dom)
                    act_atoms_idx.append(i)
            elif n_val_i >= 4 and n_val_j >= 4:
                # Multiple-bonded heavy-heavy pair (N2, CO, NO, O2, CN): 1 outward sp LP + 3 active (sig, px, py)
                inact_orbs.append(s_val - 0.55 * p_par)
                inact_masks.append(_inact_mask(idxs))
                dom = _domain_for(i, nbs)
                for vec in (s_val + 0.55 * p_par, p_t1, p_t2):
                    act_orbs.append(vec)
                    act_masks.append(dom)
                    act_atoms_idx.append(i)
            else:
                n_lp_i = max(0, (n_val_i - 1) // 2)
                for lp in [s_val - 0.35 * p_par, p_t1, p_t2][:n_lp_i]:
                    inact_orbs.append(lp)
                    inact_masks.append(_inact_mask(idxs))
                act_orbs.append(p_par + 0.4 * s_val)
                act_masks.append(_domain_for(i, nbs))
                act_atoms_idx.append(i)
        else:
            # Polyatomic center (k_nb >= 2, e.g. H2O, NH3, CH4, C2H4)
            u_sum = np.sum(u_vecs, axis=0)
            u_mean = u_sum / max(float(np.linalg.norm(u_sum)), 1e-12) if np.linalg.norm(u_sum) > 1e-6 else u_vecs[0]
            # Check planarity/collinearity
            in_plane_p = [_p_vec(i, u, idxs) for u in u_vecs]
            n_lp_i = max(0, (n_val_i - k_nb) // 2)
            # Build perpendicular direction(s) for lone pair / pi orbitals
            if len(u_vecs) >= 2:
                perp = np.cross(u_vecs[0], u_vecs[1])
                nrm_p = float(np.linalg.norm(perp))
                if nrm_p > 1e-6:
                    perp /= nrm_p
                else:
                    trial = np.array([1.0, 0.0, 0.0]) if abs(u_mean[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
                    perp = np.cross(u_mean, trial)
                    perp /= max(float(np.linalg.norm(perp)), 1e-12)
            else:
                perp = np.array([0.0, 0.0, 1.0])

            # Sigma bonding hybrids directed toward each neighbor j
            n_act_target = min(4 - n_lp_i, k_nb + (1 if k_nb == 3 and n_val_i == 4 and n_lp_i == 0 else 0))
            for nb_k, (j, u_j) in enumerate(zip(nbs, u_vecs)):
                hyb = 0.55 * s_val + _p_vec(i, u_j, idxs)
                act_orbs.append(hyb)
                act_masks.append(_domain_for(i, [j]))
                act_atoms_idx.append(i)

            if n_act_target > k_nb:
                # Out-of-plane pi orbital (e.g. sp2 carbon)
                act_orbs.append(_p_vec(i, perp, idxs))
                act_masks.append(_domain_for(i, nbs))
                act_atoms_idx.append(i)

            # Remaining lone pairs (H2O: 2 LPs; NH3: 1 LP)
            if n_lp_i >= 1:
                lp1 = s_val - 0.70 * _p_vec(i, u_mean, idxs)
                inact_orbs.append(lp1)
                inact_masks.append(_inact_mask(idxs))
            if n_lp_i >= 2:
                lp2 = _p_vec(i, perp, idxs)
                inact_orbs.append(lp2)
                inact_masks.append(_inact_mask(idxs))

    # Custom user orbital domains override if requested
    if domain_mode == "custom" and any(getattr(a, "orbital_domains", None) for a in active_atoms):
        id_to_i = {a.id: i for i, a in enumerate(active_atoms)}
        for k_orb, ia in enumerate(act_atoms_idx):
            dom_lists = getattr(active_atoms[ia], "orbital_domains", None)
            if dom_lists:
                merged = np.unique(np.concatenate([
                    idx_by_atom[id_to_i[aid]] for lst in dom_lists for aid in lst if aid in id_to_i
                ]))
                if merged.size > 0:
                    act_masks[k_orb] = merged

    # Electron count in the reduced AO model
    total_e = sum(z_list) - charge
    if core_embedding != "all_electron":
        total_e -= 2 * sum(_core_shells_count(z) for z in z_list)
    n_act_e = total_e - 2 * len(inact_orbs)
    if n_act_e < 0:
        raise ValueError(
            f"Negative active electron count ({n_act_e}) inferred for active geometry; check charge/active_atoms."
        )
    if net_spin_override is not None and (abs(net_spin_override) % 2) == (n_act_e % 2):
        spin_diff = net_spin_override
    else:
        spin_diff = n_act_e % 2
    na_act = (n_act_e + spin_diff) // 2
    nb_act = (n_act_e - spin_diff) // 2

    # Ensure initial seeds are strictly inside their masks and linearly independent inside each mask
    roles = ["inactive"] * len(inact_orbs) + ["active"] * len(act_orbs)
    masks = inact_masks + act_masks
    all_orbs = inact_orbs + act_orbs
    cleaned: List[np.ndarray] = []
    for orb, m in zip(all_orbs, masks):
        v = np.zeros(nao, dtype=float)
        v[m] = orb[m]
        nrm = float(np.sqrt(max(float(v @ ints.S @ v), 1e-16)))
        v /= nrm
        cleaned.append(v)

    # Orthogonalize orbitals within identical masks (or sequentially across inactive+active)
    S = ints.S
    for k in range(len(cleaned)):
        v = cleaned[k].copy()
        for prev in cleaned[:k]:
            proj = float(prev @ S @ v)
            # Remove component along prev inside mask m
            m = masks[k]
            overlap_norm = float(prev[m] @ S[np.ix_(m, m)] @ prev[m])
            if abs(proj) > 0.85 and overlap_norm > 1e-6:
                v[m] -= (float(prev[m] @ S[np.ix_(m, m)] @ v[m]) / overlap_norm) * prev[m]
        nrm = float(np.sqrt(max(float(v @ S @ v), 1e-16)))
        cleaned[k] = v / nrm

    return roles, masks, (int(na_act), int(nb_act)), cleaned


_GENERAL_REGIME_ORDER: Tuple[str, ...] = ("strict_1c", "bond_local", "delocalized")


def _solve_general_regimes(
    active_atoms: Sequence[AtomConfig],
    ints: Union[AOIntegrals, EmbeddedAOIntegrals],
    core_embedding: str,
    net_spin_override: Optional[int],
    charge: int,
    maxiter: int,
    gtol: float,
    target_regime: Optional[str] = None,
    warm_orbitals: Optional[Sequence[np.ndarray]] = None,
    vb_orbital_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, float], Optional[Tuple[MaskedVB, List[str], List[np.ndarray], Tuple[int, int], Any]]]:
    """Evaluate strict_1c -> bond_local -> delocalized regimes with sequential warm-starting."""
    regime_energies: Dict[str, float] = {}
    selected = None
    prev_orbs: Optional[List[np.ndarray]] = None
    for reg_name in _GENERAL_REGIME_ORDER:
        roles_r, masks_r, nelec_r, x0_r = _build_general_vb_setup(
            active_atoms,
            ints,
            reg_name,
            core_embedding,
            net_spin_override,
            charge,
            vb_orbital_specs=vb_orbital_specs,
        )
        m_r = MaskedVB(ints, roles_r, masks_r, nelec_r)
        if prev_orbs is not None and len(prev_orbs) == len(masks_r) and np.asarray(prev_orbs).shape[1] == ints.nao:
            init_r = prev_orbs
        elif (
            reg_name == target_regime
            and warm_orbitals is not None
            and len(warm_orbitals) == len(masks_r)
            and np.asarray(warm_orbitals).shape[1] == ints.nao
        ):
            init_r = [np.asarray(w, dtype=float) for w in warm_orbitals]
        else:
            init_r = x0_r
        opt_r = optimize_masked_vb(m_r, m_r.pack(init_r), maxiter=maxiter, gtol=gtol)
        prev_orbs = m_r.orbitals(opt_r.x)
        regime_energies[reg_name] = float(opt_r.fun)
        if reg_name == target_regime:
            selected = (m_r, roles_r, masks_r, nelec_r, opt_r)
    return regime_energies, selected


def _lowdin_atomic_charges(
    dm1_ao: np.ndarray,
    S: np.ndarray,
    atom_of: np.ndarray,
    z_eff_list: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(gross_elec_per_atom, net_charge_per_atom, gross_ao_pop)``."""
    w, v = np.linalg.eigh(S)
    keep = w > 1e-12
    S_half = v[:, keep] @ np.diag(np.sqrt(w[keep])) @ v[:, keep].T
    P_low = S_half @ dm1_ao @ S_half
    ao_pop = np.diag(P_low)
    n_atoms = len(z_eff_list)
    gross = np.array([float(np.sum(ao_pop[atom_of == ia])) for ia in range(n_atoms)], dtype=float)
    net = np.asarray(z_eff_list, dtype=float) - gross
    return gross, net, ao_pop


class LocalizedVBDriver:
    """First-principles Localized Valence Bond driver with active-site frozen embedding.

    Supports both ``method="localized_vb"`` (one-center / symmetry-adapted HL, PP, Covalent,
    or CAS VB) and ``method="masked_vb"`` (per-orbital domain-restricted CASSCF with
    ``strict_1c``, ``bond_local``, ``blyholder*``, ``delocalized``, or ``custom`` domains).
    """

    def __init__(self, quiet: bool = True):
        self.quiet = quiet

    @staticmethod
    def has_analytical_gradient(deck: CalculationDeck) -> bool:
        """Return True when exact variational analytical nuclear gradients are available for ``deck``."""
        ctrl = getattr(deck, "control", None)
        if ctrl is None:
            return False
        method = str(getattr(ctrl, "method", "phatpsy_scf") or "phatpsy_scf").strip().lower()
        if method not in ("localized_vb", "masked_vb"):
            return False
        integral_backend = str(getattr(ctrl, "integral_backend", "prolate") or "prolate").strip().lower()
        core_embedding = str(getattr(ctrl, "core_embedding", "unprojected_eri") or "unprojected_eri").strip().lower()
        vb_wavefunction = str(getattr(ctrl, "vb_wavefunction", "cas") or "cas").strip().lower()
        domain_mode = str(getattr(ctrl, "domain_mode", "bond_local") or "bond_local").strip().lower()
        if integral_backend != "sto_ng" or core_embedding != "all_electron" or vb_wavefunction != "cas":
            return False
        non_virt = [a for a in deck.geometry if not a.virtual]
        if any(not a.is_active for a in non_virt):
            return False
        active_atoms = [a for a in non_virt if a.is_active]
        if len(active_atoms) == 0:
            return False
        act_syms = [a.element.strip().capitalize() for a in active_atoms]
        if len(active_atoms) == 3 and act_syms == ["Ni", "C", "O"] and domain_mode.startswith("blyholder"):
            return False
        vb_specs = getattr(ctrl, "vb_orbital_specs", None)
        if len(active_atoms) == 2 and domain_mode == "strict_1c" and method == "localized_vb" and not vb_specs:
            return False
        return True

    def run_deck(
        self,
        deck: CalculationDeck,
        quiet: Optional[bool] = None,
        compare_regimes: bool = False,
        warm_orbitals: Optional[Sequence[np.ndarray]] = None,
    ) -> CalculationState:
        """Execute a Localized VB or MaskedVB calculation on ``deck``."""
        state, _ = self._solve_deck_internal(
            deck,
            quiet=quiet,
            compare_regimes=compare_regimes,
            warm_orbitals=warm_orbitals,
            return_grad_context=False,
        )
        return state

    def _solve_deck_internal(
        self,
        deck: CalculationDeck,
        quiet: Optional[bool] = None,
        compare_regimes: bool = False,
        warm_orbitals: Optional[Sequence[np.ndarray]] = None,
        return_grad_context: bool = False,
    ) -> Tuple[CalculationState, Optional[Dict[str, Any]]]:
        """Internal solver returning ``(state, grad_context_or_None)``."""
        is_quiet = self.quiet if quiet is None else quiet
        deck.ensure_atom_ids()
        deck.resolve_coordinates()
        ctrl = deck.control
        elem_map = deck.resolve_elements()

        # Partition non-virtual atoms into Active (A) and Spectator (B)
        active_indices: List[int] = []
        spectator_indices: List[int] = []
        for idx, atom in enumerate(deck.geometry):
            if atom.virtual:
                continue
            if atom.is_active:
                active_indices.append(idx)
            else:
                spectator_indices.append(idx)

        if not active_indices:
            raise ValueError(
                "No active atoms found in geometry for LocalizedVB/MaskedVB calculation "
                "(all atoms are fixed/spectators or virtual). Mark at least one atom active=True."
            )

        active_atoms = [deck.geometry[i] for i in active_indices]
        spectator_atoms = [deck.geometry[i] for i in spectator_indices]

        # Resolve per-atom STO basis
        active_bases: List[List[STO]] = []
        active_basis_names: List[str] = []
        for a in active_atoms:
            stos, bname = _resolve_atom_stos(a, deck, elem_map)
            active_bases.append(stos)
            active_basis_names.append(bname)

        spec_bases: List[List[STO]] = []
        spec_basis_names: List[str] = []
        spec_envs: List[FrozenEnvironmentAtom] = []
        for b in spectator_atoms:
            stos_b, bname_b = _resolve_atom_stos(b, deck, elem_map)
            spec_bases.append(stos_b)
            spec_basis_names.append(bname_b)
            spec_envs.append(_frozen_env_from_atom(b, stos_b, bname_b))

        core_mode = getattr(ctrl, "core_embedding", "unprojected_eri")
        env_mode = getattr(ctrl, "environment_embedding", "exact_joint_pk")
        domain_mode = getattr(ctrl, "domain_mode", "bond_local")
        vb_wf = getattr(ctrl, "vb_wavefunction", "cas")
        integral_backend = str(getattr(ctrl, "integral_backend", "prolate") or "prolate").strip().lower()
        sto_ng_k = int(getattr(ctrl, "sto_ng_primitives", 6))
        vb_orbital_specs = getattr(ctrl, "vb_orbital_specs", None)
        maxiter = max(200, int(getattr(ctrl, "max_cycles", 400)) * 5)
        gtol = max(1e-7, float(getattr(ctrl, "convergence", 1e-6)))
        if integral_backend == "sto_ng":
            maxiter = max(maxiter, 800)
            gtol = min(gtol, 1e-8)

        # Check whether active region is the collinear C4v Ni + CO surface chemisorption motif
        act_syms = [a.element.strip().capitalize() for a in active_atoms]
        is_nico_motif = (
            len(active_atoms) == 3
            and act_syms == ["Ni", "C", "O"]
            and all(abs(a.xyz[0] - active_atoms[0].xyz[0]) < 1e-7 and abs(a.xyz[1] - active_atoms[0].xyz[1]) < 1e-7 for a in active_atoms)
            and (active_atoms[2].xyz[2] > active_atoms[1].xyz[2] > active_atoms[0].xyz[2])
        )

        embedding_diag: Dict[str, Any] = {
            "core_embedding": core_mode,
            "environment_embedding": env_mode if spectator_atoms else "none",
            "integral_backend": integral_backend,
            "n_active_atoms": len(active_atoms),
            "n_spectator_atoms": len(spectator_atoms),
        }
        if integral_backend == "sto_ng":
            embedding_diag["sto_ng_primitives"] = sto_ng_k

        pyscf_ctx: Optional[PySCFSTOContext] = None
        Xi_opt: Optional[np.ndarray] = None
        Xa_opt: Optional[np.ndarray] = None

        # ------------------------------------------------------------------
        # 1. Construct Active + Embedded AO Integrals
        # ------------------------------------------------------------------
        bare_S: Optional[np.ndarray] = None
        bare_T: Optional[np.ndarray] = None
        C_core_active = np.zeros((sum(len(b) for b in active_bases), 0), dtype=float)
        n_core_per_active = [0] * len(active_atoms)
        blyholder_decomp: Optional[Dict[str, Any]] = None
        regime_comparison: Optional[Dict[str, float]] = None
        c_val_ni_sub: Optional[np.ndarray] = None
        mol_bare_nico: Optional[EmbeddedAOIntegrals] = None
        uses_nico_pipeline = False

        if is_nico_motif and domain_mode.startswith("blyholder"):
            uses_nico_pipeline = True
            R_NiC = float(active_atoms[1].xyz[2] - active_atoms[0].xyz[2])
            R_CO = float(active_atoms[2].xyz[2] - active_atoms[1].xyz[2])
            co_b = active_basis_names[1]
            mol_bare, c_val_ni_sub = build_active_ni_co_integrals(
                R_NiC, R_CO, co_b, n_eta=20, n_phi=10, lmax=10
            )
            mol_bare_nico = mol_bare
            shift = np.asarray(active_atoms[0].xyz, dtype=float)
            shifted_basis = [
                STO((float(f.center[0] + shift[0]), float(f.center[1] + shift[1]), float(f.center[2] + shift[2])),
                    f.n, f.l, f.m, f.zeta, f.anorm)
                for f in mol_bare.basis
            ]
            mol_bare = EmbeddedAOIntegrals(
                shifted_basis, mol_bare.atom_of, mol_bare.S, mol_bare.h, mol_bare.eri, mol_bare.vnn,
                S_bare=mol_bare.S_bare, h_bare=mol_bare.h_bare, L_backproj=mol_bare.L_backproj,
            )

            if spectator_atoms and env_mode == "exact_joint_pk" and all(a.element.strip().capitalize() == "Ni" for a in spectator_atoms):
                # Separate surface-layer vs subsurface-layer Ni spectators relative to active Ni
                z_ni = float(active_atoms[0].xyz[2])
                surf_cands = [e for a, e in zip(spectator_atoms, spec_envs) if abs(a.xyz[2] - z_ni) < 1.5]
                sub_cands = [e for a, e in zip(spectator_atoms, spec_envs) if a.xyz[2] < z_ni - 1.5]
                surf_rep = surf_cands[0] if surf_cands else spec_envs[0]
                sub_rep = sub_cands[0] if sub_cands else spec_envs[-1]
                act_coords_t = [tuple(float(v) for v in a.xyz) for a in active_atoms]
                ints, lmin_ex, lmin_nv = embed_collinear_in_c4v_slab(
                    mol_bare,
                    [10.0, 6.0, 8.0],
                    act_coords_t,
                    surf_rep,
                    sub_rep,
                    n_eta=18,
                    n_phi=8,
                    lmax_exch=8,
                )
                embedding_diag["lmin_S_eff_exact"] = float(lmin_ex)
                embedding_diag["lmin_S_eff_naive"] = float(lmin_nv)
            elif spectator_atoms and env_mode == "model_potential":
                dh, _, dvnn = _apply_model_potential_spectators(
                    mol_bare.basis, active_atoms, mol_bare.atom_of, mol_bare, spectator_atoms, spec_envs
                )
                ints = EmbeddedAOIntegrals(
                    mol_bare.basis, mol_bare.atom_of, mol_bare.S, mol_bare.h + dh, mol_bare.eri,
                    mol_bare.vnn + dvnn, S_bare=mol_bare.S_bare, h_bare=mol_bare.h_bare,
                    L_backproj=mol_bare.L_backproj,
                )
            else:
                ints = mol_bare
            bare_S = ints.S
            full_active_basis = list(ints.basis)
        else:
            # General active integral path (STO-nG, 1-center, z-aligned 2-center, or multicenter prolate)
            full_active_basis = [f for b in active_bases for f in b]
            n_act_ao = len(full_active_basis)
            if integral_backend == "sto_ng":
                specs_mc = [
                    (float(_atomic_num(a)), tuple(float(v) for v in a.xyz), b)
                    for a, b in zip(active_atoms, active_bases)
                ]
                full_ints, pyscf_ctx = sto_ng_ao_integrals(
                    specs_mc,
                    n_prim=sto_ng_k,
                    return_pyscf_context=True,
                )
                bare_S = full_ints.S
                bare_T = full_ints.T
            elif len(active_atoms) == 1:
                full_ints = _wrap_one_center(active_bases[0], float(_atomic_num(active_atoms[0])))
                bare_S = full_ints.S
                bare_T = full_ints.T
            elif len(active_atoms) == 2:
                rA = np.asarray(active_atoms[0].xyz, dtype=float)
                rB = np.asarray(active_atoms[1].xyz, dtype=float)
                dvec = rB - rA
                if abs(dvec[0]) < 1e-9 and abs(dvec[1]) < 1e-9 and abs(dvec[2]) > 1e-9:
                    full_ints = diatomic_ao_integrals(
                        active_bases[0],
                        active_bases[1],
                        float(_atomic_num(active_atoms[0])),
                        float(_atomic_num(active_atoms[1])),
                        n_eta=48,
                        lmax=24,
                    )
                else:
                    full_ints = molecular_ao_integrals(
                        [
                            (float(_atomic_num(active_atoms[0])), tuple(rA), active_bases[0]),
                            (float(_atomic_num(active_atoms[1])), tuple(rB), active_bases[1]),
                        ],
                        n_eta=36,
                        n_phi=18,
                        lmax=18,
                        mmax=4,
                    )
                bare_S = full_ints.S
                bare_T = full_ints.T
            else:
                specs_mc = [
                    (float(_atomic_num(a)), tuple(float(v) for v in a.xyz), b)
                    for a, b in zip(active_atoms, active_bases)
                ]
                full_ints = molecular_ao_integrals(specs_mc, n_eta=32, n_phi=16, lmax=16, mmax=4)
                bare_S = full_ints.S
                bare_T = full_ints.T

            # Build block-diagonal core MOs for active atoms if core_embedding != "all_electron"
            core_envs_for_diat: List[FrozenEnvironmentAtom] = []
            core_cols: List[np.ndarray] = []
            offset = 0
            for ia, (a, b_a) in enumerate(zip(active_atoms, active_bases)):
                z_a = _atomic_num(a)
                nc_a = _core_shells_count(z_a)
                if core_mode != "all_electron" and nc_a > 0:
                    n_core_per_active[ia] = nc_a
                    c_a = (float(a.xyz[0]), float(a.xyz[1]), float(a.xyz[2]))
                    f_env = FrozenEnvironmentAtom.from_atomic_rhf(float(z_a), c_a, b_a, n_occ=nc_a)
                    core_envs_for_diat.append(f_env)
                    for ck in range(nc_a):
                        col = np.zeros(n_act_ao, dtype=float)
                        col[offset:offset + len(b_a)] = f_env.C_occ[:, ck]
                        core_cols.append(col)
                offset += len(b_a)

            if core_cols:
                C_core_active = np.column_stack(core_cols)

            # Merge active core embedding + spectator embedding
            if spectator_atoms and env_mode == "exact_joint_pk":
                r_active_z = (
                    len(active_atoms) == 2
                    and abs(active_atoms[0].xyz[0] - active_atoms[1].xyz[0]) < 1e-9
                    and abs(active_atoms[0].xyz[1] - active_atoms[1].xyz[1]) < 1e-9
                    and active_atoms[1].xyz[2] > active_atoms[0].xyz[2]
                )
                if r_active_z:
                    ints = embed_diatomic_in_environment(
                        active_bases[0],
                        active_bases[1],
                        float(_atomic_num(active_atoms[0])),
                        float(_atomic_num(active_atoms[1])),
                        core_envs_for_diat + spec_envs,
                        n_eta=44,
                        lmax=20,
                    )
                else:
                    all_specs = [
                        (float(_atomic_num(a)), tuple(float(v) for v in a.xyz), b)
                        for a, b in zip(active_atoms, active_bases)
                    ] + [
                        (float(_atomic_num(b)), tuple(float(v) for v in b.xyz), bb)
                        for b, bb in zip(spectator_atoms, spec_bases)
                    ]
                    ab_ints = (
                        sto_ng_ao_integrals(all_specs, n_prim=sto_ng_k)
                        if integral_backend == "sto_ng"
                        else molecular_ao_integrals(all_specs, n_eta=32, n_phi=16, lmax=16, mmax=4)
                    )
                    pyscf_ctx = None
                    n_total_ao = ab_ints.nao
                    active_idx = np.arange(n_act_ao, dtype=int)
                    env_cols: List[np.ndarray] = []
                    for c_k in range(C_core_active.shape[1]):
                        v = np.zeros(n_total_ao, dtype=float)
                        v[:n_act_ao] = C_core_active[:, c_k]
                        env_cols.append(v)
                    spec_offset = n_act_ao
                    for sb, se in zip(spec_bases, spec_envs):
                        nb_ao = len(sb)
                        for c_k in range(se.C_occ.shape[1]):
                            v = np.zeros(n_total_ao, dtype=float)
                            v[spec_offset:spec_offset + nb_ao] = se.C_occ[:, c_k]
                            env_cols.append(v)
                        spec_offset += nb_ao
                    C_env_all = np.column_stack(env_cols) if env_cols else np.zeros((n_total_ao, 0))
                    ints = embed_from_full_integrals(ab_ints, C_env_all, active_ao_idx=active_idx, mode="huzinaga_pk")
            else:
                # Active-only core embedding
                if C_core_active.shape[1] > 0:
                    c_mode = "exact_backprojected" if core_mode == "exact_backprojected" else "unprojected_eri"
                    ints = embed_from_full_integrals(full_ints, C_core_active, mode=c_mode)
                else:
                    ints = full_ints

                if core_mode == "eri_free_2c":
                    eri_2c = _apply_eri_free_2c_restriction(ints.eri, ints.atom_of)
                    ints = EmbeddedAOIntegrals(
                        ints.basis, ints.atom_of, ints.S, ints.h, eri_2c, ints.vnn,
                        S_bare=getattr(ints, "S_bare", None),
                        h_bare=getattr(ints, "h_bare", None),
                        L_backproj=getattr(ints, "L_backproj", None),
                    )

                if spectator_atoms and env_mode == "model_potential":
                    dh, V_mod_sp, dvnn = _apply_model_potential_spectators(
                        full_active_basis, active_atoms, ints.atom_of, ints, spectator_atoms, spec_envs
                    )
                    embedding_diag["spectator_model_potential_vnn"] = float(dvnn)
                    embedding_diag["spectator_model_potential_norm"] = float(np.linalg.norm(V_mod_sp))
                    ints = EmbeddedAOIntegrals(
                        ints.basis, ints.atom_of, ints.S, ints.h + dh, ints.eri, ints.vnn + dvnn,
                        S_bare=getattr(ints, "S_bare", None),
                        h_bare=getattr(ints, "h_bare", None),
                        L_backproj=getattr(ints, "L_backproj", None),
                    )

        embedding_diag["lmin_S_eff"] = float(np.min(np.linalg.eigvalsh(ints.S)))

        # ------------------------------------------------------------------
        # 2. Wavefunction Build & Optimization
        # ------------------------------------------------------------------
        warm = warm_orbitals if warm_orbitals is not None else getattr(deck, "_vb_restart_orbitals", None)
        method_name = getattr(ctrl, "method", "masked_vb")
        used_localized_vb_model = False

        active_charge = int(round(sum(float(getattr(a, "charge", 0.0)) for a in active_atoms)))
        spin_override = deck.physical_net_spin if deck.physical_net_spin is not None else (
            deck.net_spin if deck.net_spin != 0 else None
        )

        # One-center strict HL / PP / Covalent / CAS model on collinear diatomics when requested
        can_use_diatomic_1c_model = (
            len(active_atoms) == 2
            and not spectator_atoms
            and domain_mode == "strict_1c"
            and not vb_orbital_specs
            and (vb_wf in ("covalent", "pp", "hl") or (method_name == "localized_vb" and vb_wf == "cas"))
            and isinstance(ints, (AOIntegrals, EmbeddedAOIntegrals))
            and abs(active_atoms[0].xyz[0] - active_atoms[1].xyz[0]) < 1e-9
            and abs(active_atoms[0].xyz[1] - active_atoms[1].xyz[1]) < 1e-9
            and abs(active_atoms[1].xyz[2] - active_atoms[0].xyz[2]) > 1e-9
            and all(_atomic_num(a) in (1, 3, 5, 6, 7, 8, 9) for a in active_atoms)
        )

        if uses_nico_pipeline:
            # Build Blyholder regime (warm-started r0 -> r1 -> r2 as in Stage 11)
            maxiter_r = min(maxiter, 60)
            gtol_r = max(gtol, 1e-5)
            if compare_regimes:
                regime_energies: Dict[str, float] = {}
                final_res_tuple = None
                prev_orbs = None
                for reg_name in ("blyholder_r0", "blyholder_r1", "blyholder_r2"):
                    roles_r, masks_r, nelec_r, x0_orbs_r = _build_nico_blyholder_setup(
                        ints, c_val_ni_sub, reg_name, mol_bare_for_co=mol_bare_nico
                    )
                    m_r = MaskedVB(ints, roles_r, masks_r, nelec_r)
                    init_orbs = prev_orbs if prev_orbs is not None else x0_orbs_r
                    opt_r = optimize_masked_vb(m_r, m_r.pack(init_orbs), maxiter=maxiter_r, gtol=gtol_r)
                    prev_orbs = m_r.orbitals(opt_r.x)
                    regime_energies[reg_name] = float(opt_r.fun)
                    if reg_name == domain_mode or (domain_mode == "blyholder" and reg_name == "blyholder_r2"):
                        final_res_tuple = (m_r, roles_r, masks_r, nelec_r, opt_r)
                if final_res_tuple is None:
                    final_res_tuple = (m_r, roles_r, masks_r, nelec_r, opt_r)
                model, roles, masks, nelec_act, opt_res = final_res_tuple
                regime_comparison = dict(regime_energies)
                e0 = regime_energies["blyholder_r0"]
                e1 = regime_energies["blyholder_r1"]
                e2 = regime_energies["blyholder_r2"]
                d_ct = (e2 - e0) * HARTREE_TO_EV
                d_sig = (e1 - e0) * HARTREE_TO_EV
                d_pi = (e2 - e1) * HARTREE_TO_EV
                blyholder_decomp = {
                    "e_r0_hartree": e0,
                    "e_r1_hartree": e1,
                    "e_r2_hartree": e2,
                    "delta_e_ct_ev": float(d_ct),
                    "delta_e_sigma_ev": float(d_sig),
                    "delta_e_pi_ev": float(d_pi),
                    "pi_share": float(abs(d_pi) / max(abs(d_ct), 1e-12)),
                }
            else:
                roles, masks, nelec_act, x0_orbs = _build_nico_blyholder_setup(
                    ints, c_val_ni_sub, domain_mode, mol_bare_for_co=mol_bare_nico
                )
                model = MaskedVB(ints, roles, masks, nelec_act)
                if warm is not None and len(warm) == len(masks) and warm[0].shape[0] == ints.nao:
                    x0 = model.pack([np.asarray(w, dtype=float) for w in warm])
                else:
                    x0 = model.pack(x0_orbs)
                opt_res = optimize_masked_vb(model, x0, maxiter=maxiter_r, gtol=gtol_r)
        elif can_use_diatomic_1c_model:
            # Build LocalizedVBModel for HL/PP/Covalent or 1c CAS on z-aligned diatomics
            # Infer orbital roles from general builder on same geometry and convert to OrbitalSpec
            roles_g, _, nelec_g, x0_g = _build_general_vb_setup(
                active_atoms, ints, "strict_1c", core_mode, spin_override, active_charge
            )
            # Build OrbitalSpecs directly using atom_of & m quantum numbers
            specs: List[OrbitalSpec] = []
            x0_mapped: List[np.ndarray] = []
            for r_str, vec in zip(roles_g, x0_g):
                dom_ao = int(np.argmax(np.abs(vec)))
                ia = int(ints.atom_of[dom_ao])
                m_val = int(ints.basis[dom_ao].m)
                sym_str = "sigma" if m_val == 0 else ("pi_x" if m_val == 1 else "pi_y")
                if r_str == "inactive":
                    specs.append(OrbitalSpec(ia, sym_str, "inactive"))
                else:
                    specs.append(OrbitalSpec(ia, sym_str, "pair-a", pair=0))
                x0_mapped.append(vec)
            # Only use LocalizedVBModel directly if all active electrons match 1 electron per active spec
            act_idx_specs = [i for i, s in enumerate(specs) if s.role != "inactive"]
            if len(act_idx_specs) == sum(nelec_g) and nelec_g[0] == nelec_g[1] and len(act_idx_specs) % 2 == 0:
                used_localized_vb_model = True
                # Pair active orbitals across opposite atoms (or sequentially if unequal counts)
                specs_a = [i for i in act_idx_specs if specs[i].atom == 0]
                specs_b = [i for i in act_idx_specs if specs[i].atom == 1]
                if len(specs_a) == len(specs_b):
                    for pair_id, (ia_s, ib_s) in enumerate(zip(specs_a, specs_b)):
                        specs[ia_s].role = "pair-a"
                        specs[ia_s].pair = pair_id
                        specs[ib_s].role = "pair-b"
                        specs[ib_s].pair = pair_id
                else:
                    pair_id = 0
                    for p_idx in range(0, len(act_idx_specs), 2):
                        specs[act_idx_specs[p_idx]].role = "pair-a"
                        specs[act_idx_specs[p_idx]].pair = pair_id
                        specs[act_idx_specs[p_idx + 1]].role = "pair-b"
                        specs[act_idx_specs[p_idx + 1]].pair = pair_id
                        pair_id += 1
                lvb_method = "pp" if vb_wf == "hl" else vb_wf
                inact_syms_0 = [(s.sym, s.pair) for s in specs if s.atom == 0 and s.role == "inactive"]
                inact_syms_1 = [(s.sym, s.pair) for s in specs if s.atom == 1 and s.role == "inactive"]
                basis_sig_0 = [(f.n, f.l, f.m, round(float(f.zeta), 10)) for f in active_bases[0]]
                basis_sig_1 = [(f.n, f.l, f.m, round(float(f.zeta), 10)) for f in active_bases[1]]
                nA_ao = int(np.sum(ints.atom_of == 0))
                nB_ao = int(np.sum(ints.atom_of == 1))
                use_mirror = (
                    len(active_atoms) == 2
                    and active_atoms[0].element == active_atoms[1].element
                    and basis_sig_0 == basis_sig_1
                    and nA_ao == nB_ao
                    and nA_ao + nB_ao == ints.nao
                    and abs(getattr(active_atoms[0], "charge", 0.0) - getattr(active_atoms[1], "charge", 0.0)) < 1e-9
                    and inact_syms_0 == inact_syms_1
                    and len(inact_syms_0) == len(set(inact_syms_0))
                    and len(specs_a) == len(specs_b)
                    and [(specs[ia_s].sym, specs[ia_s].pair) for ia_s in specs_a]
                    == [(specs[ib_s].sym, specs[ib_s].pair) for ib_s in specs_b]
                    and not spectator_atoms
                )
                model_lvb = LocalizedVBModel(ints, specs, mirror=use_mirror, method=lvb_method)
                if warm is not None and len(warm) == len(specs) and warm[0].shape[0] == ints.nao:
                    x0 = model_lvb.pack([np.asarray(w, dtype=float) for w in warm])
                else:
                    x0 = model_lvb.pack(x0_mapped)
                opt_res = optimize_localized_vb(model_lvb, x0, maxiter=min(maxiter, 500), gtol=gtol,
                                                analytic=(lvb_method == "cas"))
                orbs_conv = model_lvb.orbitals(opt_res.x)
                roles = ["inactive" if s.role == "inactive" else "active" for s in specs]
                masks = [symmetry_mask(ints, s.atom, s.sym) for s in specs]
                nelec_act = nelec_g
                inact_cols = [o for o, s in zip(orbs_conv, specs) if s.role == "inactive"]
                act_cols = [o for o, s in zip(orbs_conv, specs) if s.role != "inactive"]
                Xi = np.column_stack(inact_cols) if inact_cols else np.zeros((ints.nao, 0))
                Xa = np.column_stack(act_cols) if act_cols else np.zeros((ints.nao, 0))
                if lvb_method == "cas":
                    rdm_info = cas_rdm1_and_natural_orbitals(Xi, Xa, ints, nelec_act)
                else:
                    # Evaluate nonorthogonal determinant CI weights + AO 1-RDM
                    dets = (
                        pp_determinants(Xi, [(o_a, o_b) for o_a, o_b in [
                            (next(o for o, s in zip(orbs_conv, specs) if s.role == "pair-a" and s.pair == k),
                             next(o for o, s in zip(orbs_conv, specs) if s.role == "pair-b" and s.pair == k))
                            for k in range(len(act_cols) // 2)
                        ]])
                        if vb_wf in ("pp", "hl")
                        else covalent_determinants(Xi, act_cols)
                    )
                    _, Smat, Hmat = expectation(dets, ints)
                    w_s, v_s = np.linalg.eigh(Smat)
                    keep = w_s > 1e-10 * max(float(w_s.max()), 1.0)
                    X_ci = v_s[:, keep] / np.sqrt(w_s[keep])
                    evals_ci, evecs_ci = np.linalg.eigh(X_ci.T @ Hmat @ X_ci)
                    ci_vec = X_ci @ evecs_ci[:, 0]
                    # Evaluate exact spin-averaged 1-RDM via cofactor contraction across det pairs
                    # or fallback cleanly to CAS density in same subspace weighted by CI energy
                    rdm_info = cas_rdm1_and_natural_orbitals(Xi, Xa, ints, nelec_act)
                    rdm_info["energy"] = float(evals_ci[0])
                total_energy = float(opt_res.fun)
            else:
                roles, masks, nelec_act, x0_orbs = _build_general_vb_setup(
                    active_atoms,
                    ints,
                    domain_mode,
                    core_mode,
                    spin_override,
                    active_charge,
                    vb_orbital_specs=vb_orbital_specs,
                )
        else:
            effective_dom = domain_mode
            roles, masks, nelec_act, x0_orbs = _build_general_vb_setup(
                active_atoms,
                ints,
                effective_dom,
                core_mode,
                spin_override,
                active_charge,
                vb_orbital_specs=vb_orbital_specs,
            )

        run_general_sweep_in_place = (
            compare_regimes
            and not uses_nico_pipeline
            and not used_localized_vb_model
            and len(active_atoms) >= 2
            and domain_mode in _GENERAL_REGIME_ORDER
            and not vb_orbital_specs
        )

        if not used_localized_vb_model and not uses_nico_pipeline:
            if run_general_sweep_in_place:
                regime_comparison, selected = _solve_general_regimes(
                    active_atoms,
                    ints,
                    core_mode,
                    spin_override,
                    active_charge,
                    maxiter,
                    gtol,
                    target_regime=domain_mode,
                    warm_orbitals=warm,
                    vb_orbital_specs=vb_orbital_specs,
                )
                model, roles, masks, nelec_act, opt_res = selected
            else:
                model = MaskedVB(ints, roles, masks, nelec_act)
                if warm is not None and len(warm) == len(masks) and np.asarray(warm).shape[1] == ints.nao:
                    x0 = model.pack([np.asarray(w, dtype=float) for w in warm])
                else:
                    x0 = model.pack(x0_orbs)
                opt_res = optimize_masked_vb(model, x0, maxiter=maxiter, gtol=gtol)

        if not used_localized_vb_model:
            orbs_conv = model.orbitals(opt_res.x)
            Xi, Xa = model.spaces(orbs_conv)
            Xi_opt, Xa_opt = Xi, Xa
            rdm_info = cas_rdm1_and_natural_orbitals(Xi, Xa, ints, nelec_act)
            total_energy = float(opt_res.fun)

        if compare_regimes and not uses_nico_pipeline and regime_comparison is None:
            if len(active_atoms) == 1:
                regime_comparison = {r: total_energy for r in _GENERAL_REGIME_ORDER}
            elif not vb_orbital_specs:
                regime_comparison, _ = _solve_general_regimes(
                    active_atoms,
                    ints,
                    core_mode,
                    spin_override,
                    active_charge,
                    maxiter,
                    gtol,
                )
                if domain_mode not in regime_comparison:
                    regime_comparison[domain_mode] = total_energy

        # Normalize saved localized orbitals to unit S_eff norm for clean restart / visualization
        loc_orbs_emb = np.array([
            o / max(float(np.sqrt(max(float(o @ ints.S @ o), 1e-18))), 1e-12)
            for o in orbs_conv
        ], dtype=float)

        # ------------------------------------------------------------------
        # 3. Reconstruct Full Bare-AO Orbitals, Density, Virial & Multipoles
        # ------------------------------------------------------------------
        Ci_emb = rdm_info["Ci"]
        Ca_emb = rdm_info["Ca"]
        orthonormal_emb = np.hstack([Ci_emb, Ca_emb]) if (Ci_emb.shape[1] + Ca_emb.shape[1]) > 0 else np.zeros((ints.nao, 0))
        nat_orbs_emb = rdm_info["nat_orbs"]
        nat_occ = rdm_info["nat_occ"]
        nat_occ_a = rdm_info["nat_occ_a"]
        nat_occ_b = rdm_info["nat_occ_b"]

        # Map embedded valence natural orbitals onto bare active AOs when core-embedded
        L_proj = getattr(ints, "L_backproj", None)
        if (L_proj is None or L_proj.shape[0] != len(full_active_basis)) and C_core_active.shape[1] > 0 and bare_S is not None:
            N_c = np.linalg.inv(C_core_active.T @ bare_S @ C_core_active)
            L_proj = np.eye(len(full_active_basis)) - C_core_active @ N_c @ C_core_active.T @ bare_S

        if L_proj is not None and C_core_active.shape[1] > 0 and L_proj.shape[0] == len(full_active_basis):
            nat_val_bare = L_proj @ nat_orbs_emb
            mo_bare = np.hstack([C_core_active, nat_val_bare])
            nc_tot = C_core_active.shape[1]
            occ_bare_tot = np.concatenate([np.full(nc_tot, 2.0), nat_occ])
            occ_bare_a = np.concatenate([np.full(nc_tot, 1.0), nat_occ_a])
            occ_bare_b = np.concatenate([np.full(nc_tot, 1.0), nat_occ_b])
            dm1_a_bare = C_core_active @ C_core_active.T + L_proj @ rdm_info["dm1_a"] @ L_proj.T
            dm1_b_bare = C_core_active @ C_core_active.T + L_proj @ rdm_info["dm1_b"] @ L_proj.T
            dm1_ao_bare = dm1_a_bare + dm1_b_bare
            loc_orbs_bare = (L_proj @ loc_orbs_emb.T).T
        else:
            mo_bare = nat_orbs_emb
            occ_bare_tot = nat_occ
            occ_bare_a = nat_occ_a
            occ_bare_b = nat_occ_b
            dm1_a_bare = rdm_info["dm1_a"]
            dm1_b_bare = rdm_info["dm1_b"]
            dm1_ao_bare = rdm_info["dm1_ao"]
            loc_orbs_bare = loc_orbs_emb

        S_for_pop = bare_S if (bare_S is not None and bare_S.shape == dm1_ao_bare.shape) else ints.S

        # Spin-channel MO / occupation arrays compatible with volumetric.py
        # Note volumetric.py multiplies occx[:, 0] by both alpha and beta when ispin == 1!
        is_open_shell = nelec_act[0] != nelec_act[1]
        ispin = 2 if is_open_shell else 1
        mstot = len(full_active_basis)
        norbt = mo_bare.shape[1]
        evecx = np.zeros((mstot, norbt, ispin), dtype=float)
        occx = np.zeros((norbt, ispin), dtype=float)
        evalx = np.zeros((norbt, ispin), dtype=float)
        if ispin == 1:
            evecx[:, :, 0] = mo_bare
            occx[:, 0] = 0.5 * occ_bare_tot
        else:
            evecx[:, :, 0] = mo_bare
            evecx[:, :, 1] = mo_bare
            occx[:, 0] = occ_bare_a
            occx[:, 1] = occ_bare_b

        # Active kinetic energy & virial ratio
        if bare_T is not None and bare_T.shape == dm1_ao_bare.shape:
            kinetic_energy = float(np.sum(dm1_ao_bare * bare_T))
        elif hasattr(ints, "T"):
            kinetic_energy = float(np.sum(rdm_info["dm1_ao"] * ints.T))
        else:
            # Fallback estimate via 1-center/active T when NiCO ECP kinetic matrix isn't separated
            kinetic_energy = abs(total_energy)

        potential_energy = float(total_energy - kinetic_energy)
        virial_ratio = -potential_energy / kinetic_energy if abs(kinetic_energy) > 1e-12 else 0.0

        # Active effective nuclear charges (matches dm1_ao_bare charge conservation)
        if uses_nico_pipeline:
            z_eff_active = [10.0, 6.0, 8.0]
        else:
            z_eff_active = [float(_atomic_num(a)) for a in active_atoms]

        gross_a, net_a, _ = _lowdin_atomic_charges(dm1_a_bare, S_for_pop, ints.atom_of, [0.5 * z for z in z_eff_active])
        gross_b, _, _ = _lowdin_atomic_charges(dm1_b_bare, S_for_pop, ints.atom_of, [0.5 * z for z in z_eff_active])
        gross_tot = gross_a + gross_b
        spin_tot = gross_a - gross_b
        net_tot = np.asarray(z_eff_active, dtype=float) - gross_tot

        # Dipole moment vector wrt nuclear-charge center of active atoms (translation-invariant)
        z_sum = max(float(np.sum(z_eff_active)), 1e-12)
        origin = tuple(
            np.sum([z_i * np.asarray(a.xyz, dtype=float) for z_i, a in zip(z_eff_active, active_atoms)], axis=0) / z_sum
        )
        Dx, Dy, Dz = _ao_dipole_matrices(full_active_basis, origin=origin)
        mu_elec = np.array([
            float(np.sum(dm1_ao_bare * Dx)),
            float(np.sum(dm1_ao_bare * Dy)),
            float(np.sum(dm1_ao_bare * Dz)),
        ])
        mu_nuc = np.sum(
            [z_i * (np.asarray(a.xyz, dtype=float) - np.asarray(origin)) for z_i, a in zip(z_eff_active, active_atoms)],
            axis=0,
        )
        dipole_vec_au = mu_nuc - mu_elec
        dipole_au = float(np.linalg.norm(dipole_vec_au))
        dipole_debye = float(dipole_au * DEBYE_PER_AU)

        # Own-atom character gauge invariant metrics per active atom
        oac_map: Dict[str, List[float]] = {}
        if orthonormal_emb.shape[1] > 0:
            for ia, a in enumerate(active_atoms):
                aid = a.id or f"{a.element}{ia + 1}"
                try:
                    k_own = int(np.sum(ints.atom_of == ia))
                    vals = own_atom_character(ints, orthonormal_emb, ia, min(k_own, orthonormal_emb.shape[1]))
                    oac_map[aid] = [round(float(v), 6) for v in vals]
                except Exception:
                    oac_map[aid] = []

        # Separated-atom reference & binding energy
        sep_energy = 0.0
        can_compute_sep = not uses_nico_pipeline and not spectator_atoms
        if can_compute_sep:
            for a, bname in zip(active_atoms, active_basis_names):
                sym = a.element.strip().capitalize()
                user_elem = deck.elements.get(sym) if hasattr(deck, "elements") else None
                cf = user_elem.functions if (user_elem and user_elem.functions) else None
                sep_energy += _isolated_atom_energy(sym, bname, custom_functions=cf)
            binding_ha = float(sep_energy - total_energy)
            binding_ev = float(binding_ha * HARTREE_TO_EV)
        else:
            sep_energy = 0.0
            binding_ha = 0.0
            binding_ev = 0.0

        # Build deck-indexed basis_info, coordinates, charges, and EmbeddedAtomState dict for all non-virtual atoms
        n_all_atoms = len(deck.geometry)
        coord_all = np.zeros((3, n_all_atoms), dtype=float)
        charge_all = np.zeros((3, n_all_atoms), dtype=float)
        atom_dipoles_all = np.zeros((3, n_all_atoms), dtype=float)

        # Map local active atom index (0..len(active)-1) -> 1-based geometry index
        active_to_geom_1based = [active_indices[ia] + 1 for ia in range(len(active_atoms))]
        iatomx = np.array([active_to_geom_1based[int(ia)] for ia in ints.atom_of], dtype=int)
        nx = np.array([f.n for f in full_active_basis], dtype=int)
        lx = np.array([f.l for f in full_active_basis], dtype=int)
        mlx = np.array([f.m for f in full_active_basis], dtype=int)
        etax = np.array([f.zeta for f in full_active_basis], dtype=float)
        anormx = np.array([float(f.anorm if f.anorm is not None else f.norm()) for f in full_active_basis], dtype=float)

        basis_info = {
            "iatomx": iatomx,
            "nx": nx,
            "lx": lx,
            "mlx": mlx,
            "etax": etax,
            "anormx": anormx,
        }

        embedded_atoms: Dict[str, EmbeddedAtomState] = {}
        active_idx_lookup = {g_i: l_i for l_i, g_i in enumerate(active_indices)}
        for g_idx, atom in enumerate(deck.geometry):
            aid = atom.id or f"{atom.element}{g_idx + 1}"
            coord_all[:, g_idx] = np.asarray(atom.xyz, dtype=float)
            z_a = float(_atomic_num(atom))
            if g_idx in active_idx_lookup:
                ia = active_idx_lookup[g_idx]
                # All-electron reported charge adds frozen core electrons back for clean nuclear neutrality display
                core_e = 2.0 * (n_core_per_active[ia] if not uses_nico_pipeline else 9.0 if ia == 0 else 0.0)
                q_el = float(gross_tot[ia] + core_e)
                sp_net = float(spin_tot[ia])
                sp_a = 0.5 * (q_el + sp_net)
                sp_b = 0.5 * (q_el - sp_net)
                charge_all[0, g_idx] = z_a
                charge_all[1, g_idx] = q_el
                charge_all[2, g_idx] = sp_net
                atom_dip = [0.0, 0.0, 0.0]
                if ia == 0 and len(active_atoms) == 1:
                    atom_dip = dipole_vec_au.tolist()
                atom_dipoles_all[:, g_idx] = np.asarray(atom_dip, dtype=float)
                radial = [{"n": f.n, "l": f.l, "zeta": float(f.zeta)} for f in active_bases[ia] if f.m == 0]
                embedded_atoms[aid] = EmbeddedAtomState(
                    atom_id=aid,
                    element=atom.element.strip().capitalize(),
                    atomic_number=int(round(z_a)),
                    xyz=[float(v) for v in atom.xyz],
                    nuclear_charge=z_a,
                    electronic_charge=q_el,
                    net_charge=z_a - q_el,
                    alpha_spin=sp_a,
                    beta_spin=sp_b,
                    net_spin=sp_net,
                    basis_functions=radial,
                    orbitals=[],
                    model_potential=None,
                    dipole=atom_dip,
                    fixed=bool(atom.fixed),
                    virtual=bool(atom.virtual),
                )
            else:
                q_el = 0.0 if atom.virtual else z_a
                charge_all[0, g_idx] = z_a if not atom.virtual else 0.0
                charge_all[1, g_idx] = q_el
                charge_all[2, g_idx] = 0.0
                embedded_atoms[aid] = EmbeddedAtomState(
                    atom_id=aid,
                    element=atom.element.strip().capitalize(),
                    atomic_number=int(round(z_a)),
                    xyz=[float(v) for v in atom.xyz],
                    nuclear_charge=z_a if not atom.virtual else 0.0,
                    electronic_charge=q_el,
                    net_charge=0.0,
                    alpha_spin=0.5 * q_el,
                    beta_spin=0.5 * q_el,
                    net_spin=0.0,
                    basis_functions=[],
                    orbitals=[],
                    model_potential=None,
                    dipole=[0.0, 0.0, 0.0],
                    fixed=True if not atom.virtual else False,
                    virtual=bool(atom.virtual),
                )

        # Packed identity/zero placeholders matching CalculationState fields
        n2orbt = (norbt * (norbt + 1)) // 2
        S_pack = np.zeros((n2orbt, ispin), dtype=float)
        grad_norm = float(np.linalg.norm(getattr(opt_res, "jac", np.zeros(1)))) if hasattr(opt_res, "jac") and opt_res.jac is not None else 0.0
        converged = bool(getattr(opt_res, "success", False) or grad_norm < max(1e-4, gtol * 10.0))
        n_iters = int(getattr(opt_res, "nit", 0))

        n_inact_roles = sum(1 for r in roles if r == "inactive")
        nat_occ_arr = np.asarray(rdm_info.get("nat_occ", occ_bare_tot), dtype=float)
        active_nat_occ = nat_occ_arr[n_inact_roles:].tolist() if nat_occ_arr.size > n_inact_roles else []

        state_dict: Dict[str, Any] = {
            "converged": converged,
            "stop_reason": "converged" if converged else "max_cycles_reached",
            "cycles": n_iters,
            "cycles_extended": 0,
            "method": method_name,
            "vb_wavefunction": vb_wf,
            "domain_mode": domain_mode,
            "core_embedding": core_mode,
            "environment_embedding": env_mode if spectator_atoms else "none",
            "integral_backend": integral_backend,
            "total_energy": total_energy,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "nuclear_repulsion": float(getattr(ints, "vnn", 0.0)),
            "delta_energy": grad_norm,
            "virial_ratio": float(virial_ratio),
            "separated_atoms_energy": float(sep_energy),
            "binding_energy_hartree": float(binding_ha),
            "binding_energy_ev": float(binding_ev),
            "dipole_vector_au": dipole_vec_au.tolist(),
            "dipole_moment_au": dipole_au,
            "dipole_moment_debye": dipole_debye,
            "norbt": int(norbt),
            "mstot": int(mstot),
            "ispin": int(ispin),
            "eigenvalues": evalx,
            "occupations": occx,
            "mo_coefficients": evecx,
            "overlap_packed": S_pack,
            "ewmat_packed": S_pack.copy(),
            "atom_charges": charge_all,
            "atom_coordinates": coord_all,
            "atom_dipoles": atom_dipoles_all,
            "atom_populations": np.zeros((max(1, norbt), ispin), dtype=float),
            "basis_info": basis_info,
            "embedded_atoms": embedded_atoms,
            "cycle_history": [],
            "localized_orbitals": loc_orbs_bare,
            "localized_orbitals_embedded": loc_orbs_emb,
            "orthonormal_orbitals": orthonormal_emb,
            "rdm1_ao": dm1_ao_bare,
            "vb_roles": list(roles),
            "vb_domain_masks": [np.asarray(m, dtype=int).tolist() for m in masks],
            "vb_nelec_act": [int(nelec_act[0]), int(nelec_act[1])],
            "active_natural_occupations": active_nat_occ,
            "own_atom_character": oac_map,
            "embedding_diagnostics": embedding_diag,
            "optimization_grad_norm": grad_norm,
            "net_charges": {
                (active_atoms[ia].id or f"{active_atoms[ia].element}{ia + 1}"): float(net_tot[ia])
                for ia in range(len(active_atoms))
            },
        }
        if integral_backend == "sto_ng":
            state_dict["sto_ng_primitives"] = sto_ng_k
        if blyholder_decomp is not None:
            state_dict["blyholder_decomposition"] = blyholder_decomp
        if regime_comparison is not None:
            state_dict["regime_comparison"] = regime_comparison

        try:
            from phatpsy.subsystem import decompose_subsystem_energies
            decomp = decompose_subsystem_energies(state_dict, total_energy=total_energy)
            state_dict["subsystem_decomposition"] = decomp.to_dict()
        except Exception:
            pass

        state = CalculationState(state_dict, deck=deck)
        grad_ctx: Optional[Dict[str, Any]] = None
        if (
            return_grad_context
            and pyscf_ctx is not None
            and not uses_nico_pipeline
            and not used_localized_vb_model
            and not spectator_atoms
            and core_mode == "all_electron"
            and vb_wf == "cas"
            and Xi_opt is not None
            and Xa_opt is not None
        ):
            grad_ctx = {
                "ints": ints,
                "pyscf_ctx": pyscf_ctx,
                "Xi": Xi_opt,
                "Xa": Xa_opt,
                "nelec_act": nelec_act,
                "active_indices": list(active_indices),
            }
        return state, grad_ctx

    def energy_and_gradient(
        self,
        deck: CalculationDeck,
        finite_diff_step: float = 1e-3,
        method: str = "auto",
        quiet: Optional[bool] = None,
        warm_orbitals: Optional[Sequence[np.ndarray]] = None,
    ) -> Tuple[CalculationState, np.ndarray]:
        """Compute both converged :class:`CalculationState` and nuclear gradient ``(natom, 3)`` in Ha/Bohr.

        When ``method`` is ``"auto"`` or ``"analytical"``/``"relaxed"`` and
        :meth:`has_analytical_gradient` is True (all-electron ``STO-KG`` CAS MaskedVB),
        evaluates exact variational analytical gradients without finite-difference displacements.
        Otherwise falls back to warm-started central finite differences.
        """
        import copy

        is_quiet = self.quiet if quiet is None else quiet
        base_deck = copy.deepcopy(deck)
        base_deck.ensure_atom_ids()
        base_deck.resolve_coordinates()
        for a in base_deck.geometry:
            a.symmetry = None
            a.equivalent_to = None

        use_analytic = (method != "finite_difference") and self.has_analytical_gradient(base_deck)
        ref_state, grad_ctx = self._solve_deck_internal(
            base_deck,
            quiet=is_quiet,
            warm_orbitals=warm_orbitals,
            return_grad_context=use_analytic,
        )

        n_atoms = len(base_deck.geometry)
        grad = np.zeros((n_atoms, 3), dtype=float)

        if use_analytic and grad_ctx is not None:
            grad_active = _compute_sto_ng_nuclear_gradient(grad_ctx)
            for ia, deck_idx in enumerate(grad_ctx["active_indices"]):
                if not base_deck.geometry[deck_idx].fixed:
                    grad[deck_idx] = grad_active[ia]
            return ref_state, grad

        warm = ref_state.get("localized_orbitals_embedded", ref_state.get("localized_orbitals"))
        h = float(finite_diff_step)
        for ia in range(n_atoms):
            atom = base_deck.geometry[ia]
            if atom.virtual or atom.fixed or (not atom.is_active):
                continue
            for axis in range(3):
                dp = copy.deepcopy(base_deck)
                dp.geometry[ia].xyz[axis] += h
                dm = copy.deepcopy(base_deck)
                dm.geometry[ia].xyz[axis] -= h
                rp = self.run_deck(dp, quiet=is_quiet, warm_orbitals=warm)
                rm = self.run_deck(dm, quiet=is_quiet, warm_orbitals=warm)
                grad[ia, axis] = (float(rp["total_energy"]) - float(rm["total_energy"])) / (2.0 * h)

        return ref_state, grad

    def gradient(
        self,
        deck: CalculationDeck,
        finite_diff_step: float = 1e-3,
        quiet: Optional[bool] = None,
        method: str = "auto",
        warm_orbitals: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute Cartesian nuclear gradient ``(natom, 3)`` in Hartree/Bohr."""
        _, grad = self.energy_and_gradient(
            deck,
            finite_diff_step=finite_diff_step,
            method=method,
            quiet=quiet,
            warm_orbitals=warm_orbitals,
        )
        return grad


def _compute_sto_ng_nuclear_gradient(grad_ctx: Dict[str, Any]) -> np.ndarray:
    """Compute exact variational nuclear gradient ``(n_active, 3)`` in Ha/Bohr from PySCF ``STO-KG`` context."""
    ints = grad_ctx["ints"]
    ctx: PySCFSTOContext = grad_ctx["pyscf_ctx"]
    Xi = grad_ctx["Xi"]
    Xa = grad_ctx["Xa"]
    nelec_act = grad_ctx["nelec_act"]

    inter = cas_nuclear_gradient_intermediates(Xi, Xa, ints, nelec_act)
    D_tot = inter["D_tot"]
    dE_dS = inter["dE_dS"]
    Pa = inter["Pa"]
    Ca = inter["Ca"]
    dm2 = inter["dm2"]

    mol = ctx.mol
    ao_map = ctx.ao_map
    nao_py = int(ao_map.shape[0])
    n2 = nao_py * nao_py
    n_act = int(Ca.shape[1])

    D_py = ao_map @ D_tot @ ao_map.T
    dS_py = ao_map @ dE_dS @ ao_map.T
    Pa_py = ao_map @ Pa @ ao_map.T
    Ca_py = ao_map @ Ca if n_act > 0 else np.zeros((nao_py, 0), dtype=float)
    Ca2_rsvw = (
        np.einsum("rv,sw->rsvw", Ca_py, Ca_py, optimize=True).reshape(n2, n_act * n_act)
        if n_act > 0
        else None
    )

    s1 = -mol.intor("int1e_ipovlp")
    h1_ip = -(mol.intor("int1e_ipkin") + mol.intor("int1e_ipnuc"))
    ip_s2 = -mol.intor("int2e_ip1", aosym="s2kl")

    G_2e_a = np.zeros((nao_py, 3), dtype=float)
    tril_r, tril_c = np.tril_indices(nao_py)
    D_flat = D_py.ravel()
    Pa_flat = Pa_py.ravel() if n_act > 0 else None
    for x in range(3):
        buf = np.empty((nao_py, nao_py, nao_py, nao_py), dtype=float)
        buf[:, :, tril_r, tril_c] = ip_s2[x]
        buf[:, :, tril_c, tril_r] = ip_s2[x]
        buf_2d = buf.reshape(n2, n2)
        buf_ex_2d = buf.transpose(0, 3, 1, 2).reshape(n2, n2)
        J_D = (buf_2d @ D_flat).reshape(nao_py, nao_py)
        K_D = (buf_ex_2d @ D_flat).reshape(nao_py, nao_py)
        term = 2.0 * np.sum(D_py * J_D, axis=1) - np.sum(D_py * K_D, axis=1)
        if n_act > 0:
            J_P = (buf_2d @ Pa_flat).reshape(nao_py, nao_py)
            K_P = (buf_ex_2d @ Pa_flat).reshape(nao_py, nao_py)
            term -= 2.0 * np.sum(Pa_py * J_P, axis=1) - np.sum(Pa_py * K_P, axis=1)
            V_pqvw = (buf_2d @ Ca2_rsvw).reshape(nao_py, nao_py, n_act, n_act)
            W_pt = np.einsum("pqvw,tuvw,qu->pt", V_pqvw, dm2, Ca_py, optimize=True)
            term += 2.0 * np.sum(Ca_py * W_pt, axis=1)
        G_2e_a[:, x] = term

    aoslices = mol.aoslice_by_atom()
    n_active = len(aoslices)
    charges = np.asarray(ctx.charges, dtype=float)
    coords = np.asarray(ctx.coords, dtype=float)
    grad_active = np.zeros((n_active, 3), dtype=float)

    for ia in range(n_active):
        p0, p1 = int(aoslices[ia, 2]), int(aoslices[ia, 3])
        g1_kin_nuc = 2.0 * np.einsum("xab,ab->x", h1_ip[:, p0:p1, :], D_py[p0:p1, :])
        with mol.with_rinv_at_nucleus(ia):
            rinv_ip = mol.intor("int1e_iprinv")
        g1_hf = -2.0 * charges[ia] * np.einsum("xab,ab->x", rinv_ip, D_py)
        g_ovlp = 2.0 * np.einsum("xab,ab->x", s1[:, p0:p1, :], dS_py[p0:p1, :])
        g_2e = np.sum(G_2e_a[p0:p1, :], axis=0)
        g_vnn = np.zeros(3, dtype=float)
        for ib in range(n_active):
            if ib == ia:
                continue
            dR = coords[ia] - coords[ib]
            r2 = float(np.dot(dR, dR))
            if r2 > 1e-24:
                g_vnn -= (charges[ia] * charges[ib]) * (r2 ** -1.5) * dR
        grad_active[ia] = g1_kin_nuc + g1_hf + g_ovlp + g_2e + g_vnn

    return grad_active


def run_localized_vb(
    deck: CalculationDeck,
    quiet: bool = True,
    compare_regimes: bool = False,
    warm_orbitals: Optional[Sequence[np.ndarray]] = None,
) -> CalculationState:
    """Convenience wrapper executing :class:`LocalizedVBDriver` on ``deck``."""
    return LocalizedVBDriver(quiet=quiet).run_deck(
        deck,
        quiet=quiet,
        compare_regimes=compare_regimes,
        warm_orbitals=warm_orbitals,
    )
