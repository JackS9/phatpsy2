"""
PHATPSY 2.0: Localized-VB Active-Site Embedding in a PHATPSY Environment
(branch ``two-center-wip``, Stage 9).

Formulates the Phillips-Kleinman / Huzinaga / frozen-density embedding of an
exact-integral localized-VB active region A inside a frozen localized-orbital
environment B.

Any active-region orbital phi_i orthogonalized against the frozen doubly occupied
environment orbitals C_B (with metric S_BB^orb = C_B^T S_BB C_B and projector
D_B = C_B (S_BB^orb)^-1 C_B^T) has the Phillips-Kleinman form

    phi_i = (1 - P_B) tilde_phi_i ,    tilde_phi_i in span(A)

with AO coefficient vector c_i = L tilde_c_i, where the (N_A + N_B) x N_A
back-projection matrix is

    L = [ I_A          ]
        [ -P_BA        ],     P_BA = C_B (S_BB^orb)^-1 C_B^T S_BA .

Projecting the full Hamiltonian onto the image of L yields an effective active-region
AOIntegrals problem of dimension N_A with:
  * pseudometric        tilde_S = L^T S L = S_AA - S_AB D_B S_BA
  * effective 1e Fock   tilde_h = L^T F^B L,   F^B = h + 2 J(D_B) - K(D_B)
  * effective 2e ERIs   tilde_V = L^{x4} V     (Level E1: exact back-projected)
                        or V_AAAA              (Level E2: unprojected active ERIs)
  * scalar offset       tilde_vnn = E_core(B) + V_nn

For purely one-center environment atoms B (Level E3: ERI-free embedding), the
electrostatic potential V_Coul^B(r) of each environment atom's density D_B is
evaluated analytically in closed form via Laplace's multipole expansion and real
Gaunt coefficients (:func:`one_center_coulomb_potential`), and the one-electron
PK operator L^T (T + V_es) L + P_BA^T (-K_BB) P_BA requires zero multi-center
two-electron integrals involving the environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import legendre as npleg

from phatpsy.two_center_integrals import STO, ProlateGrid, _assoc_legendre_table
from phatpsy.volumetric import evaluate_real_sto


# ---------------------------------------------------------------------------
# Analytic one-center real spherical harmonics, Gaunt coefficients & Coulomb
# ---------------------------------------------------------------------------

def real_spherical_harmonics(pts_rel: np.ndarray, lmax: int) -> Dict[Tuple[int, int], np.ndarray]:
    """Orthonormal real spherical harmonics S_{l,m}(r_hat) with the Fortran/volumetric
    Condon-Shortley phase (-1)^|m| at relative coordinates ``pts_rel`` of shape (N, 3)."""
    x, y, z = pts_rel[:, 0], pts_rel[:, 1], pts_rel[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r == 0.0, 1.0, r)
    cos_t = np.where(r == 0.0, 1.0, np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)
    P = _assoc_legendre_table(cos_t, lmax, lmax, outer=False)
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            fac = math.sqrt((2 * l + 1) / (2.0 * math.pi * (2 if m == 0 else 1))
                            * math.factorial(l - am) / math.factorial(l + am))
            cs = -1.0 if (am % 2) == 1 else 1.0
            trig = np.ones_like(phi) if m == 0 else (np.cos(am * phi) if m > 0 else np.sin(am * phi))
            val = cs * fac * P[l, am] * trig
            if l > 0:
                val = np.where(r == 0.0, 0.0, val)
            out[(l, m)] = val
    return out


_GAUNT_CACHE: Dict[int, Dict[Tuple[int, int, int, int, int, int], float]] = {}


def _gaunt_table(lmax_orb: int) -> Dict[Tuple[int, int, int, int, int, int], float]:
    """Exact real Gaunt integrals G[(l1,m1,l2,m2,L,M)] = int S_{l1,m1} S_{l2,m2} S_{L,M} dOmega."""
    if lmax_orb in _GAUNT_CACHE:
        return _GAUNT_CACHE[lmax_orb]
    Lmax = 2 * lmax_orb
    ct, wt = npleg.leggauss(2 * Lmax + 4)
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    nphi = 4 * Lmax + 8
    phi = 2.0 * math.pi * np.arange(nphi) / nphi
    wphi = 2.0 * math.pi / nphi
    CT, PHI = np.meshgrid(ct, phi, indexing="ij")
    ST = np.sqrt(np.maximum(0.0, 1.0 - CT * CT))
    W = (wt[:, None] * np.ones((1, nphi)) * wphi).ravel()
    pts = np.stack([(ST * np.cos(PHI)).ravel(), (ST * np.sin(PHI)).ravel(), CT.ravel()], axis=1)
    Y = real_spherical_harmonics(pts, Lmax)
    tab: Dict[Tuple[int, int, int, int, int, int], float] = {}
    for l1 in range(lmax_orb + 1):
        for m1 in range(-l1, l1 + 1):
            y1 = Y[(l1, m1)]
            for l2 in range(lmax_orb + 1):
                for m2 in range(-l2, l2 + 1):
                    y12 = y1 * Y[(l2, m2)] * W
                    for L in range(abs(l1 - l2), l1 + l2 + 1):
                        if (l1 + l2 + L) % 2 != 0:
                            continue
                        for M in range(-L, L + 1):
                            g = float(np.dot(y12, Y[(L, M)]))
                            if abs(g) > 1e-14:
                                tab[(l1, m1, l2, m2, L, M)] = g
    _GAUNT_CACHE[lmax_orb] = tab
    return tab


def _radial_multipole_potential(N: int, zeta: float, L: int, r: np.ndarray) -> np.ndarray:
    """Exact radial integral
        I_{N,zeta,L}(r) = r^{-(L+1)} int_0^r r'^{N+L} e^{-zeta r'} dr'
                        + r^L       int_r^inf r'^{N-L-1} e^{-zeta r'} dr'
    with N = n1 + n2 >= L + 2.  Uses the power series for x = zeta * r < 0.5 to avoid
    catastrophic cancellation at small r."""
    x = zeta * r
    ex = np.exp(-x)
    k_in = N + L
    k_out = N - L - 1
    f_in = float(math.factorial(k_in))
    f_out = float(math.factorial(k_out))

    # Outer integral: (N-L-1)! * x^L * e^{-x} * sum_{j=0}^{N-L-1} x^j / j!
    s_out = np.zeros_like(x)
    xp = np.ones_like(x)
    for j in range(k_out + 1):
        s_out = s_out + xp / math.factorial(j)
        xp = xp * x
    outer = f_out * (x ** L) * ex * s_out

    # Inner integral: large x -> f_in / x^{L+1} * (1 - e^{-x} sum_{j=0}^{N+L} x^j / j!)
    #                 small x -> f_in * e^{-x} * sum_{m=0}^{32} x^{N+m} / (N+L+1+m)!
    small = x < 0.5
    inner = np.zeros_like(x)
    if np.any(~small):
        xl = x[~small]
        exl = ex[~small]
        s_in = np.zeros_like(xl)
        xpl = np.ones_like(xl)
        for j in range(k_in + 1):
            s_in = s_in + xpl / math.factorial(j)
            xpl = xpl * xl
        inner[~small] = f_in * (1.0 - exl * s_in) / (xl ** (L + 1))
    if np.any(small):
        xs = x[small]
        exs = ex[small]
        ser = np.zeros_like(xs)
        xps = xs ** N
        for m in range(32):
            ser = ser + xps / math.factorial(k_in + 1 + m)
            xps = xps * xs
        inner[small] = f_in * exs * ser
    return (zeta ** (-N)) * (inner + outer)


def one_center_coulomb_potential(basis: Sequence[STO], D: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Exact analytic electrostatic potential V(r) = sum_{mu,nu} D_{mu,nu} int chi_mu(r') chi_nu(r') / |r-r'| dr'
    of a one-center STO density matrix D at arbitrary 3D points ``pts`` of shape (N_p, 3)."""
    center = np.asarray(basis[0].center, dtype=float)
    pts_rel = np.asarray(pts, dtype=float) - center[None, :]
    r = np.linalg.norm(pts_rel, axis=1)
    lmax_orb = max(f.l for f in basis)
    Lmax = 2 * lmax_orb
    Y = real_spherical_harmonics(pts_rel, Lmax)
    gaunt = _gaunt_table(lmax_orb)

    rad_cache: Dict[Tuple[int, float, int], np.ndarray] = {}
    V = np.zeros(pts_rel.shape[0])
    n = len(basis)
    for mu in range(n):
        f1 = basis[mu]
        for nu in range(mu + 1):
            w = float(D[mu, nu]) if mu == nu else float(D[mu, nu] + D[nu, mu])
            if abs(w) < 1e-15:
                continue
            f2 = basis[nu]
            N = f1.n + f2.n
            zeta = round(float(f1.zeta + f2.zeta), 12)
            pref = w * f1.norm() * f2.norm()
            for L in range(abs(f1.l - f2.l), f1.l + f2.l + 1):
                if (f1.l + f2.l + L) % 2 != 0:
                    continue
                ang = np.zeros_like(r)
                any_m = False
                for M in range(-L, L + 1):
                    g = gaunt.get((f1.l, f1.m, f2.l, f2.m, L, M), 0.0)
                    if g != 0.0:
                        ang = ang + g * Y[(L, M)]
                        any_m = True
                if not any_m:
                    continue
                key = (N, zeta, L)
                if key not in rad_cache:
                    rad_cache[key] = _radial_multipole_potential(N, zeta, L, r)
                V = V + pref * (4.0 * math.pi / (2 * L + 1)) * rad_cache[key] * ang
    return V


def _slater_radial_rk(N1: int, z1: float, N2: int, z2: float, L: int) -> float:
    """Exact one-center two-electron radial integral
        R^L(N1,z1; N2,z2) = int_0^inf dr r^{N2-1} e^{-z2 r} I_{N1,z1,L}(r)."""
    z12 = z1 + z2
    k_in = N1 + L
    k_out = N1 - L - 1
    f_in = math.factorial(k_in)
    f_out = math.factorial(k_out)
    t1 = math.factorial(N2 - L - 1) / (z2 ** (N2 - L))
    for j in range(k_in + 1):
        t1 -= (z1 ** j / math.factorial(j)) * math.factorial(N2 - L - 1 + j) / (z12 ** (N2 - L + j))
    t1 *= f_in / (z1 ** (k_in + 1))
    t2 = 0.0
    for j in range(k_out + 1):
        t2 += (z1 ** j / math.factorial(j)) * math.factorial(N2 + L + j) / (z12 ** (N2 + L + 1 + j))
    t2 *= f_out / (z1 ** (k_out + 1))
    return float(t1 + t2)


def one_center_ao_matrices(basis: Sequence[STO], z_nuc: float):
    """Exact analytic one-center (S, T, V_nuc, eri) for a single atom's STO basis."""
    n = len(basis)
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    for i, f1 in enumerate(basis):
        for j, f2 in enumerate(basis):
            if f1.l != f2.l or f1.m != f2.m:
                continue
            N = f1.n + f2.n
            z = f1.zeta + f2.zeta
            nn = f1.norm() * f2.norm()
            s_ij = nn * math.factorial(N) / (z ** (N + 1))
            v_ij = nn * math.factorial(N - 1) / (z ** N)
            S[i, j] = s_ij
            V[i, j] = -z_nuc * v_ij
            c2 = f2.n * (f2.n - 1) - f2.l * (f2.l + 1)
            r_m2 = nn * math.factorial(N - 2) / (z ** (N - 1)) if c2 != 0 else 0.0
            T[i, j] = -0.5 * (f2.zeta ** 2) * s_ij + f2.n * f2.zeta * v_ij - 0.5 * c2 * r_m2

    lmax_orb = max(f.l for f in basis)
    gaunt = _gaunt_table(lmax_orb)
    rk_cache: Dict[Tuple[int, float, int, float, int], float] = {}
    eri = np.zeros((n, n, n, n))
    for p in range(n):
        fp = basis[p]
        for q in range(p + 1):
            fq = basis[q]
            N1 = fp.n + fq.n
            z1 = round(float(fp.zeta + fq.zeta), 12)
            n12 = fp.norm() * fq.norm()
            for r in range(n):
                fr = basis[r]
                for s in range(r + 1):
                    fs = basis[s]
                    N2 = fr.n + fs.n
                    z2 = round(float(fr.zeta + fs.zeta), 12)
                    val = 0.0
                    Lmin = max(abs(fp.l - fq.l), abs(fr.l - fs.l))
                    Lmax = min(fp.l + fq.l, fr.l + fs.l)
                    for L in range(Lmin, Lmax + 1):
                        if (fp.l + fq.l + L) % 2 != 0 or (fr.l + fs.l + L) % 2 != 0:
                            continue
                        ang = 0.0
                        for M in range(-L, L + 1):
                            g1 = gaunt.get((fp.l, fp.m, fq.l, fq.m, L, M), 0.0)
                            if g1 != 0.0:
                                g2 = gaunt.get((fr.l, fr.m, fs.l, fs.m, L, M), 0.0)
                                ang += g1 * g2
                        if ang == 0.0:
                            continue
                        key = (N1, z1, N2, z2, L)
                        if key not in rk_cache:
                            rk_cache[key] = _slater_radial_rk(N1, z1, N2, z2, L)
                        val += (4.0 * math.pi / (2 * L + 1)) * rk_cache[key] * ang
                    val *= n12 * fr.norm() * fs.norm()
                    eri[p, q, r, s] = eri[q, p, r, s] = eri[p, q, s, r] = eri[q, p, s, r] = val
    return S, T, V, eri


# ---------------------------------------------------------------------------
# Embedded active-region AO integrals
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedAOIntegrals:
    """Effective active-region integrals produced by embedding an active space A in a
    frozen localized-orbital environment B.  Duck-type compatible with
    :class:`phatpsy.localized_vb.AOIntegrals` and
    :class:`phatpsy.multicenter_integrals.MolecularAOIntegrals`."""
    basis: List[STO]
    atom_of: np.ndarray
    S: np.ndarray                # pseudometric tilde_S = L^T S L = S_AA - S_AB D_B S_BA
    h: np.ndarray                # effective 1e operator tilde_h on A
    eri: np.ndarray              # effective 2e ERIs on A
    vnn: float                   # E_core(B) + V_nn(A-A) + V_nn(A-B) + V_nn(B-B)
    S_bare: np.ndarray = field(repr=False, default=None)
    h_bare: np.ndarray = field(repr=False, default=None)
    L_backproj: Optional[np.ndarray] = field(repr=False, default=None)
    _Jmat: Optional[np.ndarray] = field(default=None, repr=False)
    _Kmat: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def nao(self) -> int:
        return len(self.basis)

    def jk_matrices(self):
        if self._Jmat is None:
            n = self.nao
            self._Jmat = self.eri.reshape(n * n, n * n)
            self._Kmat = np.ascontiguousarray(self.eri.transpose(0, 3, 2, 1)).reshape(n * n, n * n)
        return self._Jmat, self._Kmat


def _renormalize_active_eri(S_bare: np.ndarray, S_eff: np.ndarray, eri_bare: np.ndarray) -> np.ndarray:
    """Charge-conserving renormalization of active-region ERIs under the pseudometric S_eff
    (the subspace-invariant matrix form of PHATPSY's ``intnrm`` + ``NORMLZ`` sequence).

    Constructs W = S_bare^{-1/2} (S_bare^{-1/2} S_eff S_bare^{-1/2})^{1/2} S_bare^{1/2},
    which satisfies W^T S_bare W = S_eff (and W = I when S_eff = S_bare), so that any
    orbital orthonormalized in S_eff has exact unit norm in S_bare when contracted with
    W^{x4} eri_bare.
    """
    wb, vb = np.linalg.eigh(S_bare)
    Sb_half = (vb * np.sqrt(wb)) @ vb.T
    Sb_mhalf = (vb / np.sqrt(wb)) @ vb.T
    M = 0.5 * (Sb_mhalf @ S_eff @ Sb_mhalf + (Sb_mhalf @ S_eff @ Sb_mhalf).T)
    wm, vm = np.linalg.eigh(M)
    M_half = (vm * np.sqrt(np.maximum(wm, 0.0))) @ vm.T
    W = Sb_mhalf @ M_half @ Sb_half
    t = np.einsum("mnls,mp->pnls", eri_bare, W, optimize=True)
    t = np.einsum("pnls,nq->pqls", t, W, optimize=True)
    t = np.einsum("pqls,lr->pqrs", t, W, optimize=True)
    return np.einsum("pqrs,st->pqrt", t, W, optimize=True)


def embed_from_full_integrals(
    full_ints,
    C_env: np.ndarray,
    active_ao_idx: Optional[Sequence[int]] = None,
    mode: str = "exact_backprojected",
) -> EmbeddedAOIntegrals:
    """Construct :class:`EmbeddedAOIntegrals` on ``active_ao_idx`` from a full-system
    integral object ``full_ints`` and frozen doubly occupied environment orbitals ``C_env``
    (shape ``(nao_full, n_env)``).

    Modes:
      * ``"exact_backprojected"`` (Level E1):
            tilde_S = L^T S L,  tilde_h = L^T F^B L,  tilde_eri = L^{x4} eri_full.
        Mathematically isomorphic to running the full system with ``C_env`` frozen in
        ``Xi`` and active orbitals restricted to ``active_ao_idx``.
      * ``"unprojected_eri"`` (Level E2):
            tilde_S = L^T S L,  tilde_h = L^T F^B L,  tilde_eri = W^{x4} eri_AAAA.
        Uses only active-region ERIs ``eri_AAAA`` (renormalized by W so pseudometric-
        orthonormal orbitals conserve unit charge in S_AA) with zero environment ERIs in
        the active 2e tensor.
      * ``"huzinaga_pk"`` (Level E2b):
            tilde_S = L^T S L,
            tilde_h = F^B_AA - S_AB C_B (S_BB^orb)^-1 (C_B^T F^B C_B) (S_BB^orb)^-1 C_B^T S_BA,
            tilde_eri = W^{x4} eri_AAAA.
      * ``"bare_fock_no_pk"`` (Diagnostic control: electrostatic + exchange of B without
        orthogonality projection, demonstrating variational collapse onto B's core).
    """
    S, h = full_ints.S, full_ints.h
    n_full = full_ints.nao
    if active_ao_idx is None:
        active_ao_idx = list(range(n_full))
    idx_A = np.asarray(active_ao_idx, dtype=int)
    n_A = idx_A.size

    Jm, Km = full_ints.jk_matrices()
    J = lambda g: (Jm @ g.ravel()).reshape(g.shape)
    K = lambda g: (Km @ g.ravel()).reshape(g.shape)

    if C_env.shape[1] > 0:
        S_orb = C_env.T @ S @ C_env
        S_orb_inv = np.linalg.inv(S_orb)
        D_B = C_env @ S_orb_inv @ C_env.T
    else:
        S_orb_inv = np.zeros((0, 0))
        D_B = np.zeros((n_full, n_full))

    F_B = h + 2.0 * J(D_B) - K(D_B)
    e_core_B = float(np.sum(D_B * (h + F_B)))

    # Back-projection matrix L: shape (n_full, n_A), mapping tilde_c on A to (1 - D_B S) tilde_c
    I_A = np.zeros((n_full, n_A))
    I_A[idx_A, np.arange(n_A)] = 1.0
    L = I_A - D_B @ (S @ I_A)

    S_AA = S[np.ix_(idx_A, idx_A)]
    h_AA = h[np.ix_(idx_A, idx_A)]
    eri_AAAA = full_ints.eri[np.ix_(idx_A, idx_A, idx_A, idx_A)]

    if mode == "exact_backprojected":
        S_eff = L.T @ S @ L
        h_eff = L.T @ F_B @ L
        t = np.einsum("mnls,mp->pnls", full_ints.eri, L, optimize=True)
        t = np.einsum("pnls,nq->pqls", t, L, optimize=True)
        t = np.einsum("pqls,lr->pqrs", t, L, optimize=True)
        eri_eff = np.einsum("pqrs,st->pqrt", t, L, optimize=True)
    elif mode == "unprojected_eri":
        S_eff = L.T @ S @ L
        h_eff = L.T @ F_B @ L
        eri_eff = _renormalize_active_eri(S_AA, 0.5 * (S_eff + S_eff.T), eri_AAAA)
    elif mode == "huzinaga_pk":
        S_eff = L.T @ S @ L
        E_B = S_orb_inv @ (C_env.T @ F_B @ C_env) @ S_orb_inv
        S_AB = I_A.T @ S @ C_env
        h_eff = F_B[np.ix_(idx_A, idx_A)] - S_AB @ E_B @ S_AB.T
        eri_eff = _renormalize_active_eri(S_AA, 0.5 * (S_eff + S_eff.T), eri_AAAA)
    elif mode == "bare_fock_no_pk":
        S_eff = S_AA
        h_eff = F_B[np.ix_(idx_A, idx_A)]
        eri_eff = eri_AAAA
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")

    S_eff = 0.5 * (S_eff + S_eff.T)
    ws, vs = np.linalg.eigh(S_eff)
    S_eff = (vs * np.maximum(ws, 0.0)) @ vs.T
    S_eff = 0.5 * (S_eff + S_eff.T)
    h_eff = 0.5 * (h_eff + h_eff.T)
    basis_A = [full_ints.basis[i] for i in idx_A]
    atom_of_A = np.asarray(full_ints.atom_of)[idx_A]
    return EmbeddedAOIntegrals(
        basis=basis_A,
        atom_of=atom_of_A,
        S=S_eff,
        h=h_eff,
        eri=eri_eff,
        vnn=e_core_B + float(full_ints.vnn),
        S_bare=S_AA,
        h_bare=h_AA,
        L_backproj=L,
    )


# ---------------------------------------------------------------------------
# ERI-free embedding of an active region in one-center PHATPSY environment atoms
# ---------------------------------------------------------------------------

@dataclass
class FrozenEnvironmentAtom:
    """A frozen closed-shell environment atom/ion B (or frozen core shell on an atom),
    defined strictly by its one-center STO basis and doubly occupied orbital coefficients."""
    z_nuc: float
    center: Tuple[float, float, float]
    basis: List[STO]
    C_occ: np.ndarray            # shape (nao_B, n_occ_B), doubly occupied one-center orbitals

    @classmethod
    def from_atomic_rhf(cls, z_nuc: float, center: Sequence[float], basis: Sequence[STO], n_occ: int):
        """Solve the exact one-center atomic RHF problem on ``basis`` and freeze the
        lowest ``n_occ`` doubly occupied orbitals."""
        S, T, V, eri = one_center_ao_matrices(basis, z_nuc)
        h = T + V
        n = len(basis)
        Jm = eri.reshape(n * n, n * n)
        Km = np.ascontiguousarray(eri.transpose(0, 3, 2, 1)).reshape(n * n, n * n)
        w, v = np.linalg.eigh(S)
        X = v @ np.diag(w ** -0.5) @ v.T
        _, U = np.linalg.eigh(X.T @ h @ X)
        C = X @ U[:, :n_occ]
        for _ in range(100):
            D = C @ C.T
            F = h + (2.0 * (Jm @ D.ravel()) - (Km @ D.ravel())).reshape(n, n)
            eps, U = np.linalg.eigh(X.T @ F @ X)
            C_new = X @ U[:, :n_occ]
            if np.linalg.norm(C_new @ C_new.T - D) < 1e-12:
                C = C_new
                break
            C = C_new
        return cls(float(z_nuc), tuple(float(c) for c in center), list(basis), C)

    @classmethod
    def from_phatpsy_state(cls, state, atom_index: int, n_occ: Optional[int] = None, occ_threshold: float = 0.5):
        """Extract a :class:`FrozenEnvironmentAtom` for 1-based ``atom_index`` directly
        from a converged PHATPSY calculation state dictionary."""
        b = state["basis_info"]
        X = np.asarray(state["atom_coordinates"], dtype=float)
        chg = np.asarray(state["atom_charges"], dtype=float)
        mstot = int(state["mstot"])
        C_all = np.asarray(state["mo_coefficients"], dtype=float)[:mstot]
        occ = np.asarray(state["occupations"], dtype=float)
        iat = np.asarray(b["iatomx"])[:mstot]

        idx = np.where(iat == atom_index)[0]
        center = tuple(float(c) for c in X[:, atom_index - 1])
        z_nuc = float(chg[0, atom_index - 1])
        basis = [
            STO(center, int(b["nx"][i]), int(b["lx"][i]), int(b["mlx"][i]),
                float(b["etax"][i]), float(b["anormx"][i]))
            for i in idx
        ]
        cols = []
        for k in range(C_all.shape[1]):
            if np.max(np.abs(C_all[idx, k, 0])) > 1e-8 and np.max(np.abs(np.delete(C_all[:, k, 0], idx))) < 1e-8:
                if n_occ is not None or occ[k, 0] >= occ_threshold:
                    cols.append(C_all[idx, k, 0].copy())
                    if n_occ is not None and len(cols) >= n_occ:
                        break
        C_occ = np.column_stack(cols) if cols else np.zeros((len(basis), 0))
        return cls(z_nuc, center, basis, C_occ)



def embed_diatomic_in_environment(
    basis_a: Sequence[STO],
    basis_b: Sequence[STO],
    z_a: float,
    z_b: float,
    env_atoms: Sequence[FrozenEnvironmentAtom],
    n_eta: int = 64,
    lmax: int = 30,
    lmax_exch: int = 16,
    include_exchange: bool = True,
    include_internal_env_energy: bool = True,
) -> EmbeddedAOIntegrals:
    """Embed an active diatomic (A, B) in a collection of frozen environment atoms/cores
    using only one- and two-center integrals (Level E3).

    * Active-region 2e integrals are the bare diatomic ERIs on (A, B).
    * Electrostatic potential V_es^E(r) of each environment atom/core E is evaluated
      analytically in closed form on the active prolate grid via
      :func:`one_center_coulomb_potential` (exact 3-center Coulomb with zero fitting).
    * One-electron cross-integrals S_{A,E}, T_{A,E}, V_{es, A,E} and two-center
      core-valence exchange K(D_E) are evaluated on two-center prolate grids.
    """
    from phatpsy.localized_vb import diatomic_ao_integrals

    act_ints = diatomic_ao_integrals(basis_a, basis_b, z_a, z_b, n_eta=n_eta, lmax=lmax)
    basis_act = act_ints.basis
    n_act = act_ints.nao
    A_xyz = np.asarray(basis_a[0].center, dtype=float)
    B_xyz = np.asarray(basis_b[0].center, dtype=float)
    act_coords = [A_xyz, B_xyz]
    act_charges = [float(z_a), float(z_b)]

    # Active prolate grid for integrating V_es^env(r) over active AO pairs
    zmin_act = min(f.zeta for f in basis_act)
    g_act = ProlateGrid(A_xyz, B_xyz, zeta_min=min(zmin_act, 1.0), n_eta=n_eta)
    chi_act = np.array([g_act.evaluate(f) for f in basis_act])
    w_act = g_act.weights
    pts_act = np.stack([g_act.X.ravel(), g_act.Y.ravel(), g_act.Z.ravel()], axis=1)

    # Total electrostatic potential of all environment atoms on the active grid
    v_es_act = np.zeros(pts_act.shape[0])
    e_const = float(act_ints.vnn)

    env_data = []
    for E in env_atoms:
        R_E = np.asarray(E.center, dtype=float)
        d_A = float(np.linalg.norm(R_E - A_xyz))
        d_B = float(np.linalg.norm(R_E - B_xyz))
        same_center = (d_A < 1e-10) or (d_B < 1e-10)

        S_EE, T_EE, V_EE, eri_EE = one_center_ao_matrices(E.basis, E.z_nuc)
        n_E = len(E.basis)
        Jm_E = eri_EE.reshape(n_E * n_E, n_E * n_E)
        Km_E = np.ascontiguousarray(eri_EE.transpose(0, 3, 2, 1)).reshape(n_E * n_E, n_E * n_E)
        S_orb = E.C_occ.T @ S_EE @ E.C_occ
        S_orb_inv = np.linalg.inv(S_orb)
        D_E = E.C_occ @ S_orb_inv @ E.C_occ.T
        J_EE = (Jm_E @ D_E.ravel()).reshape(n_E, n_E)
        K_EE = (Km_E @ D_E.ravel()).reshape(n_E, n_E)
        F_EE_es = T_EE + V_EE + 2.0 * J_EE
        F_EE_atom = F_EE_es - K_EE
        e_int_E = float(np.sum(D_E * (T_EE + V_EE + F_EE_atom)))
        if include_internal_env_energy:
            e_const += e_int_E

        # Analytic electrostatic potential of E on the active grid:
        # If E is a same-center core shell on A or B, its nuclear attraction -Z_E/r_E is
        # already in act_ints.h, so only the electronic screening +V_Coul^E(r) is added.
        v_coul_E = one_center_coulomb_potential(E.basis, 2.0 * D_E, pts_act)
        if same_center:
            v_es_act = v_es_act + v_coul_E
        else:
            r_E_act = np.linalg.norm(pts_act - R_E[None, :], axis=1)
            r_E_safe = np.where(r_E_act < 1e-12, 1e-12, r_E_act)
            v_es_act = v_es_act + (-E.z_nuc / r_E_safe + v_coul_E)

        # Interaction of active nuclei with environment E:
        for Z_c, R_c in zip(act_charges, act_coords):
            d_cE = float(np.linalg.norm(R_c - R_E))
            if d_cE > 1e-10:
                v_coul_at_c = float(one_center_coulomb_potential(E.basis, 2.0 * D_E, R_c[None, :])[0])
                if same_center:
                    # Z_A * Z_B / R is already in act_ints.vnn; add only Tr[2 D_E (-Z_c / r_c)]
                    e_const -= Z_c * v_coul_at_c
                else:
                    e_const += Z_c * (E.z_nuc / d_cE - v_coul_at_c)

        env_data.append(dict(
            E=E, R_E=R_E, same_center=same_center, d_A=d_A, d_B=d_B,
            S_EE=S_EE, F_EE_es=F_EE_es, K_EE=K_EE, D_E=D_E, S_orb_inv=S_orb_inv,
            v_coul_grid=v_coul_E.reshape(g_act.shape),
        ))

    # Environment-environment interaction if multiple environment centers exist
    for i in range(len(env_data)):
        for j in range(i):
            R_i, R_j = env_data[i]["R_E"], env_data[j]["R_E"]
            d_ij = float(np.linalg.norm(R_i - R_j))
            if d_ij > 1e-10:
                E_i, E_j = env_data[i]["E"], env_data[j]["E"]
                D_i, D_j = env_data[i]["D_E"], env_data[j]["D_E"]
                # Add nuclear-nuclear repulsion only if at least one is an external center
                # (when both are same_center on A and B, Z_A * Z_B / R is already in act_ints.vnn)
                if not (env_data[i]["same_center"] and env_data[j]["same_center"]):
                    if not env_data[i]["same_center"] and not env_data[j]["same_center"]:
                        e_const += E_i.z_nuc * E_j.z_nuc / d_ij
                    elif not env_data[i]["same_center"]:
                        e_const -= E_i.z_nuc * float(one_center_coulomb_potential(E_j.basis, 2.0 * D_j, R_i[None, :])[0])
                    elif not env_data[j]["same_center"]:
                        e_const -= E_j.z_nuc * float(one_center_coulomb_potential(E_i.basis, 2.0 * D_i, R_j[None, :])[0])
                # Core-core 2e Coulomb + exchange between E_i and E_j
                if env_data[i]["same_center"] and env_data[j]["same_center"]:
                    chi_i = np.array([g_act.evaluate(f) for f in E_i.basis])
                    rho_i = np.einsum("mn,mijk,nijk->ijk", 2.0 * D_i, chi_i, chi_i)
                    e_const += float(np.sum(w_act * rho_i * env_data[j]["v_coul_grid"]))
                    if include_exchange:
                        chi_j = np.array([g_act.evaluate(f) for f in E_j.basis])
                        Ci_orth = E_i.C_occ @ np.linalg.cholesky(env_data[i]["S_orb_inv"])
                        Cj_orth = E_j.C_occ @ np.linalg.cholesky(env_data[j]["S_orb_inv"])
                        pairs_ij = []
                        for bi in range(Ci_orth.shape[1]):
                            psi_i = np.tensordot(Ci_orth[:, bi], chi_i, axes=(0, 0))
                            for bj in range(Cj_orth.shape[1]):
                                psi_j = np.tensordot(Cj_orth[:, bj], chi_j, axes=(0, 0))
                                pairs_ij.append(psi_i * psi_j)
                        if pairs_ij:
                            K_ij = g_act.coulomb_matrix(np.array(pairs_ij), lmax=lmax_exch)
                            e_const -= float(np.trace(K_ij))

    V_es_AA = np.einsum("pijk,qijk,ijk->pq", chi_act, chi_act,
                        w_act * v_es_act.reshape(g_act.shape))
    S_eff = act_ints.S.copy()
    h_eff = act_ints.h + V_es_AA

    # Total electrostatic potential on the active grid (including active nuclei)
    v_tot_act_grid = -(z_a / g_act.r_a + z_b / g_act.r_b) + v_es_act.reshape(g_act.shape)

    # First Subtract K_AA from all environment atoms on h_eff so F_AA, F_AE, F_EE all share
    # the exact same total environment Fock operator F^B = T + V_tot - sum_E K(D_E).
    K_AA_tot = np.zeros((n_act, n_act))
    for ed in env_data:
        E = ed["E"]
        S_orb_inv = ed["S_orb_inv"]
        C_orth = E.C_occ @ np.linalg.cholesky(S_orb_inv)
        if ed["same_center"]:
            chi_E = np.array([g_act.evaluate(f) for f in E.basis])
            ed["chi_E_act"] = chi_E
            ed["C_orth"] = C_orth
            if include_exchange:
                n_E = len(E.basis)
                K_AA = np.zeros((n_act, n_act))
                K_AE = np.zeros((n_act, n_E))
                for b in range(C_orth.shape[1]):
                    psi_b = np.tensordot(C_orth[:, b], chi_E, axes=(0, 0))
                    rho_A = chi_act * psi_b[None, ...]
                    rho_E = chi_E * psi_b[None, ...]
                    rho_all = np.concatenate([rho_A, rho_E], axis=0)
                    K_all = g_act.coulomb_matrix(rho_all, lmax=lmax_exch)
                    K_AA += K_all[:n_act, :n_act]
                    K_AE += K_all[:n_act, n_act:]
                K_AA_tot += K_AA
                ed["K_AE_own"] = K_AE

    if include_exchange:
        h_eff = h_eff - K_AA_tot

    # Add Phillips-Kleinman orthogonality projection + exchange across the joint environment space
    S_AB_blocks = []
    F_AB_blocks = []
    F_BB_diag = []
    for idx_e, ed in enumerate(env_data):
        E = ed["E"]
        R_E = ed["R_E"]
        D_E = ed["D_E"]
        S_orb_inv = ed["S_orb_inv"]
        F_EE = ed["F_EE_es"].copy()
        n_E = len(E.basis)
        C_orth = E.C_occ @ np.linalg.cholesky(S_orb_inv)   # orthonormal occupied orbitals on E
        ed["C_orth"] = C_orth

        if ed["same_center"]:
            chi_E = ed["chi_E_act"]
            tchi_E = np.array([g_act.kinetic_applied(f) for f in E.basis])
            ed["tchi_E_act"] = tchi_E
            S_AE = np.einsum("pijk,qijk,ijk->pq", chi_act, chi_E, w_act)
            T_AE = np.einsum("pijk,qijk,ijk->pq", chi_act, tchi_E, w_act)
            own_vnuc = (-z_a / g_act.r_a) if ed["d_A"] < 1e-10 else (-z_b / g_act.r_b)
            v_other_on_E = v_tot_act_grid - own_vnuc - ed["v_coul_grid"]
            F_EE = F_EE + np.einsum("pijk,qijk,ijk->pq", chi_E, chi_E, w_act * v_other_on_E)
            V_AE = np.einsum("pijk,qijk,ijk->pq", chi_act, chi_E, w_act * v_tot_act_grid)
            F_AE = T_AE + V_AE
            if include_exchange:
                atom_id = 0 if ed["d_A"] < 1e-10 else 1
                idx_own = np.where(act_ints.atom_of == atom_id)[0]
                if len(idx_own) == n_E:
                    F_AE = F_AE - K_AA_tot[:, idx_own]
                    F_EE = F_EE - K_AA_tot[np.ix_(idx_own, idx_own)]
                else:
                    F_AE = F_AE - ed["K_AE_own"]
                    F_EE = F_EE - ed["K_EE"]
        else:
            S_AE = np.zeros((n_act, n_E))
            F_AE = np.zeros((n_act, n_E))
            K_AA = np.zeros((n_act, n_act))
            ia_near = 0 if ed["d_A"] <= ed["d_B"] else 1
            for ia, (R_c, Z_c) in enumerate(zip(act_coords, act_charges)):
                idx_c = np.where(act_ints.atom_of == ia)[0]
                n_c = idx_c.size
                zmin_pair = min(min(basis_act[i].zeta for i in idx_c), min(f.zeta for f in E.basis), 1.0)
                g_pair = ProlateGrid(R_c, R_E, zeta_min=zmin_pair, n_eta=n_eta)
                w_p = g_pair.weights
                pts_p = np.stack([g_pair.X.ravel(), g_pair.Y.ravel(), g_pair.Z.ravel()], axis=1)
                chi_c = np.array([g_pair.evaluate(basis_act[i]) for i in idx_c])
                chi_E = np.array([g_pair.evaluate(f) for f in E.basis])
                tchi_E = np.array([g_pair.kinetic_applied(f) for f in E.basis])
                S_AE[idx_c, :] = np.einsum("pijk,qijk,ijk->pq", chi_c, chi_E, w_p)
                T_cE = np.einsum("pijk,qijk,ijk->pq", chi_c, tchi_E, w_p)
                r_A_p = np.linalg.norm(pts_p - A_xyz[None, :], axis=1)
                r_B_p = np.linalg.norm(pts_p - B_xyz[None, :], axis=1)
                r_E_p = g_pair.r_b.ravel()
                v_p = (-z_a / np.maximum(r_A_p, 1e-12)
                       - z_b / np.maximum(r_B_p, 1e-12)
                       - E.z_nuc / np.maximum(r_E_p, 1e-12)
                       + one_center_coulomb_potential(E.basis, 2.0 * D_E, pts_p))
                V_cE = np.einsum("pijk,qijk,ijk->pq", chi_c, chi_E, w_p * v_p.reshape(g_pair.shape))
                F_cE = T_cE + V_cE
                if include_exchange:
                    K_cc = np.zeros((n_c, n_c))
                    K_cE = np.zeros((n_c, n_E))
                    if ia == ia_near:
                        chi_all_p = np.array([
                            evaluate_real_sto(g_pair.X, g_pair.Y, g_pair.Z, f.center, f.n, f.l, f.m, f.zeta, f.norm())
                            for f in basis_act
                        ])
                    for b in range(C_orth.shape[1]):
                        psi_b = np.tensordot(C_orth[:, b], chi_E, axes=(0, 0))
                        rho_c = chi_c * psi_b[None, ...]
                        rho_E = chi_E * psi_b[None, ...]
                        rho_all = np.concatenate([rho_c, rho_E], axis=0)
                        K_all = g_pair.coulomb_matrix(rho_all, lmax=lmax_exch)
                        K_cc += K_all[:n_c, :n_c]
                        K_cE += K_all[:n_c, n_c:]
                        if ia == ia_near:
                            K_full = g_pair.coulomb_matrix(chi_all_p * psi_b[None, ...], lmax=lmax_exch)
                            idx_o = np.where(act_ints.atom_of == (1 - ia))[0]
                            K_AA[np.ix_(idx_c, idx_o)] += K_full[np.ix_(idx_c, idx_o)]
                            K_AA[np.ix_(idx_o, idx_c)] += K_full[np.ix_(idx_o, idx_c)]
                    K_AA[np.ix_(idx_c, idx_c)] += K_cc
                    F_cE = F_cE - K_cE
                F_AE[idx_c, :] = F_cE
                if ia == ia_near:
                    vnuc_act_on_E = (-z_a / np.maximum(r_A_p, 1e-12)
                                     - z_b / np.maximum(r_B_p, 1e-12)).reshape(g_pair.shape)
                    F_EE = F_EE + np.einsum("pijk,qijk,ijk->pq", chi_E, chi_E, w_p * vnuc_act_on_E)
            if include_exchange:
                h_eff = h_eff - K_AA
                F_EE = F_EE - ed["K_EE"]

        S_AB_blocks.append(S_AE @ C_orth)
        F_AB_blocks.append(F_AE @ C_orth)
        F_BB_diag.append(C_orth.T @ F_EE @ C_orth)

    if S_AB_blocks:
        S_AB_orb = np.hstack(S_AB_blocks)
        F_AB_orb = np.hstack(F_AB_blocks)
        offsets = np.cumsum([0] + [b.shape[1] for b in S_AB_blocks])
        n_tot_orb = int(offsets[-1])
        S_BB_orb = np.eye(n_tot_orb)
        F_BB_orb = np.zeros((n_tot_orb, n_tot_orb))
        for i, fb in enumerate(F_BB_diag):
            sl_i = slice(offsets[i], offsets[i + 1])
            F_BB_orb[sl_i, sl_i] = fb
            for j in range(i):
                sl_j = slice(offsets[j], offsets[j + 1])
                ed_i, ed_j = env_data[i], env_data[j]
                Ci, Cj = ed_i["C_orth"], ed_j["C_orth"]
                if ed_i["same_center"] and ed_j["same_center"]:
                    chi_i, chi_j = ed_i["chi_E_act"], ed_j["chi_E_act"]
                    tchi_j = ed_j["tchi_E_act"]
                    s_ij = Ci.T @ np.einsum("pijk,qijk,ijk->pq", chi_i, chi_j, w_act) @ Cj
                    f_ij_ao = np.einsum("pijk,qijk,ijk->pq", chi_i, tchi_j, w_act) + \
                              np.einsum("pijk,qijk,ijk->pq", chi_i, chi_j, w_act * v_tot_act_grid)
                    if include_exchange:
                        ai = 0 if ed_i["d_A"] < 1e-10 else 1
                        aj = 0 if ed_j["d_A"] < 1e-10 else 1
                        idx_i = np.where(act_ints.atom_of == ai)[0]
                        idx_j = np.where(act_ints.atom_of == aj)[0]
                        if len(idx_i) == len(ed_i["E"].basis) and len(idx_j) == len(ed_j["E"].basis):
                            f_ij_ao = f_ij_ao - K_AA_tot[np.ix_(idx_i, idx_j)]
                    f_ij = Ci.T @ f_ij_ao @ Cj
                    S_BB_orb[sl_i, sl_j] = s_ij
                    S_BB_orb[sl_j, sl_i] = s_ij.T
                    F_BB_orb[sl_i, sl_j] = f_ij
                    F_BB_orb[sl_j, sl_i] = f_ij.T

        S_BB_inv = np.linalg.inv(S_BB_orb)
        P_BA_orb = S_BB_inv @ S_AB_orb.T
        S_eff = S_eff - S_AB_orb @ P_BA_orb
        h_eff = h_eff - F_AB_orb @ P_BA_orb - P_BA_orb.T @ F_AB_orb.T + P_BA_orb.T @ F_BB_orb @ P_BA_orb

    S_eff = 0.5 * (S_eff + S_eff.T)
    ws, vs = np.linalg.eigh(S_eff)
    S_eff = (vs * np.maximum(ws, 0.0)) @ vs.T
    S_eff = 0.5 * (S_eff + S_eff.T)
    h_eff = 0.5 * (h_eff + h_eff.T)
    eri_eff = _renormalize_active_eri(act_ints.S, S_eff, act_ints.eri)
    return EmbeddedAOIntegrals(
        basis=basis_act,
        atom_of=act_ints.atom_of.copy(),
        S=S_eff,
        h=h_eff,
        eri=eri_eff,
        vnn=e_const,
        S_bare=act_ints.S,
        h_bare=act_ints.h,
    )
