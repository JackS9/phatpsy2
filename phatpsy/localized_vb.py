"""
PHATPSY 2.0: Exact-integral localized-orbital valence bond for a bonded pair of atoms
(option B of the two-center plan; exploratory, branch ``two-center-wip``).

Wavefunction (perfect-pairing VB with strictly one-center orbitals)::

    Psi = A[ (inactive core / lone-pair orbitals, doubly occupied)
             x prod_k ( a_k(1) b_k(2) + b_k(1) a_k(2) ) (alpha beta - beta alpha) ]

* every orbital is expanded only in the STOs of its own atom (PHATPSY locality);
* a_k lives on atom A, b_k on atom B (one spin-coupled pair per bond);
* the energy is the exact expectation value of the bare-nucleus Hamiltonian, using
  exact one- and two-center integrals from :mod:`phatpsy.two_center_integrals`
  (no model potential, no Phillips-Kleinman projector, no fitted parameters).

Expanding the pair functions gives 2^n spin-separated determinants with coefficient +1;
matrix elements between nonorthogonal determinants follow the Loewdin rules with
transition density matrices gamma = C_I (M^-1)^T C_J^T,  M = C_I^T S C_J.

The same AO integrals can be handed to PySCF (:func:`pyscf_mean_field`) for RHF / UHF
/ CASSCF references in an identical basis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from phatpsy.two_center_integrals import STO, ProlateGrid, eri_tensor


# ---------------------------------------------------------------------------
# AO integrals
# ---------------------------------------------------------------------------

@dataclass
class AOIntegrals:
    basis: List[STO]
    atom_of: np.ndarray          # atom index (0 = A, 1 = B) of each AO
    S: np.ndarray
    T: np.ndarray
    V_A: np.ndarray              # -Z_A / r_A
    V_B: np.ndarray              # -Z_B / r_B
    eri: np.ndarray              # (pq|rs), chemists' notation
    z: Tuple[float, float]
    R: float
    _Jmat: Optional[np.ndarray] = field(default=None, repr=False)
    _Kmat: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def h(self) -> np.ndarray:
        return self.T + self.V_A + self.V_B

    @property
    def vnn(self) -> float:
        return self.z[0] * self.z[1] / self.R

    @property
    def nao(self) -> int:
        return len(self.basis)

    def jk_matrices(self):
        """Reshaped ERIs so that J(g) = Jmat @ g.ravel(), K(g) = Kmat @ g.ravel()."""
        if self._Jmat is None:
            n = self.nao
            self._Jmat = self.eri.reshape(n * n, n * n)
            # K(g)_{m s} = sum_{n l} (m n | l s) g_{l n}
            self._Kmat = np.ascontiguousarray(self.eri.transpose(0, 3, 2, 1)).reshape(n * n, n * n)
        return self._Jmat, self._Kmat


def atom_basis(center, element: str, basis_name: str) -> List[STO]:
    """STOs of one atom in Fortran-core order (shells; m = -l..l)."""
    from phatpsy.basis import get_element_basis

    out = []
    for f in get_element_basis(element, basis_name).functions:
        for m in range(-f.l, f.l + 1):
            out.append(STO(tuple(float(x) for x in center), f.n, f.l, m, f.zeta))
    return out


def diatomic_ao_integrals(
    basis_a: Sequence[STO], basis_b: Sequence[STO], z_a: float, z_b: float,
    n_eta: int = 64, lmax: int = 30,
) -> AOIntegrals:
    basis = list(basis_a) + list(basis_b)
    A, B = np.array(basis_a[0].center), np.array(basis_b[0].center)
    zmin = min(f.zeta for f in basis)
    g = ProlateGrid(A, B, zeta_min=min(zmin, 1.0), n_eta=n_eta)
    chi = np.array([g.evaluate(f) for f in basis])
    tchi = np.array([g.kinetic_applied(f) for f in basis])
    w = g.weights
    S = np.einsum("pijk,qijk,ijk->pq", chi, chi, w)
    T = np.einsum("pijk,qijk,ijk->pq", chi, tchi, w)
    T = 0.5 * (T + T.T)
    VA = np.einsum("pijk,qijk,ijk->pq", chi, chi, w * (-z_a / g.r_a))
    VB = np.einsum("pijk,qijk,ijk->pq", chi, chi, w * (-z_b / g.r_b))
    eri = eri_tensor(g, chi, lmax=lmax)
    atom_of = np.array([0] * len(basis_a) + [1] * len(basis_b))
    return AOIntegrals(basis, atom_of, S, T, VA, VB, eri, (z_a, z_b), g.R)


# ---------------------------------------------------------------------------
# Nonorthogonal determinant matrix elements
# ---------------------------------------------------------------------------

def _corresponding(CI: np.ndarray, CJ: np.ndarray, S: np.ndarray, tol: float = 1e-10):
    """Loewdin corresponding orbitals for one spin block.

    Returns (sign, s_nonzero, W, Z_list) where W = sum_k cI_k cJ_k^T / s_k over non-zero
    singular values and Z_list holds cI_z cJ_z^T for (numerically) zero singular values.
    """
    n_ao = S.shape[0]
    if CI.shape[1] == 0:
        return 1.0, np.ones(0), np.zeros((n_ao, n_ao)), []
    M = CI.T @ S @ CJ
    U, s, Vt = np.linalg.svd(M)
    sign = float(np.sign(np.linalg.det(U) * np.linalg.det(Vt)))
    cI = CI @ U
    cJ = CJ @ Vt.T
    scale = max(1.0, float(s.max()))
    nz = s > tol * scale
    W = (cI[:, nz] / s[nz]) @ cJ[:, nz].T
    Z = [np.outer(cI[:, k], cJ[:, k]) for k in np.where(~nz)[0]]
    return sign, s[nz], W, Z


def determinant_pair(CaI, CbI, CaJ, CbJ, ints: AOIntegrals) -> Tuple[float, float]:
    """(S_IJ, H_IJ) for spin-separated nonorthogonal determinants I, J (H includes V_nn).

    Generalized Slater-Condon (Loewdin) rules via corresponding orbitals, valid for up
    to two vanishing singular values of the orbital overlap (beyond that H_IJ = 0).
    In transition-density form, with gamma = C_I (M^-1)^T C_J^T per spin,
        H_IJ / S_IJ = sum h gamma_t + 1/2 sum (mn|ls)[gt_mn gt_ls - sum_s g_ms g_ln] + V_nn.
    """
    S = ints.S
    sa, s_a, Wa, Za = _corresponding(CaI, CaJ, S)
    sb, s_b, Wb, Zb = _corresponding(CbI, CbJ, S)
    zeros = [(Z, 0) for Z in Za] + [(Z, 1) for Z in Zb]
    if len(zeros) > 2:
        return 0.0, 0.0
    red = sa * sb * float(np.prod(s_a) * np.prod(s_b))
    Jm, Km = ints.jk_matrices()

    def J(g):
        return (Jm @ g.ravel()).reshape(g.shape)

    def K(g):
        return (Km @ g.ravel()).reshape(g.shape)

    W = (Wa, Wb)
    if not zeros:
        gt = Wa + Wb
        e = float(np.sum(ints.h * gt)) + 0.5 * float(np.sum(J(gt) * gt))
        e -= 0.5 * (float(np.sum(K(Wa) * Wa)) + float(np.sum(K(Wb) * Wb)))
        return red, red * (e + ints.vnn)
    if len(zeros) == 1:
        Z, sz = zeros[0]
        e = float(np.sum(ints.h * Z)) + float(np.sum(J(Wa + Wb) * Z)) - float(np.sum(K(W[sz]) * Z))
        return 0.0, red * e
    (Z1, s1), (Z2, s2) = zeros
    e = float(np.sum(J(Z2) * Z1))
    if s1 == s2:
        e -= float(np.sum(K(Z2) * Z1))
    return 0.0, red * e


# ---------------------------------------------------------------------------
# Perfect-pairing localized VB energy
# ---------------------------------------------------------------------------

def pp_determinants(inactive: np.ndarray, pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
                    ionic: bool = False):
    """Spin-separated determinants (Ca, Cb, coefficient) of the perfect-pairing function.

    inactive: (nao, n_in) doubly occupied orbitals.  pairs: list of (a, b) AO vectors.
    Covalent pair function a b + b a  ->  dets (alpha a, beta b) and (alpha b, beta a).
    """
    dets = []
    for choice in product((0, 1), repeat=len(pairs)):
        ca = [inactive[:, i] for i in range(inactive.shape[1])]
        cb = list(ca)
        for (a, b), c in zip(pairs, choice):
            if c == 0:
                ca.append(a); cb.append(b)
            else:
                ca.append(b); cb.append(a)
        dets.append((np.array(ca).T, np.array(cb).T, 1.0))
    return dets


def expectation(dets, ints: AOIntegrals) -> Tuple[float, np.ndarray, np.ndarray]:
    n = len(dets)
    Smat = np.zeros((n, n))
    Hmat = np.zeros((n, n))
    for I in range(n):
        for J in range(I, n):
            s, h = determinant_pair(dets[I][0], dets[I][1], dets[J][0], dets[J][1], ints)
            Smat[I, J] = Smat[J, I] = s
            Hmat[I, J] = Hmat[J, I] = h
    c = np.array([d[2] for d in dets])
    return float(c @ Hmat @ c / (c @ Smat @ c)), Smat, Hmat


def pp_energy(inactive, pairs, ints: AOIntegrals) -> float:
    return expectation(pp_determinants(inactive, pairs), ints)[0]


def covalent_determinants(inactive: np.ndarray, active: Sequence[np.ndarray]):
    """All Sz = 0 determinants with every active orbital singly occupied (the full
    covalent spin-coupling space; its lowest root is the spin-coupled VB energy for
    these orbitals)."""
    from itertools import combinations

    n = len(active)
    ina = [inactive[:, i] for i in range(inactive.shape[1])]
    dets = []
    for alpha in combinations(range(n), n // 2):
        beta = [k for k in range(n) if k not in alpha]
        ca = ina + [active[k] for k in alpha]
        cb = ina + [active[k] for k in beta]
        dets.append((np.array(ca).T, np.array(cb).T, 1.0))
    return dets


def generalized_ci_energy(dets, ints: AOIntegrals) -> Tuple[float, np.ndarray]:
    _, Smat, Hmat = expectation(dets, ints)
    w, v = np.linalg.eigh(Smat)
    keep = w > 1e-10 * w.max()
    X = v[:, keep] / np.sqrt(w[keep])
    e, u = np.linalg.eigh(X.T @ Hmat @ X)
    return float(e[0]), X @ u[:, 0]


def _sym_orth_cols(C: np.ndarray, S: np.ndarray) -> np.ndarray:
    M = C.T @ S @ C
    w, v = np.linalg.eigh(M)
    return C @ (v @ np.diag(w ** -0.5) @ v.T)


def cas_energy(inactive: np.ndarray, active: Sequence[np.ndarray], ints: AOIntegrals,
               nelec_act: Optional[Tuple[int, int]] = None) -> float:
    """Full CI in the span of the one-center active orbitals (covalent + all ionic
    structures), with doubly occupied one-center inactive orbitals.  Evaluated in a
    Loewdin-orthonormalized basis (the energy is invariant to that choice)."""
    from pyscf import ao2mo, fci

    S, h = ints.S, ints.h
    Ci = _sym_orth_cols(inactive, S) if inactive.shape[1] else inactive
    Ca = np.array(active).T
    if Ci.shape[1]:
        Ca = Ca - Ci @ (Ci.T @ S @ Ca)
    Ca = _sym_orth_cols(Ca, S)
    nact = Ca.shape[1]
    if nelec_act is None:
        nelec_act = (nact // 2, nact - nact // 2)
    Jm, Km = ints.jk_matrices()
    if Ci.shape[1]:
        Dc = Ci @ Ci.T
        Jc = (Jm @ Dc.ravel()).reshape(Dc.shape)
        Kc = (Km @ Dc.ravel()).reshape(Dc.shape)
        fock_core = h + 2 * Jc - Kc
        e_core = float(np.sum(Dc * (h + fock_core)))
    else:
        fock_core = h
        e_core = 0.0
    h1 = Ca.T @ fock_core @ Ca
    eri = ao2mo.incore.full(ao2mo.restore(8, ints.eri, ints.nao), Ca)
    e, _ = fci.direct_spin1.kernel(h1, eri, nact, nelec_act, ecore=e_core + ints.vnn, conv_tol=1e-12)
    return float(e)


def orthonormal_mo_guess(Xi: np.ndarray, Xa: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Orthonormal [inactive | active | virtual] MO matrix spanning the same inactive and
    active subspaces as the one-center orbitals (used to seed delocalized CASSCF so that
    its energy is guaranteed not to exceed the localized one)."""
    n = S.shape[0]
    Ci = _sym_orth_cols(Xi, S) if Xi.shape[1] else Xi
    Ya = Xa - Ci @ (Ci.T @ S @ Xa) if Ci.shape[1] else Xa
    Ca = _sym_orth_cols(Ya, S)
    occ = np.hstack([Ci, Ca])
    P = np.eye(n) - occ @ occ.T @ S
    w, v = np.linalg.eigh(P.T @ S @ P)
    Cv = P @ v[:, w > 1e-8]
    Cv = _sym_orth_cols(Cv, S)
    return np.hstack([occ, Cv])


def cas_energy_and_gradient(Xi: np.ndarray, Xa: np.ndarray, ints: AOIntegrals,
                            nelec_act: Optional[Tuple[int, int]] = None):
    """CI-in-active-span energy and its analytic gradient with respect to the raw
    (nonorthogonal, unnormalized) inactive and active orbital coefficients.

    Construction:
        D_c = X_i (X_i^T S X_i)^-1 X_i^T                  inactive projector density
        Y   = (1 - D_c S) X_a ,  C_a = Y (Y^T S Y)^-1/2   active orbitals
        E   = sum D_c (h + F_c) + sum_pq (C_a^T F_c C_a)_pq g_pq + 1/2 sum (pq|rs) G_pqrs + V_nn
    with F_c = h + 2J(D_c) - K(D_c) and (g, G) the FCI 1-/2-RDMs (Hellmann-Feynman: the
    CI vector is variational, so RDMs are held fixed).  E depends only on span(X_i) and
    span(X_i, X_a), which gives the projected gradients below.
    Returns (E, dE/dX_i, dE/dX_a).
    """
    from pyscf import ao2mo, fci

    S, h = ints.S, ints.h
    n = ints.nao
    Jm, Km = ints.jk_matrices()
    J = lambda g: (Jm @ g.ravel()).reshape(g.shape)
    K = lambda g: (Km @ g.ravel()).reshape(g.shape)
    I = np.eye(n)

    if Xi.shape[1]:
        Ni = np.linalg.inv(Xi.T @ S @ Xi)
        Dc = Xi @ Ni @ Xi.T
    else:
        Ni = np.zeros((0, 0))
        Dc = np.zeros((n, n))
    Fc = h + 2 * J(Dc) - K(Dc)
    e_core = float(np.sum(Dc * (h + Fc)))

    Y = Xa - Dc @ S @ Xa
    w, v = np.linalg.eigh(Y.T @ S @ Y)
    T = v @ np.diag(w ** -0.5) @ v.T
    Ca = Y @ T
    nact = Ca.shape[1]
    if nelec_act is None:
        nelec_act = (nact // 2, nact - nact // 2)
    h1 = Ca.T @ Fc @ Ca
    eri_act = ao2mo.incore.full(ao2mo.restore(8, ints.eri, n), Ca)
    e, civec = fci.direct_spin1.kernel(h1, eri_act, nact, nelec_act, ecore=e_core + ints.vnn, conv_tol=1e-12)
    dm1, dm2 = fci.direct_spin1.make_rdm12(civec, nact, nelec_act)

    # dE/dC_a (unconstrained, RDMs fixed)
    t = np.einsum("mnls,nq->mqls", ints.eri, Ca, optimize=True)
    t = np.einsum("mqls,lr->mqrs", t, Ca, optimize=True)
    t = np.einsum("mqrs,st->mqrt", t, Ca, optimize=True)
    Q = np.einsum("mqrt,pqrt->mp", t, dm2, optimize=True)
    G_Ca = 2 * Fc @ Ca @ dm1 + 2 * Q
    # through Loewdin (span-only dependence) and the inactive projection
    G_Y = (I - S @ Ca @ Ca.T) @ G_Ca @ T
    G_Xa = (I - S @ Dc) @ G_Y
    # dE/dD_c: explicit core terms + active energy through F_c + dependence of Y on D_c
    Pa = Ca @ dm1 @ Ca.T
    A = 2 * Fc + 2 * J(Pa) - K(Pa)
    A -= 0.5 * (S @ Xa @ G_Y.T + G_Y @ Xa.T @ S)
    G_Xi = 2 * (I - S @ Dc) @ A @ Xi @ Ni if Xi.shape[1] else np.zeros_like(Xi)
    return float(e), G_Xi, G_Xa


def structure_ci_energy(inactive, pairs, ints: AOIntegrals) -> Tuple[float, np.ndarray]:
    """Variational mixing of the 2^n covalent determinant products is NOT used; instead
    this returns the lowest eigenvalue over the space spanned by the PP determinants
    (a cheap upper bound on full-valence spin-coupling within the same orbitals)."""
    dets = pp_determinants(inactive, pairs)
    _, Smat, Hmat = expectation(dets, ints)
    w, v = np.linalg.eigh(Smat)
    keep = w > 1e-10 * w.max()
    X = v[:, keep] / np.sqrt(w[keep])
    e, u = np.linalg.eigh(X.T @ Hmat @ X)
    return float(e[0]), X @ u[:, 0]


# ---------------------------------------------------------------------------
# One-center, symmetry-adapted orbital parametrization
# ---------------------------------------------------------------------------

def symmetry_mask(ints: AOIntegrals, atom: int, sym: str) -> np.ndarray:
    """AO indices on ``atom`` compatible with sigma / pi_x / pi_y symmetry about the
    bond axis (bond along z).  m = 0 -> sigma, m = +1 -> pi_x, m = -1 -> pi_y."""
    msel = {"sigma": 0, "pi_x": 1, "pi_y": -1}[sym]
    return np.array([i for i, f in enumerate(ints.basis) if ints.atom_of[i] == atom and f.m == msel])


@dataclass
class OrbitalSpec:
    atom: int
    sym: str
    role: str                    # "inactive" or "pair-a" / "pair-b"
    pair: int = -1


class LocalizedVBModel:
    """Maps a flat parameter vector to one-center orbitals and evaluates the PP energy.

    If ``mirror`` is True (homonuclear, bond along z, A at lower z), atom-B orbitals are
    the z -> -z mirror images of atom-A orbitals, halving the parameter count.
    """

    def __init__(self, ints: AOIntegrals, specs: Sequence[OrbitalSpec], mirror: bool = False,
                 method: str = "pp"):
        self.ints = ints
        self.specs = list(specs)
        self.mirror = mirror
        self.method = method
        self.masks = [symmetry_mask(ints, s.atom, s.sym) for s in self.specs]
        self.free = [i for i, s in enumerate(self.specs) if not (mirror and s.atom == 1)]
        self.offsets = np.cumsum([0] + [len(self.masks[i]) for i in self.free])
        if mirror:
            nA = int(np.sum(ints.atom_of == 0))
            self.parity = np.array([(-1) ** (f.l + f.m) for f in ints.basis[:nA]], dtype=float)
            self.partner = {}
            for i, s in enumerate(self.specs):
                if s.atom == 1:
                    j = next(j for j, t in enumerate(self.specs)
                             if t.atom == 0 and t.sym == s.sym and t.role.split("-")[0] == s.role.split("-")[0]
                             and (t.pair == s.pair))
                    self.partner[i] = j

    @property
    def nparam(self) -> int:
        return int(self.offsets[-1])

    def orbitals(self, x: np.ndarray) -> List[np.ndarray]:
        n = self.ints.nao
        orbs: List[Optional[np.ndarray]] = [None] * len(self.specs)
        for k, i in enumerate(self.free):
            v = np.zeros(n)
            v[self.masks[i]] = x[self.offsets[k]:self.offsets[k + 1]]
            orbs[i] = v
        if self.mirror:
            nA = int(np.sum(self.ints.atom_of == 0))
            for i, j in self.partner.items():
                v = np.zeros(n)
                v[nA:] = self.parity * orbs[j][:nA]
                orbs[i] = v
        return orbs

    def split(self, orbs):
        inactive = np.array([o for o, s in zip(orbs, self.specs) if s.role == "inactive"]).T
        npair = 1 + max([s.pair for s in self.specs] + [-1])
        pairs = []
        for k in range(npair):
            a = next(o for o, s in zip(orbs, self.specs) if s.role == "pair-a" and s.pair == k)
            b = next(o for o, s in zip(orbs, self.specs) if s.role == "pair-b" and s.pair == k)
            pairs.append((a, b))
        if inactive.size == 0:
            inactive = np.zeros((self.ints.nao, 0))
        return inactive, pairs

    def energy(self, x: np.ndarray) -> float:
        inactive, pairs = self.split(self.orbitals(x))
        if self.method == "pp":
            return pp_energy(inactive, pairs, self.ints)
        active = [a for a, _ in pairs] + [b for _, b in pairs]
        if self.method == "covalent":
            return generalized_ci_energy(covalent_determinants(inactive, active), self.ints)[0]
        if self.method == "cas":
            return cas_energy(inactive, active, self.ints)
        raise ValueError(self.method)

    def pack(self, orbs) -> np.ndarray:
        return np.concatenate([orbs[i][self.masks[i]] for i in self.free])

    def energy_and_grad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Analytic gradient (method 'cas' only)."""
        if self.method != "cas":
            raise NotImplementedError("analytic gradient implemented for method='cas'")
        orbs = self.orbitals(x)
        inactive_idx = [i for i, s in enumerate(self.specs) if s.role == "inactive"]
        npair = 1 + max([s.pair for s in self.specs] + [-1])
        a_idx = [next(i for i, s in enumerate(self.specs) if s.role == "pair-a" and s.pair == k) for k in range(npair)]
        b_idx = [next(i for i, s in enumerate(self.specs) if s.role == "pair-b" and s.pair == k) for k in range(npair)]
        act_idx = a_idx + b_idx
        Xi = np.array([orbs[i] for i in inactive_idx]).T if inactive_idx else np.zeros((self.ints.nao, 0))
        Xa = np.array([orbs[i] for i in act_idx]).T
        e, Gi, Ga = cas_energy_and_gradient(Xi, Xa, self.ints)
        gorb = {}
        for c, i in enumerate(inactive_idx):
            gorb[i] = Gi[:, c]
        for c, i in enumerate(act_idx):
            gorb[i] = Ga[:, c]
        if self.mirror:
            nA = int(np.sum(self.ints.atom_of == 0))
            for i, j in self.partner.items():
                g = np.zeros(self.ints.nao)
                g[:nA] = self.parity * gorb[i][nA:]
                gorb[j] = gorb[j] + g
        grad = np.concatenate([gorb[i][self.masks[i]] for i in self.free])
        return e, grad


def optimize(model: LocalizedVBModel, x0: np.ndarray, maxiter: int = 400, gtol: float = 1e-6, verbose=False,
             analytic: Optional[bool] = None):
    """BFGS over one-center orbital parameters.  Uses the analytic gradient for
    method='cas' (unless analytic=False), central finite differences otherwise."""
    from scipy.optimize import minimize

    if analytic is None:
        analytic = model.method == "cas"
    if analytic:
        return minimize(model.energy_and_grad, x0, jac=True, method="BFGS",
                        options={"maxiter": maxiter, "gtol": gtol, "disp": verbose})
    return _optimize_fd(model, x0, maxiter, gtol, verbose)


def _optimize_fd(model: LocalizedVBModel, x0: np.ndarray, maxiter: int = 400, gtol: float = 1e-6, verbose=False):
    from scipy.optimize import minimize

    def f(x):
        return model.energy(x)

    def grad(x, h=1e-6):
        g = np.zeros_like(x)
        for i in range(x.size):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (f(xp) - f(xm)) / (2 * h)
        return g

    res = minimize(f, x0, jac=grad, method="BFGS", options={"maxiter": maxiter, "gtol": gtol, "disp": verbose})
    return res


# ---------------------------------------------------------------------------
# PySCF references in the identical exact STO basis
# ---------------------------------------------------------------------------

def pyscf_mean_field(ints: AOIntegrals, nelec: Tuple[int, int], kind: str = "RHF", h: Optional[np.ndarray] = None,
                     dm0=None):
    from pyscf import ao2mo, gto, scf

    mol = gto.M(verbose=0)
    mol.nelectron = int(sum(nelec))
    mol.spin = int(nelec[0] - nelec[1])
    mol.incore_anyway = True
    mol.nao_nr = lambda *a, **k: ints.nao
    mol.energy_nuc = lambda *a, **k: ints.vnn if h is None else 0.0
    hh = ints.h if h is None else h
    mf = {"RHF": scf.RHF, "UHF": scf.UHF, "ROHF": scf.ROHF}[kind](mol)
    mf.get_hcore = lambda *a: hh
    mf.get_ovlp = lambda *a: ints.S
    mf._eri = ao2mo.restore(8, ints.eri, ints.nao)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.kernel(dm0=dm0)
    return mf
