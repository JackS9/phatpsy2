"""
PHATPSY 2.0: Two-Center STO Integral Engine (prolate spheroidal quadrature).

Exploratory prototype (branch ``two-center-wip``, Stage 1 of the two-center
resonance plan).  Provides one- and two-electron integrals between real Slater-type
orbitals (STOs) located on two centers A and B, which the single-center PHATPSY
energy functional does not currently evaluate:

* one-electron:  <a|b>, <a|T|b>, <a|-1/r_A|b>, <a|-1/r_B|b>  (a, b on A and/or B)
* two-electron:  (rho_1 | rho_2) for arbitrary densities built from STOs on A and B,
  e.g. Coulomb (aa|bb), exchange (ab|ab), hybrid (aa|ab), one-center (aa|aa).

Method
------
Prolate spheroidal coordinates with foci at A and B (A at z=-R/2, B at z=+R/2 in the
bond frame)::

    xi  = (r_A + r_B) / R  in [1, inf),   eta = (r_A - r_B) / R  in [-1, 1],   phi
    dV  = (R/2)^3 (xi^2 - eta^2) dxi deta dphi

* xi:  composite Gauss-Legendre panels in u = xi - 1, geometrically graded toward
  u = 0 (spectral cumulative integration inside each panel);
* eta: Gauss-Legendre;  phi: trapezoid (exact for trigonometric polynomials).

Two-electron integrals use the Neumann expansion of 1/r12 in prolate spheroidal
coordinates (Neumann 1878; Ruedenberg, J. Chem. Phys. 19, 1459 (1951)):

    1/r12 = (2/R) sum_{L,M} s_M (2L+1) eps_M [(L-M)!/(L+M)!]^2
            P_L^M(xi_<) Q_L^M(xi_>) P_L^M(eta_1) P_L^M(eta_2) cos M(phi_1 - phi_2)

The sign convention s_M is fixed by the self-test ``neumann_selftest``, which compares
the truncated series against 1/|r1 - r2| directly.

STO conventions (radial normalization, real-harmonic ordering m = 0, +1(x), -1(y), ...,
Condon-Shortley phase (-1)^|m|) are those of :func:`phatpsy.volumetric.evaluate_real_sto`,
which match the Fortran core, so PHATPSY ``basis_info`` MO coefficients can be used
directly.  All quantities are in atomic units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import legendre as npleg

from phatpsy.volumetric import evaluate_real_sto


# ---------------------------------------------------------------------------
# Real STO description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class STO:
    """A real Slater-type orbital chi_{n l m}(r - center) with exponent zeta."""

    center: Tuple[float, float, float]
    n: int
    l: int
    m: int
    zeta: float
    anorm: Optional[float] = None

    def norm(self) -> float:
        if self.anorm is not None and self.anorm > 0.0:
            return float(self.anorm)
        return (2.0 * self.zeta) ** (self.n + 0.5) / math.sqrt(math.factorial(2 * self.n))


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------

def _panel_cumulative_matrix(order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on [-1, 1] and the spectral cumulative-integration
    matrix C with  int_{-1}^{t_i} g(t) dt  ~=  sum_j C[i, j] g(t_j)."""
    t, w = npleg.leggauss(order)
    V = npleg.legvander(t, order - 1)          # V[i, k] = P_k(t_i)
    Vinv = np.linalg.inv(V)                    # l_j(t) = sum_k Vinv[k, j] P_k(t)
    # Antiderivatives of P_k from -1:  k=0 -> t+1 ;  k>=1 -> (P_{k+1} - P_{k-1})/(2k+1)
    Pk = npleg.legvander(t, order)             # P_0..P_order at nodes
    IP = np.zeros((order, order))
    IP[:, 0] = t + 1.0
    for k in range(1, order):
        IP[:, k] = (Pk[:, k + 1] - Pk[:, k - 1]) / (2 * k + 1)
    C = IP @ Vinv
    return t, w, C


def _assoc_legendre_table(x: np.ndarray, lmax: int, mmax: int, outer: bool) -> np.ndarray:
    """P_L^M(x) without Condon-Shortley phase, shape (lmax+1, mmax+1, len(x)).

    outer=False: Ferrers functions (1-x^2)^{M/2} d^M P_L/dx^M  for |x| <= 1
    outer=True : (x^2-1)^{M/2} d^M P_L/dx^M  for x >= 1
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((lmax + 1, mmax + 1, x.size))
    fac = np.abs(1.0 - x * x) if not outer else (x * x - 1.0)
    fac = np.sqrt(np.maximum(fac, 0.0))
    for L in range(lmax + 1):
        c = np.zeros(L + 1)
        c[L] = 1.0
        for M in range(min(L, mmax) + 1):
            d = npleg.legder(c, M) if M > 0 else c
            out[L, M] = npleg.legval(x, d) * fac ** M
    return out


def _legendre_q_table(xi: np.ndarray, lmax: int, mmax: int) -> np.ndarray:
    """Q_L^M(xi) for xi > 1 (type-3, no Condon-Shortley phase):
    Q_L^M = (xi^2-1)^{M/2} d^M Q_L / dxi^M.  Shape (lmax+1, mmax+1, len(xi)).

    Q_L (M=0): upward recurrence for xi close to 1 (where it is stable), Miller's
    downward recurrence elsewhere, normalized to Q_0 = artanh(1/xi).
    M > 0:     Q_L^{M+1} = (xi^2-1)^{-1/2} [ (L-M) xi Q_L^M - (L+M) Q_{L-1}^M ].
    """
    xi = np.asarray(xi, dtype=np.float64)
    nx = xi.size
    Q = np.zeros((lmax + 1, mmax + 1, nx))
    q0 = np.arctanh(1.0 / xi)

    near = xi < 1.01
    # --- upward recurrence near xi = 1 ---
    if np.any(near):
        x = xi[near]
        qm1 = q0[near]
        Q[0, 0, near] = qm1
        if lmax >= 1:
            qc = x * qm1 - 1.0
            Q[1, 0, near] = qc
            for L in range(1, lmax):
                qn = ((2 * L + 1) * x * qc - L * qm1) / (L + 1)
                Q[L + 1, 0, near] = qn
                qm1, qc = qc, qn
    # --- Miller downward recurrence ---
    far = ~near
    if np.any(far):
        x = xi[far]
        ntop = lmax + 400
        qp1 = np.zeros_like(x)
        qc = np.full_like(x, 1e-280)
        vals = np.zeros((lmax + 1, x.size))
        for n in range(ntop, 0, -1):
            # Q_{n-1} = [(2n+1) x Q_n - (n+1) Q_{n+1}] / n
            qm1 = ((2 * n + 1) * x * qc - (n + 1) * qp1) / n
            if n - 1 <= lmax:
                vals[n - 1] = qm1
            if n <= lmax:
                vals[n] = qc
            qp1, qc = qc, qm1
            big = np.abs(qc) > 1e250
            if np.any(big):
                s = np.where(big, 1e-250, 1.0)
                qc = qc * s
                qp1 = qp1 * s
                vals = vals * s
        vals = vals * (q0[far] / vals[0])
        Q[:, 0, far] = vals
    # --- M recurrence ---
    root = np.sqrt(xi * xi - 1.0)
    for M in range(mmax):
        for L in range(M + 1, lmax + 1):
            Q[L, M + 1] = ((L - M) * xi * Q[L, M] - (L + M) * Q[L - 1, M]) / root
    return Q


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

class ProlateGrid:
    """Prolate spheroidal product grid for a pair of centers A and B."""

    def __init__(
        self,
        center_a: Sequence[float],
        center_b: Sequence[float],
        zeta_min: float = 0.5,
        n_eta: int = 48,
        n_phi: int = 16,
        panel_order: int = 16,
    ) -> None:
        self.A = np.asarray(center_a, dtype=np.float64)
        self.B = np.asarray(center_b, dtype=np.float64)
        d = self.B - self.A
        self.R = float(np.linalg.norm(d))
        if self.R < 1e-10:
            raise ValueError("ProlateGrid requires two distinct centers.")
        self.mid = 0.5 * (self.A + self.B)

        # Bond frame: e_z along A->B
        ez = d / self.R
        trial = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ex = trial - np.dot(trial, ez) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        self.rot = np.column_stack([ex, ey, ez])

        # --- xi panels in u = xi - 1 ---
        u_max = max(40.0 / (max(zeta_min, 1e-3) * self.R), 8.0)
        breaks = [0.0] + [10.0 ** k for k in range(-10, 0)] + [0.3, 1.0]
        b = 1.0
        while b < u_max:
            b *= 2.0
            breaks.append(b)
        breaks = np.asarray(breaks)
        t, w, C = _panel_cumulative_matrix(panel_order)
        u_nodes, u_w, blocks = [], [], []
        for a0, a1 in zip(breaks[:-1], breaks[1:]):
            h = 0.5 * (a1 - a0)
            u_nodes.append(a0 + h * (t + 1.0))
            u_w.append(h * w)
            blocks.append(h * C)
        self.u = np.concatenate(u_nodes)
        self.w_xi = np.concatenate(u_w)
        self.xi = 1.0 + self.u
        nxi = self.xi.size
        # Global cumulative matrices:
        #   (Ccum @ g)[i] = int_1^{xi_i} g dxi      (Crev @ g)[i] = int_{xi_i}^inf g dxi
        # Crev is assembled from the tail so that int_xi^inf is never formed as a
        # difference of two large numbers (P_L(xi) grows like xi^L at large L).
        all_w = np.concatenate(u_w)
        Ccum = np.zeros((nxi, nxi))
        Crev = np.zeros((nxi, nxi))
        off = 0
        for blk, pw in zip(blocks, u_w):
            k = blk.shape[0]
            Ccum[off:off + k, :off] = all_w[:off][None, :]
            Ccum[off:off + k, off:off + k] = blk
            Crev[off:off + k, off + k:] = all_w[off + k:][None, :]
            Crev[off:off + k, off:off + k] = pw[None, :] - blk
            off += k
        self.Ccum = Ccum
        self.Crev = Crev
        # Panel data for cumulative integrals to arbitrary xi (potentials_at)
        self.breaks = breaks
        self.panel_order = panel_order
        self._gl_t, self._gl_w = t, w
        self._lag_inv = np.linalg.inv(npleg.legvander(t, panel_order - 1))

        # --- eta, phi ---
        self.eta, self.w_eta = npleg.leggauss(n_eta)
        self.n_phi = n_phi
        self.phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
        self.w_phi = np.full(n_phi, 2.0 * np.pi / n_phi)

        XI, ETA, PHI = np.meshgrid(self.xi, self.eta, self.phi, indexing="ij")
        h = 0.5 * self.R
        rho = h * np.sqrt(np.maximum((XI * XI - 1.0) * (1.0 - ETA * ETA), 0.0))
        xb = rho * np.cos(PHI)
        yb = rho * np.sin(PHI)
        zb = h * XI * ETA
        pts = self.rot @ np.vstack([xb.ravel(), yb.ravel(), zb.ravel()])
        self.X = (pts[0] + self.mid[0]).reshape(XI.shape)
        self.Y = (pts[1] + self.mid[1]).reshape(XI.shape)
        self.Z = (pts[2] + self.mid[2]).reshape(XI.shape)
        self.r_a = h * (XI + ETA)
        self.r_b = h * (XI - ETA)
        self.jac = h ** 3 * (XI * XI - ETA * ETA)
        self.weights = (
            self.w_xi[:, None, None] * self.w_eta[None, :, None] * self.w_phi[None, None, :]
        ) * self.jac
        self._XI, self._ETA = XI, ETA
        self._neumann_cache: dict = {}

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.X.shape

    # ------------------------------------------------------------------
    # Basis function evaluation
    # ------------------------------------------------------------------
    def evaluate(self, f: STO) -> np.ndarray:
        """Real STO on the grid in the Fortran-core (Condon-Shortley) convention.

        ``evaluate_real_sto`` already applies the phase (-1)^|m| used by the Fortran
        two-center kernels; checked against ``gensab``/``rotsab`` for generic off-axis
        O(DZP)--O(DZP) and O(DZP)--H(DZP) pairs (tests/test_two_center_integrals.py).
        """
        return evaluate_real_sto(self.X, self.Y, self.Z, f.center, f.n, f.l, f.m, f.zeta, f.norm())

    def kinetic_applied(self, f: STO) -> np.ndarray:
        """(T f)(r) = -1/2 nabla^2 f, using
        nabla^2 chi_nlm = [ (n(n-1) - l(l+1))/r^2 - 2 n zeta / r + zeta^2 ] chi_nlm."""
        chi = self.evaluate(f)
        r = np.sqrt((self.X - f.center[0]) ** 2 + (self.Y - f.center[1]) ** 2 + (self.Z - f.center[2]) ** 2)
        c2 = f.n * (f.n - 1) - f.l * (f.l + 1)
        lap = (c2 / (r * r) if c2 != 0 else 0.0) - 2.0 * f.n * f.zeta / r + f.zeta ** 2
        return -0.5 * lap * chi

    def integrate(self, g: np.ndarray) -> float:
        return float(np.sum(g * self.weights))

    # ------------------------------------------------------------------
    # Coulomb (Neumann expansion)
    # ------------------------------------------------------------------
    def _neumann_tables(self, lmax: int, mmax: int):
        key = (lmax, mmax)
        if key not in self._neumann_cache:
            Peta = _assoc_legendre_table(self.eta, lmax, mmax, outer=False)
            Pxi = _assoc_legendre_table(self.xi, lmax, mmax, outer=True)
            Qxi = _legendre_q_table(self.xi, lmax, mmax)
            self._neumann_cache[key] = (Peta, Pxi, Qxi)
        return self._neumann_cache[key]

    # Sign factor s_M of the Neumann expansion for our P/Q conventions (see neumann_selftest)
    NEUMANN_SIGN = staticmethod(lambda M: (-1.0) ** M)

    def _coef(self, L: int, M: int) -> float:
        eps = 1.0 if M == 0 else 2.0
        ratio = math.exp(math.lgamma(L - M + 1) - math.lgamma(L + M + 1))
        return (2.0 / self.R) * self.NEUMANN_SIGN(M) * (2 * L + 1) * eps * ratio * ratio

    def _project(self, rho: np.ndarray, Peta: np.ndarray, lmax: int, mmax: int):
        """f[L, M, c](xi) = int deta dphi (xi^2-eta^2)(R/2)^3 rho P_L^M(eta) T_M(phi)."""
        h3 = (0.5 * self.R) ** 3
        g = rho * (self._XI ** 2 - self._ETA ** 2) * h3      # (nxi, neta, nphi)
        cosm = np.cos(np.outer(np.arange(mmax + 1), self.phi)) * self.w_phi[None, :]
        sinm = np.sin(np.outer(np.arange(mmax + 1), self.phi)) * self.w_phi[None, :]
        gc = np.einsum("ijk,mk->ijm", g, cosm)              # (nxi, neta, M)
        gs = np.einsum("ijk,mk->ijm", g, sinm)
        we = self.w_eta
        fc = np.einsum("ijm,lmj,j->lmi", gc, Peta, we)      # (L, M, nxi)
        fs = np.einsum("ijm,lmj,j->lmi", gs, Peta, we)
        return fc, fs

    def coulomb(self, rho1: np.ndarray, rho2: np.ndarray, lmax: int = 30, mmax: int = 4) -> float:
        """(rho1 | rho2) = int int rho1(1) rho2(2) / r12 via the Neumann expansion."""
        mmax = min(mmax, lmax, self.n_phi // 2 - 1)
        Peta, Pxi, Qxi = self._neumann_tables(lmax, mmax)
        f1c, f1s = self._project(rho1, Peta, lmax, mmax)
        f2c, f2s = self._project(rho2, Peta, lmax, mmax)
        total = 0.0
        wx = self.w_xi
        for L in range(lmax + 1):
            for M in range(min(L, mmax) + 1):
                P, Q = Pxi[L, M], Qxi[L, M]
                c = self._coef(L, M)
                for f1, f2 in ((f1c[L, M], f2c[L, M]), (f1s[L, M], f2s[L, M])):
                    if not np.any(f1) or not np.any(f2):
                        continue
                    A = self.Ccum @ (f2 * P)                # int_1^{xi} f2 P
                    B = self.Crev @ (f2 * Q)                # int_xi^inf f2 Q
                    total += c * np.dot(wx, f1 * (Q * A + P * B))
        return float(total)

    def coulomb_matrix(self, densities: np.ndarray, lmax: int = 30, mmax: int = 4, chunk: int = 32) -> np.ndarray:
        """Batched Coulomb integrals G[p, q] = (rho_p | rho_q) for a stack of densities
        (shape (n, *grid.shape)).  Same Neumann algorithm as :meth:`coulomb`, vectorized
        over densities."""
        mmax = min(mmax, lmax, self.n_phi // 2 - 1)
        Peta, Pxi, Qxi = self._neumann_tables(lmax, mmax)
        n = densities.shape[0]
        fc = np.zeros((lmax + 1, mmax + 1, n, self.xi.size))
        fs = np.zeros_like(fc)
        for s in range(0, n, chunk):
            for k in range(s, min(n, s + chunk)):
                c, si = self._project(densities[k], Peta, lmax, mmax)
                fc[:, :, k] = c
                fs[:, :, k] = si
        G = np.zeros((n, n))
        wx = self.w_xi
        CcT, CrT = self.Ccum.T, self.Crev.T
        for L in range(lmax + 1):
            for M in range(min(L, mmax) + 1):
                P, Q = Pxi[L, M], Qxi[L, M]
                c = self._coef(L, M)
                for F in (fc[L, M], fs[L, M]):
                    if not np.any(F):
                        continue
                    A = (F * P) @ CcT          # (n, nxi): int_1^xi f P
                    B = (F * Q) @ CrT          # int_xi^inf f Q
                    G += c * (F * wx) @ (Q * A + P * B).T
        return 0.5 * (G + G.T)

    def frame_coordinates(self, points: np.ndarray):
        """(xi, eta, phi) of lab-frame points (shape (npts, 3)) in this grid's frame."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        ra = np.linalg.norm(p - self.A, axis=1)
        rb = np.linalg.norm(p - self.B, axis=1)
        xi = np.maximum((ra + rb) / self.R, 1.0 + 1e-14)
        eta = np.clip((ra - rb) / self.R, -1.0, 1.0)
        local = (p - self.mid) @ self.rot
        phi = np.arctan2(local[:, 1], local[:, 0])
        return xi, eta, phi

    def potentials_at(self, densities: np.ndarray, points: np.ndarray, lmax: int = 24, mmax: int = 4,
                      chunk: int = 4096) -> np.ndarray:
        """Electrostatic potentials V_k(r) = int rho_k(r') / |r - r'| dr' of densities given on
        this grid, evaluated at arbitrary lab points, from the same Neumann expansion used
        by :meth:`coulomb` (exact up to the angular truncation and xi quadrature):

            V(r) = sum_LMT c_LM [Q_L^M(xi) A_LMT(xi) + P_L^M(xi) B_LMT(xi)] P_L^M(eta) T_M(phi),
            A = int_1^xi f P dxi',   B = int_xi^inf f Q dxi'.

        Cumulative integrals to an arbitrary xi use the panel Lagrange interpolant of the
        integrand (spectral within each panel).  Returns array (n_densities, n_points).
        """
        dens = np.asarray(densities)
        if dens.ndim == 3:
            dens = dens[None]
        n = dens.shape[0]
        mmax = min(mmax, lmax, self.n_phi // 2 - 1)
        Peta, Pxi, Qxi = self._neumann_tables(lmax, mmax)
        fc = np.zeros((lmax + 1, mmax + 1, n, self.xi.size))
        fs = np.zeros_like(fc)
        for k in range(n):
            c, s = self._project(dens[k], Peta, lmax, mmax)
            fc[:, :, k], fs[:, :, k] = c, s
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        npts = pts.shape[0]
        V = np.zeros((n, npts))
        npan = len(self.breaks) - 1
        order = self.panel_order
        pw = np.array([0.5 * (self.breaks[i + 1] - self.breaks[i]) * self._gl_w for i in range(npan)])  # (npan, order)
        for s0 in range(0, npts, chunk):
            p = pts[s0:s0 + chunk]
            xi, eta, phi = self.frame_coordinates(p)
            u = xi - 1.0
            beyond = u >= self.breaks[-1]
            kp = np.clip(np.searchsorted(self.breaks, u, side="right") - 1, 0, npan - 1)
            a = self.breaks[kp]
            hh = 0.5 * (self.breaks[kp + 1] - a)
            tt = np.clip((u - a) / hh - 1.0, -1.0, 1.0)
            Pk = npleg.legvander(tt, order)
            IP = np.zeros((tt.size, order))
            IP[:, 0] = tt + 1.0
            for k in range(1, order):
                IP[:, k] = (Pk[:, k + 1] - Pk[:, k - 1]) / (2 * k + 1)
            W = hh[:, None] * (IP @ self._lag_inv)                    # (np, order): int_{panel start}^{xi}
            Wrest = pw[kp] - W                                         # int_{xi}^{panel end}
            Pp = _assoc_legendre_table(xi, lmax, mmax, outer=True)
            Qp = _legendre_q_table(xi, lmax, mmax)
            Pe = _assoc_legendre_table(eta, lmax, mmax, outer=False)
            for L in range(lmax + 1):
                for M in range(min(L, mmax) + 1):
                    c = self._coef(L, M)
                    for F, trig in ((fc[L, M], np.cos(M * phi)), (fs[L, M], np.sin(M * phi))):
                        if not np.any(F):
                            continue
                        FP = (F * Pxi[L, M]).reshape(n, npan, order)
                        FQ = (F * Qxi[L, M]).reshape(n, npan, order)
                        totP = np.einsum("npj,pj->np", FP, pw)
                        totQ = np.einsum("npj,pj->np", FQ, pw)
                        preP = np.concatenate([np.zeros((n, 1)), np.cumsum(totP, axis=1)[:, :-1]], axis=1)
                        sufQ = np.concatenate([np.cumsum(totQ[:, ::-1], axis=1)[:, ::-1][:, 1:], np.zeros((n, 1))], axis=1)
                        A = preP[:, kp] + np.einsum("nij,ij->ni", FP[:, kp, :], W)
                        B = sufQ[:, kp] + np.einsum("nij,ij->ni", FQ[:, kp, :], Wrest)
                        if np.any(beyond):
                            A[:, beyond] = preP[:, -1:] + totP[:, -1:]
                            B[:, beyond] = 0.0
                        V[:, s0:s0 + chunk] += c * (Qp[L, M] * A + Pp[L, M] * B) * (Pe[L, M] * trig)
        return V

    def neumann_selftest(self, lmax: int = 60, n_pairs: int = 8, seed: int = 0) -> float:
        """Max relative error of the truncated Neumann series vs 1/|r1-r2| at random
        pairs of points with well-separated xi values."""
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n_pairs):
            x1, x2 = 1.0 + rng.uniform(0.05, 0.8), 1.0 + rng.uniform(1.2, 2.5)
            e1, e2 = rng.uniform(-0.95, 0.95, 2)
            p1, p2 = rng.uniform(0, 2 * np.pi, 2)
            mm = 6
            Pe = _assoc_legendre_table(np.array([e1, e2]), lmax, mm, outer=False)
            Px = _assoc_legendre_table(np.array([x1, x2]), lmax, mm, outer=True)
            Qx = _legendre_q_table(np.array([x1, x2]), lmax, mm)
            s = 0.0
            for L in range(lmax + 1):
                for M in range(min(L, mm) + 1):
                    s += self._coef(L, M) * Px[L, M, 0] * Qx[L, M, 1] * Pe[L, M, 0] * Pe[L, M, 1] * math.cos(M * (p1 - p2))
            h = 0.5 * self.R

            def cart(x, e, p):
                rr = h * math.sqrt((x * x - 1) * (1 - e * e))
                return np.array([rr * math.cos(p), rr * math.sin(p), h * x * e])

            exact = 1.0 / np.linalg.norm(cart(x1, e1, p1) - cart(x2, e2, p2))
            worst = max(worst, abs(s - exact) / exact)
        return worst


# ---------------------------------------------------------------------------
# Matrix-level conveniences
# ---------------------------------------------------------------------------

def one_electron_matrices(grid: ProlateGrid, basis: Sequence[STO], z_a: float, z_b: float):
    """Overlap S, kinetic T, and nuclear attraction V = -z_a/r_A - z_b/r_B over ``basis``."""
    chi = np.array([grid.evaluate(f) for f in basis])
    tchi = np.array([grid.kinetic_applied(f) for f in basis])
    w = grid.weights
    vnuc = -(z_a / grid.r_a + z_b / grid.r_b)
    n = len(basis)
    S = np.einsum("pijk,qijk,ijk->pq", chi, chi, w)
    T = np.einsum("pijk,qijk,ijk->pq", chi, tchi, w)
    T = 0.5 * (T + T.T)
    V = np.einsum("pijk,qijk,ijk->pq", chi, chi, w * vnuc)
    return S, T, V, chi


def eri_tensor(grid: ProlateGrid, orbitals: np.ndarray, lmax: int = 30, mmax: int = 4) -> np.ndarray:
    """Full (pq|rs) tensor (chemists' notation) for a set of functions given as grid
    values, shape (n, *grid.shape).  Uses the batched Neumann solver over the n(n+1)/2
    pair densities."""
    n = orbitals.shape[0]
    pairs = [(p, q) for p in range(n) for q in range(p + 1)]
    dens = np.array([orbitals[p] * orbitals[q] for p, q in pairs])
    G = grid.coulomb_matrix(dens, lmax=lmax, mmax=mmax)
    eri = np.zeros((n, n, n, n))
    for i, (p, q) in enumerate(pairs):
        for j, (r, s) in enumerate(pairs):
            v = G[i, j]
            eri[p, q, r, s] = eri[q, p, r, s] = eri[p, q, s, r] = eri[q, p, s, r] = v
    return eri
