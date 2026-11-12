"""
PHATPSY 2.0: multi-center STO integrals from pair-density potentials (exploratory,
branch ``two-center-wip``).

Every integral over real STOs on any number of atoms is reduced to one local, exact
operation: the electrostatic potential of a *pair density* rho_mn = chi_m chi_n, computed
in that pair's own prolate-spheroidal frame (foci = the atoms of m and n) by the Neumann
expansion (see :mod:`phatpsy.two_center_integrals`), acting on

* a nucleus C:          <m| 1/r_C |n>  = V_mn(R_C)                (three-center attraction)
* another pair density: (mn|ls) = int V_mn rho_ls dV             (three-/four-center ERIs)

with every density integrated in a frame in which its own centers are foci (so nuclear
cusps are always handled exactly by the quadrature).  No fitting of any kind.

Cross-frame Coulomb integrals use the separable form of the Neumann potential:
    (s|t) = sum_LMT c_LM sum_nodes [ FP_s(node) K^A_t(node) + FQ_s(node) K^B_t(node) ]
where FP_s, FQ_s are the source-frame projections of rho_s (times P, Q on the source xi
nodes) and the target kernels K^A_t, K^B_t collect rho_t over its own quadrature points,
binned by the source-frame xi panel in which each target point falls.  Cost therefore
scales with (#target densities x #target points), independent of the number of sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import legendre as npleg

from phatpsy.two_center_integrals import (STO, ProlateGrid, _assoc_legendre_table,
                                          _legendre_q_table)
from phatpsy.volumetric import evaluate_real_sto


@dataclass
class MolecularAOIntegrals:
    basis: List[STO]
    atom_of: np.ndarray
    charges: np.ndarray
    coords: np.ndarray
    S: np.ndarray
    T: np.ndarray
    V: np.ndarray                 # total nuclear attraction sum_C -Z_C/r_C
    eri: np.ndarray
    _Jmat: Optional[np.ndarray] = field(default=None, repr=False)
    _Kmat: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def h(self):
        return self.T + self.V

    @property
    def nao(self):
        return len(self.basis)

    @property
    def vnn(self):
        e = 0.0
        for i in range(len(self.charges)):
            for j in range(i):
                e += self.charges[i] * self.charges[j] / np.linalg.norm(self.coords[i] - self.coords[j])
        return float(e)

    def jk_matrices(self):
        if self._Jmat is None:
            n = self.nao
            self._Jmat = self.eri.reshape(n * n, n * n)
            self._Kmat = np.ascontiguousarray(self.eri.transpose(0, 3, 2, 1)).reshape(n * n, n * n)
        return self._Jmat, self._Kmat


def _eval(f: STO, X, Y, Z):
    return evaluate_real_sto(X, Y, Z, f.center, f.n, f.l, f.m, f.zeta, f.norm())


def _kin(f: STO, X, Y, Z):
    chi = _eval(f, X, Y, Z)
    r = np.sqrt((X - f.center[0]) ** 2 + (Y - f.center[1]) ** 2 + (Z - f.center[2]) ** 2)
    c2 = f.n * (f.n - 1) - f.l * (f.l + 1)
    lap = (c2 / (r * r) if c2 != 0 else 0.0) - 2.0 * f.n * f.zeta / r + f.zeta ** 2
    return -0.5 * lap * chi


class _Frame:
    def __init__(self, i, j, grid: ProlateGrid):
        self.i, self.j, self.grid = i, j, grid
        self.pairs: List[Tuple[int, int]] = []


def _cross_coulomb(src: _Frame, proj, tgt: _Frame, chi_fn, lmax, mmax, chunk=4096):
    """G[s, t] for densities s in src (given by projections) and t in tgt."""
    g = src.grid
    fc, fs = proj                                   # (L, M, n_s, nxi)
    n_s = fc.shape[2]
    n_t = len(tgt.pairs)
    _, Pxi, Qxi = g._neumann_tables(lmax, mmax)
    npan = len(g.breaks) - 1
    order = g.panel_order
    nxi = g.xi.size
    pw = np.array([0.5 * (g.breaks[q + 1] - g.breaks[q]) * g._gl_w for q in range(npan)])

    modes_L, modes_M, modes_sin, modes_c, F_list = [], [], [], [], []
    for L in range(lmax + 1):
        for M in range(min(L, mmax) + 1):
            c = g._coef(L, M)
            for is_sin, Fsrc in ((0, fc[L, M]), (1, fs[L, M])):
                if not np.any(Fsrc):
                    continue
                modes_L.append(L)
                modes_M.append(M)
                modes_sin.append(is_sin)
                modes_c.append(c)
                F_list.append(Fsrc)
    if not F_list:
        return np.zeros((n_s, n_t))
    modes_L = np.array(modes_L)
    modes_M = np.array(modes_M)
    modes_sin = np.array(modes_sin)
    modes_c = np.array(modes_c)
    F_all = np.stack(F_list, axis=0)                # (K, n_s, nxi)
    K = len(modes_L)

    tg = tgt.grid
    X, Y, Z, W8 = tg.X.ravel(), tg.Y.ravel(), tg.Z.ravel(), tg.weights.ravel()
    keep = W8 != 0.0
    X, Y, Z, W8 = X[keep], Y[keep], Z[keep], W8[keep]
    pts_all = np.stack([X, Y, Z], axis=1)
    xi_all, eta_all, phi_all = g.frame_coordinates(pts_all)
    u_all = xi_all - 1.0
    kp_all = np.searchsorted(g.breaks, u_all, side="right") - 1
    beyond_all = u_all >= g.breaks[-1]
    kp_all = np.where(beyond_all, npan, np.clip(kp_all, 0, npan - 1))

    perm = np.argsort(kp_all, kind="stable")
    pts_all = pts_all[perm]
    W8 = W8[perm]
    xi_all = xi_all[perm]
    eta_all = eta_all[perm]
    phi_all = phi_all[perm]
    u_all = u_all[perm]
    kp_all = kp_all[perm]
    edges = np.searchsorted(kp_all, np.arange(npan + 2))

    pa = np.array([p[0] for p in tgt.pairs])
    pb = np.array([p[1] for p in tgt.pairs])

    TA = np.zeros((K, n_t, npan + 1))
    TB = np.zeros((K, n_t, npan + 1))
    KA = np.zeros((K, npan, n_t, order))
    KB = np.zeros((K, npan, n_t, order))

    m_vals = np.arange(mmax + 1)
    for q in range(npan + 1):
        s_q, e_q = int(edges[q]), int(edges[q + 1])
        if s_q == e_q:
            continue
        for s0 in range(s_q, e_q, chunk):
            s1 = min(s0 + chunk, e_q)
            sl = slice(s0, s1)
            pts = pts_all[sl]
            chi, idx = chi_fn(pts)                  # (n_bf_t, np), map global -> row
            pa_idx = [idx[m] for m in pa]
            pb_idx = [idx[n] for n in pb]
            rho = chi[pa_idx] * chi[pb_idx] * W8[sl][None, :]   # (n_t, m)

            xi = xi_all[sl]
            eta = eta_all[sl]
            phi = phi_all[sl]
            Pp = _assoc_legendre_table(xi, lmax, mmax, outer=True)
            Qp = _legendre_q_table(xi, lmax, mmax)
            Pe = _assoc_legendre_table(eta, lmax, mmax, outer=False)

            m_phi = m_vals[:, None] * phi[None, :]
            trig_table = np.stack([np.cos(m_phi), np.sin(m_phi)], axis=0)
            ang = Pe[modes_L, modes_M] * trig_table[modes_sin, modes_M]
            PhiA = Qp[modes_L, modes_M] * ang
            PhiB = Pp[modes_L, modes_M] * ang

            TA[:, :, q] += PhiA @ rho.T
            TB[:, :, q] += PhiB @ rho.T

            if q < npan:
                u = u_all[sl]
                a = g.breaks[q]
                hh = 0.5 * (g.breaks[q + 1] - a)
                tt = np.clip((u - a) / hh - 1.0, -1.0, 1.0)
                Pk = npleg.legvander(tt, order)
                IP = np.zeros((tt.size, order))
                IP[:, 0] = tt + 1.0
                for k in range(1, order):
                    IP[:, k] = (Pk[:, k + 1] - Pk[:, k - 1]) / (2 * k + 1)
                Wa = hh * (IP @ g._lag_inv)
                Wb = pw[q][None, :] - Wa
                Ra = np.transpose(rho[:, :, None] * Wa[None, :, :], (1, 0, 2)).reshape(-1, n_t * order)
                Rb = np.transpose(rho[:, :, None] * Wb[None, :, :], (1, 0, 2)).reshape(-1, n_t * order)
                KA[:, q] += (PhiA @ Ra).reshape(K, n_t, order)
                KB[:, q] += (PhiB @ Rb).reshape(K, n_t, order)

    SA = np.cumsum(TA[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:]
    SB = np.concatenate([np.zeros((K, n_t, 1)), np.cumsum(TB[:, :, :npan], axis=2)[:, :, :-1]], axis=2)

    KA += np.transpose(SA, (0, 2, 1))[:, :, :, None] * pw[None, :, None, :]
    KB += np.transpose(SB, (0, 2, 1))[:, :, :, None] * pw[None, :, None, :]

    KA_xi = np.transpose(KA, (0, 1, 3, 2)).reshape(K * nxi, n_t)
    KB_xi = np.transpose(KB, (0, 1, 3, 2)).reshape(K * nxi, n_t)

    Pxi_m = Pxi[modes_L, modes_M]
    Qxi_m = Qxi[modes_L, modes_M]
    FP = (F_all * (modes_c[:, None, None] * Pxi_m[:, None, :])).transpose(1, 0, 2).reshape(n_s, K * nxi)
    FQ = (F_all * (modes_c[:, None, None] * Qxi_m[:, None, :])).transpose(1, 0, 2).reshape(n_s, K * nxi)
    return FP @ KA_xi + FQ @ KB_xi


def molecular_ao_integrals(atoms: Sequence[Tuple[float, Sequence[float], List[STO]]],
                           n_eta: int = 40, n_phi: int = 24, lmax: int = 24, mmax: int = 6,
                           verbose: bool = False) -> MolecularAOIntegrals:
    """Exact multi-center integrals over real STOs.

    atoms: sequence of (Z, xyz, basis STO list centred on xyz).
    """
    import itertools
    import time

    natom = len(atoms)
    charges = np.array([float(a[0]) for a in atoms])
    coords = np.array([np.asarray(a[1], float) for a in atoms])
    basis: List[STO] = []
    atom_of = []
    for ia, a in enumerate(atoms):
        basis += list(a[2])
        atom_of += [ia] * len(a[2])
    atom_of = np.array(atom_of)
    nao = len(basis)
    zmin = {ia: min(f.zeta for f in atoms[ia][2]) for ia in range(natom)}

    frames: Dict[Tuple[int, int], _Frame] = {}
    for i, j in itertools.combinations(range(natom), 2):
        frames[(i, j)] = _Frame(i, j, ProlateGrid(coords[i], coords[j], zeta_min=min(zmin[i], zmin[j], 1.0),
                                                  n_eta=n_eta, n_phi=n_phi))
    nearest = {}
    for i in range(natom):
        d = [np.linalg.norm(coords[i] - coords[j]) if j != i else np.inf for j in range(natom)]
        nearest[i] = int(np.argmin(d))
    owner = {}
    for m in range(nao):
        for n in range(m + 1):
            i, j = atom_of[m], atom_of[n]
            key = (min(i, j), max(i, j)) if i != j else (min(i, nearest[i]), max(i, nearest[i]))
            frames[key].pairs.append((m, n))
            owner[(m, n)] = key

    def frame_bf(fr: _Frame):
        return [k for k in range(nao) if atom_of[k] in (fr.i, fr.j)]

    # --- per-frame basis values, one-electron integrals, projections, same-frame ERIs
    S = np.zeros((nao, nao)); T = np.zeros((nao, nao)); V = np.zeros((nao, nao))
    proj = {}
    G = {}
    t0 = time.time()
    for key, fr in frames.items():
        g = fr.grid
        bfs = frame_bf(fr)
        idx = {k: r for r, k in enumerate(bfs)}
        chi = np.array([_eval(basis[k], g.X, g.Y, g.Z) for k in bfs])
        tchi = np.array([_kin(basis[k], g.X, g.Y, g.Z) for k in bfs])
        w = g.weights
        foci_v = -(charges[fr.i] / g.r_a + charges[fr.j] / g.r_b)
        dens = []
        for (m, n) in fr.pairs:
            rho = chi[idx[m]] * chi[idx[n]]
            S[m, n] = S[n, m] = np.sum(rho * w)
            tv = 0.5 * (np.sum(chi[idx[m]] * tchi[idx[n]] * w) + np.sum(chi[idx[n]] * tchi[idx[m]] * w))
            T[m, n] = T[n, m] = tv
            V[m, n] = V[n, m] = np.sum(rho * foci_v * w)
            dens.append(rho)
        dens = np.array(dens)
        mm = min(mmax, lmax, g.n_phi // 2 - 1)
        Peta, _, _ = g._neumann_tables(lmax, mm)
        proj[key] = g._project_batch(dens, Peta, lmax, mm)
        # nuclei that are not foci of this frame: potential of each pair density at R_C
        others = [c for c in range(natom) if c not in (fr.i, fr.j)]
        if others:
            Vc = g.potentials_at(dens, coords[others], lmax=lmax, mmax=mm, proj=proj[key])
            for k, (m, n) in enumerate(fr.pairs):
                v = -float(np.dot(charges[others], Vc[k]))
                V[m, n] += v
                if m != n:
                    V[n, m] += v
        G[(key, key)] = g.coulomb_matrix(dens, lmax=lmax, mmax=mm, proj=proj[key])
        del dens, chi, tchi
        if verbose:
            print(f"  frame {key}: {len(fr.pairs)} pair densities ({time.time()-t0:.0f}s)", flush=True)

    # --- cross-frame ERIs
    keys = list(frames)
    for a_i, ka in enumerate(keys):
        for kb in keys[a_i + 1:]:
            src, tgt = frames[ka], frames[kb]
            bfs_t = frame_bf(tgt)
            idx_t = {k: r for r, k in enumerate(bfs_t)}

            def chi_fn(pts, bfs_t=bfs_t, idx_t=idx_t):
                vals = np.array([_eval(basis[k], pts[:, 0], pts[:, 1], pts[:, 2]) for k in bfs_t])
                return vals, idx_t

            mm = min(mmax, lmax, src.grid.n_phi // 2 - 1)
            G[(ka, kb)] = _cross_coulomb(src, proj[ka], tgt, chi_fn, lmax, mm)
            if verbose:
                print(f"  cross {ka}->{kb} ({time.time()-t0:.0f}s)", flush=True)

    # --- assemble ERI tensor
    pos = {key: {p: r for r, p in enumerate(frames[key].pairs)} for key in keys}
    eri = np.zeros((nao, nao, nao, nao))
    for (ka, kb), blk in G.items():
        for (m, n), r in pos[ka].items():
            for (l, s), c in pos[kb].items():
                v = blk[r, c]
                for p, q in ((m, n), (n, m)):
                    for x, y in ((l, s), (s, l)):
                        eri[p, q, x, y] = v
                        eri[x, y, p, q] = v
    return MolecularAOIntegrals(basis, atom_of, charges, coords, S, T, V, eri)

