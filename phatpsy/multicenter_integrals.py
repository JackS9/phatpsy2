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
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        self.rho_meridian: Optional[np.ndarray] = None
        self.pts_nz: Optional[np.ndarray] = None
        self.w_nz: Optional[np.ndarray] = None
        self.chi_nz: Optional[np.ndarray] = None
        self.bfs_idx: Optional[Dict[int, int]] = None
        self._src_prep: Dict[Tuple[int, int], Any] = {}


def _is_same_axis(g1: ProlateGrid, g2: ProlateGrid) -> bool:
    if not np.allclose(g1.rot, g2.rot, atol=1e-12):
        return False
    return float(np.linalg.norm(np.cross(g2.mid - g1.mid, g1.rot[:, 2]))) < 1e-12


def _cross_coulomb_collinear(src: _Frame, proj, tgt: _Frame, lmax: int, mmax: int) -> np.ndarray:
    """Exact 2D meridian contraction for _cross_coulomb when src and tgt share the same bond axis."""
    g = src.grid
    tg = tgt.grid
    fc, fs = proj
    n_s = fc.shape[2]
    n_t = len(tgt.pairs)
    _, Pxi, Qxi = g._neumann_tables(lmax, mmax)
    npan = len(g.breaks) - 1
    order = g.panel_order
    pw = np.array([0.5 * (g.breaks[q + 1] - g.breaks[q]) * g._gl_w for q in range(npan)])

    X2, Y2, Z2 = tg.X[:, :, 0].ravel(), tg.Y[:, :, 0].ravel(), tg.Z[:, :, 0].ravel()
    pts2 = np.stack([X2, Y2, Z2], axis=1)
    xi_all, eta_all, _ = g.frame_coordinates(pts2)
    u_all = xi_all - 1.0
    kp_all = np.searchsorted(g.breaks, u_all, side="right") - 1
    beyond_all = u_all >= g.breaks[-1]
    kp_all = np.where(beyond_all, npan, np.clip(kp_all, 0, npan - 1))

    perm = np.argsort(kp_all, kind="stable")
    xi_all, eta_all, u_all, kp_all = xi_all[perm], eta_all[perm], u_all[perm], kp_all[perm]
    edges = np.searchsorted(kp_all, np.arange(npan + 2))
    rho_mer = tgt.rho_meridian[:, perm, :, :]  # (n_t, N2d, 2, mmax+1)

    Pp_all = _assoc_legendre_table(xi_all, lmax, mmax, outer=True)
    Qp_all = _legendre_q_table(xi_all, lmax, mmax)
    Pe_all = _assoc_legendre_table(eta_all, lmax, mmax, outer=False)

    N2d = xi_all.size
    Wa_all = np.zeros((N2d, order))
    Wb_all = np.zeros((N2d, order))
    for q in range(npan):
        s_q, e_q = int(edges[q]), int(edges[q + 1])
        if s_q == e_q:
            continue
        sl = slice(s_q, e_q)
        u = u_all[sl]
        a = g.breaks[q]
        hh = 0.5 * (g.breaks[q + 1] - a)
        tt = np.clip((u - a) / hh - 1.0, -1.0, 1.0)
        Pk = npleg.legvander(tt, order)
        IP = np.zeros((tt.size, order))
        IP[:, 0] = tt + 1.0
        for k in range(1, order):
            IP[:, k] = (Pk[:, k + 1] - Pk[:, k - 1]) / (2 * k + 1)
        Wa_all[sl] = hh * (IP @ g._lag_inv)
        Wb_all[sl] = pw[q][None, :] - Wa_all[sl]

    G_out = np.zeros((n_s, n_t))
    for M in range(mmax + 1):
        L_vals = np.arange(M, lmax + 1)
        K = L_vals.size
        c_vals = np.array([g._coef(int(L), M) for L in L_vals])
        Pxi_m = Pxi[L_vals, M]
        Qxi_m = Qxi[L_vals, M]
        ang = Pe_all[L_vals, M]
        PhiA_all = Qp_all[L_vals, M] * ang
        PhiB_all = Pp_all[L_vals, M] * ang

        for is_sin, F_M in ((0, fc[:, M]), (1, fs[:, M])):
            if not np.any(F_M):
                continue
            rho_M = rho_mer[:, :, is_sin, M]  # (n_t, N2d)
            if not np.any(rho_M):
                continue
            F_sub = F_M[L_vals]  # (K, n_s, nxi)
            FP_pan = (F_sub * (c_vals[:, None, None] * Pxi_m[:, None, :])).reshape(K, n_s, npan, order).transpose(1, 2, 0, 3)
            FQ_pan = (F_sub * (c_vals[:, None, None] * Qxi_m[:, None, :])).reshape(K, n_s, npan, order).transpose(1, 2, 0, 3)
            IP_src = np.sum(FP_pan * pw[None, :, None, :], axis=3)
            IQ_src = np.sum(FQ_pan * pw[None, :, None, :], axis=3)
            UA = np.zeros((n_s, npan + 1, K))
            UA[:, 1:, :] = np.cumsum(IP_src, axis=1)
            UB = np.zeros((n_s, npan + 1, K))
            UB[:, :npan - 1, :] = np.cumsum(IQ_src[:, ::-1, :], axis=1)[:, ::-1, :][:, 1:, :]

            V_M = np.zeros((n_s, N2d))
            for q in range(npan + 1):
                s_q, e_q = int(edges[q]), int(edges[q + 1])
                if s_q == e_q:
                    continue
                sl = slice(s_q, e_q)
                PhiA = PhiA_all[:, sl]
                PhiB = PhiB_all[:, sl]
                V_q = UA[:, q, :] @ PhiA + UB[:, q, :] @ PhiB
                if q < npan:
                    Wa = Wa_all[sl]
                    Wb = Wb_all[sl]
                    MA = (PhiA[:, None, :] * Wa.T[None, :, :]).reshape(K * order, -1)
                    MB = (PhiB[:, None, :] * Wb.T[None, :, :]).reshape(K * order, -1)
                    V_q += FP_pan[:, q].reshape(n_s, K * order) @ MA + FQ_pan[:, q].reshape(n_s, K * order) @ MB
                V_M[:, sl] = V_q
            G_out += V_M @ rho_M.T
    return G_out


def _prepare_neumann_source(src: _Frame, proj, lmax: int, mmax: int):
    """Precompute source-frame Neumann panel projections once per (src, lmax, mmax)."""
    cache_key = (int(lmax), int(mmax))
    if cache_key in src._src_prep:
        return src._src_prep[cache_key]

    g = src.grid
    fc, fs = proj                                   # (L, M, n_s, nxi)
    n_s = fc.shape[2]
    _, Pxi, Qxi = g._neumann_tables(lmax, mmax)
    npan = len(g.breaks) - 1
    order = g.panel_order
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
        src._src_prep[cache_key] = None
        return None

    modes_L_arr = np.array(modes_L)
    modes_M_arr = np.array(modes_M)
    modes_sin_arr = np.array(modes_sin)
    modes_c_arr = np.array(modes_c)
    F_all = np.stack(F_list, axis=0)                # (K, n_s, nxi)
    K = len(modes_L_arr)

    Pxi_m = Pxi[modes_L_arr, modes_M_arr]
    Qxi_m = Qxi[modes_L_arr, modes_M_arr]
    FP_pan = (F_all * (modes_c_arr[:, None, None] * Pxi_m[:, None, :])).reshape(K, n_s, npan, order).transpose(1, 2, 0, 3)
    FQ_pan = (F_all * (modes_c_arr[:, None, None] * Qxi_m[:, None, :])).reshape(K, n_s, npan, order).transpose(1, 2, 0, 3)
    IP_src = np.sum(FP_pan * pw[None, :, None, :], axis=3)
    IQ_src = np.sum(FQ_pan * pw[None, :, None, :], axis=3)
    UA = np.zeros((n_s, npan + 1, K))
    UA[:, 1:, :] = np.cumsum(IP_src, axis=1)
    UB = np.zeros((n_s, npan + 1, K))
    UB[:, :npan - 1, :] = np.cumsum(IQ_src[:, ::-1, :], axis=1)[:, ::-1, :][:, 1:, :]
    FP_flat = [np.ascontiguousarray(FP_pan[:, q, :, :].reshape(n_s, K * order)) for q in range(npan)]
    FQ_flat = [np.ascontiguousarray(FQ_pan[:, q, :, :].reshape(n_s, K * order)) for q in range(npan)]

    prep = (modes_L_arr, modes_M_arr, modes_sin_arr, K, npan, order, pw, UA, UB, FP_flat, FQ_flat)
    src._src_prep[cache_key] = prep
    return prep


def _cross_coulomb(src: _Frame, proj, tgt: _Frame, chi_fn, lmax, mmax, chunk=2048):
    """G[s, t] for densities s in src (given by projections) and t in tgt."""
    if tgt.rho_meridian is not None and _is_same_axis(src.grid, tgt.grid):
        return _cross_coulomb_collinear(src, proj, tgt, lmax, mmax)

    g = src.grid
    fc, _ = proj                                    # (L, M, n_s, nxi)
    n_s = fc.shape[2]
    n_t = len(tgt.pairs)
    prep = _prepare_neumann_source(src, proj, lmax, mmax)
    if prep is None:
        return np.zeros((n_s, n_t))
    modes_L, modes_M, modes_sin, K, npan, order, pw, UA, UB, FP_flat, FQ_flat = prep

    tg = tgt.grid
    if tgt.pts_nz is not None and tgt.w_nz is not None:
        pts_base = tgt.pts_nz
        W8_base = tgt.w_nz
        use_cached_chi = (tgt.chi_nz is not None and tgt.bfs_idx is not None)
    else:
        X, Y, Z, W8_raw = tg.X.ravel(), tg.Y.ravel(), tg.Z.ravel(), tg.weights.ravel()
        keep = W8_raw != 0.0
        pts_base = np.stack([X[keep], Y[keep], Z[keep]], axis=1)
        W8_base = W8_raw[keep]
        use_cached_chi = False

    xi_all, eta_all, phi_all = g.frame_coordinates(pts_base)
    u_all = xi_all - 1.0
    kp_all = np.searchsorted(g.breaks, u_all, side="right") - 1
    beyond_all = u_all >= g.breaks[-1]
    kp_all = np.where(beyond_all, npan, np.clip(kp_all, 0, npan - 1))

    perm = np.argsort(kp_all, kind="stable")
    W8 = W8_base[perm]
    xi_all = xi_all[perm]
    eta_all = eta_all[perm]
    phi_all = phi_all[perm]
    u_all = u_all[perm]
    kp_all = kp_all[perm]
    edges = np.searchsorted(kp_all, np.arange(npan + 2))

    if use_cached_chi:
        pa_arr = np.array([tgt.bfs_idx[p[0]] for p in tgt.pairs], dtype=int)
        pb_arr = np.array([tgt.bfs_idx[p[1]] for p in tgt.pairs], dtype=int)
        chi_all_perm = np.ascontiguousarray(tgt.chi_nz[:, perm])
    else:
        pts_all = pts_base[perm]
        pa = np.array([p[0] for p in tgt.pairs])
        pb = np.array([p[1] for p in tgt.pairs])

    m_vals = np.arange(mmax + 1)
    G_out = np.zeros((n_s, n_t))

    for q in range(npan + 1):
        s_q, e_q = int(edges[q]), int(edges[q + 1])
        if s_q == e_q:
            continue
        UA_q = UA[:, q, :]
        UB_q = UB[:, q, :]
        if q < npan:
            FP_q = FP_flat[q]
            FQ_q = FQ_flat[q]
            a_q = g.breaks[q]
            hh_q = 0.5 * (g.breaks[q + 1] - a_q)
        for s0 in range(s_q, e_q, chunk):
            s1 = min(s0 + chunk, e_q)
            sl = slice(s0, s1)
            if use_cached_chi:
                rho = chi_all_perm[pa_arr, sl] * (chi_all_perm[pb_arr, sl] * W8[sl][None, :])
            else:
                pts = pts_all[sl]
                chi, idx = chi_fn(pts)                  # (n_bf_t, np), map global -> row
                pa_idx = [idx[m] for m in pa]
                pb_idx = [idx[n] for n in pb]
                rho = chi[pa_idx] * (chi[pb_idx] * W8[sl][None, :])   # (n_t, m)

            xi = xi_all[sl]
            eta = eta_all[sl]
            phi = phi_all[sl]
            Pp = _assoc_legendre_table(xi, lmax, mmax, outer=True)
            Qp = _legendre_q_table(xi, lmax, mmax)
            Pe = _assoc_legendre_table(eta, lmax, mmax, outer=False)

            m_phi = m_vals[:, None] * phi[None, :]
            trig_table = np.stack([np.cos(m_phi), np.sin(m_phi)], axis=0)
            ang = Pe[modes_L, modes_M] * trig_table[modes_sin, modes_M]
            PhiA = np.ascontiguousarray(Qp[modes_L, modes_M] * ang)
            PhiB = np.ascontiguousarray(Pp[modes_L, modes_M] * ang)

            V_pts = UA_q @ PhiA + UB_q @ PhiB
            if q < npan:
                u = u_all[sl]
                tt = np.clip((u - a_q) / hh_q - 1.0, -1.0, 1.0)
                Pk = npleg.legvander(tt, order)
                IP = np.zeros((tt.size, order))
                IP[:, 0] = tt + 1.0
                for k in range(1, order):
                    IP[:, k] = (Pk[:, k + 1] - Pk[:, k - 1]) / (2 * k + 1)
                Wa = hh_q * (IP @ g._lag_inv)
                Wb = pw[q][None, :] - Wa
                MA = np.ascontiguousarray((PhiA[:, None, :] * Wa.T[None, :, :]).reshape(K * order, -1))
                MB = np.ascontiguousarray((PhiB[:, None, :] * Wb.T[None, :, :]).reshape(K * order, -1))
                V_pts += FP_q @ MA + FQ_q @ MB
            G_out += V_pts @ rho.T
    return G_out


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
        w_flat = w.ravel()
        keep_nz = w_flat != 0.0
        fr.pts_nz = np.stack([g.X.ravel()[keep_nz], g.Y.ravel()[keep_nz], g.Z.ravel()[keep_nz]], axis=1)
        fr.w_nz = w_flat[keep_nz]
        fr.chi_nz = np.ascontiguousarray(chi.reshape(len(bfs), -1)[:, keep_nz])
        fr.bfs_idx = idx
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
        if any(k2 != key and _is_same_axis(g, frames[k2].grid) for k2 in frames):
            h3 = (0.5 * g.R) ** 3
            jac2d = (g.xi[:, None] ** 2 - g.eta[None, :] ** 2) * (h3 * g.w_xi[:, None] * g.w_eta[None, :])
            csT = np.vstack([
                np.cos(np.outer(np.arange(mm + 1), g.phi)) * g.w_phi[None, :],
                np.sin(np.outer(np.arange(mm + 1), g.phi)) * g.w_phi[None, :],
            ]).T
            n_p = dens.shape[0]
            gcs = (dens.reshape(-1, g.n_phi) @ csT).reshape(n_p, g.xi.size, g.eta.size, 2, mm + 1)
            tol = (1e-13 * 2.0 * np.pi) * np.max(np.abs(dens), axis=3)[..., None, None]
            gcs[np.abs(gcs) <= tol] = 0.0
            gcs *= jac2d[None, :, :, None, None]
            fr.rho_meridian = gcs.reshape(n_p, g.xi.size * g.eta.size, 2, mm + 1)
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
            mm = min(mmax, lmax, src.grid.n_phi // 2 - 1)
            G[(ka, kb)] = _cross_coulomb(src, proj[ka], tgt, None, lmax, mm)
            if verbose:
                print(f"  cross {ka}->{kb} ({time.time()-t0:.0f}s)", flush=True)
        # Free per-source cached Neumann tables as soon as source frame ka finishes
        frames[ka]._src_prep.clear()

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


# Convenience alias matching multicenter naming in phase documentation
multicenter_ao_integrals = molecular_ao_integrals


# ---------------------------------------------------------------------------
# Stewart (1970) L2-optimal STO-KG analytic Gaussian representation
# ---------------------------------------------------------------------------

_STEWART_CACHE: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

# Precomputed IEEE-754 L2-optimal Stewart (1970) expansions for standard shells
# (1 <= n <= 4, 0 <= l < n, K in {3, 4, 6, 8}) obtained via reduced-QR Variable
# Projection with exact Golub-Pereyra analytical Jacobians. Tabulating standard
# shells eliminates BLAS/libm optimizer path differences across CPU/OS targets.
_STEWART_TABULATED: Dict[Tuple[int, int, int], Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    (1, 0, 3): ((
        0.10981750937567349, 0.40577114977042816, 2.2276605315094145,
    ), (
        0.4446345358647228, 0.5353281449100443, 0.1543289709041496,
    )),
    (1, 0, 4): ((
        0.08801862689933045, 0.26520340602638665, 0.9546182537112489, 5.216844367941977,
    ), (
        0.2916254345911361, 0.5328461137432163, 0.2601413595512105, 0.05675242275026748,
    )),
    (1, 0, 6): ((
        0.06510954017450822, 0.1580884174292812, 0.4070989062671859, 1.185056548374838, 4.2359156624714185, 23.103032311158298,
    ), (
        0.1303340880916898, 0.4164915341664037, 0.370562796993462, 0.16853830128607106, 0.04936149134564872, 0.009163595896956431,
    )),
    (1, 0, 8): ((
        0.052940633964914385, 0.11411094479798387, 0.2509528565520742, 0.586147510985127, 1.4960712948547796, 4.3439479444429585, 15.512965132620051, 84.5781729759865,
    ), (
        0.06159114936437505, 0.28926575960683387, 0.3777167805190561, 0.2522578363401453, 0.11338306948968166, 0.038690665466409285, 0.010166357287909876, 0.0018237083243798473,
    )),
    (2, 0, 3): ((
        0.060183322663161794, 0.1567622102782533, 2.58157840777903,
    ), (
        0.4581786282787323, 0.5960385405112001, -0.05994474927158796,
    )),
    (2, 0, 4): ((
        0.06125744534648636, 0.16072806881807397, 2.0002431021959555, 11.615255245023233,
    ), (
        0.4770079979788344, 0.5805587173159834, -0.054720525241105884, -0.011984117649004858,
    )),
    (2, 0, 6): ((
        0.04416183980553907, 0.09260298407584037, 0.20403357305411823, 1.4267860450396308, 5.077140575967911, 27.684961864731452,
    ), (
        0.1712994701052178, 0.5621061303452498, 0.33462711679484924, -0.05150303323517754, -0.020670241581938226, -0.0041512778997159945,
    )),
    (2, 0, 8): ((
        0.03406083284242161, 0.06279711939868601, 0.11723920919431066, 0.23151673012540894, 1.2207232966013382, 3.485333415492766, 12.399908781636379, 67.58364892507242,
    ), (
        0.04955008606839366, 0.3242061704275561, 0.47043676617443647, 0.22755540539922525, -0.046622190179446896, -0.024669511974868854, -0.007416931909927064, -0.0013823721877124738,
    )),
    (2, 1, 3): ((
        0.08009805817069672, 0.23591945362627753, 0.9192379187370487,
    ), (
        0.4223071823164882, 0.5661708834536082, 0.1623948511018804,
    )),
    (2, 1, 4): ((
        0.06543927105531522, 0.16437186357187314, 0.4662622290365279, 1.7982610237345602,
    ), (
        0.26323149664542056, 0.5517873114175589, 0.2857455479016867, 0.0571317010573523,
    )),
    (2, 1, 6): ((
        0.0494822019322023, 0.10466559897495263, 0.22889327930440878, 0.5475665419318311, 1.5303296979306502, 5.868286224240664,
    ), (
        0.10518552437257724, 0.4012362944982654, 0.40498631574619465, 0.1898399993036839, 0.051441045132676826, 0.007924232973801085,
    )),
    (2, 1, 8): ((
        0.04075185507877051, 0.0782823327373878, 0.1510814600018058, 0.30413323014721766, 0.6546898063059168, 1.5563682925216522, 4.337105238364374, 16.607711489600288,
    ), (
        0.04436024510940972, 0.25541953008832713, 0.3930840563497997, 0.28719719137414657, 0.12949517632858262, 0.04083923617056748, 0.009220135731774373, 0.0013183711220773693,
    )),
    (3, 0, 3): ((
        0.0326952912912096, 0.069244214117717, 0.5641487705722992,
    ), (
        0.22618420265758674, 0.86127616070862, -0.178257797394638,
    )),
    (3, 0, 4): ((
        0.037605450689980335, 0.0764332087379857, 0.42624974894195966, 1.5132655717442958,
    ), (
        0.35896273345726465, 0.7518511179816038, -0.17245169561339954, -0.03295496412498451,
    )),
    (3, 0, 6): ((
        0.02724741123315423, 0.047973737796886164, 0.08636686952418687, 0.35933497914549445, 0.9200611478967873, 3.2730320438146725,
    ), (
        0.07223633359761061, 0.5015350980563255, 0.5534527716221308, -0.15878560952220375, -0.05639325664728753, -0.006775596547632846,
    )),
    (3, 0, 8): ((
        0.017612038553957748, 0.031740586759737004, 0.05330766286756541, 0.09174551497417839, 0.32761430876978953, 0.7184440644860712, 1.857760769430785, 6.631277782385126,
    ), (
        0.002991047167720999, 0.1398695313600235, 0.5301379973045215, 0.4617266797555186, -0.1442887146792656, -0.06886044681722286, -0.014880103372724376, -0.0014784875550403922,
    )),
    (3, 1, 3): ((
        0.05739585034658854, 0.148935959074114, 2.692880350485527,
    ), (
        0.5450015133787047, 0.5218564273415661, -0.010619457856170033,
    )),
    (3, 1, 4): ((
        0.04184253873061267, 0.08655487965439311, 0.19150757239562113, 1.8531802271450146,
    ), (
        0.2144986534090308, 0.5846750878760927, 0.2755177570474164, -0.014342493953594531,
    )),
    (3, 1, 6): ((
        0.03315424284658269, 0.06076408936380016, 0.11317418571511623, 0.2248434863164828, 1.3407869206812242, 5.077973416001455,
    ), (
        0.07411456453822861, 0.3908813091496372, 0.4485358225548301, 0.1639395738215564, -0.01419488334839527, -0.0033299299874387975,
    )),
    (3, 1, 8): ((
        0.02714307087487428, 0.04584499471863949, 0.07720515855881102, 0.13438004741769877, 0.24802424869302694, 1.143714803211496, 3.140645187937419, 11.99470214981546,
    ), (
        0.021524700866833908, 0.19584134776771142, 0.41320747398774305, 0.3351923910985532, 0.11296803324289514, -0.013215027736219454, -0.004843452896069824, -0.0008104714875903286,
    )),
    (3, 2, 3): ((
        0.06386630049560549, 0.1639595886718329, 0.5229112272385814,
    ), (
        0.40567795619889385, 0.5847984803283809, 0.1686596035732886,
    )),
    (3, 2, 4): ((
        0.05286755864247115, 0.11875688798526582, 0.2920461074026051, 0.918584657281425,
    ), (
        0.24324232692261394, 0.5601358025621775, 0.30455813890945094, 0.057990578675227986,
    )),
    (3, 2, 6): ((
        0.04058484414837671, 0.07877734872590139, 0.1559114500031202, 0.33113275893441135, 0.7981488148262914, 2.488297030705572,
    ), (
        0.08902828052762414, 0.38431002987490026, 0.4266269064120536, 0.20721390737554496, 0.053867990154938164, 0.00728382746064413,
    )),
    (3, 2, 8): ((
        0.03371880108152618, 0.0602497980345211, 0.10728071383111257, 0.19714706031306553, 0.3823973843870193, 0.8047659288456942, 1.930668359075428, 6.003640875775423,
    ), (
        0.03426782922189298, 0.2279298877143574, 0.3963177272687814, 0.31285146373566786, 0.1438326240097824, 0.0434715711034384, 0.008839808231554683, 0.0010633977400070413,
    )),
    (4, 0, 3): ((
        0.021952946804967077, 0.04448178023974761, 0.22679387522780012,
    ), (
        0.12566617071542063, 1.0567446646117935, -0.334904832538419,
    )),
    (4, 0, 4): ((
        0.028290666007355716, 0.05081097451770038, 0.1663217175662325, 0.3242212829898812,
    ), (
        0.35178112076265844, 0.8909873786315412, -0.2845426860824948, -0.11206828260006228,
    )),
    (4, 0, 6): ((
        0.018740931598930834, 0.031634003192742524, 0.05277666505396369, 0.17179054849979344, 0.3605788797788711, 3.232838649697535,
    ), (
        0.024871792911316484, 0.4389248040003435, 0.7812787295873841, -0.31306273106061167, -0.08666390080724813, 0.0013748174858108013,
    )),
    (4, 0, 8): ((
        0.021845745237506077, 0.03522483640476271, 0.056354436562464076, 0.15280354341301974, 0.2592388971481135, 0.45762624122957507, 2.2119133556027504, 7.708228329307841,
    ), (
        0.07201343427485446, 0.5393064953309111, 0.6536043759085384, -0.2585616397143177, -0.13096832812939435, -0.030549205379813944, 0.001639828007215354, 0.0002512987333964065,
    )),
    (4, 1, 3): ((
        0.036533409209879936, 0.07430216915591793, 0.48596922250698904,
    ), (
        0.3932639490215395, 0.660417223844277, -0.061478234064791096,
    )),
    (4, 1, 4): ((
        0.03706272180613458, 0.0755315605673706, 0.43276193188205886, 1.49260798867507,
    ), (
        0.4117923811627317, 0.6451518205602921, -0.060133109103116936, -0.006035216188448785,
    )),
    (4, 1, 6): ((
        0.028169297788041305, 0.049582483230893884, 0.08847434509639682, 0.34155413976237387, 0.7960947971265062, 2.389722722042276,
    ), (
        0.12049704793983484, 0.5433958179284214, 0.4053115574817401, -0.05958513407907496, -0.016574649376753094, -0.0016659134281219038,
    )),
    (4, 1, 8): ((
        0.022775384849123954, 0.036798267425349585, 0.05904965148524065, 0.0970892878422029, 0.3044272920595265, 0.6071115156637771, 1.3856105644941037, 4.110392493106068,
    ), (
        0.029356529426673026, 0.26633792803308487, 0.4979813656716958, 0.28067797424367685, -0.0549136558266306, -0.023137610578580475, -0.004510340473002624, -0.00040175855212913966,
    )),
    (4, 2, 3): ((
        0.03949855551951926, 0.08040647351616798, 0.177771721936635,
    ), (
        0.2595768928142991, 0.6042409176760274, 0.23085527165580383,
    )),
    (4, 2, 4): ((
        0.040006349411381974, 0.08197240871290792, 0.18234612754019322, 1.995825439319509,
    ), (
        0.27178112345911326, 0.6058047352817892, 0.21770958880221203, -0.0028167025985190055,
    )),
    (4, 2, 6): ((
        0.032326288822916126, 0.05904190360618531, 0.11014679416268625, 0.22095930249771165, 1.3416482979839837, 4.634239438628019,
    ), (
        0.10812501885302205, 0.46216725089204536, 0.4159646940076226, 0.11086704859532559, -0.0035667778874502, -0.00047498428414763393,
    )),
    (4, 2, 8): ((
        0.021225734703640024, 0.033572783594106696, 0.052665144260788624, 0.08450863135404821, 0.1415253002147737, 0.253087380072861, 1.1554636375766565, 3.547112984170108,
    ), (
        0.005939904109273312, 0.09147516640465983, 0.30954883321492926, 0.39641374068405155, 0.2335676109135619, 0.060792321841760845, -0.0038050013093708602, -0.0007724202796404869,
    )),
    (4, 3, 3): ((
        0.05349995700829912, 0.12493805291169041, 0.3483826932039481,
    ), (
        0.39293955707528944, 0.5973380643230946, 0.17378567131864345,
    )),
    (4, 3, 4): ((
        0.04473508827473985, 0.09298346810725458, 0.2074585797018585, 0.56916701392241,
    ), (
        0.22847964934988, 0.5639423877251972, 0.3191828994188242, 0.05902730760650064,
    )),
    (4, 3, 6): ((
        0.03474556653651223, 0.06350097123209685, 0.11731468027986315, 0.22965650357661374, 0.5004907204175262, 1.3577180150485308,
    ), (
        0.07787514290313483, 0.36881125795144115, 0.4411388893440462, 0.2217065833706896, 0.05634263914460299, 0.0069302346796564175,
    )),
    (4, 3, 8): ((
        0.029063702736375048, 0.04935014803641281, 0.08309497019744257, 0.14339654330265444, 0.2589021599374259, 0.5011594666502731, 1.0854414333734883, 2.9339714049268384,
    ), (
        0.0277873509102484, 0.20618124526796372, 0.39428730221476416, 0.33230877563814315, 0.1565553076734799, 0.04623636766932739, 0.00872579213308278, 0.0009172182028933367,
    )),
}


def _compute_stewart_sto_varpro(n_i: int, l_i: int, k_i: int) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Stewart (1970) L2 GTO expansion via reduced-QR Variable Projection.

    Uses mapped Gauss-Legendre radial quadrature on ``[0, inf)`` and exact
    Golub-Pereyra analytical Jacobians for ``r = (Q Q^T - I) y`` inside
    Levenberg-Marquardt least squares, avoiding the ill-conditioned normal-equation
    Gram matrix and finite-difference gradient noise.
    """
    from numpy.polynomial.legendre import leggauss
    from scipy.optimize import least_squares

    x_gl, w_gl = leggauss(160)
    t_gl = 0.5 * (x_gl + 1.0)
    wt_gl = 0.5 * w_gl
    r_gl = t_gl / (1.0 - t_gl)
    dr_gl = wt_gl / ((1.0 - t_gl) ** 2)
    sqrt_w = np.sqrt(dr_gl) * r_gl
    r2 = r_gl ** 2

    sto_norm = (2.0 ** (n_i + 0.5)) / math.sqrt(math.factorial(2 * n_i))
    y = sqrt_w * sto_norm * (r_gl ** (n_i - 1)) * np.exp(-r_gl)
    gto_pref = math.sqrt(2.0 / math.gamma(l_i + 1.5))

    def _eval_A(log_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = np.exp(np.clip(log_a, -25.0, 25.0))
        nrm_g = gto_pref * ((2.0 * a) ** (0.5 * l_i + 0.75))
        arg = -r2[:, None] * a[None, :]
        A = (sqrt_w * (r_gl ** l_i))[:, None] * np.exp(np.clip(arg, -700.0, 0.0)) * nrm_g[None, :]
        return a, A

    def _res(log_a: np.ndarray) -> np.ndarray:
        _, A = _eval_A(log_a)
        Q, _ = np.linalg.qr(A, mode="reduced")
        return Q @ (Q.T @ y) - y

    def _jac(log_a: np.ndarray) -> np.ndarray:
        a, A = _eval_A(log_a)
        Q, R = np.linalg.qr(A, mode="reduced")
        c = np.linalg.solve(R, Q.T @ y)
        r = A @ c - y
        D = ((0.5 * l_i + 0.75) - r2[:, None] * a[None, :]) * A
        Dc = D * c[None, :]
        P_perp_Dc = Dc - Q @ (Q.T @ Dc)
        Dt_r = D.T @ r
        term2 = Q @ np.linalg.solve(R.T, np.diag(Dt_r))
        return P_perp_Dc - term2

    starts = [
        np.linspace(math.log(0.05 / max(n_i, 1)), math.log(45.0 / max(n_i * n_i, 1)), k_i),
        np.linspace(math.log(0.03 / max(n_i, 1)), math.log(8.0 / max(n_i * n_i, 1)), k_i),
    ]
    best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    for log_a0 in starts:
        sol = least_squares(_res, log_a0, jac=_jac, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=1000)
        a_cand, _ = _eval_A(sol.x)
        order_idx = np.argsort(a_cand)
        a_cand = np.ascontiguousarray(a_cand[order_idx], dtype=float)
        if k_i > 1 and np.min(np.diff(a_cand) / np.maximum(a_cand[:-1], 1e-30)) < 1e-4:
            continue
        _, A_ord = _eval_A(np.log(a_cand))
        Q_ord, R_ord = np.linalg.qr(A_ord, mode="reduced")
        c_cand = np.ascontiguousarray(np.linalg.solve(R_ord, Q_ord.T @ y), dtype=float)
        G_ord = (2.0 * np.sqrt(np.outer(a_cand, a_cand)) / (a_cand[:, None] + a_cand[None, :])) ** (l_i + 1.5)
        c_cand /= math.sqrt(max(float(c_cand @ G_ord @ c_cand), 1e-30))
        l2_err = float(np.sum(sol.fun ** 2))
        if best is None or l2_err < best[0]:
            best = (l2_err, a_cand, c_cand)

    if best is None:
        sol = least_squares(_res, starts[0], jac=_jac, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=1000)
        a_opt, _ = _eval_A(sol.x)
        order_idx = np.argsort(a_opt)
        a_opt = np.ascontiguousarray(a_opt[order_idx], dtype=float)
        _, A_ord = _eval_A(np.log(a_opt))
        Q_ord, R_ord = np.linalg.qr(A_ord, mode="reduced")
        c_opt = np.ascontiguousarray(np.linalg.solve(R_ord, Q_ord.T @ y), dtype=float)
        G_ord = (2.0 * np.sqrt(np.outer(a_opt, a_opt)) / (a_opt[:, None] + a_opt[None, :])) ** (l_i + 1.5)
        c_opt /= math.sqrt(max(float(c_opt @ G_ord @ c_opt), 1e-30))
        return a_opt, c_opt

    return best[1], best[2]


def stewart_sto_expansion(n: int, l: int, n_prim: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit-exponent (``zeta=1.0``) Stewart L2-optimal GTO expansion ``(exponents, coeffs)``.

    Minimizes the least-squares orbital residual

    .. math::
        1 - \\langle R_n | \\chi_{\\text{GTO}} \\rangle^2
        = 1 - \\mathbf{b}(\\boldsymbol{\\alpha})^T
              \\mathbf{G}(\\boldsymbol{\\alpha})^{-1}
              \\mathbf{b}(\\boldsymbol{\\alpha})

    in :math:`L^2(\\mathbb{R}^3)` with respect to :math:`\\log \\boldsymbol{\\alpha}`
    (R. F. Stewart, *J. Chem. Phys.* **52**, 431–438 (1970)).  For any physical STO
    exponent ``zeta``, scale exponents as ``zeta**2 * exponents``.
    """
    key = (int(n), int(l), int(n_prim))
    n_i, l_i, k_i = key
    if n_i < 1 or l_i < 0 or l_i >= n_i or k_i < 1:
        raise ValueError(f"Invalid STO parameters for Stewart expansion: n={n_i}, l={l_i}, n_prim={k_i}")

    if key in _STEWART_CACHE:
        return _STEWART_CACHE[key]

    tab = _STEWART_TABULATED.get(key)
    if tab is not None:
        a_opt = np.ascontiguousarray(tab[0], dtype=float)
        c_opt = np.ascontiguousarray(tab[1], dtype=float)
    else:
        a_opt, c_opt = _compute_stewart_sto_varpro(n_i, l_i, k_i)

    _STEWART_CACHE[key] = (a_opt, c_opt)
    return a_opt, c_opt


@dataclass
class PySCFSTOContext:
    """Mappings between PySCF spherical-harmonic AO integrals and PHATPSY real-STO order."""

    mol: Any
    ao_map: np.ndarray             # Shape (nao_pyscf, nao_phatpsy); O_ph = ao_map.T @ O_py @ ao_map
    charges: np.ndarray
    coords: np.ndarray
    is_perm: bool
    perm_idx: Optional[np.ndarray] = None
    perm_scale: Optional[np.ndarray] = None


def sto_ng_ao_integrals(
    atoms: Sequence[Tuple[float, Sequence[float], List[STO]]],
    n_prim: int = 6,
    return_pyscf_context: bool = False,
) -> Union[MolecularAOIntegrals, Tuple[MolecularAOIntegrals, PySCFSTOContext]]:
    """Analytic multi-center AO integrals via Stewart (1970) ``STO-KG`` expansions in PySCF.

    Every unique radial shell ``(atom_idx, n, l, zeta)`` is represented by a single
    Condon-Shortley-compatible PySCF spherical Gaussian shell (``cart=False``) using
    the L2-optimal exponents and contraction coefficients from :func:`stewart_sto_expansion`.
    Two-electron integrals are evaluated analytically with 8-fold permutation symmetry
    (``mol.intor("int2e", aosym="s8")``) so the resulting PES is :math:`C^\\infty` smooth,
    strictly rotationally/translationally invariant, and free of nearest-frame discontinuities.
    """
    from pyscf import ao2mo, gto

    natom = len(atoms)
    charges = np.array([float(a[0]) for a in atoms], dtype=float)
    coords = np.array([np.asarray(a[1], float) for a in atoms], dtype=float)
    basis: List[STO] = []
    atom_of_list: List[int] = []
    for ia, a in enumerate(atoms):
        basis.extend(list(a[2]))
        atom_of_list.extend([ia] * len(a[2]))
    atom_of = np.asarray(atom_of_list, dtype=int)
    nao = len(basis)

    from phatpsy.schema import ATOMIC_NUMBERS

    num_to_sym = {v: k for k, v in ATOMIC_NUMBERS.items()}
    pyscf_atoms: List[List[Any]] = []
    mol_basis_final: Dict[str, Any] = {}
    ao_shell_loc: Dict[int, Tuple[int, int, int, int]] = {}  # ao_idx -> (sh_idx, l, m, atom_idx)

    global_sh_idx = 0
    z_int_sum = 0
    for ia in range(natom):
        z_int = int(round(charges[ia]))
        z_int_sum += z_int
        sym = num_to_sym.get(z_int, "H")
        lbl = f"{sym}@{ia + 1}"
        pyscf_atoms.append([lbl, [float(coords[ia, 0]), float(coords[ia, 1]), float(coords[ia, 2])]])
        atom_shells: Dict[Tuple[int, int, float], int] = {}
        atom_bas_list: List[Any] = []
        for ao_idx in range(nao):
            if atom_of[ao_idx] != ia:
                continue
            f = basis[ao_idx]
            sh_key = (int(f.n), int(f.l), round(float(f.zeta), 12))
            if sh_key not in atom_shells:
                a0, c0 = stewart_sto_expansion(int(f.n), int(f.l), n_prim=int(n_prim))
                exps = a0 * (float(f.zeta) ** 2)
                atom_bas_list.append([int(f.l)] + [[float(e), float(c)] for e, c in zip(exps, c0)])
                atom_shells[sh_key] = global_sh_idx
                global_sh_idx += 1
            ao_shell_loc[ao_idx] = (atom_shells[sh_key], int(f.l), int(f.m), ia)
        mol_basis_final[lbl] = atom_bas_list

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.cart = False
    mol.charge = 0
    mol.spin = z_int_sum % 2
    mol.verbose = 0
    mol.atom = pyscf_atoms
    mol.basis = mol_basis_final
    mol.build(dump_input=False, parse_arg=False)

    ao_loc = mol.ao_loc_nr()
    nao_py = int(mol.nao_nr())
    ao_map = np.zeros((nao_py, nao), dtype=float)
    for ao_idx in range(nao):
        sh_idx, l_val, m_val, _ = ao_shell_loc[ao_idx]
        ao0 = int(ao_loc[sh_idx])
        if l_val == 0:
            comp, phase = 0, 1.0
        elif l_val == 1:
            if m_val == 1:
                comp, phase = 0, -1.0
            elif m_val == -1:
                comp, phase = 1, -1.0
            elif m_val == 0:
                comp, phase = 2, 1.0
            else:
                raise ValueError(f"Unsupported p magnetic quantum number m={m_val}")
        else:
            comp = m_val + l_val
            phase = -1.0 if (abs(m_val) % 2) == 1 else 1.0
        ao_map[ao0 + comp, ao_idx] = phase

    S_py = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
    T_py = np.asarray(mol.intor("int1e_kin"), dtype=float)
    V_py = np.asarray(mol.intor("int1e_nuc"), dtype=float)
    eri_py = np.asarray(ao2mo.restore(1, mol.intor("int2e", aosym="s8"), nao_py), dtype=float)

    # Normalize each STO column so diag(S) == 1.0 to machine precision
    diag_raw = np.einsum("mi,mn,ni->i", ao_map, S_py, ao_map)
    col_scale = 1.0 / np.sqrt(np.maximum(diag_raw, 1e-30))
    ao_map = ao_map * col_scale[None, :]

    S = ao_map.T @ S_py @ ao_map
    S = 0.5 * (S + S.T)
    T = ao_map.T @ T_py @ ao_map
    T = 0.5 * (T + T.T)
    V = ao_map.T @ V_py @ ao_map
    V = 0.5 * (V + V.T)

    # Fast exact ERI index map when every column has 1 nonzero (standard complete subshells)
    nnz_per_col = np.sum(np.abs(ao_map) > 0.0, axis=0)
    is_perm = bool(nao_py == nao and np.all(nnz_per_col == 1))
    perm_idx: Optional[np.ndarray] = None
    perm_scale: Optional[np.ndarray] = None
    if np.all(nnz_per_col == 1):
        perm_idx = np.argmax(np.abs(ao_map), axis=0)
        perm_scale = ao_map[perm_idx, np.arange(nao)]
        eri = (
            eri_py[np.ix_(perm_idx, perm_idx, perm_idx, perm_idx)]
            * perm_scale[:, None, None, None]
            * perm_scale[None, :, None, None]
            * perm_scale[None, None, :, None]
            * perm_scale[None, None, None, :]
        )
    else:
        eri = np.einsum("ai,bj,ck,dl,abcd->ijkl", ao_map, ao_map, ao_map, ao_map, eri_py, optimize=True)

    ints = MolecularAOIntegrals(basis, atom_of, charges, coords, S, T, V, eri)
    if return_pyscf_context:
        ctx = PySCFSTOContext(
            mol=mol,
            ao_map=ao_map,
            charges=charges,
            coords=coords,
            is_perm=is_perm,
            perm_idx=perm_idx,
            perm_scale=perm_scale,
        )
        return ints, ctx
    return ints


