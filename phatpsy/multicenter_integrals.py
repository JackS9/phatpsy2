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
    if key in _STEWART_CACHE:
        return _STEWART_CACHE[key]

    from numpy.polynomial.legendre import leggauss
    from scipy.optimize import minimize

    n_i, l_i, k_i = key
    if n_i < 1 or l_i < 0 or l_i >= n_i or k_i < 1:
        raise ValueError(f"Invalid STO parameters for Stewart expansion: n={n_i}, l={l_i}, n_prim={k_i}")

    x_gl, w_gl = leggauss(120)
    t_gl = 0.5 * (x_gl + 1.0)
    wt_gl = 0.5 * w_gl
    r_gl = t_gl / (1.0 - t_gl)
    dr_gl = wt_gl / ((1.0 - t_gl) ** 2)

    sto_norm = (2.0 ** (n_i + 0.5)) / math.sqrt(math.factorial(2 * n_i))
    R_sto = sto_norm * (r_gl ** (n_i - 1)) * np.exp(-r_gl) * (r_gl ** 2) * dr_gl
    gto_pref = math.sqrt(2.0 / math.gamma(l_i + 1.5))

    def _fit_mats(log_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.exp(np.clip(log_a, -25.0, 25.0))
        a_sum = a[:, None] + a[None, :]
        a_prod = a[:, None] * a[None, :]
        G = (2.0 * np.sqrt(a_prod) / a_sum) ** (l_i + 1.5)
        nrm_g = gto_pref * ((2.0 * a) ** (0.5 * l_i + 0.75))
        arg = -a[:, None] * (r_gl[None, :] ** 2)
        G_rad = nrm_g[:, None] * (r_gl[None, :] ** l_i) * np.exp(np.clip(arg, -700.0, 0.0))
        b = G_rad @ R_sto
        return a, G + 1e-14 * np.eye(k_i), b

    def _obj(log_a: np.ndarray) -> float:
        _, G_reg, b = _fit_mats(log_a)
        c = np.linalg.solve(G_reg, b)
        return float(max(0.0, 1.0 - float(b @ c)))

    log_a0 = np.linspace(math.log(0.05 / max(n_i, 1)), math.log(45.0 / max(n_i * n_i, 1)), k_i)
    res = minimize(_obj, log_a0, method="BFGS", options={"gtol": 1e-11, "maxiter": 600})
    a_opt, G_reg, b = _fit_mats(res.x)
    c_opt = np.linalg.solve(G_reg, b)
    order_idx = np.argsort(a_opt)
    a_opt = np.ascontiguousarray(a_opt[order_idx], dtype=float)
    c_opt = np.ascontiguousarray(c_opt[order_idx], dtype=float)
    G_ord = (2.0 * np.sqrt(np.outer(a_opt, a_opt)) / (a_opt[:, None] + a_opt[None, :])) ** (l_i + 1.5)
    c_opt /= math.sqrt(max(float(c_opt @ G_ord @ c_opt), 1e-30))
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


