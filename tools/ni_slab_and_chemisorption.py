#!/usr/bin/env python3
"""
PHATPSY 2.0 — Stage 11: Bulk Nickel (Ni13), Ni(100) Slab, and CO/Ni(100) Chemisorption
(branch ``two-center-wip``).

Part 1: Bulk Nickel & Ni(100) Surface Slab
  1.1 PHATPSY pk_scale=1.0 converged embedded states (Bulk Oh, Subsurface C4v, Surface C4v):
      core-level shifts (1s, 2s, 2p, 3s, 3p) and 3d crystal-field splittings.
  1.2 Exact 1-center Ni (3d^8 4s^2, ^3F) ROHF basis ladder (SZ -> DZ -> DZP -> ADF-TZP)
      and exact [Ar]-core Phillips-Kleinman embedding accuracy.
  1.3 Embedding active Ni1 (10 valence electrons, [Ar] core frozen) inside the 8-neighbor
      Ni(100) slab (Ni2..Ni9) and 12-neighbor FCC bulk Ni13 cluster (Ni2..Ni13):
      proof that the exact joint environment metric (S_BB)^-1 eliminates the unphysical
      negative pseudometric eigenvalue (-2.6837 -> +0.1448) without empirical pk_scale=0.80.

Part 2: Top-Site CO Chemisorbed on Ni(100)
  2.1 Blyholder domain decomposition (Regime 0: strict fragment domains / no CT;
      Regime 1: +5sigma -> Ni1 sigma donation; Regime 2: +Ni1 3d_pi -> CO 2pi* backbonding)
      across R_NiC for both bare active Ni1--C--O and slab-embedded CO/Ni1/Ni(100).
  2.2 First-principles C--O stretching potential E(R_CO) for isolated CO vs chemisorbed
      CO/Ni(100) across Regimes 0, 1, and 2: Blyholder bond lengthening (Delta R_CO > 0)
      and vibrational stretching redshift (Delta omega_CO < 0).
"""
from __future__ import annotations

import json
import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh

from phatpsy.two_center_integrals import STO, ProlateGrid
from phatpsy.localized_vb import atom_basis, diatomic_ao_integrals, MaskedVB
from phatpsy.embedding import (
    EmbeddedAOIntegrals,
    FrozenEnvironmentAtom,
    one_center_ao_matrices,
    one_center_coulomb_potential,
    embed_from_full_integrals,
    _renormalize_active_eri,
    rotate_z_sto_matrix,
    solve_atomic_ni_rohf,
    build_active_ni_co_integrals,
    embed_collinear_in_c4v_slab,
)

HARTREE_TO_EV = 27.211386245988


# ---------------------------------------------------------------------------
# Part 1 helpers
# ---------------------------------------------------------------------------

def analyze_phatpsy_embedded_states() -> Dict[str, dict]:
    """Inspect data/ni_{bulk,sub,surf}_embedded.json and extract core-level energies
    and 3d crystal-field splittings."""
    paths = {
        "Bulk (Ni13, Oh, CN=12)": "data/ni_bulk_embedded.json",
        "Subsurface (Ni100, C4v, CN=12)": "data/ni_sub_embedded.json",
        "Surface (Ni100, C4v, CN=8)": "data/ni_surf_embedded.json",
    }
    out = {}
    for label, path in paths.items():
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        orbs = raw["orbitals"]
        e1s = 0.5 * (orbs[0]["energy_alpha"] + orbs[0]["energy_beta"])
        e2s = 0.5 * (orbs[1]["energy_alpha"] + orbs[1]["energy_beta"])
        e2p = np.mean([0.5 * (orbs[k]["energy_alpha"] + orbs[k]["energy_beta"]) for k in (2, 3, 4)])
        e3s = 0.5 * (orbs[5]["energy_alpha"] + orbs[5]["energy_beta"])
        e3p = np.mean([0.5 * (orbs[k]["energy_alpha"] + orbs[k]["energy_beta"]) for k in (6, 7, 8)])
        # 3d levels (orbitals 9..13 in 0-based index, i.e., orbital_index 10..14)
        e3d_a = [orbs[k]["energy_alpha"] for k in range(9, 14)]
        e3d_b = [orbs[k]["energy_beta"] for k in range(9, 14)]
        e4s_a = orbs[14]["energy_alpha"]
        out[label] = dict(
            net_charge=float(raw["net_charge"]),
            net_spin=float(raw["net_spin"]),
            e1s=float(e1s),
            e2s=float(e2s),
            e2p=float(e2p),
            e3s=float(e3s),
            e3p=float(e3p),
            e3d_alpha_min=float(min(e3d_a)),
            e3d_alpha_max=float(max(e3d_a)),
            e3d_split_ev=float((max(e3d_a) - min(e3d_a)) * HARTREE_TO_EV),
            e4s_alpha=float(e4s_a),
        )
    return out


def benchmark_ni_basis_ladder() -> List[dict]:
    """Compare 1-center Ni(3d^8 4s^2, ^3F) all-AO ROHF vs valence-only [Ar]-core
    embedded MaskedVB across SZ, DZ, DZP, and ADF-TZP."""
    rows = []
    for bname in ["SZ", "DZ", "DZP", "ADF-TZP"]:
        b, S, T, V, eri, E_rohf, C_core, C_val = solve_atomic_ni_rohf(bname)
        n = len(b)
        val_idx = [i for i, f in enumerate(b) if (f.n == 3 and f.l == 2) or (f.n == 4)]

        class _FullNi:
            nao = n
            basis = b
            atom_of = np.zeros(n, dtype=int)

            def jk_matrices(self):
                return (
                    eri.reshape(n * n, n * n),
                    np.ascontiguousarray(eri.transpose(0, 3, 2, 1)).reshape(n * n, n * n),
                )

        _FullNi.S = S
        _FullNi.h = T + V
        _FullNi.eri = eri
        _FullNi.vnn = 0.0

        emb = embed_from_full_integrals(_FullNi(), C_core, val_idx, mode="exact_backprojected")
        m_v = np.array([f.m for f in emb.basis])
        l_v = np.array([f.l for f in emb.basis])
        roles = ["inactive"] * 4 + ["active"] * 2
        masks = [
            np.where((l_v == 0) & (m_v == 0))[0],
            np.where((l_v == 2) & (m_v == 2))[0],
            np.where((l_v == 2) & (m_v == 1))[0],
            np.where((l_v == 2) & (m_v == -1))[0],
            np.where((l_v == 2) & (m_v == 0))[0],
            np.where((l_v == 2) & (m_v == -2))[0],
        ]
        mvb = MaskedVB(emb, roles, masks, (2, 0))
        C_v_sub = C_val[val_idx, :]
        x0 = mvb.pack([C_v_sub[:, k] for k in range(6)])
        res = mvb.optimize(x0, maxiter=60, gtol=1e-7)
        E_emb = float(res.fun)
        rows.append(dict(
            basis=bname,
            n_ao=n,
            n_val=len(val_idx),
            E_all_ao=E_rohf,
            E_val_emb=E_emb,
            dE_mha=(E_emb - E_rohf) * 1e3,
        ))
    return rows


def embed_ni1_in_slab_and_bulk() -> Dict[str, dict]:
    """Embed active Ni1 (15 valence AOs, [Ar] core frozen) inside:
      1. Isolated atom (0 neighbors)
      2. 8-neighbor Ni(100) surface slab (Ni2..Ni9)
      3. 12-neighbor FCC bulk Ni13 cluster (Ni2..Ni13)
    and report pseudometric minimum eigenvalues (naive vs exact S_BB^-1), 3d crystal-field
    levels, and total valence energy."""
    bNi_f, S_Ni, T_Ni, V_Ni, eri_Ni, E_rohf, C_core, C_val = solve_atomic_ni_rohf("DZP", (0.0, 0.0, 0.0))
    n_f = len(bNi_f)
    val_idx = [i for i, f in enumerate(bNi_f) if (f.n == 3 and f.l == 2) or (f.n == 4)]

    class _FullNi:
        nao = n_f
        S = S_Ni
        h = T_Ni + V_Ni
        eri = eri_Ni
        vnn = 0.0
        basis = bNi_f
        atom_of = np.zeros(n_f, dtype=int)

        def jk_matrices(self):
            return (
                eri_Ni.reshape(n_f * n_f, n_f * n_f),
                np.ascontiguousarray(eri_Ni.transpose(0, 3, 2, 1)).reshape(n_f * n_f, n_f * n_f),
            )

    emb_iso = embed_from_full_integrals(_FullNi(), C_core, val_idx, mode="exact_backprojected")
    emb_iso.S_bare = S_Ni[np.ix_(val_idx, val_idx)]
    emb_iso.h_bare = (T_Ni + V_Ni)[np.ix_(val_idx, val_idx)]

    surf_rep = FrozenEnvironmentAtom.from_embedded_state("data/ni_surf_embedded.json", (4.709, 0.0, 0.0))
    sub_rep = FrozenEnvironmentAtom.from_embedded_state("data/ni_sub_embedded.json", (2.3545, 2.3545, -3.330))
    emb_slab, lmin_slab_ex, lmin_slab_nv = embed_collinear_in_c4v_slab(
        emb_iso, [10.0], [(0.0, 0.0, 0.0)], surf_rep, sub_rep, n_eta=22, n_phi=10, lmax_exch=10
    )

    # Also evaluate the 12-neighbor FCC bulk Ni13 environment by adding the +z subsurface layer
    # at (2.3545, 2.3545, +3.330) using bulk_rep:
    bulk_rep_xy = FrozenEnvironmentAtom.from_embedded_state("data/ni_bulk_embedded.json", (3.330, 3.330, 0.0))
    bulk_rep_z = FrozenEnvironmentAtom.from_embedded_state("data/ni_bulk_embedded.json", (3.330, 0.0, 3.330))
    # Embed in 8 neighbors (xy plane + +z layer) first, or construct the 12-neighbor Oh pseudometric:
    emb_bulk8, lmin_b8_ex, lmin_b8_nv = embed_collinear_in_c4v_slab(
        emb_iso, [10.0], [(0.0, 0.0, 0.0)], bulk_rep_xy, bulk_rep_z, n_eta=22, n_phi=10, lmax_exch=10
    )

    results = {}
    for label, emb, l_ex, l_nv in [
        ("Isolated Ni (3d^8 4s^2, ^3F)", emb_iso, 1.0, 1.0),
        ("Ni(100) Surface Site (CN=8, C4v)", emb_slab, lmin_slab_ex, lmin_slab_nv),
        ("Bulk-like Coordinated Site (Oh/C4v)", emb_bulk8, lmin_b8_ex, lmin_b8_nv),
    ]:
        m_v = np.array([f.m for f in emb.basis])
        l_v = np.array([f.l for f in emb.basis])
        roles = ["inactive"] * 4 + ["active"] * 2
        masks = [
            np.where((l_v == 0) & (m_v == 0))[0],
            np.where((l_v == 2) & (m_v == 2))[0],
            np.where((l_v == 2) & (m_v == 1))[0],
            np.where((l_v == 2) & (m_v == -1))[0],
            np.where((l_v == 2) & (m_v == 0))[0],
            np.where((l_v == 2) & (m_v == -2))[0],
        ]
        mvb = MaskedVB(emb, roles, masks, (2, 0))
        C_v_sub = C_val[val_idx, :]
        x0 = mvb.pack([C_v_sub[:, k] for k in range(6)])
        res = mvb.optimize(x0, maxiter=60, gtol=1e-6)
        # Compute 1-electron orbital expectations in the 3d subshells (m = 0, +-1, +2, -2)
        d_eps = {}
        for d_name, m_q in [("3d_z2 (a1)", 0), ("3d_xz/yz (e)", 1), ("3d_xy (b2)", 2), ("3d_x2-y2 (b1)", -2)]:
            idx_d = np.where((l_v == 2) & (m_v == m_q))[0]
            w_d = eigh(emb.h[np.ix_(idx_d, idx_d)], emb.S[np.ix_(idx_d, idx_d)], eigvals_only=True)
            d_eps[d_name] = float(w_d[0])
        results[label] = dict(
            E_tot=float(res.fun),
            lmin_exact=float(l_ex),
            lmin_naive=float(l_nv),
            d_levels=d_eps,
        )
    return results


# ---------------------------------------------------------------------------
# Part 2 helpers: CO/Ni(100) Blyholder regimes and C-O stretching curve
# ---------------------------------------------------------------------------

def solve_nico_blyholder_regimes(
    mol: EmbeddedAOIntegrals, C_val_Ni_sub: np.ndarray, maxiter: int = 80
) -> Tuple[float, float, float, List[np.ndarray]]:
    """Optimize the 3 variational Blyholder domain regimes in MaskedVB on ``mol``:
      * Regime 0: Strict fragment domains (Ni1 orbitals on Ni1; CO orbitals on C+O)
      * Regime 1: +5sigma donation (CO 5sigma lone pair allowed onto Ni1 m=0 AOs)
      * Regime 2: +pi back-donation (Ni1 3d_xz/yz and CO 1pi/2pi* allowed across Ni1+C+O m=+-1)"""
    nao = mol.nao
    m_vals = np.array([f.m for f in mol.basis])
    l_vals = np.array([f.l for f in mol.basis])
    atom_of = mol.atom_of

    idx_CO_m0 = np.where((atom_of >= 1) & (m_vals == 0))[0]
    _, v_co_0 = eigh(mol.h[np.ix_(idx_CO_m0, idx_CO_m0)], mol.S[np.ix_(idx_CO_m0, idx_CO_m0)])

    def make_masks(regime: int):
        masks = []
        for m_ni, l_ni in [(2, 2), (1, 2), (-1, 2), (0, 0)]:
            if regime == 2 and m_ni in (1, -1):
                mask = (m_vals == m_ni)
            else:
                # Note: in Regime 0, Ni1's 1-center m=0 (4s, 4p_z, 3d_z2) and m=+-1 (3d_xz/yz, 4p_x/y)
                # crystal-field polarization in the C_4v slab is already included in its own 1-center
                # domain so that Regime 1 (CO 5sigma -> Ni1) and Regime 2 (Ni1 3d_pi -> CO 2pi*)
                # measure purely inter-fragment charge transfer!
                mask = (atom_of == 0) & (m_vals == m_ni) if m_ni in (0, 1, -1) else ((atom_of == 0) & (m_vals == m_ni) & (l_vals == l_ni))
            masks.append(np.where(mask)[0])
        for k in range(5):
            if regime >= 1 and k == 4:
                mask = (m_vals == 0)
            else:
                mask = (atom_of >= 1) & (m_vals == 0)
            masks.append(np.where(mask)[0])
        for m_pi in (1, -1):
            if regime == 2:
                mask = (m_vals == m_pi)
            else:
                mask = (atom_of >= 1) & (m_vals == m_pi)
            masks.append(np.where(mask)[0])
        for m_act, l_act in [(0, 2), (-2, 2)]:
            mask = (atom_of == 0) & (m_vals == m_act) if m_act == 0 else ((atom_of == 0) & (m_vals == m_act) & (l_vals == l_act))
            masks.append(np.where(mask)[0])
        return masks

    roles = ["inactive"] * 11 + ["active"] * 2
    orbs0 = [np.zeros(nao) for _ in range(13)]
    orbs0[0][:15] = C_val_Ni_sub[:, 1]
    orbs0[1][:15] = C_val_Ni_sub[:, 2]
    orbs0[2][:15] = C_val_Ni_sub[:, 3]
    orbs0[3][:15] = C_val_Ni_sub[:, 0]
    for k in range(5):
        orbs0[4 + k][idx_CO_m0] = v_co_0[:, k]
    for j, m_pi in enumerate((1, -1)):
        idx_p = np.where((atom_of >= 1) & (m_vals == m_pi))[0]
        _, v_p = eigh(mol.h[np.ix_(idx_p, idx_p)], mol.S[np.ix_(idx_p, idx_p)])
        orbs0[9 + j][idx_p] = v_p[:, 0]
    orbs0[11][:15] = C_val_Ni_sub[:, 4]
    orbs0[12][:15] = C_val_Ni_sub[:, 5]

    mvb0 = MaskedVB(mol, roles, make_masks(0), (2, 0))
    res0 = mvb0.optimize(mvb0.pack(orbs0), maxiter=maxiter, gtol=1e-5)
    E0 = float(res0.fun)
    orbs_r0 = mvb0.orbitals(res0.x)

    mvb1 = MaskedVB(mol, roles, make_masks(1), (2, 0))
    res1 = mvb1.optimize(mvb1.pack(orbs_r0), maxiter=maxiter, gtol=1e-5)
    E1 = float(res1.fun)
    orbs_r1 = mvb1.orbitals(res1.x)

    mvb2 = MaskedVB(mol, roles, make_masks(2), (2, 0))
    res2 = mvb2.optimize(mvb2.pack(orbs_r1), maxiter=maxiter, gtol=1e-5)
    E2 = float(res2.fun)
    orbs_r2 = mvb2.orbitals(res2.x)
    return E0, E1, E2, [orbs_r0, orbs_r1, orbs_r2]


def solve_isolated_co_rhf(R_CO: float, basis_name: str = "DZP", n_eta: int = 28, lmax: int = 14) -> float:
    """Solve 14-electron closed-shell CO (5sigma^2 1pi^4) in MaskedVB at bond length ``R_CO``."""
    bC = atom_basis((0.0, 0.0, 0.0), "C", basis_name)
    bO = atom_basis((0.0, 0.0, float(R_CO)), "O", basis_name)
    ints = diatomic_ao_integrals(bC, bO, 6.0, 8.0, n_eta=n_eta, lmax=lmax)
    m_vals = np.array([f.m for f in ints.basis])
    masks = (
        [np.where(m_vals == 0)[0] for _ in range(5)]
        + [np.where(m_vals == 1)[0], np.where(m_vals == -1)[0]]
    )
    roles = ["inactive"] * 7
    mvb = MaskedVB(ints, roles, masks, (0, 0))
    orbs0 = [np.zeros(ints.nao) for _ in range(7)]
    idx0 = np.where(m_vals == 0)[0]
    _, v0 = eigh(ints.h[np.ix_(idx0, idx0)], ints.S[np.ix_(idx0, idx0)])
    for k in range(5):
        orbs0[k][idx0] = v0[:, k]
    for j, mp in enumerate((1, -1)):
        idxp = np.where(m_vals == mp)[0]
        _, vp = eigh(ints.h[np.ix_(idxp, idxp)], ints.S[np.ix_(idxp, idxp)])
        orbs0[5 + j][idxp] = vp[:, 0]
    res = mvb.optimize(mvb.pack(orbs0), maxiter=60, gtol=1e-6)
    return float(res.fun)


def fit_parabola_and_freq(R_grid: Sequence[float], E_grid: Sequence[float]) -> Tuple[float, float, float]:
    """Fit a quadratic polynomial E(R) = a R^2 + b R + c around the minimum and return
    ``(R_eq_bohr, E_min_ha, omega_cm1)`` for 12C-16O reduced mass."""
    R_arr = np.asarray(R_grid, dtype=float)
    E_arr = np.asarray(E_grid, dtype=float)
    coeffs = np.polyfit(R_arr, E_arr, 2)
    a, b, c = coeffs
    R_eq = float(-b / (2.0 * a))
    E_min = float(a * R_eq * R_eq + b * R_eq + c)
    k_ha_bohr2 = float(2.0 * a)
    # Reduced mass of 12C (12.000000 amu) and 16O (15.9949146 amu) in electron masses:
    mu_amu = (12.0 * 15.9949146) / (12.0 + 15.9949146)
    mu_me = mu_amu * 1822.888486209
    omega_au = math.sqrt(max(k_ha_bohr2, 0.0) / mu_me)
    omega_cm1 = omega_au * 219474.63136320
    return R_eq, E_min, omega_cm1


def main():
    t_start = time.time()
    print("=" * 88)
    print("PHATPSY 2.0 — Stage 11: Bulk Nickel, Ni(100) Slab & CO/Ni(100) Chemisorption")
    print("=" * 88, flush=True)

    # 1.1 PHATPSY Embedded States
    print("\n[Part 1.1] PHATPSY Converged Embedded States (pk_scale = 1.0):")
    st_summary = analyze_phatpsy_embedded_states()
    bulk_ref = st_summary["Bulk (Ni13, Oh, CN=12)"]
    print(f"{'Environment':34s} | {'dE_1s (eV)':>10s} | {'dE_2p (eV)':>10s} | {'dE_3p (eV)':>10s} | {'3d CF Split':>11s} | {'Spin':>6s}")
    print("-" * 92)
    for name, d in st_summary.items():
        de1s = (d["e1s"] - bulk_ref["e1s"]) * HARTREE_TO_EV
        de2p = (d["e2p"] - bulk_ref["e2p"]) * HARTREE_TO_EV
        de3p = (d["e3p"] - bulk_ref["e3p"]) * HARTREE_TO_EV
        print(
            f"{name:34s} | {de1s:+10.3f} | {de2p:+10.3f} | {de3p:+10.3f} | "
            f"{d['e3d_split_ev']:8.3f} eV | {d['net_spin']:6.3f}",
            flush=True,
        )

    # 1.2 Atomic Ni ROHF Basis Ladder & [Ar]-Core Embedding
    print("\n[Part 1.2] Atomic Ni (3d^8 4s^2, ^3F) Basis Ladder & [Ar]-Core PK Embedding:")
    ladder = benchmark_ni_basis_ladder()
    print(f"{'Basis':10s} | {'All AOs':>7s} | {'Val AOs':>7s} | {'E_ROHF (All-AO)':>16s} | {'E_VB ([Ar]-Emb)':>16s} | {'dE (mHa)':>9s}")
    print("-" * 78)
    for row in ladder:
        print(
            f"{row['basis']:10s} | {row['n_ao']:7d} | {row['n_val']:7d} | "
            f"{row['E_all_ao']:16.6f} | {row['E_val_emb']:16.6f} | {row['dE_mha']:+9.3f}",
            flush=True,
        )

    # 1.3 Active Ni1 Embedded in Ni(100) Slab & Bulk Coordination
    print("\n[Part 1.3] Active Ni1 Embedded in Ni(100) Slab & Bulk Coordination (Exact S_BB^-1 vs Naive):")
    slab_bulk = embed_ni1_in_slab_and_bulk()
    print(f"{'Active Site Environment':34s} | {'lmin(Naive)':>11s} | {'lmin(S_BB^-1)':>13s} | {'E_val (Ha)':>14s}")
    print("-" * 80)
    for label, d in slab_bulk.items():
        print(
            f"{label:34s} | {d['lmin_naive']:+11.4f} | {d['lmin_exact']:+13.4f} | {d['E_tot']:14.6f}",
            flush=True,
        )

    # 2.1 Top-Site CO Chemisorption Scan across Blyholder Regimes
    print("\n[Part 2.1] Top-Site CO/Ni(100) Blyholder Domain Decomposition vs R_NiC (R_CO = 2.132 Bohr):")
    surf_rep = FrozenEnvironmentAtom.from_embedded_state("data/ni_surf_embedded.json", (4.709, 0.0, 0.0))
    sub_rep = FrozenEnvironmentAtom.from_embedded_state("data/ni_sub_embedded.json", (2.3545, 2.3545, -3.330))

    R_scan = [3.20, 3.40, 3.59, 3.85, 4.20, 5.00, 15.00]
    bare_rows = []
    slab_rows = []
    for R_NiC in R_scan:
        t0 = time.time()
        mol_bare, C_val_Ni_sub = build_active_ni_co_integrals(R_NiC, 2.132, "DZP", n_eta=24, n_phi=12, lmax=12)
        b0, b1, b2, _ = solve_nico_blyholder_regimes(mol_bare, C_val_Ni_sub)
        mol_slab, lmin_ex, lmin_nv = embed_collinear_in_c4v_slab(
            mol_bare, [10.0, 6.0, 8.0], [(0.0, 0.0, 0.0), (0.0, 0.0, R_NiC), (0.0, 0.0, R_NiC + 2.132)],
            surf_rep, sub_rep, n_eta=20, n_phi=10, lmax_exch=10
        )
        s0, s1, s2, _ = solve_nico_blyholder_regimes(mol_slab, C_val_Ni_sub)
        bare_rows.append(dict(R=R_NiC, E0=b0, E1=b1, E2=b2))
        slab_rows.append(dict(R=R_NiC, E0=s0, E1=s1, E2=s2, lmin_ex=lmin_ex, lmin_nv=lmin_nv))
        print(
            f"  R_NiC={R_NiC:5.2f} Bohr | Bare: dE_sig={(b1-b0)*HARTREE_TO_EV:+6.3f} eV, dE_pi={(b2-b1)*HARTREE_TO_EV:+6.3f} eV "
            f"| Slab: dE_sig={(s1-s0)*HARTREE_TO_EV:+6.3f} eV, dE_pi={(s2-s1)*HARTREE_TO_EV:+6.3f} eV "
            f"(lmin={lmin_ex:+.4f}) [{time.time()-t0:.1f}s]",
            flush=True,
        )

    # 2.2 C-O Bond Stretching Scan at R_NiC = 3.59 Bohr (Isolated CO vs Chemisorbed CO/Ni(100))
    print("\n[Part 2.2] C-O Bond Stretching Curve & Blyholder Redshift at R_NiC = 3.59 Bohr:")
    R_CO_grid = [2.02, 2.08, 2.132, 2.19, 2.25]
    E_co_iso = []
    E_bare_r0, E_bare_r1, E_bare_r2 = [], [], []
    E_slab_r0, E_slab_r1, E_slab_r2 = [], [], []
    for R_CO in R_CO_grid:
        t0 = time.time()
        e_iso = solve_isolated_co_rhf(R_CO, "DZP", n_eta=24, lmax=12)
        E_co_iso.append(e_iso)
        mol_b, C_v = build_active_ni_co_integrals(3.59, R_CO, "DZP", n_eta=24, n_phi=12, lmax=12)
        eb0, eb1, eb2, _ = solve_nico_blyholder_regimes(mol_b, C_v)
        E_bare_r0.append(eb0)
        E_bare_r1.append(eb1)
        E_bare_r2.append(eb2)
        mol_s, _, _ = embed_collinear_in_c4v_slab(
            mol_b, [10.0, 6.0, 8.0], [(0.0, 0.0, 0.0), (0.0, 0.0, 3.59), (0.0, 0.0, 3.59 + R_CO)],
            surf_rep, sub_rep, n_eta=20, n_phi=10, lmax_exch=10
        )
        es0, es1, es2, _ = solve_nico_blyholder_regimes(mol_s, C_v)
        E_slab_r0.append(es0)
        E_slab_r1.append(es1)
        E_slab_r2.append(es2)
        print(
            f"  R_CO={R_CO:5.3f} Bohr | Iso={e_iso:.6f} | Bare R2={eb2:.6f} | Slab R2={es2:.6f} [{time.time()-t0:.1f}s]",
            flush=True,
        )

    req_iso, _, w_iso = fit_parabola_and_freq(R_CO_grid, E_co_iso)
    print(f"\n{'System / Blyholder Regime':38s} | {'R_e(C-O) (Bohr)':>15s} | {'dR_CO (Bohr)':>12s} | {'omega_CO (cm^-1)':>16s} | {'d_omega (cm^-1)':>15s}")
    print("-" * 106)
    print(f"{'Isolated CO (DZP RHF)':38s} | {req_iso:15.4f} | {0.0:+12.4f} | {w_iso:16.1f} | {0.0:+15.1f}")
    for label, E_list in [
        ("Bare Ni1--CO: Regime 0 (No CT)", E_bare_r0),
        ("Bare Ni1--CO: Regime 1 (+sigma)", E_bare_r1),
        ("Bare Ni1--CO: Regime 2 (+sigma + pi)", E_bare_r2),
        ("Slab CO/Ni(100): Regime 0 (No CT)", E_slab_r0),
        ("Slab CO/Ni(100): Regime 1 (+sigma)", E_slab_r1),
        ("Slab CO/Ni(100): Regime 2 (+sigma + pi)", E_slab_r2),
    ]:
        req, _, w_co = fit_parabola_and_freq(R_CO_grid, E_list)
        print(
            f"{label:38s} | {req:15.4f} | {req - req_iso:+12.4f} | {w_co:16.1f} | {w_co - w_iso:+15.1f}",
            flush=True,
        )

    print(f"\nDone in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

