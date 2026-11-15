"""Regression tests for Localized VB analytical gradients, STO-NG integrals, P-RFO, and EO -> AA study."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from phatpsy.two_center_integrals import ProlateGrid, STO
from phatpsy.dimer import MullerBrownSurface, _bofill_update, prfo_step
from phatpsy.multicenter_integrals import (
    _Frame,
    _cross_coulomb,
    _eval,
    molecular_ao_integrals,
    sto_ng_ao_integrals,
)
from phatpsy.schema import AtomConfig, CalculationDeck, ControlConfig
from phatpsy.vb_driver import LocalizedVBDriver, _build_general_vb_setup, _resolve_atom_stos

REPO_ROOT = Path(__file__).resolve().parent.parent


def _real_stos_at(element: str, xyz: list[float], basis_name: str = "SZ") -> list[STO]:
    deck = CalculationDeck(
        geometry=[AtomConfig(element=element, xyz=xyz, basis=basis_name)],
        control=ControlConfig.localized_vb(basis=basis_name),
    )
    elem_map = deck.resolve_elements()
    stos, _ = _resolve_atom_stos(deck.geometry[0], deck, elem_map)
    return stos


def test_cross_coulomb_cached_vs_uncached_parity():
    """Cached Neumann source + target grid quantities match uncached _cross_coulomb to < 1e-13."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.5, 1.3, 0.2],
        ],
        dtype=float,
    )
    basis = [STO(tuple(coords[ia]), 1, 0, 0, 1.0) for ia in range(3)]
    lmax, mmax = 8, 3

    g_src = ProlateGrid(coords[0], coords[1], zeta_min=1.0, n_eta=12, n_phi=10)
    g_tgt = ProlateGrid(coords[0], coords[2], zeta_min=1.0, n_eta=12, n_phi=10)

    src = _Frame(0, 1, g_src)
    src.pairs = [(0, 1), (0, 0)]
    bfs_s = [0, 1]
    idx_s = {k: r for r, k in enumerate(bfs_s)}
    chi_s = np.array([_eval(basis[k], g_src.X, g_src.Y, g_src.Z) for k in bfs_s])
    dens_s = np.array([chi_s[idx_s[m]] * chi_s[idx_s[n]] for m, n in src.pairs])
    Peta_s, _, _ = g_src._neumann_tables(lmax, mmax)
    proj_src = g_src._project_batch(dens_s, Peta_s, lmax, mmax)

    tgt = _Frame(0, 2, g_tgt)
    tgt.pairs = [(0, 2), (2, 2)]
    bfs_t = [0, 2]
    idx_t = {k: r for r, k in enumerate(bfs_t)}
    chi_t = np.array([_eval(basis[k], g_tgt.X, g_tgt.Y, g_tgt.Z) for k in bfs_t])
    w_flat = g_tgt.weights.ravel()
    keep_nz = w_flat != 0.0
    tgt.pts_nz = np.stack([g_tgt.X.ravel()[keep_nz], g_tgt.Y.ravel()[keep_nz], g_tgt.Z.ravel()[keep_nz]], axis=1)
    tgt.w_nz = w_flat[keep_nz]
    tgt.chi_nz = np.ascontiguousarray(chi_t.reshape(len(bfs_t), -1)[:, keep_nz])
    tgt.bfs_idx = idx_t

    G_cached = _cross_coulomb(src, proj_src, tgt, None, lmax, mmax)

    # Now bypass all frame caches and recompute via chi_fn
    src._src_prep.clear()
    tgt.pts_nz = None
    tgt.w_nz = None
    tgt.chi_nz = None
    tgt.bfs_idx = None

    def _chi_fn(pts: np.ndarray):
        chi = np.array([_eval(basis[k], pts[:, 0], pts[:, 1], pts[:, 2]) for k in bfs_t])
        return chi, idx_t

    G_uncached = _cross_coulomb(src, proj_src, tgt, _chi_fn, lmax, mmax)
    assert np.max(np.abs(G_cached - G_uncached)) < 1e-13


def test_sto_ng_invariance_and_convergence():
    """STO-KG converges smoothly toward exact prolate quadrature and is rigid-motion invariant."""
    coords_h2o = [
        [0.0, 0.0, 0.0],
        [1.48, 0.05, 1.08],
        [-1.39, -0.04, 1.15],
    ]
    specs = [
        (8.0, coords_h2o[0], _real_stos_at("O", coords_h2o[0], "SZ")),
        (1.0, coords_h2o[1], _real_stos_at("H", coords_h2o[1], "SZ")),
        (1.0, coords_h2o[2], _real_stos_at("H", coords_h2o[2], "SZ")),
    ]
    ex = molecular_ao_integrals(specs, n_eta=32, n_phi=16, lmax=16, mmax=4)
    s4 = sto_ng_ao_integrals(specs, n_prim=4)
    s8 = sto_ng_ao_integrals(specs, n_prim=8)
    err4 = float(np.max(np.abs(s4.S - ex.S)))
    err8 = float(np.max(np.abs(s8.S - ex.S)))
    assert err8 < err4
    assert err8 < 2.0e-4

    # Check rigid translation + SO(3) rotation invariance of MaskedVB E & gradient
    ctrl = ControlConfig.localized_vb(
        method="masked_vb",
        vb_wavefunction="cas",
        domain_mode="bond_local",
        core_embedding="all_electron",
        integral_backend="sto_ng",
        sto_ng_primitives=6,
        convergence=1e-9,
        max_cycles=400,
    )
    deck0 = CalculationDeck(
        geometry=[
            AtomConfig(id="O1", element="O", xyz=coords_h2o[0], basis="SZ"),
            AtomConfig(id="H1", element="H", xyz=coords_h2o[1], basis="SZ"),
            AtomConfig(id="H2", element="H", xyz=coords_h2o[2], basis="SZ"),
        ],
        control=ctrl,
    )
    axis = np.array([0.3, -0.7, 0.65], dtype=float)
    axis /= np.linalg.norm(axis)
    theta = 0.73
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    shift = np.array([1.7, -0.9, 2.3], dtype=float)

    coords_rot = (np.asarray(coords_h2o) @ R.T + shift[None, :]).tolist()
    deck_rot = CalculationDeck(
        geometry=[
            AtomConfig(id="O1", element="O", xyz=coords_rot[0], basis="SZ"),
            AtomConfig(id="H1", element="H", xyz=coords_rot[1], basis="SZ"),
            AtomConfig(id="H2", element="H", xyz=coords_rot[2], basis="SZ"),
        ],
        control=ctrl,
    )
    drv = LocalizedVBDriver(quiet=True)
    st0, g0 = drv.energy_and_gradient(deck0)
    st1, g1 = drv.energy_and_gradient(deck_rot)
    assert st0.converged and st1.converged
    assert abs(st1.total_energy - st0.total_energy) < 1e-10
    assert float(np.max(np.abs(g1 - g0 @ R.T))) < 1e-8


def test_localized_vb_analytical_gradient_vs_fd_h2o():
    """Analytical CAS/MaskedVB nuclear gradients match central FD on H2O for auto and explicit specs."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.48, 0.05, 1.08],
            [-1.39, -0.04, 1.15],
        ],
        dtype=float,
    )
    specs_list = [
        {"role": "inactive", "atoms": ["O1"]},
        {"role": "inactive", "atoms": ["O1"]},
        {"role": "inactive", "atoms": ["O1"]},
        {"role": "active", "atoms": ["O1", "H1", "H2"]},
        {"role": "active", "atoms": ["O1", "H1", "H2"]},
        {"role": "active", "atoms": ["O1", "H1", "H2"]},
        {"role": "active", "atoms": ["O1", "H1", "H2"]},
    ]
    drv = LocalizedVBDriver(quiet=True)
    step = 1.0e-4

    for use_specs, check_pairs in [
        (None, [(ia, ax) for ia in range(3) for ax in range(3)]),
        (specs_list, [(1, 0), (1, 2)]),
    ]:
        ctrl = ControlConfig.localized_vb(
            method="masked_vb",
            vb_wavefunction="cas",
            domain_mode="bond_local",
            core_embedding="all_electron",
            integral_backend="sto_ng",
            sto_ng_primitives=6,
            vb_orbital_specs=use_specs,
            convergence=1e-9,
            max_cycles=400,
        )

        def _mk(c: np.ndarray) -> CalculationDeck:
            return CalculationDeck(
                geometry=[
                    AtomConfig(id="O1", element="O", xyz=c[0].tolist(), basis="SZ"),
                    AtomConfig(id="H1", element="H", xyz=c[1].tolist(), basis="SZ"),
                    AtomConfig(id="H2", element="H", xyz=c[2].tolist(), basis="SZ"),
                ],
                control=ctrl,
            )

        st0, g_an = drv.energy_and_gradient(_mk(coords))
        assert st0.converged
        assert float(np.max(np.abs(np.sum(g_an, axis=0)))) < 1e-12

        warm = st0.get("localized_orbitals_embedded", st0.get("localized_orbitals"))
        for ia, ax in check_pairs:
            c_p = coords.copy()
            c_m = coords.copy()
            c_p[ia, ax] += step
            c_m[ia, ax] -= step
            e_p = drv.run_deck(_mk(c_p), warm_orbitals=warm).total_energy
            e_m = drv.run_deck(_mk(c_m), warm_orbitals=warm).total_energy
            g_fd = (e_p - e_m) / (2.0 * step)
            assert abs(g_an[ia, ax] - g_fd) < 1e-6


def test_acetaldehyde_carbonyl_vb_setup():
    """Terminal C=O in Acetaldehyde keeps 2 inactive O lone pairs + 3x 1s cores (5 inactive, 14 active)."""
    aa_coords = [
        [1.4675, 0.0, -0.7621],
        [-1.5168, 0.0, -0.7990],
        [-2.8642, 0.0, 1.3557],
        [2.2925, 1.7867, -1.7538],
        [2.2925, -1.7867, -1.7538],
        [2.1971, 0.0, 1.3735],
        [-2.5886, 0.0, -2.7504],
    ]
    atoms = [
        AtomConfig(id=aid, element=el, xyz=aa_coords[i], basis="SZ", active=True)
        for i, (aid, el) in enumerate(
            [("C1", "C"), ("C2", "C"), ("O1", "O"), ("H1", "H"), ("H2", "H"), ("H3", "H"), ("H4", "H")]
        )
    ]
    deck = CalculationDeck(geometry=atoms, control=ControlConfig.localized_vb(basis="SZ"))
    elem_map = deck.resolve_elements()
    specs = []
    for a in atoms:
        stos, _ = _resolve_atom_stos(a, deck, elem_map)
        specs.append((6.0 if a.element == "C" else (8.0 if a.element == "O" else 1.0), tuple(a.xyz), stos))
    ints = sto_ng_ao_integrals(specs, n_prim=6)
    roles, _masks, nelec_act, _orbs = _build_general_vb_setup(
        atoms, ints, domain_mode="bond_local", core_embedding="all_electron", net_spin_override=0, charge=0
    )
    assert roles.count("inactive") == 5
    assert roles.count("active") == 14
    assert nelec_act == (7, 7)


def test_prfo_step_muller_brown_saddle1():
    """Baker P-RFO + Bofill SR1/PSB updates locate Muller-Brown SADDLE_1 cleanly."""
    x = np.array([-0.75, 0.58], dtype=float)
    eps = 1e-5

    def _num_hess(r: np.ndarray) -> np.ndarray:
        H = np.zeros((2, 2), dtype=float)
        for k in range(2):
            rp = r.copy()
            rm = r.copy()
            rp[k] += eps
            rm[k] -= eps
            H[:, k] = (MullerBrownSurface.gradient(rp) - MullerBrownSurface.gradient(rm)) / (2.0 * eps)
        return 0.5 * (H + H.T)

    g = MullerBrownSurface.gradient(x)
    H = _num_hess(x)
    for _ in range(15):
        if np.linalg.norm(g) < 1e-8:
            break
        dx = prfo_step(g, H, trust_radius=0.15)
        x_next = x + dx
        g_next = MullerBrownSurface.gradient(x_next)
        H = _bofill_update(H, dx, g_next - g)
        x, g = x_next, g_next

    assert np.linalg.norm(x - MullerBrownSurface.SADDLE_1) < 1e-5
    assert np.linalg.norm(g) < 1e-7
    evals = np.linalg.eigvalsh(_num_hess(x))
    assert evals[0] < 0.0 < evals[1]


def test_eo_to_aa_stationary_decks_live_anti_circular():
    """Live evaluation of EO, TS, and AA YAML decks reproduces zero forces and variational ordering."""
    summary_path = REPO_ROOT / "data" / "eo_to_aa_localized_vb" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    drv = LocalizedVBDriver(quiet=True)

    results = {}
    for key, fname in [
        ("EO", "EO_localized_vb.yaml"),
        ("TS", "EO_to_AA_TS_localized_vb.yaml"),
        ("AA", "AA_localized_vb.yaml"),
    ]:
        deck = CalculationDeck.from_yaml(REPO_ROOT / "examples" / fname)
        st, g = drv.energy_and_gradient(deck)
        assert st.converged is True
        assert float(np.linalg.norm(g)) < 5.0e-4
        nat_occ = np.asarray(st["active_natural_occupations"], dtype=float)
        assert nat_occ.shape == (4,)
        assert np.all((nat_occ >= -1e-6) & (nat_occ <= 2.0 + 1e-6))
        assert abs(float(np.sum(nat_occ)) - 4.0) < 1e-4
        assert abs(float(st.total_energy) - float(summary["stationary_points"][key]["energy_ha"])) < 1e-5
        results[key] = float(st.total_energy)

    assert results["AA"] < results["EO"] < results["TS"]


def test_eo_to_aa_two_step_ts1_biradical_and_irc1_live():
    """Live evaluation of Step-1 TS1, BIRAD intermediate, and Step-1 ring-opening IRC."""
    import copy

    data_dir = REPO_ROOT / "data" / "eo_to_aa_localized_vb"
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    irc1 = json.loads((data_dir / "irc_step1_ring_opening.json").read_text(encoding="utf-8"))
    drv = LocalizedVBDriver(quiet=True)

    sp = summary["stationary_points"]
    for key in ("EO", "TS1", "BIRAD", "TS", "AA"):
        assert sp[key]["converged"] is True
    assert sp["BIRAD"]["num_imaginary"] == 0
    assert sp["TS1"]["num_imaginary"] == 1

    live_e = {}
    for key, fname in [
        ("TS1", "EO_to_Biradical_TS1_localized_vb.yaml"),
        ("BIRAD", "EO_Biradical_localized_vb.yaml"),
    ]:
        deck = CalculationDeck.from_yaml(REPO_ROOT / "examples" / fname)
        st, g = drv.energy_and_gradient(deck)
        assert st.converged is True
        assert float(np.linalg.norm(g)) < 5.0e-4
        assert abs(float(st.total_energy) - float(sp[key]["energy_ha"])) < 1.0e-5
        nat_occ = np.asarray(st["active_natural_occupations"], dtype=float)
        assert nat_occ.shape == (4,)
        assert abs(float(np.sum(nat_occ)) - 4.0) < 1.0e-4
        if key == "BIRAD":
            assert 0.95 < float(nat_occ[1]) < 1.05
            assert 0.95 < float(nat_occ[2]) < 1.05
        live_e[key] = float(st.total_energy)

    e_aa = float(sp["AA"]["energy_ha"])
    e_eo = float(sp["EO"]["energy_ha"])
    e_ts2 = float(sp["TS"]["energy_ha"])
    assert e_aa < e_eo < live_e["BIRAD"] < live_e["TS1"] < e_ts2

    pts = irc1["points"]
    assert len(pts) >= 10
    s_vals = [float(p["s_bohr_amu_half"]) for p in pts]
    assert all(s_vals[i] < s_vals[i + 1] for i in range(len(s_vals) - 1))
    peak_idx = int(np.argmax([float(p["energy_ha"]) for p in pts]))
    assert abs(float(pts[peak_idx]["s_bohr_amu_half"])) < 1e-8

    eo_end = pts[0]
    birad_end = pts[-1]
    assert eo_end["metrics"]["R_C1_O1_ang"] < 1.68
    assert eo_end["metrics"]["angle_C1_C2_O1_deg"] < 66.0
    assert birad_end["metrics"]["R_C1_O1_ang"] > 2.50
    assert birad_end["metrics"]["angle_C1_C2_O1_deg"] > 108.0

    # Live anti-circular energy verification on an interior ring-closing (s < 0) IRC 1 point
    interior_pts = [
        p for p in pts
        if float(p["s_bohr_amu_half"]) < -0.5 and float(p["metrics"]["R_C1_O1_ang"]) < 2.20
    ]
    assert interior_pts
    sample_pt = interior_pts[len(interior_pts) // 2]
    eo_deck = CalculationDeck.from_yaml(REPO_ROOT / "examples" / "EO_localized_vb.yaml")
    sample_deck = copy.deepcopy(eo_deck)
    for atom, xyz in zip(sample_deck.geometry, sample_pt["coordinates"]):
        atom.xyz = [float(c) for c in xyz]
    st_sample = drv.run_deck(sample_deck)
    assert st_sample.converged is True
    assert abs(float(st_sample.total_energy) - float(sample_pt["energy_ha"])) < 1.0e-5

