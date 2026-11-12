"""
Stage 9 Benchmark: Localized-VB Active-Site Embedding in a PHATPSY Environment.

Evaluates the three-level Phillips-Kleinman / frozen-density embedding hierarchy:
  * Level E1 (exact_backprojected): exact L^T S L, exact L^T F^B L, exact L^{x4} V
  * Level E2 (unprojected_eri):     exact L^T S L, exact L^T F^B L, unprojected V_AAAA
  * Level E3 (eri_free_2c):         embed_diatomic_in_environment (analytic 1c Coulomb + 2c PK)
  * Control  (bare_fock_no_pk):     F^B_AA without PK orthogonality projection

across:
  1. Core-valence embedding in heavy diatomics (LiH, HF, CO, N2 in DZP) vs All-Electron VB.
  2. Spectroscopic constants (R_e, omega_e, D_e) of LiH (DZP) across the PES.
  3. External PHATPSY environment embedding: H2 bond approaching a frozen PHATPSY He atom
     extracted via FrozenEnvironmentAtom.from_phatpsy_state.
"""
from __future__ import annotations

import numpy as np

from phatpsy.driver import PhatpsyDriver
from phatpsy.embedding import (
    FrozenEnvironmentAtom,
    embed_diatomic_in_environment,
    embed_from_full_integrals,
)
from phatpsy.localized_vb import (
    LocalizedVBModel,
    MaskedVB,
    atom_basis,
    diatomic_ao_integrals,
    optimize,
)
from phatpsy.multicenter_integrals import molecular_ao_integrals
from tools.localized_vb_scan import NELEC, atom_reference, build_model, initial_orbitals


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_MEV = 27211.386245988
AMU = 1822.888486
MU_LIH = 7.016003 * 1.007825 / 8.023828

DIATOMICS = {
    "LiH": ("Li", "H", 3.0154),
    "HF":  ("F",  "H", 1.7328),
    "CO":  ("C",  "O", 2.1322),
    "N2":  ("N",  "N", 2.0744),
}


def fit_pes(R: np.ndarray, E: np.ndarray, mu_amu: float):
    c = np.polyfit(R, E, min(3, len(R) - 1))
    r = np.linspace(R.min(), R.max(), 4001)
    e = np.polyval(c, r)
    k = int(np.argmin(e))
    k2 = np.polyval(np.polyder(c, 2), r[k])
    we = float(np.sqrt(max(0.0, k2) / (mu_amu * AMU)) * 219474.63)
    return dict(R_e=float(r[k]), omega_e_cm1=we, E_min=float(e[k]))


def valence_specs_and_seed(el_a: str, el_b: str, ints, x_all_opt: np.ndarray, all_model: LocalizedVBModel):
    """Build a valence-only LocalizedVBModel (dropping 1s core inactive specs, pair == -1
    on heavy atoms Z > 2) and seed it from the optimized all-electron orbitals."""
    all_orbs = all_model.orbitals(x_all_opt)
    z_a, z_b = NELEC[el_a], NELEC[el_b]
    core_indices = []
    if z_a > 2:
        core_indices.append(next(i for i, s in enumerate(all_model.specs)
                                 if s.atom == 0 and s.role == "inactive" and s.sym == "sigma" and s.pair == -1))
    if z_b > 2:
        core_indices.append(next(i for i, s in enumerate(all_model.specs)
                                 if s.atom == 1 and s.role == "inactive" and s.sym == "sigma" and s.pair == -1))
    val_specs = [s for i, s in enumerate(all_model.specs) if i not in core_indices]
    val_orbs = [o for i, o in enumerate(all_orbs) if i not in core_indices]
    val_model = LocalizedVBModel(ints, val_specs, mirror=all_model.mirror, method="cas")
    return val_model, val_model.pack(val_orbs)


def run_core_valence_table(basis_name: str = "DZP"):
    print("=" * 98)
    print(f"1. CORE-VALENCE EMBEDDING ON HEAVY DIATOMICS ({basis_name})")
    print("=" * 98)
    header = (f"{'Mol':<5} {'R':>6} {'E_all_VB':>14} {'E1 (exact PK)':>14} "
              f"{'E2 (unproj V)':>14} {'E3 (2c no-ERI)':>14} {'No-PK collapse':>15}")
    print(header)
    print("-" * len(header))

    for mol_name, (e_a, e_b, R) in DIATOMICS.items():
        z_a, z_b = float(NELEC[e_a]), float(NELEC[e_b])
        b_a = atom_basis((0.0, 0.0, 0.0), e_a, basis_name)
        b_b = atom_basis((0.0, 0.0, R), e_b, basis_name)
        ints = diatomic_ao_integrals(b_a, b_b, z_a, z_b, n_eta=56, lmax=28)

        # All-electron Localized VB
        m_all = build_model(ints, e_a, e_b, mirror=(e_a == e_b))
        m_all.method = "cas"
        x0 = m_all.pack(initial_orbitals(m_all))
        res_all = optimize(m_all, x0, maxiter=400)

        # Frozen 1s core environment atoms from atomic RHF
        env_atoms = []
        C_env_cols = []
        n_a = len(b_a)
        if z_a > 2:
            env_a = FrozenEnvironmentAtom.from_atomic_rhf(z_a, (0.0, 0.0, 0.0), b_a, n_occ=1)
            env_atoms.append(env_a)
            col = np.zeros(ints.nao); col[:n_a] = env_a.C_occ[:, 0]
            C_env_cols.append(col)
        if z_b > 2:
            env_b = FrozenEnvironmentAtom.from_atomic_rhf(z_b, (0.0, 0.0, R), b_b, n_occ=1)
            env_atoms.append(env_b)
            col = np.zeros(ints.nao); col[n_a:] = env_b.C_occ[:, 0]
            C_env_cols.append(col)
        C_env = np.column_stack(C_env_cols)

        # E1: Exact back-projected frozen core
        emb_e1 = embed_from_full_integrals(ints, C_env, mode="exact_backprojected")
        m_e1, x_val0 = valence_specs_and_seed(e_a, e_b, emb_e1, res_all.x, m_all)
        res_e1 = optimize(m_e1, x_val0, maxiter=400)

        # E2: Unprojected active ERIs
        emb_e2 = embed_from_full_integrals(ints, C_env, mode="unprojected_eri")
        m_e2 = LocalizedVBModel(emb_e2, m_e1.specs, mirror=m_e1.mirror, method="cas")
        res_e2 = optimize(m_e2, x_val0, maxiter=400)

        # E3: ERI-free 2-center embedding
        emb_e3 = embed_diatomic_in_environment(b_a, b_b, z_a, z_b, env_atoms, n_eta=56, lmax=28)
        m_e3 = LocalizedVBModel(emb_e3, m_e1.specs, mirror=m_e1.mirror, method="cas")
        res_e3 = optimize(m_e3, x_val0, maxiter=400)

        # Control: Bare Fock without PK projection
        emb_nopk = embed_from_full_integrals(ints, C_env, mode="bare_fock_no_pk")
        m_nopk = LocalizedVBModel(emb_nopk, m_e1.specs, mirror=m_e1.mirror, method="cas")
        res_nopk = optimize(m_nopk, x_val0, maxiter=400)

        print(f"{mol_name:<5} {R:6.3f} {res_all.fun:14.6f} {res_e1.fun:14.6f} "
              f"{res_e2.fun:14.6f} {res_e3.fun:14.6f} {res_nopk.fun:15.6f}", flush=True)
        print(f"      dE vs E1 (mHa):         {0.0:14.3f} "
              f"{(res_e2.fun - res_e1.fun)*1e3:14.3f} {(res_e3.fun - res_e1.fun)*1e3:14.3f} "
              f"{(res_nopk.fun - res_e1.fun)*1e3:15.1f}", flush=True)


def run_lih_pes_comparison(basis_name: str = "DZP"):
    print("\n" + "=" * 98)
    print(f"2. LiH ({basis_name}) POTENTIAL ENERGY CURVE & SPECTROSCOPIC CONSTANTS")
    print("=" * 98)
    grid = np.array([2.5, 2.75, 3.015, 3.25, 3.5, 4.0])
    E_all, E_e1, E_e2, E_e3 = [], [], [], []

    print(f"{'R (Bohr)':>8} {'All-Elec VB':>14} {'Frozen E1':>14} {'Frozen E2':>14} {'Frozen E3 (2c)':>15}", flush=True)
    e_atoms = None
    for R in grid:
        b_a = atom_basis((0.0, 0.0, 0.0), "Li", basis_name)
        b_b = atom_basis((0.0, 0.0, R), "H", basis_name)
        ints = diatomic_ao_integrals(b_a, b_b, 3.0, 1.0, n_eta=56, lmax=28)
        if e_atoms is None:
            e_atoms = atom_reference(ints, 0, 3, 1) + atom_reference(ints, 1, 1, 1)

        m_all = build_model(ints, "Li", "H", mirror=False)
        m_all.method = "cas"
        x0 = m_all.pack(initial_orbitals(m_all))
        r_all = optimize(m_all, x0, maxiter=300)

        env_a = FrozenEnvironmentAtom.from_atomic_rhf(3.0, (0.0, 0.0, 0.0), b_a, n_occ=1)
        C_env = np.zeros((ints.nao, 1)); C_env[:len(b_a), 0] = env_a.C_occ[:, 0]

        emb_e1 = embed_from_full_integrals(ints, C_env, mode="exact_backprojected")
        m_e1, x_v0 = valence_specs_and_seed("Li", "H", emb_e1, r_all.x, m_all)
        r_e1 = optimize(m_e1, x_v0, maxiter=300)

        emb_e2 = embed_from_full_integrals(ints, C_env, mode="unprojected_eri")
        m_e2 = LocalizedVBModel(emb_e2, m_e1.specs, mirror=False, method="cas")
        r_e2 = optimize(m_e2, x_v0, maxiter=300)

        emb_e3 = embed_diatomic_in_environment(b_a, b_b, 3.0, 1.0, [env_a], n_eta=56, lmax=28)
        m_e3 = LocalizedVBModel(emb_e3, m_e1.specs, mirror=False, method="cas")
        r_e3 = optimize(m_e3, x_v0, maxiter=300)

        E_all.append(r_all.fun); E_e1.append(r_e1.fun); E_e2.append(r_e2.fun); E_e3.append(r_e3.fun)
        print(f"{R:8.3f} {r_all.fun:14.6f} {r_e1.fun:14.6f} {r_e2.fun:14.6f} {r_e3.fun:15.6f}", flush=True)

    print("-" * 72)
    for label, arr in [
        ("All-Electron VB", np.array(E_all)),
        ("Frozen-Core E1 (exact PK)", np.array(E_e1)),
        ("Frozen-Core E2 (unproj ERI)", np.array(E_e2)),
        ("Frozen-Core E3 (2c ERI-free)", np.array(E_e3)),
    ]:
        fit = fit_pes(grid, arr, MU_LIH)
        de = (e_atoms - fit["E_min"]) * HARTREE_TO_EV
        print(f"{label:<30}: R_e = {fit['R_e']:.4f} Bohr, omega_e = {fit['omega_e_cm1']:6.1f} cm^-1, D_e = {de:.3f} eV", flush=True)


def run_h2_he_substrate_embedding(basis_name: str = "DZ"):
    print("\n" + "=" * 98)
    print(f"3. EXTERNAL PHATPSY ENVIRONMENT EMBEDDING: H2 + Frozen He ({basis_name})")
    print("=" * 98)
    R_hh = 1.401
    distances = [2.5, 3.0, 3.5, 4.0, 5.0, 8.0]

    # Run a genuine PHATPSY SCF calculation on isolated He to extract its FrozenEnvironmentAtom
    drv = PhatpsyDriver(quiet=True)
    he_state = drv.run_molecule([("He", 0.0, 0.0, 0.0)], basis=basis_name, spin_unrestricted=False)
    he_phatpsy = FrozenEnvironmentAtom.from_phatpsy_state(he_state, atom_index=1, n_occ=1)

    print(f"{'d(H2-He)':>9} {'Full 3c VB':>14} {'E1 (exact PK)':>14} {'E2 (unproj V)':>14} "
          f"{'E3 (2c no-3cERI)':>17} {'No-PK collapse':>15}")
    print("-" * 88)

    rows = []
    for d in distances:
        c_h1 = (0.0, 0.0, -R_hh / 2)
        c_h2 = (0.0, 0.0, R_hh / 2)
        c_he = (0.0, 0.0, R_hh / 2 + d)

        b_h1 = atom_basis(c_h1, "H", basis_name)
        b_h2 = atom_basis(c_h2, "H", basis_name)
        b_he = atom_basis(c_he, "He", basis_name)

        # Full 3-center integrals for E1/E2 reference
        m_ints = molecular_ao_integrals([(1.0, c_h1, b_h1), (1.0, c_h2, b_h2), (2.0, c_he, b_he)],
                                        n_eta=40, n_phi=20, lmax=20, mmax=4)
        idx_h1 = np.where(m_ints.atom_of == 0)[0]
        idx_h2 = np.where(m_ints.atom_of == 1)[0]
        idx_he = np.where(m_ints.atom_of == 2)[0]
        idx_act = np.concatenate([idx_h1, idx_h2])

        # Move PHATPSY He environment atom to c_he
        he_env = FrozenEnvironmentAtom(2.0, c_he, b_he, he_phatpsy.C_occ.copy())
        C_env = np.zeros((m_ints.nao, 1))
        C_env[idx_he, 0] = he_env.C_occ[:, 0]

        # 0. Full 3-center Localized VB (all 3 atoms optimized)
        o1 = np.zeros(m_ints.nao); o1[idx_h1[0]] = 1.0
        o2 = np.zeros(m_ints.nao); o2[idx_h2[0]] = 1.0
        ohe = C_env[:, 0].copy()
        m_full = MaskedVB(m_ints, ["inactive", "active", "active"], [idx_he, idx_h1, idx_h2], (1, 1))
        e_full = m_full.optimize(m_full.pack([ohe, o1, o2])).fun

        # 1. E1: Exact back-projected frozen He
        emb_e1 = embed_from_full_integrals(m_ints, C_env, active_ao_idx=idx_act, mode="exact_backprojected")
        ih1_a = np.where(emb_e1.atom_of == 0)[0]
        ih2_a = np.where(emb_e1.atom_of == 1)[0]
        oa1 = np.zeros(emb_e1.nao); oa1[ih1_a[0]] = 1.0
        oa2 = np.zeros(emb_e1.nao); oa2[ih2_a[0]] = 1.0
        m_e1 = MaskedVB(emb_e1, ["active", "active"], [ih1_a, ih2_a], (1, 1))
        r_e1 = m_e1.optimize(m_e1.pack([oa1, oa2]))

        # 2. E2: Unprojected active ERIs
        emb_e2 = embed_from_full_integrals(m_ints, C_env, active_ao_idx=idx_act, mode="unprojected_eri")
        m_e2 = MaskedVB(emb_e2, ["active", "active"], [ih1_a, ih2_a], (1, 1))
        r_e2 = m_e2.optimize(m_e2.pack([oa1, oa2]))

        # 3. E3: Pure 2-center ERI-free embedding (embed_diatomic_in_environment)
        emb_e3 = embed_diatomic_in_environment(b_h1, b_h2, 1.0, 1.0, [he_env], n_eta=48, lmax=24)
        m_e3 = MaskedVB(emb_e3, ["active", "active"], [ih1_a, ih2_a], (1, 1))
        r_e3 = m_e3.optimize(m_e3.pack([oa1, oa2]))

        # 4. Control: No PK projection
        emb_nopk = embed_from_full_integrals(m_ints, C_env, active_ao_idx=idx_act, mode="bare_fock_no_pk")
        m_nopk = MaskedVB(emb_nopk, ["active", "active"], [ih1_a, ih2_a], (1, 1))
        r_nopk = m_nopk.optimize(m_nopk.pack([oa1, oa2]))

        rows.append((d, e_full, r_e1.fun, r_e2.fun, r_e3.fun, r_nopk.fun))
        print(f"{d:9.2f} {e_full:14.6f} {r_e1.fun:14.6f} {r_e2.fun:14.6f} "
              f"{r_e3.fun:17.6f} {r_nopk.fun:15.6f}", flush=True)

    print("\nInteraction Energy dE(d) = E(d) - E(8.0 Bohr) in meV:")
    print(f"{'d(H2-He)':>9} {'Full 3c VB':>14} {'E1 (exact PK)':>14} {'E2 (unproj V)':>14} "
          f"{'E3 (2c no-3cERI)':>17} {'No-PK collapse':>15}")
    ref = rows[-1]
    for r in rows:
        d = r[0]
        de = [(r[k] - ref[k]) * HARTREE_TO_MEV for k in range(1, 6)]
        print(f"{d:9.2f} {de[0]:14.2f} {de[1]:14.2f} {de[2]:14.2f} {de[3]:17.2f} {de[4]:15.2f}", flush=True)


if __name__ == "__main__":
    run_core_valence_table("DZP")
    run_lih_pes_comparison("DZP")
    run_h2_he_substrate_embedding("DZ")
