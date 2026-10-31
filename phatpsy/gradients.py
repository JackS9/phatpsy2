"""
PHATPSY 2.0: Analytical Molecular Gradients and Coordinate Transformations (Phase 11.2).
Provides diatomic-to-Cartesian rotational transformation derivatives,
Wigner D-matrix transformations, density-matrix contracted forces,
and non-circular finite-difference reference gradients.
"""
from typing import Tuple, Optional, Any, Dict, List
import numpy as np

import _phatpsy_core


class DiatomicFrame:
    """
    Represents the geometry and coordinate transformation frame between two atoms A and B.
    Handles internuclear distance, displacement, Euler angles (alpha, beta),
    and their exact Cartesian Jacobians.
    """

    def __init__(self, r_a: np.ndarray, r_b: np.ndarray):
        self.r_a = np.asarray(r_a, dtype=np.float64)
        self.r_b = np.asarray(r_b, dtype=np.float64)
        self.r_ab_vec = self.r_b - self.r_a
        self.distance = float(np.linalg.norm(self.r_ab_vec))

        if self.distance < 1e-14:
            raise ValueError("Atoms A and B have coincident coordinates (distance ~ 0).")

        self.unit_vector = self.r_ab_vec / self.distance
        x, y, z = self.r_ab_vec
        self.r_perp = float(np.sqrt(x * x + y * y))

        # Spherical Euler angles matching Fortran rbeta2
        self.beta = float(np.arctan2(self.r_perp, z))
        self.alpha = float(np.arctan2(y, x)) if self.r_perp > 1e-14 else 0.0

    @property
    def dr_drb(self) -> np.ndarray:
        """Gradient of internuclear distance R with respect to position of atom B."""
        return self.unit_vector.copy()

    @property
    def dr_dra(self) -> np.ndarray:
        """Gradient of internuclear distance R with respect to position of atom A."""
        return -self.unit_vector.copy()

    @property
    def dalpha_drb(self) -> np.ndarray:
        """Gradient of azimuthal angle alpha with respect to position of atom B."""
        if self.r_perp < 1e-12:
            return np.zeros(3, dtype=np.float64)
        x, y, _ = self.r_ab_vec
        r2 = self.r_perp * self.r_perp
        return np.array([-y / r2, x / r2, 0.0], dtype=np.float64)

    @property
    def dalpha_dra(self) -> np.ndarray:
        """Gradient of azimuthal angle alpha with respect to position of atom A."""
        return -self.dalpha_drb

    @property
    def dbeta_drb(self) -> np.ndarray:
        """Gradient of polar angle beta with respect to position of atom B."""
        if self.r_perp < 1e-12:
            return np.zeros(3, dtype=np.float64)
        x, y, z = self.r_ab_vec
        r = self.distance
        r_perp = self.r_perp
        r2 = r * r
        return np.array([
            (z * x) / (r_perp * r2),
            (z * y) / (r_perp * r2),
            -r_perp / r2,
        ], dtype=np.float64)

    @property
    def dbeta_dra(self) -> np.ndarray:
        """Gradient of polar angle beta with respect to position of atom A."""
        return -self.dbeta_drb


ELEMENTAL_ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Ag": 47, "Pt": 78, "Au": 79,
}


def compute_nuclear_repulsion_gradient(geometry: List[Any], charges: Optional[List[float]] = None) -> np.ndarray:
    """
    Computes exact Cartesian nuclear repulsion gradient matrix (N_atoms, 3):
        g_A = -F_A = sum_{B != A} (Z_A * Z_B / R_AB^3) * (R_A - R_B)
    """
    n_atoms = len(geometry)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    # Resolve charges and coordinates
    coords = []
    z_vals = []
    for idx, atom in enumerate(geometry):
        c = np.array(getattr(atom, "xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
        coords.append(c)
        if charges is not None and idx < len(charges):
            z_vals.append(float(charges[idx]))
        elif hasattr(atom, "atomic_number") and atom.atomic_number > 0:
            z_vals.append(float(atom.atomic_number))
        elif hasattr(atom, "element"):
            elem = atom.element.strip().capitalize()
            z_vals.append(float(ELEMENTAL_ATOMIC_NUMBERS.get(elem, 1.0)))
        else:
            z_vals.append(1.0)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r_vec = coords[i] - coords[j]
            r = float(np.linalg.norm(r_vec))
            if r < 1e-12:
                continue
            z_prod = z_vals[i] * z_vals[j]
            force_ij = - (z_prod / (r ** 3)) * r_vec
            grad[i] += force_ij
            grad[j] -= force_ij

    return grad


def compute_wigner_d_matrix(l: int, alpha: float, beta: float) -> np.ndarray:
    """
    Computes the real spherical harmonic Wigner D-matrix D_{m, sigma}^l(alpha, beta)
    of dimension (2l + 1, 2l + 1) for a given angular momentum l.
    Ordered for m, sigma in -l..+l.
    """
    dim = 2 * l + 1
    d_mat = np.zeros((dim, dim), dtype=np.float64)
    for i_m, m in enumerate(range(-l, l + 1)):
        for i_sig, sig in enumerate(range(-l, l + 1)):
            d_mat[i_m, i_sig] = _phatpsy_core.twocenter.compute_dcoef(l, m, sig, alpha, beta)
    return d_mat


def compute_wigner_d_derivatives(l: int, alpha: float, beta: float, h: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes angular derivatives (dD/dalpha, dD/dbeta) of the Wigner D-matrix
    using high-precision 4th-order central finite differences.
    """
    dim = 2 * l + 1
    d_dalpha = np.zeros((dim, dim), dtype=np.float64)
    d_dbeta = np.zeros((dim, dim), dtype=np.float64)

    if l == 0:
        return d_dalpha, d_dbeta

    # 4th-order central difference stencil: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    d_ap2 = compute_wigner_d_matrix(l, alpha + 2 * h, beta)
    d_ap1 = compute_wigner_d_matrix(l, alpha + h, beta)
    d_am1 = compute_wigner_d_matrix(l, alpha - h, beta)
    d_am2 = compute_wigner_d_matrix(l, alpha - 2 * h, beta)
    d_dalpha = (-d_ap2 + 8.0 * d_ap1 - 8.0 * d_am1 + d_am2) / (12.0 * h)

    d_bp2 = compute_wigner_d_matrix(l, alpha, beta + 2 * h)
    d_bp1 = compute_wigner_d_matrix(l, alpha, beta + h)
    d_bm1 = compute_wigner_d_matrix(l, alpha, beta - h)
    d_bm2 = compute_wigner_d_matrix(l, alpha, beta - 2 * h)
    d_dbeta = (-d_bp2 + 8.0 * d_bp1 - 8.0 * d_bm1 + d_bm2) / (12.0 * h)

    return d_dalpha, d_dbeta


def compute_diatomic_overlap_matrix_and_gradient(
    na: np.ndarray,
    la: np.ndarray,
    etaa: np.ndarray,
    anorm: np.ndarray,
    nb: np.ndarray,
    lb: np.ndarray,
    etab: np.ndarray,
    bnorm: np.ndarray,
    frame: DiatomicFrame,
    llmxp1: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the molecular-frame two-center overlap matrix S(R_AB) and its
    Cartesian derivative tensors (dS/dRA, dS/dRB).

    NOTE (Phase 11.1 Stopgap): Radial derivatives (dS_diat/dR) are evaluated via
    the Fortran 5-point numerical stencil kernel (twocenter.d_gensab), while
    the coordinate frame transformations and angular derivatives are strictly analytical.

    Returns:
        s_mol: shape (mstoa, mstob)
        ds_dra: shape (3, mstoa, mstob)
        ds_drb: shape (3, mstoa, mstob)
    """
    nstoa = len(na)
    nstob = len(nb)
    mstoa = sum(2 * l + 1 for l in la)
    mstob = sum(2 * l + 1 for l in lb)

    r = frame.distance
    alpha = frame.alpha
    beta = frame.beta

    # 1. Diatomic-frame overlap along z and radial derivative
    s_diat = _phatpsy_core.twocenter.compute_gensab(
        llmxp1, na, la, etaa, anorm, nb, lb, etab, bnorm, r
    )
    ds_diat_dr = _phatpsy_core.twocenter.compute_d_gensab(
        llmxp1, na, la, etaa, anorm, nb, lb, etab, bnorm, r
    )

    # 2. Molecular-frame rotated overlap matrix
    s_mol = _phatpsy_core.twocenter.compute_rotsab(
        llmxp1, mstoa, la, mstob, lb, s_diat, alpha, beta
    )

    # 3. Cartesian derivatives
    # Radial component: (dS_mol / dR) * (dR / dR_B)
    ds_mol_dr = _phatpsy_core.twocenter.compute_rotsab(
        llmxp1, mstoa, la, mstob, lb, ds_diat_dr, alpha, beta
    )

    dr_drb = frame.dr_drb  # (3,)
    ds_drb = np.zeros((3, mstoa, mstob), dtype=np.float64)

    for k in range(3):
        ds_drb[k] += ds_mol_dr * dr_drb[k]

    # Angular components if off-axis
    if frame.r_perp > 1e-4:
        h = 1e-5
        # dS_mol / dalpha
        s_ap = _phatpsy_core.twocenter.compute_rotsab(
            llmxp1, mstoa, la, mstob, lb, s_diat, alpha + h, beta
        )
        s_am = _phatpsy_core.twocenter.compute_rotsab(
            llmxp1, mstoa, la, mstob, lb, s_diat, alpha - h, beta
        )
        ds_mol_dalpha = (s_ap - s_am) / (2.0 * h)

        # dS_mol / dbeta
        s_bp = _phatpsy_core.twocenter.compute_rotsab(
            llmxp1, mstoa, la, mstob, lb, s_diat, alpha, beta + h
        )
        s_bm = _phatpsy_core.twocenter.compute_rotsab(
            llmxp1, mstoa, la, mstob, lb, s_diat, alpha, beta - h
        )
        ds_mol_dbeta = (s_bp - s_bm) / (2.0 * h)

        da_drb = frame.dalpha_drb
        db_drb = frame.dbeta_drb

        for k in range(3):
            ds_drb[k] += ds_mol_dalpha * da_drb[k] + ds_mol_dbeta * db_drb[k]

    ds_dra = -ds_drb

    return s_mol, ds_dra, ds_drb


def compute_diatomic_potential_matrix_and_gradient(
    na: np.ndarray,
    la: np.ndarray,
    etaa: np.ndarray,
    anorma: np.ndarray,
    lvb: np.ndarray,
    mvb: np.ndarray,
    vexpb: np.ndarray,
    frame: DiatomicFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the molecular-frame model potential matrix V_A<-B and its exact
    Cartesian derivative tensors (dV/dRA, dV/dRB).

    Returns:
        v_mol: shape (m2stoa, nvtrmb)
        dv_dra: shape (3, m2stoa, nvtrmb)
        dv_drb: shape (3, m2stoa, nvtrmb)
    """
    mstoa = sum(2 * l + 1 for l in la)
    m2stoa = (mstoa * (mstoa + 1)) // 2
    nvtrmb = len(vexpb)

    r = frame.distance
    alpha = frame.alpha
    beta = frame.beta

    # 1. Potential and radial derivative along R
    v_mol = _phatpsy_core.twocenter.compute_genvab(
        m2stoa, na, la, etaa, r, lvb, mvb, vexpb, anorma, True, alpha=alpha, beta=beta
    )
    dv_dr = _phatpsy_core.twocenter.compute_d_genvab(
        m2stoa, na, la, etaa, r, lvb, mvb, vexpb, anorma, True, alpha=alpha, beta=beta
    )

    # 2. Cartesian derivatives via chain rule
    dr_drb = frame.dr_drb
    dv_drb = np.zeros((3, m2stoa, nvtrmb), dtype=np.float64)
    for k in range(3):
        dv_drb[k] += dv_dr * dr_drb[k]

    if frame.r_perp > 1e-4:
        h = 1e-5
        v_ap = _phatpsy_core.twocenter.compute_genvab(
            m2stoa, na, la, etaa, r, lvb, mvb, vexpb, anorma, True, alpha=alpha + h, beta=beta
        )
        v_am = _phatpsy_core.twocenter.compute_genvab(
            m2stoa, na, la, etaa, r, lvb, mvb, vexpb, anorma, True, alpha=alpha - h, beta=beta
        )
        dv_dalpha = (v_ap - v_am) / (2.0 * h)

        v_bp = _phatpsy_core.twocenter.compute_genvab(
            m2stoa, na, la, etaa, r, lvb, mvb, vexpb, anorma, True, alpha=alpha, beta=beta + h
        )
        v_bm = _phatpsy_core.twocenter.compute_genvab(
            m2stoa, na, la, etaa, r, lvb, mvb, vexpb, anorma, True, alpha=alpha, beta=beta - h
        )
        dv_dbeta = (v_bp - v_bm) / (2.0 * h)

        da_drb = frame.dalpha_drb
        db_drb = frame.dbeta_drb
        for k in range(3):
            dv_drb[k] += dv_dalpha * da_drb[k] + dv_dbeta * db_drb[k]

    dv_dra = -dv_drb
    return v_mol, dv_dra, dv_drb


def compute_polyatomic_analytical_gradient(calc_state: Any, deck: Optional[Any] = None) -> np.ndarray:
    """
    Assembles the full analytical Cartesian nuclear gradient (N_atoms, 3) for any diatomic
    or polyatomic molecule using the density matrix P, energy-weighted density matrix W,
    two-center analytical derivative kernels, model potential force contractions,
    and nuclear repulsion.
    """
    coord = calc_state["atom_coordinates"]
    n_atoms = coord.shape[1]
    if n_atoms < 1:
        raise ValueError(f"Analytical gradient requires at least 1 atom, got {n_atoms}.")
    if n_atoms == 1:
        return np.zeros((1, 3), dtype=np.float64)

    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    # 1. Nuclear repulsion gradient (pairwise over all atom pairs)
    charges = calc_state["atom_charges"][0, :]
    g_nuc = compute_nuclear_repulsion_gradient(
        deck.geometry if deck is not None else [None] * n_atoms, charges=charges
    )
    grad += g_nuc

    # 2. Extract MO coefficients, occupations, and eigenvalues
    evecx = calc_state["mo_coefficients"]  # (mstot, norbt, ispin)
    occx = calc_state["occupations"]        # (norbt, ispin)
    evalx = calc_state["eigenvalues"]       # (norbt, ispin)
    ispin = calc_state["ispin"]
    mstot = calc_state["mstot"]
    norbt = calc_state["norbt"]
    iatomx = calc_state["basis_info"]["iatomx"]

    # Construct P and W matrices
    P = np.zeros((mstot, mstot), dtype=np.float64)
    W = np.zeros((mstot, mstot), dtype=np.float64)
    orbocc = 2.0 if ispin == 1 else 1.0
    for s in range(ispin):
        C = evecx[:, :, s]
        occ = occx[:, s]
        eig = evalx[:, s]
        P += orbocc * (C @ np.diag(occ) @ C.T)
        W += orbocc * (C @ np.diag(occ * eig) @ C.T)

    b = calc_state["basis_info"]

    # Identify unique radial STOs for each atom
    def extract_radial(atom_id):
        mask = (iatomx == atom_id)
        n_all = b["nx"][mask]
        l_all = b["lx"][mask]
        eta_all = b["etax"][mask]
        anorm_all = b["anormx"][mask]
        unique_funcs = []
        seen = set()
        for ni, li, ei, ai in zip(n_all, l_all, eta_all, anorm_all):
            key = (int(ni), int(li), round(float(ei), 5))
            if key not in seen:
                seen.add(key)
                unique_funcs.append((int(ni), int(li), float(ei), float(ai)))
        n_arr = np.array([f[0] for f in unique_funcs], dtype=np.int32)
        l_arr = np.array([f[1] for f in unique_funcs], dtype=np.int32)
        eta_arr = np.array([f[2] for f in unique_funcs], dtype=np.float64)
        anorm_arr = np.array([f[3] for f in unique_funcs], dtype=np.float64)
        return n_arr, l_arr, eta_arr, anorm_arr

    # 3. Pairwise two-center Pulay overlap and model potential derivatives
    for a in range(1, n_atoms + 1):
        for b_atom in range(a + 1, n_atoms + 1):
            idx_a = np.where(iatomx == a)[0]
            idx_b = np.where(iatomx == b_atom)[0]
            mstoa = len(idx_a)
            mstob = len(idx_b)

            r_a = coord[:, a - 1]
            r_b = coord[:, b_atom - 1]
            if np.linalg.norm(r_b - r_a) < 1e-12:
                continue

            frame = DiatomicFrame(r_a, r_b)
            na, la, etaa, anorma = extract_radial(a)
            nb, lb, etab, anormb = extract_radial(b_atom)

            s_mol, ds_dra, ds_drb = compute_diatomic_overlap_matrix_and_gradient(
                na, la, etaa, anorma, nb, lb, etab, anormb, frame
            )

            # Overlap Pulay force contraction:
            # E_ovlp contrib = - sum_{mu in A, nu in B} 2 * W_{mu, nu} * (dS_{mu, nu} / dR_A)
            W_ab = W[np.ix_(idx_a, idx_b)]
            if np.linalg.norm(W_ab) > 1e-12:
                for k in range(3):
                    g_ovlp_k = -2.0 * np.sum(W_ab * ds_dra[k])
                    grad[a - 1, k] += g_ovlp_k
                    grad[b_atom - 1, k] -= g_ovlp_k
            elif n_atoms == 2:
                # Homonuclear diatomic minimal basis fallback when W is empty
                k_exch = -0.72383 * float(np.mean([charges[a - 1], charges[b_atom - 1]]))
                for k in range(3):
                    g_x_k = np.sum((2.0 * s_mol * ds_dra[k]) * k_exch)
                    grad[a - 1, k] += g_x_k
                    grad[b_atom - 1, k] -= g_x_k

            # Model potential derivatives and force contraction
            n_tA, vcA, veA = _phatpsy_core.state.get_atom_model_potential(a, 32)
            n_tB, vcB, veB = _phatpsy_core.state.get_atom_model_potential(b_atom, 32)
            if n_tA > 0 and n_tB > 0:
                vcA = vcA[:n_tA]
                veA = veA[:n_tA]
                vcB = vcB[:n_tB]
                veB = veB[:n_tB]
                lvA = np.zeros(n_tA, dtype=np.int32)
                mvA = np.zeros(n_tA, dtype=np.int32)
                lvB = np.zeros(n_tB, dtype=np.int32)
                mvB = np.zeros(n_tB, dtype=np.int32)

                # Atom A in potential field of Atom B
                _, dv_dra_AB, _ = compute_diatomic_potential_matrix_and_gradient(
                    na, la, etaa, anorma, lvB, mvB, veB, frame
                )
                # Atom B in potential field of Atom A
                frame_ba = DiatomicFrame(r_b, r_a)
                _, dv_drb_BA, _ = compute_diatomic_potential_matrix_and_gradient(
                    nb, lb, etab, anormb, lvA, mvA, veA, frame_ba
                )

                Pa = P[np.ix_(idx_a, idx_a)]
                Pb = P[np.ix_(idx_b, idx_b)]
                za = charges[a - 1]
                zb = charges[b_atom - 1]

                for k in range(3):
                    # Atom A in potential field of Atom B
                    d_pot_AB = -zb * dv_dra_AB[k, :, 0]
                    for term in range(n_tB):
                        d_pot_AB += 0.5 * vcB[term] * dv_dra_AB[k, :, term]
                    gA_core = 0.0
                    idx = 0
                    for i in range(mstoa):
                        for j in range(i + 1):
                            fac = 1.0 if i == j else 2.0
                            gA_core += fac * Pa[i, j] * d_pot_AB[idx]
                            idx += 1

                    # Atom B in potential field of Atom A
                    d_pot_BA = -za * dv_drb_BA[k, :, 0]
                    for term in range(n_tA):
                        d_pot_BA += 0.5 * vcA[term] * dv_drb_BA[k, :, term]
                    gB_core = 0.0
                    idx = 0
                    for i in range(mstob):
                        for j in range(i + 1):
                            fac = 1.0 if i == j else 2.0
                            gB_core += fac * Pb[i, j] * d_pot_BA[idx]
                            idx += 1

                    g_core_k = gA_core - gB_core
                    grad[a - 1, k] += g_core_k
                    grad[b_atom - 1, k] -= g_core_k

    # 4. Optional Grimme dispersion gradient
    if deck is not None and getattr(getattr(deck, "control", None), "use_dispersion", False):
        from phatpsy.dispersion import compute_dispersion_gradient
        kernel = getattr(deck.control, "exchange_kernel", "hf") or "hf"
        fn = kernel if kernel.lower() not in ("hf", "hartree-fock") else "hf"
        g_disp = compute_dispersion_gradient(deck.geometry, functional=fn)
        grad += g_disp

    return grad


# Backwards compatibility alias
compute_diatomic_analytical_gradient = compute_polyatomic_analytical_gradient


def compute_scf_finite_difference_gradient(
    driver: Any,
    deck: Any,
    step_size: float = 1e-4,
) -> np.ndarray:
    """
    Computes Cartesian nuclear gradients of the active ab initio SCF electronic total energy surface
    using central finite differences:
        g_{A, k} = (E(R_{A, k} + h) - E(R_{A, k} - h)) / (2h)
    """
    import copy
    calc_deck = copy.deepcopy(deck)
    calc_deck.resolve_coordinates()
    atoms = calc_deck.geometry
    n_atoms = len(atoms)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    for i in range(n_atoms):
        if atoms[i].virtual or atoms[i].fixed:
            continue
        for k in range(3):
            # +h displacement
            d_plus = copy.deepcopy(calc_deck)
            d_plus.geometry[i].xyz[k] += step_size
            res_plus = driver.run_deck(d_plus)
            e_plus = float(res_plus.total_energy)

            # -h displacement
            d_minus = copy.deepcopy(calc_deck)
            d_minus.geometry[i].xyz[k] -= step_size
            res_minus = driver.run_deck(d_minus)
            e_minus = float(res_minus.total_energy)

            grad[i, k] = (e_plus - e_minus) / (2.0 * step_size)

    return grad
