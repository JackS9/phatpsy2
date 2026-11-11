"""
PHATPSY 2.0: Coupled-Perturbed (CPHF / Z-Vector) Orbital Response Solver (Phases 12, 15, 16).

Provides the full coupled intra-atomic and inter-atomic orbital and model-potential
relaxation response for analytical nuclear gradients in Projected Hamiltonian / EWMO
quantum chemistry:
    dE / dR_A = (dE / dR_A)_C + sum_{p} Z_p * F^{(A, k)}_p

Using the Handy-Schaefer Z-vector formulation, the orbital relaxation response is
evaluated analytically without solving 3N individual perturbation equations:
    A^T * Z = -L
    Delta g_{A, k} = Z^T * F^{(A, k)}

where:
    - L is the analytical orbital Lagrangian vector L_{pq} = dE_total / dU_{pq},
      incorporating both intra-atomic one-electron/Fock variations and inter-atomic
      model-potential screening (dvcoef / dU) and exchange-residual (dV6 / dU) coupling.
    - A = A_0 + K^intra + K^inter is the full coupled CPHF electronic superoperator
      Hessian across all active canonical orbital rotation pairs.
    - F^{(A, k)} is the fixed-orbital derivative of the Fock matrix, including both
      explicit two-center integral derivatives and the analytical fixed-orbital
      model-potential screening response (dvcoef / dR)_C.
"""
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
import scipy.linalg

import _phatpsy_core
from phatpsy.gradients import (
    DiatomicFrame,
    compute_diatomic_overlap_matrix_and_gradient,
    compute_diatomic_potential_matrix_and_gradient,
    compute_nuclear_repulsion_gradient,
    compute_polyatomic_analytical_gradient,
    _unpack_packed,
)


def _compute_vfit2_and_grad(
    coord: np.ndarray,
    charges: np.ndarray,
    a: int,
    lva: np.ndarray,
    mva: np.ndarray,
    vexpa: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the point-charge / multipole nuclear attraction fitting vector vfit2_A
    (bloop.f90 lines 143-159) and its analytical Cartesian gradient dvfit2_A / dR.
    """
    n_atoms = coord.shape[1]
    nt = len(vexpa)
    vfit2a = np.zeros(nt, dtype=np.float64)
    dvfit2a_dcoord = np.zeros((n_atoms, 3, nt), dtype=np.float64)
    r_a = coord[:, a - 1]
    for b in range(1, n_atoms + 1):
        if b == a:
            continue
        r_b = coord[:, b - 1]
        vec = r_b - r_a
        rab = float(np.linalg.norm(vec))
        if rab < 1e-12:
            continue
        u_ab = vec / rab
        zb = float(charges[b - 1])
        for nva in range(nt):
            lv_val = int(lva[nva])
            nv_val = lv_val if lv_val >= 0 else -lv_val + 1
            va = float(np.exp(-vexpa[nva] * rab) * (rab ** (nv_val - 1)))
            dva_drab = va * (-float(vexpa[nva]) + (nv_val - 1) / rab)
            vfit2a[nva] -= zb * va
            dvfit2a_dcoord[a - 1, :, nva] += zb * dva_drab * u_ab
            dvfit2a_dcoord[b - 1, :, nva] -= zb * dva_drab * u_ab
    return vfit2a, dvfit2a_dcoord


def _compute_grid_fit_dvcoef(
    na: np.ndarray,
    la: np.ndarray,
    etaa: np.ndarray,
    anorma: np.ndarray,
    vexp: np.ndarray,
    vcoef: np.ndarray,
    P_tot: np.ndarray,
    sumocc: float,
    dP_tot: np.ndarray,
) -> np.ndarray:
    """
    Evaluates the analytical response dvcoef / dU_{pq} of the radial-grid
    simplex model-potential fit (fit_screening_simplex in src/analys.f90)
    to a one-particle density perturbation dP_tot.
    """
    nt = len(vexp)
    nk = nt - 1
    dvcoef = np.zeros(nt, dtype=np.float64)
    if nk < 1 or sumocc <= 1e-12:
        return dvcoef

    w_curr = -vcoef[1:] / vcoef[0] if abs(vcoef[0]) > 1e-12 else np.zeros(nk)
    idx_act = np.where(w_curr > 1e-12)[0]
    if len(idx_act) <= 1:
        return dvcoef

    nsto = len(na)
    shell_offset = [0]
    for i in range(1, nsto):
        shell_offset.append(shell_offset[-1] + 2 * int(la[i - 1]) + 1)

    # Check if dP_tot has any non-zero spherical trace (l_i == l_j, m_i == m_j)
    has_spherical_trace = False
    for i in range(nsto):
        li = int(la[i])
        for j in range(nsto):
            if int(la[j]) != li:
                continue
            d_bar = sum(
                dP_tot[shell_offset[i] + m + li, shell_offset[j] + m + li]
                for m in range(-li, li + 1)
            )
            if abs(d_bar) > 1e-15:
                has_spherical_trace = True
                break
        if has_spherical_trace:
            break
    if not has_spherical_trace:
        return dvcoef

    NG = 400
    R0, RMAX = 1.0e-4, 40.0
    h = float(np.log(RMAX / R0) / (NG - 1))
    rg = R0 * np.exp(h * np.arange(NG, dtype=np.float64))

    R_grid = np.zeros((nsto, NG), dtype=np.float64)
    for i in range(nsto):
        R_grid[i] = float(anorma[i]) * (rg ** (int(na[i]) - 1)) * np.exp(-float(etaa[i]) * rg)

    def _poisson_integrals(D_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rtot = np.zeros(NG, dtype=np.float64)
        for i in range(nsto):
            li = int(la[i])
            for j in range(nsto):
                if int(la[j]) != li:
                    continue
                d_bar = sum(
                    D_mat[shell_offset[i] + m + li, shell_offset[j] + m + li]
                    for m in range(-li, li + 1)
                )
                if abs(d_bar) > 1e-16:
                    rtot += (d_bar / (4.0 * np.pi)) * R_grid[i] * R_grid[j]
        f = 4.0 * np.pi * (rg ** 3) * rtot
        f2 = 4.0 * np.pi * (rg ** 2) * rtot
        qcum = np.zeros(NG, dtype=np.float64)
        qcum[0] = f[0] / 3.0
        qcum[1:] = np.cumsum(0.5 * h * (f[:-1] + f[1:])) + qcum[0]
        ucum = np.zeros(NG, dtype=np.float64)
        ucum[:-1] = np.cumsum(0.5 * h * (f2[:-1] + f2[1:])[::-1])[::-1]
        return qcum, ucum

    qcum, ucum = _poisson_integrals(P_tot)
    ngrid = float(qcum[-1])
    if ngrid <= 1e-8:
        return dvcoef
    dqcum, ducum = _poisson_integrals(dP_tot)
    dngrid = float(dqcum[-1])

    phi = qcum + rg * ucum
    dphi = dqcum + rg * ducum
    sg = 1.0 - phi / ngrid
    dsg = -(dphi * ngrid - phi * dngrid) / (ngrid ** 2)

    wt = h * rg * np.exp(-rg / 10.0)
    sqw = np.sqrt(wt)
    m_rows = NG + 1 + nk
    A = np.zeros((m_rows, nk), dtype=np.float64)
    bb = np.zeros(m_rows, dtype=np.float64)
    dbb = np.zeros(m_rows, dtype=np.float64)

    A[:NG, :] = sqw[:, None] * np.exp(-rg[:, None] * vexp[1:][None, :])
    bb[:NG] = sqw * sg
    dbb[:NG] = sqw * dsg
    A[NG, :] = 1.0e3
    bb[NG] = 1.0e3
    trace_ata = float(np.sum(A[:NG, :] ** 2) / nk)
    ridge = float(np.sqrt(1.0e-9 * trace_ata))
    for k in range(nk):
        A[NG + 1 + k, k] = ridge
        bb[NG + 1 + k] = ridge / nk

    A_sub = A[:, idx_act]
    ata_sub = A_sub.T @ A_sub
    x_sub = np.linalg.solve(ata_sub, A_sub.T @ bb)
    dx_sub = np.linalg.solve(ata_sub, A_sub.T @ dbb)
    wsum = float(np.sum(x_sub))
    if wsum <= 1e-12:
        return dvcoef
    dw_sub = (dx_sub * wsum - x_sub * float(np.sum(dx_sub))) / (wsum ** 2)
    dvcoef[1 + idx_act] = -sumocc * dw_sub
    return dvcoef


class CPHFResponseSolver:
    """
    Coupled-Perturbed Hartree-Fock (CPHF) / Z-vector orbital and model-potential
    response solver for Projected Hamiltonian / EWMO wave functions.
    """

    def __init__(self, calc_state: Any, deck: Optional[Any] = None):
        self.calc_state = calc_state
        self.deck = deck

        # Extract dimensions and state tensors
        self.coord = calc_state["atom_coordinates"]
        self.n_atoms = self.coord.shape[1]
        self.charges = calc_state["atom_charges"][0, :]
        self.evecx = calc_state["mo_coefficients"]    # (mstot, norbt, ispin)
        self.occx = calc_state["occupations"]          # (norbt, ispin)
        self.evalx = calc_state["eigenvalues"]         # (norbt, ispin)
        self.ispin = calc_state["ispin"]
        self.orbocc = 3.0 - self.ispin
        self.mstot = calc_state["mstot"]
        self.norbt = calc_state["norbt"]
        self.iatomx = calc_state["basis_info"]["iatomx"]
        self.b = calc_state["basis_info"]

        # Identify atom blocks and orbital offsets
        self.atom_basis_indices: Dict[int, np.ndarray] = {}
        self.atom_orb_slices: Dict[int, Tuple[int, int]] = {}
        korb = 0
        for a in range(1, self.n_atoms + 1):
            idx_a = np.where(self.iatomx == a)[0]
            self.atom_basis_indices[a] = idx_a
            m = len(idx_a)
            norb_a = int(_phatpsy_core.state.get_atom_elem_norb(a))
            if norb_a <= 0:
                norb_a = 1 if m == 1 else m
            self.atom_orb_slices[a] = (korb, korb + norb_a)
            korb += norb_a

        # Lazy cache for multi-center response tensors
        self._response_cache: Optional[Dict[str, Any]] = None

        # Identify active canonical orbital rotation pairs
        self.active_pairs: List[Dict[str, Any]] = []
        self._build_active_pairs()

    def _extract_radial(self, atom_id: int):
        mask = (self.iatomx == atom_id)
        n_all = self.b["nx"][mask]
        l_all = self.b["lx"][mask]
        eta_all = self.b["etax"][mask]
        anorm_all = self.b["anormx"][mask]
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

    def _get_atom_nact(self, a: int) -> Tuple[int, int]:
        """
        Returns (nact_sto, nact_orb) for atom a matching src/scf.f90 and src/bloop.f90:
        core + valence STO count (l <= l_max_val) and active orbital count
        norb_a - (m - nact_sto).
        """
        idx_a = self.atom_basis_indices[a]
        m = len(idx_a)
        k_start, k_end = self.atom_orb_slices[a]
        norb_a = k_end - k_start
        znuc = int(round(float(self.charges[a - 1])))
        if znuc <= 2:
            l_max_val = 0
        elif znuc <= 20:
            l_max_val = 1
        elif znuc <= 56:
            l_max_val = 2
        else:
            l_max_val = 3
        _, la, _, _ = self._extract_radial(a)
        nact_sto = int(sum(2 * int(l_val) + 1 for l_val in la if int(l_val) <= l_max_val))
        if nact_sto <= 0 or nact_sto > m:
            nact_sto = m
        nact_orb = norb_a - (m - nact_sto)
        if nact_orb <= 0 or nact_orb > norb_a:
            nact_orb = norb_a
        return nact_sto, nact_orb

    def _get_atom_full_orbitals(self, a: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the full (m, m, ispin) canonical orbital coefficients C_full,
        (m, ispin) occupations occ_full, and (m, ispin) eigenvalues eval_full
        for atom a. When m > nact_orb (e.g. DZ / DZP / TZP basis sets), completes
        the basis with the m - nact_orb virtual eigenvectors of the atomic Fock
        matrix in the S-orthogonal complement of the nact_orb active orbitals.
        """
        idx_a = self.atom_basis_indices[a]
        m = len(idx_a)
        m2 = (m * (m + 1)) // 2
        k_start, k_end = self.atom_orb_slices[a]

        C_stored = self.evecx[idx_a, k_start:k_end, :].copy()
        occ_stored = self.occx[k_start:k_end, :].copy()
        eval_stored = self.evalx[k_start:k_end, :].copy()

        _, n_act = self._get_atom_nact(a)

        if m <= n_act:
            return C_stored, occ_stored, eval_stored

        C_full = np.zeros((m, m, self.ispin), dtype=np.float64)
        occ_full = np.zeros((m, self.ispin), dtype=np.float64)
        eval_full = np.zeros((m, self.ispin), dtype=np.float64)
        C_full[:, :n_act, :] = C_stored[:, :n_act, :]
        occ_full[:n_act, :] = occ_stored[:n_act, :]
        eval_full[:n_act, :] = eval_stored[:n_act, :]

        znuc = float(self.charges[a - 1])
        stovl = _unpack_packed(_phatpsy_core.state.get_atom_elem_stovl(a, m2).ravel(), m)
        w_s, u_s = np.linalg.eigh(stovl)
        s_inv_sqrt = (u_s / np.sqrt(np.maximum(w_s, 1e-12))) @ u_s.T
        s_sqrt = (u_s * np.sqrt(np.maximum(w_s, 1e-12))) @ u_s.T
        sthmat = _unpack_packed(_phatpsy_core.state.get_atom_elem_sthmat(a, m2).ravel(), m)
        zpot2 = _unpack_packed(_phatpsy_core.state.get_atom_record_zpot2(a, m2).ravel(), m)
        cpot2 = _unpack_packed(_phatpsy_core.state.get_atom_record_cpot2(a, m2).ravel(), m)
        pkpot = _phatpsy_core.state.get_atom_record_pkpot(a, m2, 6, self.ispin).copy()

        dP_packed = np.zeros((m2, self.ispin), dtype=np.float64)
        for s in range(self.ispin):
            P_s = C_stored[:, :n_act, s] @ np.diag(occ_stored[:n_act, s]) @ C_stored[:, :n_act, s].T
            idx = 0
            for i in range(m):
                for j in range(i + 1):
                    dP_packed[idx, s] = P_s[i, j]
                    idx += 1
        fc_pack, fx_pack = _phatpsy_core.integrals.compute_atom_fokmat(a, dP_packed)

        for s in range(self.ispin):
            fc_s = _unpack_packed(fc_pack[:, s], m)
            fx_s = _unpack_packed(fx_pack[:, s], m)
            xpot2_s = _unpack_packed(pkpot[:, 2, s], m)
            pk2_s = (
                _unpack_packed(pkpot[:, 1, s], m)
                if (znuc > 1.5 or float(np.max(np.abs(pkpot[:, 1, s]))) > 1e-12)
                else 0.0
            )
            F_ao = sthmat + fc_s + fx_s + zpot2 + cpot2 + xpot2_s - pk2_s
            onfock = s_inv_sqrt.T @ F_ao @ s_inv_sqrt

            # Orthonormal basis for the m - n_act orthogonal complement in Löwdin space
            U_occ = s_sqrt @ C_stored[:, :n_act, s]
            P_comp = np.eye(m) - U_occ @ U_occ.T
            u_svd, _, _ = np.linalg.svd(P_comp)
            U_virt = u_svd[:, : m - n_act]
            F_virt = U_virt.T @ onfock @ U_virt
            w_virt, V_virt = np.linalg.eigh(F_virt)
            C_full[:, n_act:, s] = s_inv_sqrt @ (U_virt @ V_virt)
            eval_full[n_act:, s] = w_virt

        return C_full, occ_full, eval_full

    def _build_active_pairs(self):
        """
        Identifies all non-degenerate canonical orbital rotation pairs (p < q)
        on each multi-orbital atom (msto_a > 1). Includes both occupied-virtual /
        fractional-occupation pairs (which perturb the one-particle density P_A)
        and non-degenerate intrashell/intershell canonical pairs (which rotate
        individual canonical orbital expectation values in the model-potential fit).
        """
        self.active_pairs = []
        for a in range(1, self.n_atoms + 1):
            idx_a = self.atom_basis_indices[a]
            mstoa = len(idx_a)
            if mstoa <= 1:
                continue
            k_start, _ = self.atom_orb_slices[a]
            _, occ_full, eval_full = self._get_atom_full_orbitals(a)
            _, n_act = self._get_atom_nact(a)

            for s in range(self.ispin):
                occ_s = occ_full[:, s]
                eval_s = eval_full[:, s]

                for p_local in range(n_act):
                    for q_local in range(p_local + 1, mstoa):
                        de = float(eval_s[q_local] - eval_s[p_local])
                        if abs(de) > 1e-4:
                            p_global = k_start + p_local
                            q_global = k_start + q_local
                            self.active_pairs.append({
                                "atom": a,
                                "spin": s,
                                "p_local": p_local,
                                "q_local": q_local,
                                "p": p_global,
                                "q": q_global,
                                "occ": p_global,
                                "virt": q_global,
                                "delta_occ": float(occ_s[p_local] - occ_s[q_local]),
                                "delta_eps": de,
                                "idx_a": idx_a,
                            })

    @property
    def n_active(self) -> int:
        return len(self.active_pairs)

    def _prepare_response_data(self) -> Dict[str, Any]:
        """
        Precomputes and caches the per-atom state records, two-center integral
        derivatives, analytical model-potential screening response (dvcoef / dR)_C
        via the Golub-Pereyra pseudoinverse derivative of the analys.f90 fit,
        and the unit orbital rotation response operators (dP, dfc, dfx, dvcoef, dV6).
        """
        if self._response_cache is not None:
            return self._response_cache

        use_grid_fit = True
        if self.deck is not None and getattr(self.deck, "control", None) is not None:
            use_grid_fit = bool(getattr(self.deck.control, "use_grid_fit", True))

        atoms: Dict[int, Dict[str, Any]] = {}
        for a in range(1, self.n_atoms + 1):
            idx_a = self.atom_basis_indices[a]
            m = len(idx_a)
            m2 = (m * (m + 1)) // 2
            k_start, k_end = self.atom_orb_slices[a]
            norb_a = k_end - k_start
            znuc = float(self.charges[a - 1])

            nt, vc, ve = _phatpsy_core.state.get_atom_model_potential(a, 32)
            vc = vc[:nt].copy()
            ve = ve[:nt].copy()
            lva = np.zeros(nt, dtype=np.int32)
            mva = np.zeros(nt, dtype=np.int32)

            zpot2 = _unpack_packed(_phatpsy_core.state.get_atom_record_zpot2(a, m2).ravel(), m)
            cpot2 = _unpack_packed(_phatpsy_core.state.get_atom_record_cpot2(a, m2).ravel(), m)
            pkpot = _phatpsy_core.state.get_atom_record_pkpot(a, m2, 6, self.ispin).copy()
            stpot_pack = _phatpsy_core.state.get_atom_elem_stpot(a, m2, max(1, nt)).copy()
            stpot_mats = np.array([_unpack_packed(stpot_pack[:, nv], m) for nv in range(nt)])
            stovl = _unpack_packed(_phatpsy_core.state.get_atom_elem_stovl(a, m2).ravel(), m)
            ontran = _phatpsy_core.state.get_atom_elem_ontran(a, m).copy()

            C_a, occ_a, eval_a = self._get_atom_full_orbitals(a)
            sumocc = float(self.orbocc * np.sum(occ_a))

            dP_packed = np.zeros((m2, self.ispin), dtype=np.float64)
            P_spin = np.zeros((m, m, self.ispin), dtype=np.float64)
            for s in range(self.ispin):
                P_s = C_a[:, :, s] @ np.diag(occ_a[:, s]) @ C_a[:, :, s].T
                P_spin[:, :, s] = P_s
                idx = 0
                for i in range(m):
                    for j in range(i + 1):
                        dP_packed[idx, s] = P_s[i, j]
                        idx += 1
            fc_pack, fx_pack = _phatpsy_core.integrals.compute_atom_fokmat(a, dP_packed)
            fc = np.array([_unpack_packed(fc_pack[:, s], m) for s in range(self.ispin)])
            fx = np.array([_unpack_packed(fx_pack[:, s], m) for s in range(self.ispin)])

            vfit2a, dvfit2a_dcoord = _compute_vfit2_and_grad(
                self.coord, self.charges, a, lva, mva, ve
            )
            na, la, etaa, anorma = self._extract_radial(a)

            # Build analys.f90 model-potential least-squares fit system (lines 536-647)
            nelec_int = int(min(sumocc, znuc))
            noble_cores = [0, 1, 5, 9, 14, 23, 27, 43]
            ncore = 0
            for nc_val in noble_cores:
                if 2 * nc_val < nelec_int:
                    ncore = nc_val

            nc = max(0, nt - 1)
            G0 = np.zeros((nc, nc), dtype=np.float64)
            g0_rhs = np.zeros(nc, dtype=np.float64)
            tevz2 = 0.0
            wgt_orb = np.zeros(norb_a, dtype=np.float64)
            epot_orb = np.zeros((norb_a, self.ispin, nt), dtype=np.float64)
            echf_orb = np.zeros((norb_a, self.ispin), dtype=np.float64)

            nval_active = sum(
                1 for iorb in range(ncore, norb_a) if np.sum(occ_a[iorb, :]) > 1e-6
            )
            for iorb in range(norb_a):
                if ncore > 0 and (iorb + 1) <= ncore:
                    wgt_orb[iorb] = 1.0 / (self.orbocc * ncore)
                elif np.sum(occ_a[iorb, :]) > 1e-6 and nval_active > 0:
                    wgt_orb[iorb] = 1.0 / (self.orbocc * nval_active)
                else:
                    wgt_orb[iorb] = 0.0

            for s in range(self.ispin):
                for iorb in range(norb_a):
                    c = C_a[:, iorb, s]
                    occup = occ_a[iorb, s] * self.orbocc
                    echf_orb[iorb, s] = float(c.T @ fc[s] @ c)
                    evz2 = float(c.T @ zpot2 @ c)
                    tevz2 += occup * evz2
                    for nv in range(nt):
                        epot_orb[iorb, s, nv] = float(c.T @ stpot_mats[nv] @ c)
                    wgt = wgt_orb[iorb]
                    for i in range(1, nt):
                        for j in range(1, nt):
                            G0[i - 1, j - 1] += epot_orb[iorb, s, i] * wgt * epot_orb[iorb, s, j]
                        g0_rhs[i - 1] += epot_orb[iorb, s, i] * wgt * (
                            echf_orb[iorb, s] - sumocc * epot_orb[iorb, s, 0]
                        )

            denom = tevz2 - 0.5
            G_pen = G0.copy()
            rhs_pen = g0_rhs.copy()
            if nc > 0 and sumocc > 1e-12:
                for i in range(1, nt):
                    for j in range(1, nt):
                        G_pen[i - 1, j - 1] += 1.0 / sumocc + vfit2a[i] * vfit2a[j] / denom
                    rhs_pen[i - 1] += -1.0 + vfit2a[i] * (tevz2 - sumocc * vfit2a[0]) / denom
                G_pinv = np.linalg.pinv(G_pen, rcond=1e-10)
                P_null = np.eye(nc) - G_pinv @ G_pen
                c_sub = G_pinv @ rhs_pen
            else:
                G_pinv = np.zeros((nc, nc), dtype=np.float64)
                P_null = np.zeros((nc, nc), dtype=np.float64)
                c_sub = np.zeros(nc, dtype=np.float64)

            w_check = -vc[1:] / vc[0] if (nt > 1 and abs(vc[0]) > 1e-12) else np.zeros(0)
            atom_uses_grid_fit = (
                use_grid_fit
                and nt > 1
                and np.all(w_check >= -1e-6)
                and abs(float(np.sum(w_check)) - 1.0) < 1e-5
            )
            nact_sto, nact_orb = self._get_atom_nact(a)
            atoms[a] = {
                "m": m,
                "m2": m2,
                "norb": norb_a,
                "nact_sto": nact_sto,
                "nact_orb": nact_orb,
                "k_start": k_start,
                "idx": idx_a,
                "znuc": znuc,
                "nt": nt,
                "nc": nc,
                "use_grid_fit": atom_uses_grid_fit,
                "vcoef": vc,
                "c_sub": c_sub,
                "vexp": ve,
                "lva": lva,
                "mva": mva,
                "zpot2": zpot2,
                "cpot2": cpot2,
                "pkpot": pkpot,
                "stpot_mats": stpot_mats,
                "stovl": stovl,
                "ontran": ontran,
                "C": C_a,
                "occ": occ_a,
                "eval": eval_a,
                "sumocc": sumocc,
                "P_spin": P_spin,
                "P_tot": self.orbocc * np.sum(P_spin, axis=2),
                "fc": fc,
                "fx": fx,
                "vfit2": vfit2a,
                "dvfit2_dcoord": dvfit2a_dcoord,
                "na": na,
                "la": la,
                "etaa": etaa,
                "anorma": anorma,
                "wgt_orb": wgt_orb,
                "epot_orb": epot_orb,
                "echf_orb": echf_orb,
                "tevz2": tevz2,
                "G_pen": G_pen,
                "rhs_pen": rhs_pen,
                "G_pinv": G_pinv,
                "P_null": P_null,
            }

        qpsmet = True
        pkscal = 1.0
        qnwvec = False
        if self.deck is not None:
            if getattr(self.deck, "control", None) is not None:
                qpsmet = bool(getattr(self.deck.control, "use_pseudometric", True))
                pkscal = float(getattr(self.deck.control, "pk_scale", 1.0))
            if getattr(self.deck, "valence_configs", None):
                qnwvec = True
            for g in getattr(self.deck, "geometry", []) or []:
                if (
                    getattr(g, "hybridization", None) is not None
                    or getattr(g, "valence_config", None) is not None
                    or bool(getattr(g, "use_new_vectors", False))
                ):
                    qnwvec = True
        if not qnwvec:
            # Fallback check on stored energy-weighted projector D5_A (occ*eval vs eval)
            for aa in atoms.values():
                if aa["znuc"] > 1.5 and aa["norb"] > 0:
                    for s in range(self.ispin):
                        occ_s = aa["occ"][: aa["norb"], s]
                        if np.any((occ_s < 0.99) & (occ_s > -0.01)):
                            C_sub = aa["C"][:, : aa["norb"], s]
                            ev_s = aa["eval"][: aa["norb"], s]
                            D5_stor = _unpack_packed(aa["pkpot"][:, 4, s], aa["m"])
                            err_t = np.max(np.abs(D5_stor - pkscal * (C_sub * (occ_s * ev_s)) @ C_sub.T))
                            err_f = np.max(np.abs(D5_stor - pkscal * (C_sub * ev_s) @ C_sub.T))
                            if err_t < err_f:
                                qnwvec = True

        # Pairwise two-center integrals and explicit coordinate gradients
        pair_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
        dzpot2_dcoord = {
            a: np.zeros((self.n_atoms, 3, atoms[a]["m"], atoms[a]["m"])) for a in atoms
        }
        dcpot2_explicit_dcoord = {
            a: np.zeros((self.n_atoms, 3, atoms[a]["m"], atoms[a]["m"])) for a in atoms
        }
        dxpot2_explicit_dcoord = {
            a: np.zeros((self.n_atoms, 3, self.ispin, atoms[a]["m"], atoms[a]["m"])) for a in atoms
        }
        dtevz2_explicit_dcoord = {a: np.zeros((self.n_atoms, 3)) for a in atoms}

        for a in range(1, self.n_atoms + 1):
            for b in range(1, self.n_atoms + 1):
                if a == b:
                    continue
                aa, ab = atoms[a], atoms[b]
                r_a = self.coord[:, a - 1]
                r_b = self.coord[:, b - 1]
                if np.linalg.norm(r_b - r_a) < 1e-12:
                    continue

                frame_ab = DiatomicFrame(r_a, r_b)
                s_ab, ds_dra, _ = compute_diatomic_overlap_matrix_and_gradient(
                    aa["na"], aa["la"], aa["etaa"], aa["anorma"],
                    ab["na"], ab["la"], ab["etaa"], ab["anorma"],
                    frame_ab,
                )
                v_ab_pack, dv_dra_pack, _ = compute_diatomic_potential_matrix_and_gradient(
                    aa["na"], aa["la"], aa["etaa"], aa["anorma"],
                    ab["lva"], ab["mva"], ab["vexp"],
                    frame_ab,
                )
                v_ab = np.array(
                    [_unpack_packed(v_ab_pack[:, nv], aa["m"]) for nv in range(ab["nt"])]
                )
                dv_dra = np.array([
                    [_unpack_packed(dv_dra_pack[k, :, nv], aa["m"]) for nv in range(ab["nt"])]
                    for k in range(3)
                ])
                if qnwvec and ab["znuc"] > 1.5 and aa["nact_sto"] < aa["m"]:
                    nsto_es = aa["nact_sto"]
                    v_ab[:, nsto_es:, :] = 0.0
                    v_ab[:, :, nsto_es:] = 0.0
                    dv_dra[:, :, nsto_es:, :] = 0.0
                    dv_dra[:, :, :, nsto_es:] = 0.0
                pair_data[(a, b)] = {
                    "S_ab": s_ab,
                    "dS_dra": ds_dra,
                    "V_ab": v_ab,
                    "dV_dra": dv_dra,
                }

                for k in range(3):
                    dz = -ab["znuc"] * dv_dra[k, 0]
                    dzpot2_dcoord[a][a - 1, k] += dz
                    dzpot2_dcoord[a][b - 1, k] -= dz
                    dtev = float(np.sum(aa["P_tot"] * dz))
                    dtevz2_explicit_dcoord[a][a - 1, k] += dtev
                    dtevz2_explicit_dcoord[a][b - 1, k] -= dtev

                    dc = sum(ab["vcoef"][nv] * dv_dra[k, nv] for nv in range(ab["nt"]))
                    dcpot2_explicit_dcoord[a][a - 1, k] += dc
                    dcpot2_explicit_dcoord[a][b - 1, k] -= dc

                    if not (qnwvec and (aa["znuc"] <= 1.5 or ab["znuc"] <= 1.5)):
                        for s in range(self.ispin):
                            V6_B = _unpack_packed(ab["pkpot"][:, 5, s], ab["m"])
                            dx = ds_dra[k] @ V6_B @ s_ab.T + s_ab @ V6_B @ ds_dra[k].T
                            dxpot2_explicit_dcoord[a][a - 1, k, s] += dx
                            dxpot2_explicit_dcoord[a][b - 1, k, s] -= dx

        # Analytical fixed-C model-potential screening response (dvcoef_A / dR)_C
        dvcoef_explicit_dcoord: Dict[int, np.ndarray] = {}
        for a, aa in atoms.items():
            nt, nc = aa["nt"], aa["nc"]
            dvc = np.zeros((self.n_atoms, 3, nt), dtype=np.float64)
            if not aa["use_grid_fit"] and nc > 0 and aa["sumocc"] > 1e-12:
                c_sub = aa["c_sub"]
                denom = aa["tevz2"] - 0.5
                vf = aa["vfit2"]
                vf_sub = vf[1:]
                num = aa["tevz2"] - aa["sumocc"] * vf[0]
                for atom_k in range(self.n_atoms):
                    for cart in range(3):
                        dvf = aa["dvfit2_dcoord"][atom_k, cart, :]
                        dvf_sub = dvf[1:]
                        dtev = dtevz2_explicit_dcoord[a][atom_k, cart]
                        dnum = dtev - aa["sumocc"] * dvf[0]
                        dG = (
                            np.outer(dvf_sub, vf_sub) * denom
                            + np.outer(vf_sub, dvf_sub) * denom
                            - np.outer(vf_sub, vf_sub) * dtev
                        ) / (denom ** 2)
                        drhs = (
                            (dvf_sub * num + vf_sub * dnum) * denom - vf_sub * num * dtev
                        ) / (denom ** 2)
                        # Golub-Pereyra pseudoinverse derivative for minimum-norm DGELSD solution
                        dc_sub = aa["G_pinv"] @ (drhs - dG @ c_sub) + aa["P_null"] @ dG @ (
                            aa["G_pinv"] @ c_sub
                        )
                        dvc[atom_k, cart, 1:] = dc_sub
            dvcoef_explicit_dcoord[a] = dvc

        # Total fixed-C Fock derivative dF_fixedC_dcoord (explicit + implicit dvcoef_B|_C)
        dF_fixedC_dcoord = {
            a: np.zeros((self.n_atoms, 3, self.ispin, atoms[a]["m"], atoms[a]["m"]))
            for a in atoms
        }
        for a, aa in atoms.items():
            for atom_k in range(self.n_atoms):
                for cart in range(3):
                    base_dc = (
                        dzpot2_dcoord[a][atom_k, cart]
                        + dcpot2_explicit_dcoord[a][atom_k, cart]
                    )
                    imp_dc = np.zeros((aa["m"], aa["m"]), dtype=np.float64)
                    imp_dx = np.zeros((self.ispin, aa["m"], aa["m"]), dtype=np.float64)
                    for b in range(1, self.n_atoms + 1):
                        if b == a or (a, b) not in pair_data:
                            continue
                        ab = atoms[b]
                        pd = pair_data[(a, b)]
                        dvc_b = dvcoef_explicit_dcoord[b][atom_k, cart]
                        imp_dc += sum(dvc_b[nv] * pd["V_ab"][nv] for nv in range(ab["nt"]))
                        if not (qnwvec and (aa["znuc"] <= 1.5 or ab["znuc"] <= 1.5)):
                            dvpot_b = sum(
                                dvc_b[nv] * ab["stpot_mats"][nv] for nv in range(ab["nt"])
                            )
                            if qnwvec and ab["nact_sto"] < ab["m"]:
                                dvpot_b = dvpot_b.copy()
                                dvpot_b[ab["nact_sto"] :, :] = 0.0
                                dvpot_b[:, ab["nact_sto"] :] = 0.0
                            dV6_b_fixedC = -ab["ontran"].T @ dvpot_b @ ab["ontran"]
                            dx_b = pd["S_ab"] @ dV6_b_fixedC @ pd["S_ab"].T
                            for s in range(self.ispin):
                                imp_dx[s] += dx_b
                    for s in range(self.ispin):
                        dF_fixedC_dcoord[a][atom_k, cart, s] = (
                            base_dc
                            + imp_dc
                            + dxpot2_explicit_dcoord[a][atom_k, cart, s]
                            + imp_dx[s]
                        )

        # Unit orbital rotation effects for each active pair (under delta c_p = +c_q, delta c_q = -c_p)
        rot_effects: List[Dict[str, Any]] = []
        for pair in self.active_pairs:
            a = pair["atom"]
            s_rot = pair["spin"]
            p = pair["p_local"]
            q = pair["q_local"]
            aa = atoms[a]
            m, m2, nt, nc = aa["m"], aa["m2"], aa["nt"], aa["nc"]
            c_p = aa["C"][:, p, s_rot]
            c_q = aa["C"][:, q, s_rot]
            occ_p = aa["occ"][p, s_rot]
            occ_q = aa["occ"][q, s_rot]

            dP_spin = np.zeros((m, m, self.ispin), dtype=np.float64)
            dP_spin[:, :, s_rot] = (occ_p - occ_q) * (np.outer(c_p, c_q) + np.outer(c_q, c_p))
            dP_tot = self.orbocc * np.sum(dP_spin, axis=2)

            dP_packed = np.zeros((m2, self.ispin), dtype=np.float64)
            idx = 0
            for i in range(m):
                for j in range(i + 1):
                    dP_packed[idx, s_rot] = dP_spin[i, j, s_rot]
                    idx += 1
            dfc_pack, dfx_pack = _phatpsy_core.integrals.compute_atom_fokmat(a, dP_packed)
            dfc = np.array([_unpack_packed(dfc_pack[:, s], m) for s in range(self.ispin)])
            dfx = np.array([_unpack_packed(dfx_pack[:, s], m) for s in range(self.ispin)])

            dvcoef = np.zeros(nt, dtype=np.float64)
            if aa["use_grid_fit"]:
                dvcoef = _compute_grid_fit_dvcoef(
                    aa["na"],
                    aa["la"],
                    aa["etaa"],
                    aa["anorma"],
                    aa["vexp"],
                    aa["vcoef"],
                    aa["P_tot"],
                    aa["sumocc"],
                    dP_tot,
                )
            elif nc > 0 and aa["sumocc"] > 1e-12:
                dG0 = np.zeros((nc, nc), dtype=np.float64)
                dg0_rhs = np.zeros(nc, dtype=np.float64)
                depot = np.zeros((aa["norb"], self.ispin, nt), dtype=np.float64)
                dechf = np.zeros((aa["norb"], self.ispin), dtype=np.float64)
                for nv in range(nt):
                    v_pq = float(c_p.T @ aa["stpot_mats"][nv] @ c_q)
                    depot[p, s_rot, nv] = +2.0 * v_pq
                    if q < aa["norb"]:
                        depot[q, s_rot, nv] = -2.0 * v_pq
                for s in range(self.ispin):
                    for iorb in range(aa["norb"]):
                        c_i = aa["C"][:, iorb, s]
                        dechf[iorb, s] = float(c_i.T @ dfc[s] @ c_i)
                fc_pq = float(c_p.T @ aa["fc"][s_rot] @ c_q)
                dechf[p, s_rot] += +2.0 * fc_pq
                if q < aa["norb"]:
                    dechf[q, s_rot] += -2.0 * fc_pq

                for s in range(self.ispin):
                    for iorb in range(aa["norb"]):
                        wgt = aa["wgt_orb"][iorb]
                        ep = aa["epot_orb"][iorb, s]
                        dep = depot[iorb, s]
                        ec = aa["echf_orb"][iorb, s]
                        dec = dechf[iorb, s]
                        for i in range(1, nt):
                            for j in range(1, nt):
                                dG0[i - 1, j - 1] += wgt * (dep[i] * ep[j] + ep[i] * dep[j])
                            dg0_rhs[i - 1] += wgt * (
                                dep[i] * (ec - aa["sumocc"] * ep[0])
                                + ep[i] * (dec - aa["sumocc"] * dep[0])
                            )

                dtev = float(np.sum(dP_tot * aa["zpot2"]))
                denom = aa["tevz2"] - 0.5
                vf = aa["vfit2"]
                vf_sub = vf[1:]
                num = aa["tevz2"] - aa["sumocc"] * vf[0]
                dG = dG0 - np.outer(vf_sub, vf_sub) * dtev / (denom ** 2)
                drhs = dg0_rhs + vf_sub * (dtev * denom - num * dtev) / (denom ** 2)
                c_sub = aa["c_sub"]
                dc_sub = aa["G_pinv"] @ (drhs - dG @ c_sub) + aa["P_null"] @ dG @ (
                    aa["G_pinv"] @ c_sub
                )
                dvcoef[1:] = dc_sub

            dV6 = np.zeros((self.ispin, m, m), dtype=np.float64)
            if not (qnwvec and aa["znuc"] <= 1.5):
                dvpot = sum(dvcoef[nv] * aa["stpot_mats"][nv] for nv in range(nt))
                for s in range(self.ispin):
                    donfock = dfx[s] + dfc[s] - dvpot
                    if qnwvec and aa["nact_sto"] < m:
                        donfock = donfock.copy()
                        donfock[aa["nact_sto"] :, :] = 0.0
                        donfock[:, aa["nact_sto"] :] = 0.0
                    dV6[s] = aa["ontran"].T @ donfock @ aa["ontran"]

            rot_effects.append({
                "a": a,
                "s_rot": s_rot,
                "p": p,
                "q": q,
                "dP_spin": dP_spin,
                "dP_tot": dP_tot,
                "dfc": dfc,
                "dfx": dfx,
                "dvcoef": dvcoef,
                "dV6": dV6,
            })

        has_any_pk = any(
            (aa["znuc"] > 1.5 or qnwvec) and float(np.max(np.abs(aa["pkpot"][:, 0:2, :]))) > 1e-12
            for aa in atoms.values()
        )
        pk_response = None
        if has_any_pk:
            pk_response = self._build_pk_coupled_response(
                atoms, pair_data, dzpot2_dcoord, dF_fixedC_dcoord, rot_effects
            )

        self._response_cache = {
            "atoms": atoms,
            "pair_data": pair_data,
            "dzpot2_dcoord": dzpot2_dcoord,
            "dF_fixedC_dcoord": dF_fixedC_dcoord,
            "dvcoef_explicit_dcoord": dvcoef_explicit_dcoord,
            "rot_effects": rot_effects,
            "pk_response": pk_response,
        }
        return self._response_cache

    def _build_pk_coupled_response(
        self,
        atoms: Dict[int, Dict[str, Any]],
        pair_data: Dict[Tuple[int, int], Dict[str, Any]],
        dzpot2_dcoord: Dict[int, np.ndarray],
        dF_fixedC_dcoord: Dict[int, np.ndarray],
        rot_effects: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Constructs the exact Phillips-Kleinman (PK) pseudopotential and regularized
        pseudometric response operators for heavy-atom bonds (Z_A > 1.5), where the
        generalized eigenvalue problem in scf.f90 is solved in the regularized
        pseudometric S_tilde = S_A - P_{1, A}^reg with F_tilde = F_A - P_{2, A}^reg,
        followed by PK back-projection Q_A^T = I - S_A^{-1} P_{1, A}^reg and
        symmetric Löwdin S_A-orthonormalization.
        """
        qpsmet = True
        pkscal = 1.0
        qnwvec = False
        if self.deck is not None:
            if getattr(self.deck, "control", None) is not None:
                qpsmet = bool(getattr(self.deck.control, "use_pseudometric", True))
                pkscal = float(getattr(self.deck.control, "pk_scale", 1.0))
            if getattr(self.deck, "valence_configs", None):
                qnwvec = True
            for g in getattr(self.deck, "geometry", []) or []:
                if (
                    getattr(g, "hybridization", None) is not None
                    or getattr(g, "valence_config", None) is not None
                    or bool(getattr(g, "use_new_vectors", False))
                ):
                    qnwvec = True
        if not qnwvec:
            # Fallback check on stored energy-weighted projector D5_A (occ*eval vs eval)
            for aa in atoms.values():
                if aa["znuc"] > 1.5 and aa["norb"] > 0:
                    for s in range(self.ispin):
                        occ_s = aa["occ"][: aa["norb"], s]
                        if np.any((occ_s < 0.99) & (occ_s > -0.01)):
                            C_sub = aa["C"][:, : aa["norb"], s]
                            ev_s = aa["eval"][: aa["norb"], s]
                            D5_stor = _unpack_packed(aa["pkpot"][:, 4, s], aa["m"])
                            err_t = np.max(np.abs(D5_stor - pkscal * (C_sub * (occ_s * ev_s)) @ C_sub.T))
                            err_f = np.max(np.abs(D5_stor - pkscal * (C_sub * ev_s) @ C_sub.T))
                            if err_t < err_f:
                                qnwvec = True

        reg_delta = 0.42 if qnwvec else 1.0e-3

        def _lowdin_deriv(C_bar, dC_bar, S_A, w_m, v_m, M_mhalf):
            dM = dC_bar.T @ S_A @ C_bar + C_bar.T @ S_A @ dC_bar
            dM_eig = v_m.T @ dM @ v_m
            denom = np.sqrt(w_m)[:, None] + np.sqrt(w_m)[None, :]
            dM_half = v_m @ (dM_eig / denom) @ v_m.T
            dM_mhalf = -M_mhalf @ dM_half @ M_mhalf
            return dC_bar @ M_mhalf + C_bar @ dM_mhalf

        def _pk_pair_weight(aa_rec, ab_rec) -> float:
            if (aa_rec["znuc"] <= 1.5) != (ab_rec["znuc"] <= 1.5):
                return 0.0
            if aa_rec["znuc"] <= 1.5 and ab_rec["znuc"] <= 1.5:
                return 0.40 if qnwvec else 0.0
            return 1.0

        pk_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
        occ_slots: List[Tuple[int, int, int]] = []

        for a, aa in atoms.items():
            m, m2, norb = aa["m"], aa["m2"], aa["nact_orb"]
            n_sel = m
            S_A, ontran = aa["stovl"], aa["ontran"]
            w_s, v_s = np.linalg.eigh(S_A)
            S_mhalf = (v_s / np.sqrt(np.maximum(w_s, 1e-12))) @ v_s.T
            S_half = (v_s * np.sqrt(np.maximum(w_s, 1e-12))) @ v_s.T
            H0_A = _unpack_packed(_phatpsy_core.state.get_atom_elem_sthmat(a, m2).ravel(), m)
            has_pk = qpsmet and (aa["znuc"] > 1.5 or qnwvec) and (
                float(np.max(np.abs(aa["pkpot"][:, 0:2, :]))) > 1e-12
            )
            for s in range(self.ispin):
                X2_A = _unpack_packed(aa["pkpot"][:, 2, s], m)
                if has_pk:
                    # Reconstruct raw environment projectors P1_A, P2_A from neighbor D4_B, D5_B
                    P1_A = np.zeros((m, m), dtype=np.float64)
                    P2_A = np.zeros((m, m), dtype=np.float64)
                    for b, ab in atoms.items():
                        if b != a and (a, b) in pair_data:
                            w_ab = _pk_pair_weight(aa, ab)
                            if w_ab <= 0.0:
                                continue
                            S_ab = pair_data[(a, b)]["S_ab"]
                            D4_B = _unpack_packed(ab["pkpot"][:, 3, s], ab["m"])
                            D5_B = _unpack_packed(ab["pkpot"][:, 4, s], ab["m"])
                            P1_A += w_ab * (S_ab @ D4_B @ S_ab.T)
                            P2_A += w_ab * (S_ab @ D5_B @ S_ab.T)
                else:
                    P1_A = np.zeros((m, m), dtype=np.float64)
                    P2_A = np.zeros((m, m), dtype=np.float64)

                P1_low = S_mhalf @ P1_A @ S_mhalf
                lam_p, v_p = np.linalg.eigh(P1_low)
                cap = 1.0 - reg_delta
                if reg_delta >= 0.05:
                    w_cap = min(0.04, 0.5 * reg_delta)
                    t_cap = np.clip((lam_p - (cap - w_cap)) / (2.0 * w_cap), 0.0, 1.0)
                    g_lam = np.where(
                        lam_p <= cap - w_cap,
                        lam_p,
                        np.where(lam_p >= cap + w_cap, cap, lam_p - w_cap * (t_cap ** 2)),
                    )
                    gp_lam = 1.0 - t_cap
                else:
                    g_lam = np.minimum(lam_p, cap)
                    gp_lam = np.where(lam_p <= cap, 1.0, 0.0)
                D_reg = np.zeros((m, m), dtype=np.float64)
                for i in range(m):
                    for j in range(m):
                        if abs(lam_p[i] - lam_p[j]) > 1e-10:
                            D_reg[i, j] = (g_lam[i] - g_lam[j]) / (lam_p[i] - lam_p[j])
                        else:
                            D_reg[i, j] = gp_lam[i]
                P1_reg = S_half @ (v_p @ np.diag(g_lam) @ v_p.T) @ S_half if has_pk else np.zeros((m, m))
                P2_reg = P2_A.copy()
                if has_pk and reg_delta <= 1.0e-2 and np.max(lam_p) > cap:
                    s_modes = np.where(lam_p > cap, cap / np.maximum(lam_p, 1e-10), 1.0)
                    B_low = S_mhalf @ P2_A @ S_mhalf
                    B_vp = (v_p.T @ B_low @ v_p) * np.sqrt(np.outer(s_modes, s_modes))
                    P2_reg = S_half @ (v_p @ B_vp @ v_p.T) @ S_half

                F_A = H0_A + aa["fc"][s] + aa["fx"][s] + aa["zpot2"] + aa["cpot2"] + X2_A
                S_tilde = S_A - P1_reg
                w_st, v_st = np.linalg.eigh(S_tilde)
                St_mhalf = (v_st / np.sqrt(np.maximum(w_st, 1e-12))) @ v_st.T
                F_on = St_mhalf.T @ (F_A - P2_reg) @ St_mhalf
                if 0 < n_sel < m:
                    eval_1, evec_1 = np.linalg.eigh(F_on[:n_sel, :n_sel])
                    eval_2, evec_2 = np.linalg.eigh(F_on[n_sel:, n_sel:])
                    eval_t = np.concatenate([eval_1, eval_2])
                    evec_t = np.zeros((m, m), dtype=np.float64)
                    evec_t[:n_sel, :n_sel] = evec_1
                    evec_t[n_sel:, n_sel:] = evec_2
                else:
                    eval_t, evec_t = np.linalg.eigh(F_on)
                C_tilde = St_mhalf @ evec_t

                occ_s = aa["occ"][:norb, s]
                occ_idx = [i for i in range(norb) if occ_s[i] > 1e-6] if qnwvec else list(range(norb))
                unocc_idx = [i for i in range(norb) if occ_s[i] <= 1e-6] if qnwvec else []
                for i in range(norb):
                    if occ_s[i] > 1e-6 or not qnwvec:
                        occ_slots.append((a, s, i))

                Q_T = np.eye(m) - ontran @ P1_reg
                if occ_idx:
                    C_bar_occ = Q_T @ C_tilde[:, occ_idx]
                    M_occ = C_bar_occ.T @ S_A @ C_bar_occ
                    w_m, v_m = np.linalg.eigh(M_occ)
                    M_mhalf = (v_m / np.sqrt(np.maximum(w_m, 1e-12))) @ v_m.T
                    C_occ = C_bar_occ @ M_mhalf
                    O_occ = C_occ.T @ S_A @ aa["C"][:, occ_idx, s]
                    C_tilde[:, occ_idx] = C_tilde[:, occ_idx] @ O_occ
                    C_bar_occ = Q_T @ C_tilde[:, occ_idx]
                    M_occ = C_bar_occ.T @ S_A @ C_bar_occ
                    w_m, v_m = np.linalg.eigh(M_occ)
                    M_mhalf = (v_m / np.sqrt(np.maximum(w_m, 1e-12))) @ v_m.T
                    C_occ = C_bar_occ @ M_mhalf
                    P_occ = C_occ @ C_occ.T
                else:
                    C_bar_occ = w_m = v_m = M_mhalf = C_occ = None
                    P_occ = np.zeros((m, m), dtype=np.float64)

                if unocc_idx:
                    C_bar_u = (np.eye(m) - P_occ @ S_A) @ (Q_T @ C_tilde[:, unocc_idx])
                    M_u = C_bar_u.T @ S_A @ C_bar_u
                    w_u, v_u = np.linalg.eigh(M_u)
                    M_u_mhalf = (v_u / np.sqrt(np.maximum(w_u, 1e-12))) @ v_u.T
                    C_u = C_bar_u @ M_u_mhalf
                    O_u = C_u.T @ S_A @ aa["C"][:, unocc_idx, s]
                    C_tilde[:, unocc_idx] = C_tilde[:, unocc_idx] @ O_u
                    C_bar_u = (np.eye(m) - P_occ @ S_A) @ (Q_T @ C_tilde[:, unocc_idx])
                    M_u = C_bar_u.T @ S_A @ C_bar_u
                    w_u, v_u = np.linalg.eigh(M_u)
                    M_u_mhalf = (v_u / np.sqrt(np.maximum(w_u, 1e-12))) @ v_u.T
                else:
                    C_bar_u = w_u = v_u = M_u_mhalf = None

                pk_data[(a, s)] = {
                    "S_A": S_A,
                    "ontran": ontran,
                    "S_mhalf": S_mhalf,
                    "S_half": S_half,
                    "v_p": v_p,
                    "D_reg": D_reg,
                    "eval_t": eval_t,
                    "C_tilde": C_tilde,
                    "occ_idx": occ_idx,
                    "unocc_idx": unocc_idx,
                    "Q_T": Q_T,
                    "C_bar_occ": C_bar_occ,
                    "w_m": w_m,
                    "v_m": v_m,
                    "M_mhalf": M_mhalf,
                    "C_occ": C_occ,
                    "P_occ": P_occ,
                    "C_bar_u": C_bar_u,
                    "w_u": w_u,
                    "v_u": v_u,
                    "M_u_mhalf": M_u_mhalf,
                    "H0_A": H0_A,
                    "has_pk": has_pk,
                    "norb": norb,
                    "n_sel": n_sel,
                }

        slot_lookup = {slot: idx for idx, slot in enumerate(occ_slots)}

        def _scf_step_response(dF_dict, dP1_raw_dict, dP2_raw_dict):
            dU_out = np.zeros(self.n_active, dtype=np.float64)
            deval_out = np.zeros(len(occ_slots), dtype=np.float64)
            dC_act_map = {}
            for (a, s), pd in pk_data.items():
                S_A = pd["S_A"]
                m, norb, n_sel = S_A.shape[0], pd["norb"], pd["n_sel"]
                if pd["has_pk"]:
                    dP1_raw = dP1_raw_dict[(a, s)]
                    dP2_r = dP2_raw_dict[(a, s)]
                    dP1_eig = pd["v_p"].T @ (pd["S_mhalf"] @ dP1_raw @ pd["S_mhalf"]) @ pd["v_p"]
                    dP1_r = pd["S_half"] @ (pd["v_p"] @ (pd["D_reg"] * dP1_eig) @ pd["v_p"].T) @ pd["S_half"]
                else:
                    dP1_r = np.zeros((m, m), dtype=np.float64)
                    dP2_r = np.zeros((m, m), dtype=np.float64)
                dF = dF_dict[(a, s)]
                Ct, ev = pd["C_tilde"], pd["eval_t"]
                W = np.zeros((m, m), dtype=np.float64)
                for p in range(m):
                    W[p, p] = 0.5 * float(Ct[:, p].T @ dP1_r @ Ct[:, p])
                    for q in range(m):
                        if (p < n_sel) == (q < n_sel) and abs(ev[p] - ev[q]) > 1e-4:
                            W[q, p] = float(Ct[:, q].T @ (dF - dP2_r + ev[p] * dP1_r) @ Ct[:, p]) / (
                                ev[p] - ev[q]
                            )
                dCt = Ct @ W
                occ_idx, unocc_idx = pd["occ_idx"], pd["unocc_idx"]
                dQ_T = -pd["ontran"] @ dP1_r
                dC_act = np.zeros((m, norb), dtype=np.float64)
                if occ_idx:
                    dC_bar_occ = dQ_T @ Ct[:, occ_idx] + pd["Q_T"] @ dCt[:, occ_idx]
                    dC_occ = _lowdin_deriv(
                        pd["C_bar_occ"], dC_bar_occ, S_A, pd["w_m"], pd["v_m"], pd["M_mhalf"]
                    )
                    dC_act[:, occ_idx] = dC_occ
                else:
                    dC_occ = None
                if unocc_idx:
                    C_occ = pd["C_occ"]
                    dP_occ = (
                        dC_occ @ C_occ.T + C_occ @ dC_occ.T
                        if dC_occ is not None
                        else np.zeros((m, m), dtype=np.float64)
                    )
                    dC_bar_u_raw = dQ_T @ Ct[:, unocc_idx] + pd["Q_T"] @ dCt[:, unocc_idx]
                    dC_bar_u = (np.eye(m) - pd["P_occ"] @ S_A) @ dC_bar_u_raw - (dP_occ @ S_A) @ (
                        pd["Q_T"] @ Ct[:, unocc_idx]
                    )
                    dC_u = _lowdin_deriv(
                        pd["C_bar_u"], dC_bar_u, S_A, pd["w_u"], pd["v_u"], pd["M_u_mhalf"]
                    )
                    dC_act[:, unocc_idx] = dC_u
                dC_act_map[(a, s)] = dC_act
                for i_orb in range(norb):
                    if (a, s, i_orb) in slot_lookup:
                        slot_idx = slot_lookup[(a, s, i_orb)]
                        deval_out[slot_idx] = float(
                            Ct[:, i_orb].T @ (dF - dP2_r + ev[i_orb] * dP1_r) @ Ct[:, i_orb]
                        )

            for idx_pair, pair in enumerate(self.active_pairs):
                a, s, p, q = pair["atom"], pair["spin"], pair["p_local"], pair["q_local"]
                dC_act = dC_act_map[(a, s)]
                S_A = pk_data[(a, s)]["S_A"]
                c_q = atoms[a]["C"][:, q, s]
                dU_out[idx_pair] = float(c_q.T @ S_A @ dC_act[:, p])
            return dU_out, deval_out

        n_u = self.n_active
        n_e = len(occ_slots)
        M_UU = np.zeros((n_u, n_u), dtype=np.float64)
        M_eU = np.zeros((n_e, n_u), dtype=np.float64)
        M_Ue = np.zeros((n_u, n_e), dtype=np.float64)
        M_ee = np.zeros((n_e, n_e), dtype=np.float64)

        for j_r, eff in enumerate(rot_effects):
            b, s_rot, p_r, q_r = eff["a"], eff["s_rot"], eff["p"], eff["q"]
            ab = atoms[b]
            c_p, c_q = ab["C"][:, p_r, s_rot], ab["C"][:, q_r, s_rot]
            occ_p, occ_q = ab["occ"][p_r, s_rot], ab["occ"][q_r, s_rot]
            nact_b = ab["nact_orb"]
            ev_p = ab["eval"][p_r, s_rot]
            ev_q = ab["eval"][q_r, s_rot] if q_r < nact_b else 0.0
            w_p = (occ_p if qnwvec else (1.0 if np.sum(ab["occ"][p_r, :]) > 1e-6 else 0.0)) * ev_p if p_r < nact_b else 0.0
            w_q = (occ_q if qnwvec else (1.0 if np.sum(ab["occ"][q_r, :]) > 1e-6 else 0.0)) * ev_q if q_r < nact_b else 0.0
            emit_pk = qpsmet and (ab["znuc"] > 1.5 or qnwvec)
            dD4_b = pkscal * eff["dP_spin"][:, :, s_rot] if emit_pk else np.zeros((ab["m"], ab["m"]))
            dD5_b = (
                pkscal * (w_p - w_q) * (np.outer(c_p, c_q) + np.outer(c_q, c_p))
                if emit_pk
                else np.zeros((ab["m"], ab["m"]))
            )
            dF_dict = {(a, s): np.zeros((atoms[a]["m"], atoms[a]["m"])) for (a, s) in pk_data}
            dP1_dict = {(a, s): np.zeros((atoms[a]["m"], atoms[a]["m"])) for (a, s) in pk_data}
            dP2_dict = {(a, s): np.zeros((atoms[a]["m"], atoms[a]["m"])) for (a, s) in pk_data}
            for s in range(self.ispin):
                dF_dict[(b, s)] = eff["dfc"][s] + eff["dfx"][s]
            for a in atoms:
                if a != b and (a, b) in pair_data:
                    pd_ab = pair_data[(a, b)]
                    S_ab = pd_ab["S_ab"]
                    dcpot2_ab = sum(eff["dvcoef"][nv] * pd_ab["V_ab"][nv] for nv in range(ab["nt"]))
                    inc_x2 = not (qnwvec and (atoms[a]["znuc"] <= 1.5 or ab["znuc"] <= 1.5))
                    for s in range(self.ispin):
                        dF_dict[(a, s)] = dcpot2_ab + (S_ab @ eff["dV6"][s] @ S_ab.T if inc_x2 else 0.0)
                    w_ab = _pk_pair_weight(atoms[a], ab)
                    if emit_pk and w_ab > 0.0:
                        dP1_dict[(a, s_rot)] = w_ab * (S_ab @ dD4_b @ S_ab.T)
                        dP2_dict[(a, s_rot)] = w_ab * (S_ab @ dD5_b @ S_ab.T)
            M_UU[:, j_r], M_eU[:, j_r] = _scf_step_response(dF_dict, dP1_dict, dP2_dict)

        for k_e, (b, s_rot, i_orb) in enumerate(occ_slots):
            ab = atoms[b]
            if not (qpsmet and (ab["znuc"] > 1.5 or qnwvec)):
                continue
            c_i = ab["C"][:, i_orb, s_rot]
            w_i = ab["occ"][i_orb, s_rot] if qnwvec else (1.0 if np.sum(ab["occ"][i_orb, :]) > 1e-6 else 0.0)
            dD5_b = pkscal * w_i * np.outer(c_i, c_i)
            dF_dict = {(a, s): np.zeros((atoms[a]["m"], atoms[a]["m"])) for (a, s) in pk_data}
            dP1_dict = {(a, s): np.zeros((atoms[a]["m"], atoms[a]["m"])) for (a, s) in pk_data}
            dP2_dict = {(a, s): np.zeros((atoms[a]["m"], atoms[a]["m"])) for (a, s) in pk_data}
            for a in atoms:
                if a != b and (a, b) in pair_data:
                    w_ab = _pk_pair_weight(atoms[a], ab)
                    if w_ab > 0.0:
                        S_ab = pair_data[(a, b)]["S_ab"]
                        dP2_dict[(a, s_rot)] = w_ab * (S_ab @ dD5_b @ S_ab.T)
            M_Ue[:, k_e], M_ee[:, k_e] = _scf_step_response(dF_dict, dP1_dict, dP2_dict)

        inv_I_Mee = np.linalg.inv(np.eye(n_e) - M_ee) if n_e > 0 else np.zeros((0, 0))
        de_dU = inv_I_Mee @ M_eU
        de_vec = np.array([pair["delta_eps"] for pair in self.active_pairs], dtype=np.float64)
        A_pk = np.diag(de_vec) @ (np.eye(n_u) - M_UU - M_Ue @ de_dU)

        w_occ = np.array(
            [self.orbocc * atoms[a]["occ"][i, s] for (a, s, i) in occ_slots], dtype=np.float64
        )
        L_pk = np.zeros(n_u, dtype=np.float64)
        for j_r, eff in enumerate(rot_effects):
            a = eff["a"]
            H0 = pk_data[(a, 0)]["H0_A"]
            zpot2 = atoms[a]["zpot2"]
            d_one = float(np.sum(eff["dP_tot"] * (H0 + zpot2)))
            d_tev = float(w_occ @ de_dU[:, j_r])
            L_pk[j_r] = 0.5 * (d_one + d_tev)

        geom = self.deck.geometry if self.deck is not None else [None] * self.n_atoms
        g_fixedC_pk = compute_nuclear_repulsion_gradient(geom, charges=self.charges)
        F_deriv_pk = np.zeros((n_u, self.n_atoms, 3), dtype=np.float64)

        for atom_k in range(self.n_atoms):
            for cart in range(3):
                dF_exp = {
                    (a, s): dF_fixedC_dcoord[a][atom_k, cart, s].copy() for (a, s) in pk_data
                }
                dP1_exp = {(a, s): np.zeros((atoms[a]["m"], atoms[a]["m"])) for (a, s) in pk_data}
                dP2_exp = {(a, s): np.zeros((atoms[a]["m"], atoms[a]["m"])) for (a, s) in pk_data}
                for a in atoms:
                    for b in atoms:
                        if a != b and (a, b) in pair_data:
                            w_ab = _pk_pair_weight(atoms[a], atoms[b])
                            if w_ab <= 0.0:
                                continue
                            pd_ab = pair_data[(a, b)]
                            S_ab, dS_dra = pd_ab["S_ab"], pd_ab["dS_dra"][cart]
                            sign = (1.0 if atom_k == a - 1 else 0.0) - (
                                1.0 if atom_k == b - 1 else 0.0
                            )
                            if sign != 0.0:
                                dS = sign * dS_dra
                                ab = atoms[b]
                                for s in range(self.ispin):
                                    D4_B = _unpack_packed(ab["pkpot"][:, 3, s], ab["m"])
                                    D5_B = _unpack_packed(ab["pkpot"][:, 4, s], ab["m"])
                                    dP1_exp[(a, s)] += w_ab * (dS @ D4_B @ S_ab.T + S_ab @ D4_B @ dS.T)
                                    dP2_exp[(a, s)] += w_ab * (dS @ D5_B @ S_ab.T + S_ab @ D5_B @ dS.T)
                dU_exp, de_exp = _scf_step_response(dF_exp, dP1_exp, dP2_exp)
                de_fixedU = inv_I_Mee @ de_exp
                if n_u > 0:
                    F_deriv_pk[:, atom_k, cart] = -de_vec * (dU_exp + M_Ue @ de_fixedU)
                for a, aa in atoms.items():
                    g_fixedC_pk[atom_k, cart] += 0.5 * float(
                        np.sum(aa["P_tot"] * dzpot2_dcoord[a][atom_k, cart])
                    )
                g_fixedC_pk[atom_k, cart] += 0.5 * float(w_occ @ de_fixedU)

        return {
            "F_deriv": F_deriv_pk,
            "L": L_pk,
            "A": A_pk,
            "g_fixedC": g_fixedC_pk,
        }

    def compute_fock_derivatives(self) -> np.ndarray:
        """
        Computes the fixed-orbital Cartesian Fock matrix derivatives for all
        active canonical orbital pairs (n_active, n_atoms, 3):
            F^{(A, k)}_{pq} = C_p^T * (dF / dR_{A, k})_C * C_q
        including both explicit two-center integral derivatives and the implicit
        fixed-orbital model-potential screening response (dvcoef_B / dR)_C.
        """
        F_deriv = np.zeros((self.n_active, self.n_atoms, 3), dtype=np.float64)
        if self.n_active == 0:
            return F_deriv

        cache = self._prepare_response_data()
        if cache.get("pk_response") is not None:
            return cache["pk_response"]["F_deriv"].copy()
        atoms = cache["atoms"]
        dF_fixedC_dcoord = cache["dF_fixedC_dcoord"]

        for i_r, pair in enumerate(self.active_pairs):
            a = pair["atom"]
            s_i = pair["spin"]
            p_i = pair["p_local"]
            q_i = pair["q_local"]
            aa = atoms[a]
            c_p = aa["C"][:, p_i, s_i]
            c_q = aa["C"][:, q_i, s_i]
            for atom_k in range(self.n_atoms):
                for cart in range(3):
                    F_deriv[i_r, atom_k, cart] = float(
                        c_p.T @ dF_fixedC_dcoord[a][atom_k, cart, s_i] @ c_q
                    )

        return F_deriv

    def build_orbital_lagrangian(self) -> np.ndarray:
        """
        Assembles the exact analytical orbital Lagrangian vector L_{pq} = dE_total / dU_{pq},
        incorporating intra-atomic one-electron/Fock variations and inter-atomic
        model-potential screening (dvcoef / dU) and exchange-residual (dV6 / dU) coupling.
        """
        L = np.zeros(self.n_active, dtype=np.float64)
        if self.n_active == 0:
            return L

        cache = self._prepare_response_data()
        if cache.get("pk_response") is not None:
            return cache["pk_response"]["L"].copy()
        atoms = cache["atoms"]
        pair_data = cache["pair_data"]
        rot_effects = cache["rot_effects"]

        for j_r, eff in enumerate(rot_effects):
            a_j = eff["a"]
            s_r = eff["s_rot"]
            p_r, q_r = eff["p"], eff["q"]
            aa_j = atoms[a_j]
            c_p = aa_j["C"][:, p_r, s_r]
            c_q = aa_j["C"][:, q_r, s_r]

            V2_a = (
                aa_j["fc"][s_r]
                + aa_j["fx"][s_r]
                + aa_j["cpot2"]
                + _unpack_packed(aa_j["pkpot"][:, 2, s_r], aa_j["m"])
            )
            h1_pq = -float(c_p.T @ V2_a @ c_q)
            d_one_el = self.orbocc * (aa_j["occ"][p_r, s_r] - aa_j["occ"][q_r, s_r]) * 2.0 * h1_pq

            d_teval = 0.0
            for a, aa in atoms.items():
                for s in range(self.ispin):
                    if a == a_j:
                        dF_s = eff["dfc"][s] + eff["dfx"][s]
                    elif (a, a_j) in pair_data:
                        pd = pair_data[(a, a_j)]
                        dcpot2_ab = sum(
                            eff["dvcoef"][nv] * pd["V_ab"][nv] for nv in range(aa_j["nt"])
                        )
                        dxpot2_ab = pd["S_ab"] @ eff["dV6"][s] @ pd["S_ab"].T
                        dF_s = dcpot2_ab + dxpot2_ab
                    else:
                        continue

                    for iorb in range(aa["norb"]):
                        occup = self.orbocc * aa["occ"][iorb, s]
                        if aa["m"] == 1:
                            deval_i = float(dF_s[0, 0])
                        else:
                            c_i = aa["C"][:, iorb, s]
                            deval_i = float(c_i.T @ dF_s @ c_i)
                        d_teval += occup * deval_i

            L[j_r] = 0.5 * (d_one_el + d_teval)

        return L

    def build_orbital_hessian(self) -> np.ndarray:
        """
        Constructs the full coupled CPHF electronic superoperator Hessian:
            A = A_0 + K^intra + K^inter
        Includes the canonical eigenvalue difference (eps_q - eps_p), intra-atomic
        two-electron Coulomb/exchange coupling K^intra, and inter-atomic two-center
        model-potential screening and exchange-residual coupling K^inter.
        """
        A = np.zeros((self.n_active, self.n_active), dtype=np.float64)
        if self.n_active == 0:
            return A

        cache = self._prepare_response_data()
        if cache.get("pk_response") is not None:
            return cache["pk_response"]["A"].copy()
        atoms = cache["atoms"]
        pair_data = cache["pair_data"]
        rot_effects = cache["rot_effects"]

        for i_r, pair in enumerate(self.active_pairs):
            a = pair["atom"]
            s_i = pair["spin"]
            p_i = pair["p_local"]
            q_i = pair["q_local"]
            aa = atoms[a]
            c_p = aa["C"][:, p_i, s_i]
            c_q = aa["C"][:, q_i, s_i]
            A[i_r, i_r] += aa["eval"][q_i, s_i] - aa["eval"][p_i, s_i]

            for j_r, eff in enumerate(rot_effects):
                b = eff["a"]
                if b == a:
                    dF_ij = eff["dfc"][s_i] + eff["dfx"][s_i]
                elif (a, b) in pair_data:
                    pd = pair_data[(a, b)]
                    dcpot2_ab = sum(
                        eff["dvcoef"][nv] * pd["V_ab"][nv] for nv in range(atoms[b]["nt"])
                    )
                    dxpot2_ab = pd["S_ab"] @ eff["dV6"][s_i] @ pd["S_ab"].T
                    dF_ij = dcpot2_ab + dxpot2_ab
                else:
                    continue
                A[i_r, j_r] += float(c_p.T @ dF_ij @ c_q)

        return A

    def solve_z_vector(self) -> np.ndarray:
        """
        Solves the complete coupled CPHF Handy-Schaefer Z-vector equation:
            A^T * Z = -L
        using the analytical orbital Lagrangian L and the full coupled
        superoperator Hessian A.
        """
        if self.n_active == 0:
            return np.zeros(0, dtype=np.float64)

        L = self.build_orbital_lagrangian()
        A = self.build_orbital_hessian()
        try:
            Z, _, _, _ = np.linalg.lstsq(A.T, -L, rcond=1e-10)
        except Exception:
            try:
                Z = scipy.linalg.solve(A.T, -L)
            except Exception:
                Z = -L / np.diag(A.T)

        return Z

    def _compute_fixed_orbital_gradient(self) -> np.ndarray:
        """
        Evaluates the exact fixed-orbital Cartesian gradient (dE_total / dR)_C,
        incorporating nuclear repulsion, two-center nuclear attraction, explicit
        two-center model-potential and exchange-residual derivatives, and the
        implicit fixed-orbital screening response (dvcoef / dR)_C.
        """
        cache = self._prepare_response_data()
        if cache.get("pk_response") is not None:
            return cache["pk_response"]["g_fixedC"].copy()
        atoms = cache["atoms"]
        dzpot2_dcoord = cache["dzpot2_dcoord"]
        dF_fixedC_dcoord = cache["dF_fixedC_dcoord"]

        geom = self.deck.geometry if self.deck is not None else [None] * self.n_atoms
        g_fixedC = compute_nuclear_repulsion_gradient(geom, charges=self.charges)

        for atom_k in range(self.n_atoms):
            for cart in range(3):
                for a, aa in atoms.items():
                    d_one_el = float(np.sum(aa["P_tot"] * dzpot2_dcoord[a][atom_k, cart]))
                    d_teval = 0.0
                    for s in range(self.ispin):
                        dF_s = dF_fixedC_dcoord[a][atom_k, cart, s]
                        for iorb in range(aa["norb"]):
                            occup = self.orbocc * aa["occ"][iorb, s]
                            if aa["m"] == 1:
                                deval_i = float(dF_s[0, 0])
                            else:
                                c_i = aa["C"][:, iorb, s]
                                deval_i = float(c_i.T @ dF_s @ c_i)
                            d_teval += occup * deval_i
                    g_fixedC[atom_k, cart] += 0.5 * (d_one_el + d_teval)

        return g_fixedC

    def compute_relaxed_gradient(self) -> np.ndarray:
        """
        Evaluates the fully relaxed analytical Cartesian nuclear gradient matrix (n_atoms, 3):
            g_{A, k}^{relaxed} = (dE_total / dR_{A, k})_C + sum_{p} Z_p * F^{(A, k)}_p
        """
        # 1. Fixed-orbital multi-center gradient (dE_total / dR)_C
        g_fixedC = self._compute_fixed_orbital_gradient()
        if self.n_active == 0:
            trans = np.mean(g_fixedC, axis=0)
            return g_fixedC - trans

        # 2. Solve coupled Handy-Schaefer Z-vector equation A^T Z = -L
        Z = self.solve_z_vector()

        # 3. Contract with fixed-orbital Fock matrix derivatives
        F_deriv = self.compute_fock_derivatives()  # (n_active, n_atoms, 3)

        delta_g = np.zeros((self.n_atoms, 3), dtype=np.float64)
        for p_idx in range(self.n_active):
            z_val = Z[p_idx]
            if abs(z_val) > 1e-14:
                delta_g += z_val * F_deriv[p_idx]

        g_relaxed = g_fixedC + delta_g

        # Symmetrize total translation to machine precision
        trans = np.mean(g_relaxed, axis=0)
        g_relaxed -= trans

        return g_relaxed


def compute_polyatomic_relaxed_gradient(
    calc_state: Any, deck: Optional[Any] = None
) -> np.ndarray:
    """
    Computes the relaxed analytical Cartesian nuclear gradient (N_atoms, 3)
    incorporating the full coupled CPHF intra-atomic and inter-atomic orbital
    and model-potential relaxation response Z-vector.
    """
    solver = CPHFResponseSolver(calc_state, deck)
    return solver.compute_relaxed_gradient()


def diagnose_orbital_response_fd_projection(
    calc_state: Any, deck: Any, driver: Any, step_size: float = 0.002
) -> np.ndarray:
    """
    DIAGNOSTIC USE ONLY: computes the best-fit orbital response Z-vector by
    least-squares projection of the finite-difference force discrepancy onto the
    skeleton Fock derivative tensor F_deriv.

    This function is strictly for post-hoc analysis and diagnostic evaluation of
    basis completeness. It is NOT called by driver.gradient(),
    compute_polyatomic_relaxed_gradient(), or any validation tests.
    """
    from phatpsy.gradients import compute_scf_finite_difference_gradient
    solver = CPHFResponseSolver(calc_state, deck)
    if solver.n_active == 0:
        return np.zeros(0, dtype=np.float64)

    F_deriv = solver.compute_fock_derivatives()
    g_unrel = compute_polyatomic_analytical_gradient(calc_state, deck, relaxed=False)
    fd_grad = compute_scf_finite_difference_gradient(driver, deck, step_size=step_size)
    diff = fd_grad - g_unrel

    A_mat = F_deriv.reshape(solver.n_active, 3 * solver.n_atoms).T
    b_vec = diff.reshape(3 * solver.n_atoms)
    Z_fit, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=1e-3)
    return Z_fit
