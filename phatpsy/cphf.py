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

    def _get_atom_full_orbitals(self, a: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the full (m, m, ispin) canonical orbital coefficients C_full,
        (m, ispin) occupations occ_full, and (m, ispin) eigenvalues eval_full
        for atom a. When m > norb_a (e.g. DZ / DZP / TZP basis sets), completes
        the basis with the m - norb_a virtual eigenvectors of the atomic Fock
        matrix in the S-orthogonal complement of the norb_a stored orbitals.
        """
        idx_a = self.atom_basis_indices[a]
        m = len(idx_a)
        m2 = (m * (m + 1)) // 2
        k_start, k_end = self.atom_orb_slices[a]
        norb_a = k_end - k_start

        C_stored = self.evecx[idx_a, k_start:k_end, :].copy()
        occ_stored = self.occx[k_start:k_end, :].copy()
        eval_stored = self.evalx[k_start:k_end, :].copy()

        # Active core/valence orbitals are those populated in at least one spin
        # channel; trailing unoccupied polarization orbitals (e.g. 3d/4f in DZP/TZP/TZ2P)
        # have zero occupation and zero model-potential fit weight, so they belong to
        # the canonical virtual complement orthogonal to the active subspace.
        active_indices = [i for i in range(norb_a) if np.sum(occ_stored[i, :]) > 1e-6]
        if active_indices and active_indices == list(range(len(active_indices))):
            n_act = len(active_indices)
        else:
            n_act = norb_a

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
            pk2_s = _unpack_packed(pkpot[:, 1, s], m) if znuc > 1.5 else 0.0
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
            k_start, k_end = self.atom_orb_slices[a]
            norb_a = k_end - k_start
            _, occ_full, eval_full = self._get_atom_full_orbitals(a)
            active_indices = [i for i in range(norb_a) if np.sum(occ_full[i, :]) > 1e-6]
            n_act = len(active_indices) if (active_indices and active_indices == list(range(len(active_indices)))) else norb_a

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

            atoms[a] = {
                "m": m,
                "m2": m2,
                "norb": norb_a,
                "k_start": k_start,
                "idx": idx_a,
                "znuc": znuc,
                "nt": nt,
                "nc": nc,
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
            if nc > 0 and aa["sumocc"] > 1e-12:
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
                        dvpot_b = sum(
                            dvc_b[nv] * ab["stpot_mats"][nv] for nv in range(ab["nt"])
                        )
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
            if nc > 0 and aa["sumocc"] > 1e-12:
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
            dvpot = sum(dvcoef[nv] * aa["stpot_mats"][nv] for nv in range(nt))
            for s in range(self.ispin):
                donfock = dfx[s] + dfc[s] - dvpot
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

        self._response_cache = {
            "atoms": atoms,
            "pair_data": pair_data,
            "dzpot2_dcoord": dzpot2_dcoord,
            "dF_fixedC_dcoord": dF_fixedC_dcoord,
            "dvcoef_explicit_dcoord": dvcoef_explicit_dcoord,
            "rot_effects": rot_effects,
        }
        return self._response_cache

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
