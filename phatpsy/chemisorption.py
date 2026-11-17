# phatpsy/chemisorption.py
"""
Transition-Metal Chemisorption, d-Band Coupling, and Multi-Site Surface Energetics (Phase 9 / Phase 18).

Modern implementation of Chapter IX of Jack A. Smith's 1978 PhD dissertation:
- C4v crystal field d-band manifold projection on transition-metal surface centers:
    a_1 (d_z^2), e (d_xz, d_yz), b_1 (d_x^2-y^2), b_2 (d_xy)
- Blyholder synergistic sigma-donation and pi-backdonation coupling indices:
    5sigma lone-pair donation into metal a_1 (d_z^2)
    metal e (d_xz, d_yz) back-donation into adsorbate 2pi* LUMO
- Multi-site chemisorption energy evaluation (E_ads) with boundary atom fixation invariant:
    E_ads = E_total(slab + ads) - E_total(clean slab) - E_total(isolated ads)
- 2D lateral potential energy surface (PES) scan across high-symmetry sites:
    Top, Bridge, and 4-Fold Hollow sites evaluated directly on the active PES
- Chemisorption vibrational shift evaluation (Delta_nu = nu_ads - nu_gas):
    Harmonic vibrational frequencies computed from potential-energy curvature with fixed substrate atoms
- Surface Core-Level Shifts (SCLS) and adsorbate 1s spectroscopy:
    Koopmans-style orbital-energy shifts (expected ~15 eV relaxation offset vs XPS),
    Mayer bond-order indices, and Bagus/Siegbahn Delta-CASSCF core-hole binding energies.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Sequence
import copy
import json
import math
import numpy as np

from phatpsy.schema import CalculationDeck, AtomConfig, ControlConfig
from phatpsy.driver import PhatpsyDriver
from phatpsy.state import CalculationState, EmbeddedAtomState
from phatpsy.subsystem import decompose_subsystem_energies, SubsystemDecomposition
from phatpsy.vibrations import compute_vibrations, VibrationResult, ATOMIC_MASSES_AMU

AMU_TO_ME: float = 1822.888486209
HARTREE_TO_CM: float = 219474.63136320
AU_FORCE_CONST_TO_MDYNE_A: float = 15.56893102


# -----------------------------------------------------------------------------
# Fundamental Conversion Constants
# -----------------------------------------------------------------------------
HARTREE_TO_KCAL_MOL: float = 627.509474
HARTREE_TO_EV: float = 27.211386
EV_TO_KCAL_MOL: float = 23.060548


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class C4vDBandProjection:
    """
    C4v crystal field d-band manifold projection on a surface transition metal center.

    Manifold classification:
      - a_1 : d_z^2 (m_l = 0), directed along surface normal (acceptor for 5sigma donation)
      - e   : d_xz, d_yz (m_l = +/- 1), pointing out of plane (donor for pi-backdonation)
      - b_1 : d_x^2-y^2 (m_l = 2), in-plane along nearest-neighbor metal bonds
      - b_2 : d_xy (m_l = -2), in-plane diagonal between nearest-neighbor metal bonds
    """
    atom_id: str
    element: str
    a1_population: float
    e_population: float
    b1_population: float
    b2_population: float
    total_d_population: float
    a1_spin_diff: float
    e_spin_diff: float
    b1_spin_diff: float
    b2_spin_diff: float
    crystal_field_splitting_ev: float
    sigma_donation_index: float
    pi_backdonation_index: float
    blyholder_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert projection to a serializable dictionary."""
        return {
            "atom_id": self.atom_id,
            "element": self.element,
            "populations": {
                "a1_dz2": round(float(self.a1_population), 4),
                "e_dxz_dyz": round(float(self.e_population), 4),
                "b1_dx2y2": round(float(self.b1_population), 4),
                "b2_dxy": round(float(self.b2_population), 4),
                "total_d": round(float(self.total_d_population), 4),
            },
            "spin_polarization": {
                "a1": round(float(self.a1_spin_diff), 4),
                "e": round(float(self.e_spin_diff), 4),
                "b1": round(float(self.b1_spin_diff), 4),
                "b2": round(float(self.b2_spin_diff), 4),
            },
            "crystal_field_splitting_ev": round(float(self.crystal_field_splitting_ev), 3),
            "blyholder_coupling": {
                "sigma_donation_index": round(float(self.sigma_donation_index), 4),
                "pi_backdonation_index": round(float(self.pi_backdonation_index), 4),
                "blyholder_ratio": round(float(self.blyholder_ratio), 4),
            },
        }


@dataclass
class ChemisorptionEnergy:
    """
    Chemisorption adsorption energy and subsystem partitioning.
    E_ads = E_total(slab + ads) - E_total(clean slab) - E_total(isolated ads)
    """
    system_name: str
    adsorption_site: str
    energy_slab_ads: float
    energy_clean_slab: float
    energy_gas_ads: float
    energy_ads_hartree: float
    energy_ads_kcal_mol: float
    energy_ads_ev: float
    subsystem_excess_energy: float
    subsystem_embedding_energy: float
    adsorbate_net_charge: float
    adsorbate_net_spin: float
    converged: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert chemisorption energy to serializable dictionary."""
        return {
            "system_name": self.system_name,
            "adsorption_site": self.adsorption_site,
            "energies_hartree": {
                "slab_with_adsorbate": round(float(self.energy_slab_ads), 6),
                "clean_slab": round(float(self.energy_clean_slab), 6),
                "isolated_adsorbate": round(float(self.energy_gas_ads), 6),
                "adsorption_energy": round(float(self.energy_ads_hartree), 6),
            },
            "adsorption_energy_kcal_mol": round(float(self.energy_ads_kcal_mol), 2),
            "adsorption_energy_ev": round(float(self.energy_ads_ev), 3),
            "subsystem": {
                "excess_energy_hartree": round(float(self.subsystem_excess_energy), 6),
                "embedding_energy_hartree": round(float(self.subsystem_embedding_energy), 6),
            },
            "adsorbate_electronic_state": {
                "net_charge": round(float(self.adsorbate_net_charge), 4),
                "net_spin": round(float(self.adsorbate_net_spin), 4),
            },
            "converged": bool(self.converged),
        }


@dataclass
class LateralSiteScanResult:
    """
    2D potential energy surface scan across high-symmetry surface adsorption sites.
    """
    system_name: str
    lattice_constant_bohr: float
    sites: Dict[str, Dict[str, Any]]
    preferred_site: str
    is_top_preferred: bool
    barrier_top_to_bridge_kcal: float
    barrier_top_to_hollow_kcal: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to dictionary."""
        return {
            "system_name": self.system_name,
            "lattice_constant_bohr": round(float(self.lattice_constant_bohr), 4),
            "sites": self.sites,
            "preferred_site": self.preferred_site,
            "is_top_preferred": bool(self.is_top_preferred),
            "site_energy_differences_kcal_mol": {
                "top_to_bridge": round(float(self.barrier_top_to_bridge_kcal), 2),
                "top_to_hollow": round(float(self.barrier_top_to_hollow_kcal), 2),
            },
        }


@dataclass
class VibrationalShiftResult:
    """
    Chemisorption vibrational frequency shift for diatomic adsorbates.
    Delta_nu = nu_ads - nu_gas
    """
    system_name: str
    adsorbate: str
    nu_gas_cm: float
    nu_ads_cm: float
    delta_nu_cm: float
    is_red_shifted: bool
    force_constant_gas_mdyne_a: float
    force_constant_ads_mdyne_a: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert vibrational shift result to dictionary."""
        return {
            "system_name": self.system_name,
            "adsorbate": self.adsorbate,
            "frequencies_cm_inv": {
                "gas_phase": round(float(self.nu_gas_cm), 1),
                "chemisorbed": round(float(self.nu_ads_cm), 1),
                "shift_delta_nu": round(float(self.delta_nu_cm), 1),
            },
            "is_red_shifted": bool(self.is_red_shifted),
            "force_constants_mdyne_a": {
                "gas_phase": round(float(self.force_constant_gas_mdyne_a), 3),
                "chemisorbed": round(float(self.force_constant_ads_mdyne_a), 3),
            },
        }


@dataclass
class SurfaceCoreLevelShifts:
    """
    Surface core-level shifts (SCLS) and adsorbate chemical shifts
    computed from Koopmans-style core orbital eigenvalues (-eps_1s).
    Note: absolute Koopmans binding energies exceed experimental XPS values
    by ~15 eV due to core-hole orbital relaxation.
    """
    system_name: str
    metal_scls_ev: float
    surface_core_level_ev: float
    bulk_core_level_ev: float
    adsorbate_n1s_be_ev: Optional[float] = None
    adsorbate_o1s_be_ev: Optional[float] = None
    adsorbate_c1s_be_ev: Optional[float] = None
    method_label: str = "koopmans_orbital_energy"
    expected_relaxation_offset_ev: float = 15.0
    koopmans_ads_1s_be_ev: Dict[str, float] = field(default_factory=dict)
    koopmans_gas_1s_be_ev: Dict[str, float] = field(default_factory=dict)
    koopmans_ads_1s_shift_ev: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SCLS result to dictionary."""
        d: Dict[str, Any] = {
            "system_name": self.system_name,
            "method_label": self.method_label,
            "expected_relaxation_offset_ev": float(self.expected_relaxation_offset_ev),
            "metal_scls_ev": round(float(self.metal_scls_ev), 3),
            "surface_core_level_ev": round(float(self.surface_core_level_ev), 2),
            "bulk_core_level_ev": round(float(self.bulk_core_level_ev), 2),
        }
        if self.adsorbate_n1s_be_ev is not None:
            d["adsorbate_n1s_be_ev"] = round(float(self.adsorbate_n1s_be_ev), 2)
        if self.adsorbate_o1s_be_ev is not None:
            d["adsorbate_o1s_be_ev"] = round(float(self.adsorbate_o1s_be_ev), 2)
        if self.adsorbate_c1s_be_ev is not None:
            d["adsorbate_c1s_be_ev"] = round(float(self.adsorbate_c1s_be_ev), 2)
        if self.koopmans_ads_1s_be_ev:
            d["koopmans_ads_1s_be_ev"] = {k: round(float(v), 3) for k, v in self.koopmans_ads_1s_be_ev.items()}
        if self.koopmans_gas_1s_be_ev:
            d["koopmans_gas_1s_be_ev"] = {k: round(float(v), 3) for k, v in self.koopmans_gas_1s_be_ev.items()}
        if self.koopmans_ads_1s_shift_ev:
            d["koopmans_ads_1s_shift_ev"] = {k: round(float(v), 3) for k, v in self.koopmans_ads_1s_shift_ev.items()}
        return d


# -----------------------------------------------------------------------------
# Core Physics Routines
# -----------------------------------------------------------------------------

def project_c4v_dband(
    state_or_deck: Union[CalculationState, CalculationDeck, str, Path],
    atom_id: str = "Ni1",
    driver: Optional[PhatpsyDriver] = None,
) -> C4vDBandProjection:
    """
    Projects the C4v crystal field d-band manifold on a surface transition-metal atom.

    Extracts populations and spin polarizations across the 5 d-orbitals:
      - a_1 : d_z^2 (m_l = 0)
      - e   : d_xz, d_yz (m_l = +/- 1)
      - b_1 : d_x^2-y^2 (m_l = 2)
      - b_2 : d_xy (m_l = -2)

    Calculates Blyholder coupling indices for synergistic 5sigma donation
    and 2pi* backdonation.
    """
    if isinstance(state_or_deck, (str, Path)):
        p = Path(state_or_deck)
        state_or_deck = CalculationDeck.from_yaml(p) if p.suffix in (".yaml", ".yml") else CalculationDeck.from_json(p)

    if isinstance(state_or_deck, CalculationDeck):
        d = driver or PhatpsyDriver(quiet=True)
        res = d.run_deck(state_or_deck)
    else:
        res = state_or_deck

    bi = res.get("basis_info", {})
    pops = res.get("atom_populations")
    if pops is None or not bi:
        raise ValueError("Calculation state does not contain basis_info or atom_populations.")

    # Determine 1-based atom index for the target atom
    target_idx = 1
    target_elem = "Ni"
    if res.deck and res.deck.geometry:
        for idx, a in enumerate(res.deck.geometry):
            if a.id == atom_id:
                target_idx = idx + 1
                target_elem = a.element
                break
    else:
        # Fallback to searching embedded atoms
        embedded = res.get("embedded_atoms", {})
        for idx, (k, v) in enumerate(embedded.items()):
            if k == atom_id or getattr(v, "atom_id", "") == atom_id:
                target_idx = idx + 1
                target_elem = getattr(v, "element", "Ni")
                break

    iatomx = np.asarray(bi.get("iatomx", []))
    lx = np.asarray(bi.get("lx", []))
    mlx = np.asarray(bi.get("mlx", []))
    atom_indices = np.where(iatomx == target_idx)[0]

    pop_a1 = 0.0
    pop_e = 0.0
    pop_b1 = 0.0
    pop_b2 = 0.0

    spin_a1 = 0.0
    spin_e = 0.0
    spin_b1 = 0.0
    spin_b2 = 0.0

    d_indices = [idx for idx in atom_indices if lx[idx] == 2]
    emb_atom = (res.get("embedded_atoms") or {}).get(atom_id)
    emb_orbs = getattr(emb_atom, "orbitals", None) if emb_atom is not None else None

    for k, idx in enumerate(d_indices):
        ml = mlx[idx]
        if emb_orbs is not None and len(emb_orbs) >= 15 and len(d_indices) == 10:
            orb = emb_orbs[5 + k]
            pa = float(orb["population_alpha"])
            pb = float(orb["population_beta"])
        else:
            if idx >= pops.shape[0]:
                continue
            pa = float(pops[idx, 0])
            pb = float(pops[idx, 1]) if pops.shape[1] > 1 else pa
        if np.isnan(pa) or np.isnan(pb):
            continue
        ptot = pa + pb
        pdiff = pa - pb

        if ml == 0:
            pop_a1 += ptot
            spin_a1 += pdiff
        elif abs(ml) == 1:
            pop_e += ptot
            spin_e += pdiff
        elif ml == 2:
            pop_b1 += ptot
            spin_b1 += pdiff
        elif ml == -2:
            pop_b2 += ptot
            spin_b2 += pdiff

    tot_d = pop_a1 + pop_e + pop_b1 + pop_b2

    if np.isnan(tot_d) or tot_d < 1e-6:
        rdm1_ao = res.get("rdm1_ao")
        ao_ovlp = res.get("ao_overlap")
        if rdm1_ao is not None and len(d_indices) > 0:
            p_arr = np.asarray(rdm1_ao, dtype=float)
            if p_arr.ndim == 2:
                pa_mat = 0.5 * p_arr
                pb_mat = 0.5 * p_arr
            else:
                pa_mat = p_arr[0]
                pb_mat = p_arr[1] if p_arr.shape[0] > 1 else p_arr[0]
            if ao_ovlp is not None:
                s_mat = np.asarray(ao_ovlp, dtype=float)
                evals, evecs = np.linalg.eigh(0.5 * (s_mat + s_mat.T))
                s_half = (evecs * np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
                pa_orth = s_half @ pa_mat @ s_half
                pb_orth = s_half @ pb_mat @ s_half
            else:
                pa_orth = pa_mat
                pb_orth = pb_mat
            pop_a1 = pop_e = pop_b1 = pop_b2 = 0.0
            spin_a1 = spin_e = spin_b1 = spin_b2 = 0.0
            for idx in d_indices:
                if idx >= pa_orth.shape[0]:
                    continue
                pa = float(pa_orth[idx, idx])
                pb = float(pb_orth[idx, idx])
                ptot = pa + pb
                pdiff = pa - pb
                ml = int(mlx[idx])
                if ml == 0:
                    pop_a1 += ptot
                    spin_a1 += pdiff
                elif abs(ml) == 1:
                    pop_e += ptot
                    spin_e += pdiff
                elif ml == 2:
                    pop_b1 += ptot
                    spin_b1 += pdiff
                elif ml == -2:
                    pop_b2 += ptot
                    spin_b2 += pdiff
            tot_d = pop_a1 + pop_e + pop_b1 + pop_b2

    if np.isnan(tot_d) or tot_d < 1e-6:
        raise ValueError(f"No valid d-orbital population data found for atom '{atom_id}'.")

    # Crystal field splitting and Blyholder indices from computed orbital populations
    eg_pop_avg = (pop_a1 + pop_b1) / 2.0
    t2g_pop_avg = (pop_e / 2.0 + pop_b2) / 2.0
    cf_splitting_ev = abs(eg_pop_avg - t2g_pop_avg) * HARTREE_TO_EV / 10.0

    # Sigma donation index from departure of a1 (dz2) from nominal d^2 occupation,
    # and pi backdonation index from holes / depletion in the e (dxz, dyz) manifold.
    n_radial_d = max(1, len(d_indices) // 5)
    nom_a1 = 2.0 * n_radial_d
    nom_e = 4.0 * n_radial_d
    sigma_index = abs(nom_a1 - pop_a1)
    pi_backdonation_index = max(0.0, nom_e - pop_e)
    blyholder_ratio = pi_backdonation_index / sigma_index if sigma_index > 1e-8 else 0.0

    return C4vDBandProjection(
        atom_id=atom_id,
        element=target_elem,
        a1_population=pop_a1,
        e_population=pop_e,
        b1_population=pop_b1,
        b2_population=pop_b2,
        total_d_population=tot_d,
        a1_spin_diff=spin_a1,
        e_spin_diff=spin_e,
        b1_spin_diff=spin_b1,
        b2_spin_diff=spin_b2,
        crystal_field_splitting_ev=cf_splitting_ev,
        sigma_donation_index=sigma_index,
        pi_backdonation_index=pi_backdonation_index,
        blyholder_ratio=blyholder_ratio,
    )


_ADSORBATE_ELEMENTS = {"C", "N", "O", "H", "S", "Cl"}
_DEFAULT_ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Cl": 17, "Ni": 28}


def _resolve_state_or_deck(
    item: Union[CalculationState, CalculationDeck, str, Path],
    driver: PhatpsyDriver,
) -> Tuple[Optional[CalculationDeck], Optional[CalculationState]]:
    if isinstance(item, (str, Path)):
        p = Path(item)
        deck = CalculationDeck.from_yaml(p) if p.suffix in (".yaml", ".yml") else CalculationDeck.from_json(p)
        return deck, None
    if isinstance(item, CalculationDeck):
        return item, None
    return item.deck, item


def _build_clean_slab_deck(
    deck_slab_ads: CalculationDeck,
    substrate_atoms: Sequence[AtomConfig],
) -> CalculationDeck:
    deck_clean = copy.deepcopy(deck_slab_ads)
    deck_clean.title = f"Clean substrate for {deck_slab_ads.title}"
    deck_clean.geometry = copy.deepcopy(list(substrate_atoms))
    sub_elem_symbols = {a.element for a in substrate_atoms}
    deck_clean.elements = {k: v for k, v in deck_clean.elements.items() if k in sub_elem_symbols}
    if getattr(deck_clean.control, "calculation_method", "phatpsy_scf") in ("localized_vb", "masked_vb"):
        deck_clean.control.target_spin = None
    if any(a.element in {"N", "H"} for a in deck_slab_ads.geometry if a not in substrate_atoms):
        # Stripping an odd-electron radical adsorbate leaves an even-electron metal substrate.
        n_e_sub = sum(
            deck_clean.elements[a.element].atomic_number
            if a.element in deck_clean.elements
            else _DEFAULT_ATOMIC_NUMBERS.get(a.element, 0)
            for a in substrate_atoms
        )
        if n_e_sub % 2 == 0 and deck_clean.net_spin == 1:
            deck_clean.net_spin = 0
    return deck_clean


def _build_isolated_adsorbate_deck(
    deck_slab_ads: CalculationDeck,
    adsorbate_atoms: Sequence[AtomConfig],
) -> CalculationDeck:
    calc_method = getattr(deck_slab_ads.control, "calculation_method", "phatpsy_scf")
    ads_syms = "".join(sorted([a.element for a in adsorbate_atoms]))
    repo_root = Path(__file__).resolve().parent.parent
    if calc_method == "phatpsy_scf":
        if ads_syms in ("CO", "OC") and (repo_root / "examples" / "CO.yaml").exists():
            return CalculationDeck.from_yaml(repo_root / "examples" / "CO.yaml")
        if ads_syms in ("NO", "ON") and (repo_root / "examples" / "NO.yaml").exists():
            return CalculationDeck.from_yaml(repo_root / "examples" / "NO.yaml")

    deck_gas = copy.deepcopy(deck_slab_ads)
    deck_gas.title = "Isolated gas-phase adsorbate"
    gas_atoms = copy.deepcopy(list(adsorbate_atoms))
    for at in gas_atoms:
        at.active = True
        at.fixed = False
        at.equivalent_to = None
        at.symmetry_op = None
        at.symmetry_axis = None
        at.embedded_state = None
    deck_gas.geometry = gas_atoms
    ads_elem_symbols = {a.element for a in gas_atoms}
    deck_gas.elements = {k: v for k, v in deck_gas.elements.items() if k in ads_elem_symbols}
    deck_gas.control.environment_embedding = "none"
    deck_gas.control.target_spin = None
    n_e = sum(
        deck_gas.elements[a.element].atomic_number
        if a.element in deck_gas.elements
        else _DEFAULT_ATOMIC_NUMBERS.get(a.element, 0)
        for a in gas_atoms
    ) - int(getattr(deck_gas, "charge", 0))
    deck_gas.net_spin = 1 if (n_e % 2 == 1) else 0
    return deck_gas


def evaluate_chemisorption_energy(
    slab_with_ads: Union[CalculationState, CalculationDeck, str, Path],
    clean_slab: Optional[Union[CalculationState, CalculationDeck, str, Path]] = None,
    isolated_ads: Optional[Union[CalculationState, CalculationDeck, str, Path]] = None,
    driver: Optional[PhatpsyDriver] = None,
    site_name: str = "top",
) -> ChemisorptionEnergy:
    """
    Evaluates multi-site chemisorption adsorption energy (E_ads) with boundary atom fixation invariant.

    E_ads = E_total(slab + ads) - E_total(clean slab) - E_total(isolated ads)
    """
    d = driver or PhatpsyDriver(quiet=True)

    # 1. Resolve slab with adsorbate
    deck_slab_ads, res_slab_ads = _resolve_state_or_deck(slab_with_ads, d)
    if res_slab_ads is None:
        res_slab_ads = d.run_deck(deck_slab_ads)
    if deck_slab_ads is None:
        raise ValueError("slab_with_ads state does not reference a CalculationDeck.")
    e_slab_ads = float(res_slab_ads.total_energy)

    geom = deck_slab_ads.geometry if deck_slab_ads else []
    adsorbate_atoms = [a for a in geom if a.element in _ADSORBATE_ELEMENTS]
    substrate_atoms = [a for a in geom if a.element not in _ADSORBATE_ELEMENTS]

    # 2. Resolve clean slab
    if clean_slab is not None:
        deck_clean, res_clean = _resolve_state_or_deck(clean_slab, d)
        if res_clean is None:
            res_clean = d.run_deck(deck_clean)
    else:
        deck_clean = _build_clean_slab_deck(deck_slab_ads, substrate_atoms)
        res_clean = d.run_deck(deck_clean)
    e_clean = float(res_clean.total_energy)

    # 3. Resolve isolated adsorbate
    if isolated_ads is not None:
        deck_gas, res_gas = _resolve_state_or_deck(isolated_ads, d)
        if res_gas is None:
            res_gas = d.run_deck(deck_gas)
    else:
        deck_gas = _build_isolated_adsorbate_deck(deck_slab_ads, adsorbate_atoms)
        res_gas = d.run_deck(deck_gas)
    e_gas = float(res_gas.total_energy)

    # Evaluate chemisorption binding energy directly from subsystem energies
    e_ads_ha = float(e_slab_ads - e_clean - e_gas)
    e_ads_kcal = e_ads_ha * HARTREE_TO_KCAL_MOL
    e_ads_ev = e_ads_ha * HARTREE_TO_EV

    decomp = res_slab_ads.subsystem_decomposition
    excess_e = float(decomp.energy_excess) if decomp else 0.0
    embed_e = float(decomp.energy_embedding) if decomp else 0.0

    # Adsorbate charge transfer (from EWMO Löwdin populations when available)
    net_q = 0.0
    net_s = 0.0
    for a in adsorbate_atoms:
        atom_st = res_slab_ads.get_embedded_atom(a.id)
        if atom_st:
            if atom_st.orbitals:
                pop_a = sum(float(o.get("population_alpha", 0.0)) for o in atom_st.orbitals)
                pop_b = sum(float(o.get("population_beta", 0.0)) for o in atom_st.orbitals)
                if not (np.isnan(pop_a) or np.isnan(pop_b)) and (pop_a + pop_b) > 0.0:
                    net_q += float(atom_st.nuclear_charge) - (pop_a + pop_b)
                    net_s += pop_a - pop_b
                    continue
            net_q += float(atom_st.net_charge)
            net_s += float(atom_st.net_spin)

    title = getattr(deck_slab_ads, "title", "Chemisorption System")

    return ChemisorptionEnergy(
        system_name=title,
        adsorption_site=site_name,
        energy_slab_ads=e_slab_ads,
        energy_clean_slab=e_clean,
        energy_gas_ads=e_gas,
        energy_ads_hartree=e_ads_ha,
        energy_ads_kcal_mol=e_ads_kcal,
        energy_ads_ev=e_ads_ev,
        subsystem_excess_energy=excess_e,
        subsystem_embedding_energy=embed_e,
        adsorbate_net_charge=net_q,
        adsorbate_net_spin=net_s,
        converged=bool(res_slab_ads.converged),
    )


def scan_lateral_chemisorption_pes(
    slab_deck: CalculationDeck,
    sites: Optional[Sequence[str]] = None,
    lattice_constant: float = 4.71,
    driver: Optional[PhatpsyDriver] = None,
    clean_slab: Optional[Union[CalculationState, CalculationDeck, str, Path]] = None,
    isolated_ads: Optional[Union[CalculationState, CalculationDeck, str, Path]] = None,
) -> LateralSiteScanResult:
    """
    Performs a 2D lateral potential energy surface (PES) scan across high-symmetry
    surface sites on the Ni(100) square lattice:
      - 'top'    : (0, 0) directly above Ni1
      - 'bridge' : (a/2, 0) halfway between Ni1 and Ni2
      - 'hollow' : (a/2, a/2) centered over 4 surface Ni atoms
    """
    d = driver or PhatpsyDriver(quiet=True)
    site_names = list(sites) if sites else ["top", "bridge", "hollow"]

    a = float(lattice_constant)
    site_coords = {
        "top": (0.0, 0.0),
        "bridge": (a / 2.0, 0.0),
        "hollow": (a / 2.0, a / 2.0),
    }

    adsorbate_atoms = [at for at in slab_deck.geometry if at.element in _ADSORBATE_ELEMENTS]
    if not adsorbate_atoms:
        raise ValueError("Could not find any adsorbate atoms in calculation deck.")

    substrate_atoms = [at for at in slab_deck.geometry if at.element not in _ADSORBATE_ELEMENTS]

    if clean_slab is not None:
        deck_clean, res_clean = _resolve_state_or_deck(clean_slab, d)
        if res_clean is None:
            res_clean = d.run_deck(deck_clean)
    else:
        res_clean = d.run_deck(_build_clean_slab_deck(slab_deck, substrate_atoms))
    e_clean = float(res_clean.total_energy)

    if isolated_ads is not None:
        deck_gas, res_gas = _resolve_state_or_deck(isolated_ads, d)
        if res_gas is None:
            res_gas = d.run_deck(deck_gas)
    else:
        res_gas = d.run_deck(_build_isolated_adsorbate_deck(slab_deck, adsorbate_atoms))
    e_gas = float(res_gas.total_energy)

    cx0 = float(np.mean([at.xyz[0] for at in adsorbate_atoms]))
    cy0 = float(np.mean([at.xyz[1] for at in adsorbate_atoms]))

    raw_site_kcal: Dict[str, float] = {}
    for s_name in site_names:
        tx, ty = site_coords.get(s_name, (0.0, 0.0))
        dx = tx - cx0
        dy = ty - cy0
        d_site = copy.deepcopy(slab_deck)
        d_site.title = f"{slab_deck.title} [{s_name.upper()} Site]"
        for at in d_site.geometry:
            if at.element in _ADSORBATE_ELEMENTS:
                at.xyz[0] += dx
                at.xyz[1] += dy
                at.equivalent_to = None
                at.symmetry_op = None
                at.symmetry_axis = None

        res_site = d.run_deck(d_site)
        e_ads_ha = float(res_site.total_energy) - e_clean - e_gas
        raw_site_kcal[s_name] = e_ads_ha * HARTREE_TO_KCAL_MOL

    preferred_site = min(raw_site_kcal, key=lambda k: raw_site_kcal[k])
    ref_kcal = raw_site_kcal["top"] if "top" in raw_site_kcal else raw_site_kcal[preferred_site]

    site_results: Dict[str, Dict[str, Any]] = {}
    for s_name in site_names:
        tx, ty = site_coords.get(s_name, (0.0, 0.0))
        e_ads_site_kcal = raw_site_kcal[s_name]
        e_rel_kcal = e_ads_site_kcal - ref_kcal
        site_results[s_name] = {
            "x_bohr": round(float(tx), 3),
            "y_bohr": round(float(ty), 3),
            "e_ads_kcal_mol": round(float(e_ads_site_kcal), 4),
            "delta_e_kcal_mol": round(float(e_rel_kcal), 4),
            "delta_e_ev": round(float(e_rel_kcal / EV_TO_KCAL_MOL), 4),
        }

    barrier_bridge = (
        float(raw_site_kcal["bridge"] - raw_site_kcal["top"])
        if ("bridge" in raw_site_kcal and "top" in raw_site_kcal)
        else 0.0
    )
    barrier_hollow = (
        float(raw_site_kcal["hollow"] - raw_site_kcal["top"])
        if ("hollow" in raw_site_kcal and "top" in raw_site_kcal)
        else 0.0
    )
    is_top = (preferred_site == "top")

    return LateralSiteScanResult(
        system_name=slab_deck.title,
        lattice_constant_bohr=a,
        sites=site_results,
        preferred_site=preferred_site,
        is_top_preferred=is_top,
        barrier_top_to_bridge_kcal=barrier_bridge,
        barrier_top_to_hollow_kcal=barrier_hollow,
    )


def _diatomic_reduced_mass_amu(el1: str, el2: str) -> float:
    m1 = float(ATOMIC_MASSES_AMU.get(el1, 14.0))
    m2 = float(ATOMIC_MASSES_AMU.get(el2, 16.0))
    return (m1 * m2) / (m1 + m2)


def _internal_diatomic_stretch_cm(
    deck: CalculationDeck,
    idx1: int,
    idx2: int,
    driver: PhatpsyDriver,
    base_energy: Optional[float] = None,
    step_bohr: float = 0.01,
) -> Tuple[float, float]:
    """
    Computes internal harmonic bond-stretch frequency (cm^-1) and force constant
    (mdyne/A) via 3-point central finite difference in internal bond-length coordinates.
    """
    at1 = deck.geometry[idx1]
    at2 = deck.geometry[idx2]
    m1 = float(ATOMIC_MASSES_AMU.get(at1.element, 14.0))
    m2 = float(ATOMIC_MASSES_AMU.get(at2.element, 16.0))
    mu_amu = (m1 * m2) / (m1 + m2)

    r1 = np.asarray(at1.xyz, dtype=float)
    r2 = np.asarray(at2.xyz, dtype=float)
    vec = r2 - r1
    dist = float(np.linalg.norm(vec))
    if dist < 1e-8:
        raise ValueError("Adsorbate atoms are coincident; cannot compute internal stretch.")
    u = vec / dist

    if base_energy is None:
        e0 = float(driver.run_deck(deck).total_energy)
    else:
        e0 = float(base_energy)

    energies = {}
    for sign in (+1.0, -1.0):
        d_shift = copy.deepcopy(deck)
        dr1 = -(m2 / (m1 + m2)) * sign * step_bohr * u
        dr2 = +(m1 / (m1 + m2)) * sign * step_bohr * u
        for idx, dr in ((idx1, dr1), (idx2, dr2)):
            a = d_shift.geometry[idx]
            a.xyz = [float(a.xyz[i] + dr[i]) for i in range(3)]
            a.equivalent_to = None
            a.symmetry_op = None
            a.symmetry_axis = None
        energies[sign] = float(driver.run_deck(d_shift).total_energy)

    k_au = (energies[+1.0] - 2.0 * e0 + energies[-1.0]) / (step_bohr ** 2)
    nu_mag = math.sqrt(abs(k_au) / (mu_amu * AMU_TO_ME)) * HARTREE_TO_CM
    nu_cm = math.copysign(nu_mag, k_au)
    k_mdyne = k_au * AU_FORCE_CONST_TO_MDYNE_A
    return float(nu_cm), float(k_mdyne)


def compute_chemisorption_vibrational_shift(
    slab_with_ads: Union[CalculationState, CalculationDeck, str, Path],
    isolated_ads: Optional[Union[CalculationDeck, str, Path]] = None,
    driver: Optional[PhatpsyDriver] = None,
    step_bohr: float = 0.01,
    use_full_hessian: Optional[bool] = None,
) -> VibrationalShiftResult:
    """
    Computes the vibrational frequency shift (Delta_nu = nu_ads - nu_gas)
    of a diatomic adsorbate upon chemisorption onto the transition-metal surface
    directly from computed potential-energy curvature.
    """
    d = driver or PhatpsyDriver(quiet=True)
    deck_slab, res_slab = _resolve_state_or_deck(slab_with_ads, d)
    if deck_slab is None:
        raise ValueError("slab_with_ads state does not reference a CalculationDeck.")

    ads_indices = [i for i, at in enumerate(deck_slab.geometry) if at.element in _ADSORBATE_ELEMENTS]
    if len(ads_indices) != 2:
        raise ValueError(
            f"Expected 2 adsorbate atoms for diatomic vibrational shift, found {len(ads_indices)}."
        )
    idx1, idx2 = ads_indices
    a1, a2 = deck_slab.geometry[idx1], deck_slab.geometry[idx2]
    ads_syms = "".join(sorted([a1.element, a2.element]))
    name = "CO" if ads_syms in ("CO", "OC") else ("NO" if ads_syms in ("NO", "ON") else f"{a1.element}{a2.element}")

    if isolated_ads is not None:
        deck_gas, res_gas = _resolve_state_or_deck(isolated_ads, d)
    else:
        deck_gas = _build_isolated_adsorbate_deck(deck_slab, [a1, a2])
        res_gas = None

    calc_method = getattr(deck_slab.control, "calculation_method", "phatpsy_scf")
    full_hess = (calc_method in ("localized_vb", "masked_vb")) if use_full_hessian is None else bool(use_full_hessian)
    mu_amu = _diatomic_reduced_mass_amu(a1.element, a2.element)

    if full_hess:
        vib_gas: VibrationResult = compute_vibrations(deck_gas, step_bohr=step_bohr, driver=d, quiet=True)
        real_gas = [m.frequency_cm for m in vib_gas.modes if m.frequency_cm > 0.0]
        if not real_gas:
            raise ValueError("Gas-phase Hessian produced no real positive stretching mode.")
        nu_gas = float(max(real_gas))
        k_gas = ((nu_gas / HARTREE_TO_CM) ** 2) * (mu_amu * AMU_TO_ME) * AU_FORCE_CONST_TO_MDYNE_A

        deck_slab_clamped = copy.deepcopy(deck_slab)
        for i, at in enumerate(deck_slab_clamped.geometry):
            at.fixed = i not in ads_indices
        vib_ads: VibrationResult = compute_vibrations(deck_slab_clamped, step_bohr=step_bohr, driver=d, quiet=True)
        real_ads = [m.frequency_cm for m in vib_ads.modes if m.frequency_cm > 0.0]
        if not real_ads:
            raise ValueError("Adsorbed-state partial Hessian produced no real positive stretching mode.")
        nu_ads = float(max(real_ads))
        k_ads = ((nu_ads / HARTREE_TO_CM) ** 2) * (mu_amu * AMU_TO_ME) * AU_FORCE_CONST_TO_MDYNE_A
    else:
        e0_gas = float(res_gas.total_energy) if res_gas is not None else None
        e0_slab = float(res_slab.total_energy) if res_slab is not None else None
        nu_gas, k_gas = _internal_diatomic_stretch_cm(
            deck_gas, 0, 1, d, base_energy=e0_gas, step_bohr=step_bohr
        )
        nu_ads, k_ads = _internal_diatomic_stretch_cm(
            deck_slab, idx1, idx2, d, base_energy=e0_slab, step_bohr=step_bohr
        )

    delta_nu = float(nu_ads - nu_gas)

    return VibrationalShiftResult(
        system_name=deck_slab.title,
        adsorbate=name,
        nu_gas_cm=nu_gas,
        nu_ads_cm=nu_ads,
        delta_nu_cm=delta_nu,
        is_red_shifted=(delta_nu < 0.0),
        force_constant_gas_mdyne_a=k_gas,
        force_constant_ads_mdyne_a=k_ads,
    )


def _extract_ni_2p_orbital_energy_ha(atom_state: EmbeddedAtomState) -> Optional[float]:
    if not atom_state or not atom_state.orbitals:
        return None
    es = [
        0.5 * (float(o["energy_alpha"]) + float(o["energy_beta"]))
        for o in atom_state.orbitals
        if "energy_alpha" in o and "energy_beta" in o
    ]
    # Ni 2p core levels reside near [-42, -24] Ha (~-33 Ha); select explicitly in case MOs are unsorted
    ni2p = [e for e in es if -42.0 <= e <= -24.0]
    if ni2p:
        return float(np.mean(ni2p))
    return None


def _extract_light_1s_be_ev(atom_state: EmbeddedAtomState) -> Optional[float]:
    if not atom_state or not atom_state.orbitals:
        return None
    es = [
        0.5 * (float(o["energy_alpha"]) + float(o["energy_beta"]))
        for o in atom_state.orbitals
        if "energy_alpha" in o and "energy_beta" in o
    ]
    if not es:
        return None
    return float(-min(es) * HARTREE_TO_EV)


def _extract_1s_map_ev(res: CalculationState) -> Dict[str, float]:
    out: Dict[str, float] = {}
    kmap = res.get("koopmans_core_energies_ha") or {}
    for k, val_ha in kmap.items():
        if k.endswith("_1s"):
            aid = k[:-3]
            out[aid] = float(-float(val_ha) * HARTREE_TO_EV)
    embedded = res.get("embedded_atoms", {}) or {}
    for aid, st in embedded.items():
        if aid in out:
            continue
        if getattr(st, "element", "") in ("C", "N", "O"):
            be = _extract_light_1s_be_ev(st)
            if be is not None:
                out[aid] = be
    return out


def compute_surface_core_level_shifts(
    res_slab: Union[CalculationState, CalculationDeck, str, Path],
    driver: Optional[PhatpsyDriver] = None,
    gas_state: Optional[Union[CalculationState, CalculationDeck, str, Path]] = None,
) -> SurfaceCoreLevelShifts:
    """
    Computes Koopmans-style surface core-level shifts (SCLS) comparing surface vs
    subsurface/bulk metal 2p orbital eigenvalues, and adsorbate N 1s / O 1s / C 1s
    Koopmans binding energies (-eps_1s).

    Note: absolute Koopmans orbital-energy binding energies overestimate XPS experiment
    by ~15 eV due to neglected final-state core-hole orbital relaxation.
    """
    d = driver or PhatpsyDriver(quiet=True)
    deck, res = _resolve_state_or_deck(res_slab, d)
    if res is None:
        res = d.run_deck(deck)

    kmap = res.get("koopmans_core_energies_ha") or {}
    embedded = res.get("embedded_atoms", {}) or {}

    surf_ni = embedded.get("Ni1") or embedded.get("Ni_surf")
    bulk_ni = embedded.get("Ni6") or embedded.get("Ni_sub") or embedded.get("Ni_bulk_fixed")

    surf_ha = None
    for key in ("Ni1_2p", "Ni_surf_2p"):
        if key in kmap:
            surf_ha = float(kmap[key])
            break
    if surf_ha is None:
        surf_ha = _extract_ni_2p_orbital_energy_ha(surf_ni)
    if surf_ha is None:
        for k, v in kmap.items():
            if k.startswith("Ni") and k.endswith("_2p"):
                surf_ha = float(v)
                break
    if surf_ha is None:
        for st in embedded.values():
            if getattr(st, "element", "") == "Ni":
                surf_ha = _extract_ni_2p_orbital_energy_ha(st)
                if surf_ha is not None:
                    break

    if surf_ha is None:
        raise ValueError(
            "Calculation state does not contain extractable Ni 2p core orbital eigenvalues "
            "(neither 'koopmans_core_energies_ha' nor 'embedded_atoms' Ni 2p orbitals)."
        )

    bulk_ha = None
    for key in ("Ni6_2p", "Ni_sub_2p", "Ni_bulk_fixed_2p"):
        if key in kmap:
            bulk_ha = float(kmap[key])
            break
    if bulk_ha is None:
        bulk_ha = _extract_ni_2p_orbital_energy_ha(bulk_ni)
    if bulk_ha is None and res.deck and res.deck.geometry:
        ni_atoms = [a for a in res.deck.geometry if a.element == "Ni"]
        if len(ni_atoms) >= 2:
            deepest = min(ni_atoms, key=lambda a: float(a.xyz[2]))
            bulk_ha = kmap.get(f"{deepest.id}_2p")
            if bulk_ha is None:
                bulk_ha = _extract_ni_2p_orbital_energy_ha(embedded.get(deepest.id))
            if bulk_ha is not None:
                bulk_ha = float(bulk_ha)

    if bulk_ha is None:
        bulk_ha = surf_ha

    surf_core_ev = float(surf_ha * HARTREE_TO_EV)
    bulk_core_ev = float(bulk_ha * HARTREE_TO_EV)
    scls_ev = float(surf_core_ev - bulk_core_ev)

    ads_1s_map = _extract_1s_map_ev(res)
    n1s_be = None
    o1s_be = None
    c1s_be = None

    geom = res.deck.geometry if res.deck else []
    elem_by_id = {a.id: a.element for a in geom}
    for aid, be in ads_1s_map.items():
        elem = elem_by_id.get(aid) or getattr(embedded.get(aid), "element", None) or aid[:1]
        if elem == "N" and n1s_be is None:
            n1s_be = be
        elif elem == "O" and o1s_be is None:
            o1s_be = be
        elif elem == "C" and c1s_be is None:
            c1s_be = be

    gas_1s_map: Dict[str, float] = {}
    shift_1s_map: Dict[str, float] = {}
    if gas_state is not None:
        deck_g, res_g = _resolve_state_or_deck(gas_state, d)
        if res_g is None:
            res_g = d.run_deck(deck_g)
        gas_1s_map = _extract_1s_map_ev(res_g)
        geom_g = res_g.deck.geometry if res_g.deck else []
        gas_by_elem: Dict[str, float] = {}
        for ga in geom_g:
            if ga.id in gas_1s_map:
                gas_by_elem[ga.element] = gas_1s_map[ga.id]
        for aid, be in ads_1s_map.items():
            elem = elem_by_id.get(aid) or aid[:1]
            if aid in gas_1s_map:
                shift_1s_map[aid] = float(be - gas_1s_map[aid])
            elif elem in gas_by_elem:
                shift_1s_map[aid] = float(be - gas_by_elem[elem])

    title = getattr(res.deck, "title", "Surface Chemisorption System")

    return SurfaceCoreLevelShifts(
        system_name=title,
        metal_scls_ev=scls_ev,
        surface_core_level_ev=surf_core_ev,
        bulk_core_level_ev=bulk_core_ev,
        adsorbate_n1s_be_ev=n1s_be,
        adsorbate_o1s_be_ev=o1s_be,
        adsorbate_c1s_be_ev=c1s_be,
        method_label="koopmans_orbital_energy",
        expected_relaxation_offset_ev=15.0,
        koopmans_ads_1s_be_ev=ads_1s_map,
        koopmans_gas_1s_be_ev=gas_1s_map,
        koopmans_ads_1s_shift_ev=shift_1s_map,
    )


# -----------------------------------------------------------------------------
# Mayer Bond Orders & CVS-Delta-CASSCF Core-Hole Spectroscopy (Phase 18)
# -----------------------------------------------------------------------------


def compute_mayer_bond_orders(
    state_or_deck: Union[CalculationState, CalculationDeck, str, Path],
    driver: Optional[PhatpsyDriver] = None,
) -> Dict[str, Any]:
    """Compute spin-resolved Mayer bond orders ``B_AB`` and total atomic valences ``V_A``.

    Uses exact spin-density matrices ``(P^alpha, P^beta)`` when available (e.g., ``rdm1_ao_spin``
    from :class:`phatpsy.vb_driver.LocalizedVBDriver`) and the active-region AO overlap matrix ``S``:
        B_AB = 2 * sum_{mu in A, nu in B} [(P^alpha S)_{mu,nu}(P^alpha S)_{nu,mu}
                                           + (P^beta S)_{mu,nu}(P^beta S)_{nu,mu}]
    For closed-shell wavefunctions (``P^alpha = P^beta = 0.5 * P_tot``), this reduces
    identically to ``sum_{mu in A, nu in B} (P_tot S)_{mu,nu}(P_tot S)_{nu,mu}``.
    """
    if isinstance(state_or_deck, (str, Path)):
        p = Path(state_or_deck)
        deck = CalculationDeck.from_yaml(p) if p.suffix in (".yaml", ".yml") else CalculationDeck.from_json(p)
        res = (driver or PhatpsyDriver(quiet=True)).run_deck(deck)
    elif isinstance(state_or_deck, CalculationDeck):
        res = (driver or PhatpsyDriver(quiet=True)).run_deck(state_or_deck)
    else:
        res = state_or_deck

    s_raw = res.get("ao_overlap")
    if s_raw is None:
        raise ValueError("CalculationState lacks 'ao_overlap'; cannot compute Mayer bond orders.")
    S = np.asarray(s_raw, dtype=float)

    rdm1_spin = res.get("rdm1_ao_spin")
    if rdm1_spin is not None:
        p_spin = np.asarray(rdm1_spin, dtype=float)
        Pa = np.asarray(p_spin[0], dtype=float)
        Pb = np.asarray(p_spin[1], dtype=float)
    else:
        rdm1_tot = res.get("rdm1_ao")
        if rdm1_tot is None:
            raise ValueError("CalculationState lacks 'rdm1_ao_spin' / 'rdm1_ao'; cannot compute Mayer bond orders.")
        P_tot = np.asarray(rdm1_tot, dtype=float)
        if P_tot.ndim == 3:
            Pa, Pb = P_tot[0], P_tot[1]
        else:
            Pa = 0.5 * P_tot
            Pb = 0.5 * P_tot

    bi = res.get("basis_info") or {}
    iatomx = np.asarray(bi.get("iatomx", []), dtype=int)
    if iatomx.size != S.shape[0]:
        raise ValueError(
            f"basis_info['iatomx'] length ({iatomx.size}) does not match AO overlap dimension ({S.shape[0]})."
        )

    deck_geom = res.deck.geometry if getattr(res, "deck", None) is not None else []
    unique_gidx = []
    for g1 in iatomx.tolist():
        if g1 not in unique_gidx:
            unique_gidx.append(int(g1))

    atom_ids: List[str] = []
    ao_groups: List[np.ndarray] = []
    for g1 in unique_gidx:
        if 1 <= g1 <= len(deck_geom):
            a_cfg = deck_geom[g1 - 1]
            aid = a_cfg.id or f"{a_cfg.element.strip().capitalize()}{g1}"
        else:
            aid = f"Atom{g1}"
        atom_ids.append(aid)
        ao_groups.append(np.where(iatomx == g1)[0])

    PSa = Pa @ S
    PSb = Pb @ S
    n_at = len(atom_ids)
    bo_mat = np.zeros((n_at, n_at), dtype=float)
    pair_bos: Dict[str, float] = {}
    for ia in range(n_at):
        idx_a = ao_groups[ia]
        for ib in range(ia + 1, n_at):
            idx_b = ao_groups[ib]
            b_val = 2.0 * float(
                np.sum(PSa[np.ix_(idx_a, idx_b)] * PSa[np.ix_(idx_b, idx_a)].T)
                + np.sum(PSb[np.ix_(idx_a, idx_b)] * PSb[np.ix_(idx_b, idx_a)].T)
            )
            bo_mat[ia, ib] = b_val
            bo_mat[ib, ia] = b_val
            pair_bos[f"{atom_ids[ia]}-{atom_ids[ib]}"] = float(b_val)

    valences = {
        atom_ids[ia]: float(np.sum(bo_mat[ia, :]) - bo_mat[ia, ia])
        for ia in range(n_at)
    }
    return {
        "atom_ids": atom_ids,
        "bond_order_matrix": bo_mat.tolist(),
        "pair_bond_orders": pair_bos,
        "atomic_valences": valences,
    }


def get_mayer_bond_order(mayer_result: Dict[str, Any], atom_id_a: str, atom_id_b: str) -> float:
    """Look up symmetric Mayer bond order ``B_AB`` between two atom IDs."""
    pairs = mayer_result.get("pair_bond_orders", {})
    if f"{atom_id_a}-{atom_id_b}" in pairs:
        return float(pairs[f"{atom_id_a}-{atom_id_b}"])
    if f"{atom_id_b}-{atom_id_a}" in pairs:
        return float(pairs[f"{atom_id_b}-{atom_id_a}"])
    atom_ids = list(mayer_result.get("atom_ids", []))
    if atom_id_a in atom_ids and atom_id_b in atom_ids:
        ia = atom_ids.index(atom_id_a)
        ib = atom_ids.index(atom_id_b)
        return float(mayer_result["bond_order_matrix"][ia][ib])
    raise KeyError(f"Atom pair ({atom_id_a}, {atom_id_b}) not found in Mayer bond order result.")


def compute_delta_casscf_core_1s_be(
    deck: Union[CalculationDeck, str, Path],
    target_atom_id: str,
    *,
    cvs_penalty_ha: float = 100.0,
    warm_ground_state: Optional[CalculationState] = None,
    maxiter: int = 350,
    gtol: float = 1e-6,
) -> Dict[str, Any]:
    """Bagus/Siegbahn Core-Valence-Separated Delta-CASSCF ``1s`` core-hole binding energy.

    Unfreezes ONLY ``target_atom_id``'s atomic ``1s`` orbital ``c_1s`` as frozen active
    orbital ``0``, projects all inactive and valence active orbitals strictly orthogonal to
    ``c_1s`` (keeping all spectator cores/atoms and non-target active cores frozen in ``C_env``),
    enforces single ``1s`` core-hole occupation ``n_1s = 1`` via the exact Hermitian CVS penalty
    ``Lambda * (n_0 - 1)^2``, and relaxes all valence orbitals variationally via BFGS.

    For open-shell ground states (``S_GS > 0``, e.g., paramagnetic ``NO(X 2Pi)``), evaluates
    both high-spin (``S_+ = S_GS + 1/2``) and low-spin (``S_- = S_GS - 1/2``) core-hole
    multiplets along with their spin-degeneracy-weighted average BE.
    """
    from pyscf import ao2mo, fci, lib

    from phatpsy.embedding import (
        FrozenEnvironmentAtom,
        embed_from_full_integrals,
    )
    from phatpsy.localized_vb import (
        bfgs_with_reset_continuation,
        build_subset_gauge_pairs,
        eval_orbital_gauge_and_hinge_penalty,
    )
    from phatpsy.multicenter_integrals import molecular_ao_integrals, sto_ng_ao_integrals
    from phatpsy.vb_driver import (
        LocalizedVBDriver,
        _atomic_num,
        _core_shells_count,
        _frozen_env_from_atom,
        _resolve_atom_stos,
    )

    if isinstance(deck, (str, Path)):
        p = Path(deck)
        calc_deck = CalculationDeck.from_yaml(p) if p.suffix in (".yaml", ".yml") else CalculationDeck.from_json(p)
    else:
        calc_deck = copy.deepcopy(deck)

    calc_deck.ensure_atom_ids()
    calc_deck.resolve_coordinates()
    elem_map = calc_deck.resolve_elements()
    ctrl = calc_deck.control

    vb_drv = LocalizedVBDriver(quiet=True)
    res_gs = warm_ground_state if warm_ground_state is not None else vb_drv.run_deck(calc_deck)
    e_gs = float(res_gs["total_energy"])

    kmap = res_gs.get("koopmans_core_energies_ha") or {}
    koopmans_be_ev: Optional[float] = (
        float(-float(kmap[f"{target_atom_id}_1s"]) * HARTREE_TO_EV)
        if f"{target_atom_id}_1s" in kmap
        else None
    )

    active_atoms = [a for a in calc_deck.geometry if not a.virtual and a.is_active]
    spectator_atoms = [a for a in calc_deck.geometry if not a.virtual and not a.is_active]
    target_matches = [
        i for i, a in enumerate(active_atoms) if (a.id or "").strip() == target_atom_id.strip()
    ]
    if not target_matches:
        raise ValueError(f"Target atom '{target_atom_id}' not found among active atoms.")
    ia_target = target_matches[0]
    z_target = int(_atomic_num(active_atoms[ia_target]))
    if not (3 <= z_target <= 10):
        raise ValueError(
            f"Delta-CASSCF 1s solver supports first-row atoms (3 <= Z <= 10); got Z={z_target} ({target_atom_id})."
        )

    active_bases = []
    for a in active_atoms:
        stos, _ = _resolve_atom_stos(a, calc_deck, elem_map)
        active_bases.append(stos)
    spec_bases = []
    spec_envs = []
    for b in spectator_atoms:
        stos_b, bname_b = _resolve_atom_stos(b, calc_deck, elem_map)
        spec_bases.append(stos_b)
        spec_envs.append(_frozen_env_from_atom(b, stos_b, bname_b))

    integral_backend = str(getattr(ctrl, "integral_backend", "prolate") or "prolate").strip().lower()
    sto_ng_k = int(getattr(ctrl, "sto_ng_primitives", 6))
    core_mode = str(getattr(ctrl, "core_embedding", "exact_backprojected") or "exact_backprojected").lower()
    env_mode = str(getattr(ctrl, "environment_embedding", "exact_joint_pk") or "exact_joint_pk").lower()
    if core_mode not in ("all_electron", "exact_backprojected"):
        raise ValueError(f"CVS-Delta-CASSCF requires core_embedding in ('all_electron', 'exact_backprojected'), got '{core_mode}'.")

    full_active_basis = [f for b in active_bases for f in b]
    n_act_ao = len(full_active_basis)
    n_total_ao = n_act_ao + sum(len(b) for b in spec_bases)

    all_specs = [
        (float(_atomic_num(a)), tuple(float(v) for v in a.xyz), b)
        for a, b in zip(active_atoms, active_bases)
    ] + [
        (float(_atomic_num(b)), tuple(float(v) for v in b.xyz), bb)
        for b, bb in zip(spectator_atoms, spec_bases)
    ]

    # Build full-system or active-system integrals
    ab_ints_h = None
    pyscf_ctx_ab_h = None
    full_ints_h = None
    pyscf_ctx_h = None
    if integral_backend == "sto_ng":
        if spectator_atoms and env_mode == "exact_backprojected":
            ab_ints_h, pyscf_ctx_ab_h = sto_ng_ao_integrals(
                all_specs,
                n_prim=sto_ng_k,
                return_pyscf_context=True,
                build_eri=(n_total_ao <= 90),
            )
            all_ord = list(active_atoms) + list(spectator_atoms)
            pyscf_ctx_ab_h.fixed_atom_indices = tuple(
                ia for ia, a_ord in enumerate(all_ord) if bool(a_ord.fixed)
            )
        else:
            specs_mc = all_specs[: len(active_atoms)]
            full_ints_h, pyscf_ctx_h = sto_ng_ao_integrals(
                specs_mc,
                n_prim=sto_ng_k,
                return_pyscf_context=True,
                build_eri=(n_act_ao <= 90 and not spectator_atoms and core_mode == "all_electron"),
            )
            pyscf_ctx_h.fixed_atom_indices = tuple(
                ia for ia, a_ord in enumerate(active_atoms) if bool(a_ord.fixed)
            )
    else:
        if spectator_atoms and env_mode == "exact_backprojected":
            ab_ints_h = molecular_ao_integrals(all_specs, n_eta=32, n_phi=16, lmax=16, mmax=4)
        else:
            full_ints_h = molecular_ao_integrals(all_specs[: len(active_atoms)], n_eta=32, n_phi=16, lmax=16, mmax=4)

    # Atomic 1s orbital vector on target atom in active AO space
    target_off = sum(len(b) for b in active_bases[:ia_target])
    target_coords = tuple(float(v) for v in active_atoms[ia_target].xyz)
    f_target_1s = FrozenEnvironmentAtom.from_atomic_rhf(
        float(z_target), target_coords, active_bases[ia_target], n_occ=1
    )
    v_1s_bare = np.zeros(n_act_ao, dtype=float)
    v_1s_bare[target_off : target_off + len(active_bases[ia_target])] = f_target_1s.C_occ[:, 0]

    # Active core columns OMITTING target_atom_id's 1s orbital when core_mode != 'all_electron'
    core_cols_hole: List[np.ndarray] = []
    off_c = 0
    for ia, (a, b_a) in enumerate(zip(active_atoms, active_bases)):
        z_a = _atomic_num(a)
        nc_a = _core_shells_count(z_a)
        if core_mode != "all_electron" and nc_a > 0 and ia != ia_target:
            c_a = (float(a.xyz[0]), float(a.xyz[1]), float(a.xyz[2]))
            f_env = FrozenEnvironmentAtom.from_atomic_rhf(float(z_a), c_a, b_a, n_occ=nc_a)
            for ck in range(nc_a):
                col = np.zeros(n_act_ao, dtype=float)
                col[off_c : off_c + len(b_a)] = f_env.C_occ[:, ck]
                core_cols_hole.append(col)
        off_c += len(b_a)
    C_core_hole = np.column_stack(core_cols_hole) if core_cols_hole else np.zeros((n_act_ao, 0), dtype=float)

    if spectator_atoms and env_mode == "exact_backprojected":
        env_cols = []
        for ck in range(C_core_hole.shape[1]):
            v = np.zeros(n_total_ao, dtype=float)
            v[:n_act_ao] = C_core_hole[:, ck]
            env_cols.append(v)
        spec_off = n_act_ao
        for sb, se in zip(spec_bases, spec_envs):
            nb_ao = len(sb)
            for ck in range(se.C_occ.shape[1]):
                v = np.zeros(n_total_ao, dtype=float)
                v[spec_off : spec_off + nb_ao] = se.C_occ[:, ck]
                env_cols.append(v)
            spec_off += nb_ao
        C_env_h = np.column_stack(env_cols) if env_cols else np.zeros((n_total_ao, 0), dtype=float)
        ints_hole = embed_from_full_integrals(
            ab_ints_h,
            C_env_h,
            active_ao_idx=np.arange(n_act_ao, dtype=int),
            mode="exact_backprojected",
            pyscf_ctx=pyscf_ctx_ab_h,
        )
    elif core_mode != "all_electron" and C_core_hole.shape[1] > 0:
        ints_hole = embed_from_full_integrals(
            full_ints_h, C_core_hole, mode="exact_backprojected", pyscf_ctx=pyscf_ctx_h
        )
    else:
        ints_hole = full_ints_h

    S_h = ints_hole.S
    h_h = ints_hole.h
    eri_h = ints_hole.eri
    eri_s8_h = getattr(ints_hole, "_eri_s8", None)
    if eri_s8_h is None:
        eri_s8_h = ao2mo.restore(8, eri_h, n_act_ao)
        setattr(ints_hole, "_eri_s8", eri_s8_h)
    Jm_h, Km_h = ints_hole.jk_matrices()

    # Normalized target 1s core orbital in embedded metric
    c_1s = v_1s_bare / math.sqrt(max(float(v_1s_bare @ S_h @ v_1s_bare), 1e-16))
    S_c1s = S_h @ c_1s  # shape (n_act_ao,)

    # Extract GS localized orbitals, roles, domain masks, and electron counts
    roles_gs = list(res_gs["vb_roles"])
    masks_gs = [np.asarray(m, dtype=int) for m in res_gs["vb_domain_masks"]]
    na_gs, nb_gs = int(res_gs["vb_nelec_act"][0]), int(res_gs["vb_nelec_act"][1])
    loc_bare = res_gs.get("localized_orbitals")
    loc_emb = res_gs.get("localized_orbitals_embedded")
    if loc_bare is not None and np.asarray(loc_bare, dtype=float).shape[1] == n_act_ao:
        loc_gs = np.asarray(loc_bare, dtype=float)
    else:
        loc_gs = np.asarray(loc_emb, dtype=float)

    # In all_electron mode, remove target 1s from GS inactive set so it isn't double-counted
    keep_idx = list(range(len(roles_gs)))
    if core_mode == "all_electron":
        inact_indices = [k for k, r in enumerate(roles_gs) if r == "inactive"]
        if inact_indices:
            best_k = max(inact_indices, key=lambda k: abs(float(loc_gs[k] @ S_h @ c_1s)))
            keep_idx.remove(best_k)

    val_roles = [roles_gs[k] for k in keep_idx]
    val_masks = [masks_gs[k] for k in keep_idx]
    val_init_orbs = [loc_gs[k] - c_1s * float(S_c1s @ loc_gs[k]) for k in keep_idx]
    off_m = np.cumsum([0] + [len(m) for m in val_masks])
    x0_val = np.concatenate([o[m] for o, m in zip(val_init_orbs, val_masks)])
    for k in range(len(val_masks)):
        s0, s1 = int(off_m[k]), int(off_m[k + 1])
        nrm_k = float(np.linalg.norm(x0_val[s0:s1]))
        if nrm_k > 1e-12:
            x0_val[s0:s1] /= nrm_k

    # Precondition valence orbitals in S_perp-orthonormal non-null subspace per mask
    # (projecting out frozen c_1s direction cleanly) with per-orbital unit-norm tether
    # and subset-gauge orthogonality.
    S_perp = 0.5 * ((S_h - np.outer(S_c1s, S_c1s)) + (S_h - np.outer(S_c1s, S_c1s)).T)
    B_hole: List[np.ndarray] = []
    P_hole: List[np.ndarray] = []
    u0_hole_parts: List[np.ndarray] = []
    for k_h, (_, m_h) in enumerate(zip(val_roles, val_masks)):
        Sm_h = 0.5 * (S_perp[np.ix_(m_h, m_h)] + S_perp[np.ix_(m_h, m_h)].T)
        wm_h, vm_h = np.linalg.eigh(Sm_h)
        for col_h in range(vm_h.shape[1]):
            imax_h = int(np.argmax(np.abs(vm_h[:, col_h])))
            if vm_h[imax_h, col_h] < 0.0:
                vm_h[:, col_h] *= -1.0
        keep_h = wm_h > 1e-8
        if not np.any(keep_h):
            keep_h = np.array([int(np.argmax(wm_h))])
        wk_h = np.maximum(wm_h[keep_h], 1e-12)
        Bk_h = vm_h[:, keep_h] * (wk_h ** -0.5)[None, :]
        Pk_h = vm_h[:, keep_h] * (wk_h ** 0.5)[None, :]
        uk_h = Pk_h.T @ np.asarray(val_init_orbs[k_h][m_h], dtype=float)
        nrm_uh = float(np.linalg.norm(uk_h))
        if nrm_uh > 1e-12:
            uk_h = uk_h / nrm_uh
        B_hole.append(Bk_h)
        P_hole.append(Pk_h)
        u0_hole_parts.append(uk_h)
    use_gauge_h = len(B_hole) > 1
    F_hole, _, gauge_pairs_h, u0_hole_parts = build_subset_gauge_pairs(
        val_roles,
        val_masks,
        B_hole,
        P_hole,
        S_perp,
        u0_hole_parts,
        orthogonalize_init=use_gauge_h,
    )
    u_off_h = np.cumsum([0] + [b.shape[1] for b in B_hole])
    u0_hole = np.concatenate(u0_hole_parts) if u0_hole_parts else np.zeros(0, dtype=float)

    # Total active orbitals in core-hole state = 1 (frozen c_1s) + GS active valence orbitals
    nact_gs = sum(1 for r in val_roles if r == "active")
    nact_tot = 1 + nact_gs
    nelec_tot_hole = na_gs + nb_gs + 1
    if abs(na_gs - nb_gs) == 0:
        sectors = [("doublet", (na_gs + 1, nb_gs), 0.5, 2)]
    else:
        n_hi, n_lo = max(na_gs, nb_gs), min(na_gs, nb_gs)
        s_gs = 0.5 * (n_hi - n_lo)
        s_plus = s_gs + 0.5
        s_minus = s_gs - 0.5
        sectors = [
            ("high_spin", (n_hi + 1, n_lo), s_plus, int(round(2.0 * s_plus + 1.0))),
            ("low_spin", (n_hi, n_lo + 1), s_minus, int(round(2.0 * s_minus + 1.0))),
        ]

    Lambda = float(cvs_penalty_ha)
    warm_u_state = [np.array(u0_hole, dtype=float, copy=True)]
    I_n = np.eye(n_act_ao)

    def _unpack_projected_spaces(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inact_cols: List[np.ndarray] = []
        act_cols: List[np.ndarray] = [c_1s]
        for k, (r, m) in enumerate(zip(val_roles, val_masks)):
            v_raw = np.zeros(n_act_ao, dtype=float)
            v_raw[m] = x[int(off_m[k]) : int(off_m[k + 1])]
            # Project strictly orthogonal to frozen c_1s: P_perp v = v - c_1s (S_c1s^T v)
            v_perp = v_raw - c_1s * float(np.dot(S_c1s, v_raw))
            if r == "inactive":
                inact_cols.append(v_perp)
            else:
                act_cols.append(v_perp)
        Xi = np.column_stack(inact_cols) if inact_cols else np.zeros((n_act_ao, 0), dtype=float)
        Xa = np.column_stack(act_cols)
        return Xi, Xa

    def _solve_sector(
        label: str, nelec_h: Tuple[int, int], s_targ: float
    ) -> Tuple[float, float, float, bool]:
        from pyscf.fci import cistring

        need_fix_spin = (label == "low_spin") or abs(s_targ - 0.5 * abs(nelec_h[0] - nelec_h[1])) > 1e-6
        last_ci: List[Optional[np.ndarray]] = [None]

        def _fci_cvs_solve(h1_b: np.ndarray, eri4: np.ndarray, e_const_b: float) -> Tuple[float, np.ndarray, np.ndarray, float]:
            h1_cvs = np.array(h1_b, dtype=float, copy=True)
            eri4_cvs = np.array(eri4, dtype=float, copy=True)
            h1_cvs[0, 0] -= Lambda
            eri4_cvs[0, 0, 0, 0] += 2.0 * Lambda
            e_const_cvs = float(e_const_b + Lambda)
            eri_act_cvs = ao2mo.restore(8, eri4_cvs, nact_tot)

            solver = fci.direct_spin1.FCI()
            solver.max_memory = max(int(solver.max_memory), int(lib.current_memory()[0]) + 4000)
            solver.conv_tol = 1e-10 if need_fix_spin else 1e-12
            solver.max_cycle = 400
            if need_fix_spin:
                solver = fci.addons.fix_spin_(solver, ss=float(s_targ * (s_targ + 1.0)), shift=5.0)
            na_h = int(cistring.num_strings(nact_tot, nelec_h[0]))
            nb_h = int(cistring.num_strings(nact_tot, nelec_h[1]))
            num_dets = na_h * nb_h
            if (not need_fix_spin) and num_dets <= 1024:
                h0_diag = solver.make_hdiag(h1_cvs, eri_act_cvs, nact_tot, nelec_h)
                addr, h0_mat = solver.pspace(h1_cvs, eri_act_cvs, nact_tot, nelec_h, h0_diag, num_dets)
                pw, pv = solver.eig(h0_mat)
                ci_full = np.zeros(num_dets, dtype=float)
                ci_full[addr] = pv[:, 0]
                civec = ci_full.reshape(na_h, nb_h)
                e_cvs = float(pw[0] + e_const_cvs)
            elif last_ci[0] is not None:
                e_cvs, civec = solver.kernel(
                    h1_cvs, eri_act_cvs, nact_tot, nelec_h, ci0=last_ci[0], ecore=e_const_cvs
                )
            elif need_fix_spin:
                e_cvs, civec = solver.kernel(
                    h1_cvs, eri_act_cvs, nact_tot, nelec_h, ecore=e_const_cvs
                )
            else:
                h0_diag = solver.make_hdiag(h1_cvs, eri_act_cvs, nact_tot, nelec_h)
                n_ps = min(int(num_dets), 1024)
                addr, h0_mat = solver.pspace(h1_cvs, eri_act_cvs, nact_tot, nelec_h, h0_diag, n_ps)
                _, pv = solver.eig(h0_mat)
                ci_full = np.zeros(num_dets, dtype=float)
                ci_full[addr] = pv[:, 0]
                e_cvs, civec = solver.kernel(
                    h1_cvs, eri_act_cvs, nact_tot, nelec_h, ci0=ci_full.reshape(na_h, nb_h), ecore=e_const_cvs
                )
            last_ci[0] = civec
            dm1, dm2 = fci.direct_spin1.make_rdm12(civec, nact_tot, nelec_h)
            e_pure = float(e_const_b + np.sum(h1_b * dm1) + 0.5 * np.sum(eri4 * dm2))
            return float(e_cvs), dm1, dm2, e_pure

        def _contract_eri4(Ca_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            Ca2 = (Ca_mat[:, None, :, None] * Ca_mat[None, :, None, :]).reshape(
                n_act_ao * n_act_ao, nact_tot * nact_tot
            )
            V4 = (Jm_h @ Ca2).reshape(n_act_ao, n_act_ao, nact_tot * nact_tot)
            t3_loc = np.einsum("mnk,nq->mqk", V4, Ca_mat, optimize=True).reshape(
                n_act_ao, nact_tot, nact_tot, nact_tot
            )
            eri4_loc = np.tensordot(Ca_mat.T, t3_loc, axes=(1, 0))
            return t3_loc, eri4_loc

        e_off_cvs = float(ints_hole.vnn + Lambda)
        c_pen_h = 1.0e-3 if use_gauge_h else 1.0e-4

        def _obj_and_grad_u(u: np.ndarray) -> Tuple[float, np.ndarray]:
            pen_e, pen_g, x = eval_orbital_gauge_and_hinge_penalty(
                u,
                u_off_h,
                B_hole,
                F_hole,
                val_masks,
                S_perp,
                gauge_pairs_h,
                c_pen=c_pen_h,
                c_hinge=1.0e-4,
                w_floor=0.05,
                use_gauge_and_hinge=use_gauge_h,
            )
            if pen_e >= 1.0e7:
                return float(pen_e), np.zeros_like(u)

            Xi, Xa = _unpack_projected_spaces(x)
            if Xi.shape[1] > 0:
                Gi = 0.5 * (Xi.T @ S_h @ Xi + (Xi.T @ S_h @ Xi).T)
                wi, vi = np.linalg.eigh(Gi)
                if wi[0] < 1e-10 or not np.all(np.isfinite(wi)):
                    return 1.0e8, np.zeros_like(u)
                Ni = (vi * (1.0 / wi)[None, :]) @ vi.T
                Dc = Xi @ Ni @ Xi.T
            else:
                Ni = np.zeros((0, 0), dtype=float)
                Dc = np.zeros((n_act_ao, n_act_ao), dtype=float)
            J_Dc = (Jm_h @ Dc.ravel()).reshape(n_act_ao, n_act_ao)
            K_Dc = (Km_h @ Dc.ravel()).reshape(n_act_ao, n_act_ao)
            Fc = h_h + 2.0 * J_Dc - K_Dc
            e_core = float(np.sum(Dc * (h_h + Fc)))

            Y = Xa - Dc @ (S_h @ Xa)
            M_ov = 0.5 * (Y.T @ S_h @ Y + (Y.T @ S_h @ Y).T)
            w_m, V_m = np.linalg.eigh(M_ov)
            if w_m[0] < 1e-10 or not np.all(np.isfinite(w_m)):
                return 1.0e8, np.zeros_like(u)
            Tm = (V_m * (1.0 / np.sqrt(w_m))[None, :]) @ V_m.T
            Ca = Y @ Tm
            h1_b = Ca.T @ Fc @ Ca
            t3, eri4 = _contract_eri4(Ca)
            e_cvs, dm1, dm2, _ = _fci_cvs_solve(h1_b, eri4, float(e_core + ints_hole.vnn))

            Q_ca = np.einsum("mqrt,pqrt->mp", t3, dm2, optimize=True)
            G_Ca = 2.0 * (Fc @ Ca @ dm1) + 2.0 * Q_ca
            G_Y = (I_n - S_h @ Ca @ Ca.T) @ G_Ca @ Tm
            G_Xa = (I_n - S_h @ Dc) @ G_Y

            Pa = Ca @ dm1 @ Ca.T
            J_Pa = (Jm_h @ Pa.ravel()).reshape(n_act_ao, n_act_ao)
            K_Pa = (Km_h @ Pa.ravel()).reshape(n_act_ao, n_act_ao)
            Ac = 2.0 * Fc + 2.0 * J_Pa - K_Pa - 0.5 * (S_h @ Xa @ G_Y.T + G_Y @ Xa.T @ S_h)
            G_Xi = 2.0 * (I_n - S_h @ Dc) @ Ac @ Xi @ Ni if Xi.shape[1] > 0 else np.zeros((n_act_ao, 0))

            gi = iter(G_Xi.T)
            ga = iter(G_Xa[:, 1:].T)  # Column 0 is frozen c_1s
            grads_u = []
            for k_i, (r, m) in enumerate(zip(val_roles, val_masks)):
                g_col = next(gi) if r == "inactive" else next(ga)
                # Adjoint of P_perp = (I - c_1s S_c1s^T): P_perp^T g = g - S_c1s (c_1s^T g)
                g_proj = g_col - S_c1s * float(np.dot(c_1s, g_col))
                grads_u.append(B_hole[k_i].T @ g_proj[m])
            gu = np.concatenate(grads_u) + pen_g
            return float((e_cvs - e_off_cvs) + pen_e), gu

        maxit_sec = int(maxiter)
        gtol_sec = max(float(gtol), 1.0e-4) if (need_fix_spin and nact_tot >= 7) else float(gtol)
        res_opt, u_opt_h_parts = bfgs_with_reset_continuation(
            _obj_and_grad_u,
            warm_u_state[0],
            u_off_h,
            maxiter=maxit_sec,
            gtol=gtol_sec,
            stat_tol=max(5e-3, gtol_sec * 50.0),
        )
        warm_u_state[0] = np.array(res_opt.x, dtype=float, copy=True)

        # Evaluate converged unpenalized core-hole energy & verify n_1s == 1.0
        x_opt_h = np.empty(int(off_m[-1]), dtype=float)
        for k_i, Bk_h in enumerate(B_hole):
            x_opt_h[int(off_m[k_i]) : int(off_m[k_i + 1])] = Bk_h @ u_opt_h_parts[k_i]
        Xi_f, Xa_f = _unpack_projected_spaces(x_opt_h)
        if Xi_f.shape[1] > 0:
            Dc_f = Xi_f @ np.linalg.inv(Xi_f.T @ S_h @ Xi_f) @ Xi_f.T
        else:
            Dc_f = np.zeros((n_act_ao, n_act_ao), dtype=float)
        J_Dcf = (Jm_h @ Dc_f.ravel()).reshape(n_act_ao, n_act_ao)
        K_Dcf = (Km_h @ Dc_f.ravel()).reshape(n_act_ao, n_act_ao)
        Fc_f = h_h + 2.0 * J_Dcf - K_Dcf
        e_core_f = float(np.sum(Dc_f * (h_h + Fc_f)))
        Y_f = Xa_f - Dc_f @ (S_h @ Xa_f)
        M_f = 0.5 * (Y_f.T @ S_h @ Y_f + (Y_f.T @ S_h @ Y_f).T)
        w_f, V_f = np.linalg.eigh(M_f)
        w_f = np.maximum(w_f, 1e-14)
        Ca_f = Y_f @ ((V_f * (1.0 / np.sqrt(w_f))[None, :]) @ V_f.T)
        _, eri4_f = _contract_eri4(Ca_f)
        last_ci[0] = None
        _, dm1_f, _, e_pure_f = _fci_cvs_solve(Ca_f.T @ Fc_f @ Ca_f, eri4_f, float(e_core_f + ints_hole.vnn))
        n_1s = float(dm1_f[0, 0])
        if abs(n_1s - 1.0) >= 1e-3:
            raise RuntimeError(
                f"CVS core-hole variational check failed on {target_atom_id} ({label}): "
                f"|n_1s - 1.0| = {abs(n_1s - 1.0):.3e} (n_1s={n_1s:.6f})"
            )
        be_ev = float((e_pure_f - e_gs) * HARTREE_TO_EV)
        jac_arr_h = np.asarray(res_opt.jac, dtype=float) if getattr(res_opt, "jac", None) is not None else np.zeros(0)
        jac_inf = float(np.max(np.abs(jac_arr_h))) if jac_arr_h.size else 0.0
        n_iters_opt = int(getattr(res_opt, "nit", maxit_sec))
        conv_s = bool(res_opt.success or jac_inf < max(5e-3, gtol_sec * 50.0) or (n_iters_opt < maxit_sec and jac_inf < 2e-2))
        return float(e_pure_f), be_ev, n_1s, conv_s

    sector_results: Dict[str, Dict[str, Any]] = {}
    num_weighted = 0.0
    den_weight = 0.0
    max_n1s_err = 0.0
    all_conv = True
    for label, nelec_h, s_targ, mult_w in sectors:
        e_hole_s, be_s, n1s_s, conv_s = _solve_sector(label, nelec_h, s_targ)
        sector_results[label] = {
            "energy_ha": round(float(e_hole_s), 8),
            "be_ev": round(float(be_s), 4),
            "n_1s_occupation": round(float(n1s_s), 8),
            "spin_multiplicity": int(mult_w),
            "converged": bool(conv_s),
        }
        num_weighted += float(mult_w) * be_s
        den_weight += float(mult_w)
        max_n1s_err = max(max_n1s_err, abs(n1s_s - 1.0))
        all_conv = all_conv and conv_s

    be_avg_ev = float(num_weighted / max(den_weight, 1.0))
    if "high_spin" in sector_results and "low_spin" in sector_results:
        be_hi = float(sector_results["high_spin"]["be_ev"])
        be_lo = float(sector_results["low_spin"]["be_ev"])
        split_ev = float(be_lo - be_hi)
    else:
        be_hi = float(be_avg_ev)
        be_lo = float(be_avg_ev)
        split_ev = 0.0

    relax_ev = float(koopmans_be_ev - be_avg_ev) if koopmans_be_ev is not None else None
    return {
        "method": "cvs_delta_casscf",
        "target_atom_id": target_atom_id,
        "ground_energy_ha": float(e_gs),
        "be_multiplet_avg_ev": float(be_avg_ev),
        "be_high_spin_ev": float(be_hi),
        "be_low_spin_ev": float(be_lo),
        "multiplet_splitting_ev": float(split_ev),
        "n_1s_occupation": float(
            sector_results[sectors[0][0]]["n_1s_occupation"]
        ),
        "max_n_1s_deviation": float(max_n1s_err),
        "koopmans_method": "koopmans_orbital_energy",
        "koopmans_be_ev": float(koopmans_be_ev) if koopmans_be_ev is not None else None,
        "expected_koopmans_offset_ev": 15.0,
        "relaxation_energy_ev": float(relax_ev) if relax_ev is not None else None,
        "cvs_penalty_ha": float(Lambda),
        "converged": bool(all_conv),
        "sectors": sector_results,
    }
