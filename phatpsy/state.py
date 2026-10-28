# phatpsy/state.py
"""
State persistence, checkpointing, warm restarts, and transferable embedded atomic states
for PHATPSY calculations.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import numpy as np

from phatpsy.basis import InitialOrbital, BasisFunction


@dataclass
class EmbeddedAtomState:
    """
    Converged localized atomic state extracted from a molecular or atomic calculation.
    Transferable for inclusion in other systems (e.g., fixed bulk or surface substrate atoms).
    """
    atom_id: Optional[str] = None
    element: str = ""
    atomic_number: int = 0
    xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    nuclear_charge: float = 0.0
    electronic_charge: float = 0.0
    net_charge: float = 0.0
    alpha_spin: float = 0.0
    beta_spin: float = 0.0
    net_spin: float = 0.0
    basis_functions: List[Dict[str, Any]] = field(default_factory=list)
    orbitals: List[Dict[str, Any]] = field(default_factory=list)
    density_matrix: Optional[List[List[float]]] = None
    model_potential: Optional[Dict[str, Any]] = None
    fixed: bool = False
    virtual: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a JSON-serializable dictionary."""
        d = asdict(self)
        d["xyz"] = [round(float(c), 6) for c in self.xyz]
        d["nuclear_charge"] = round(float(self.nuclear_charge), 6)
        d["electronic_charge"] = round(float(self.electronic_charge), 6)
        d["net_charge"] = round(float(self.net_charge), 6)
        d["alpha_spin"] = round(float(self.alpha_spin), 6)
        d["beta_spin"] = round(float(self.beta_spin), 6)
        d["net_spin"] = round(float(self.net_spin), 6)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddedAtomState":
        """Reconstruct EmbeddedAtomState from a dictionary."""
        xyz = [float(c) for c in data.get("xyz", [0.0, 0.0, 0.0])]
        return cls(
            atom_id=data.get("atom_id"),
            element=data.get("element", "").strip().capitalize(),
            atomic_number=int(data.get("atomic_number", 0)),
            xyz=xyz,
            nuclear_charge=float(data.get("nuclear_charge", 0.0)),
            electronic_charge=float(data.get("electronic_charge", 0.0)),
            net_charge=float(data.get("net_charge", 0.0)),
            alpha_spin=float(data.get("alpha_spin", 0.0)),
            beta_spin=float(data.get("beta_spin", 0.0)),
            net_spin=float(data.get("net_spin", 0.0)),
            basis_functions=list(data.get("basis_functions", [])),
            orbitals=list(data.get("orbitals", [])),
            density_matrix=data.get("density_matrix"),
            model_potential=data.get("model_potential"),
            fixed=bool(data.get("fixed", False)),
            virtual=bool(data.get("virtual", False)),
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Save embedded atom state to JSON or NPZ file."""
        target = Path(filepath).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()

        if suffix == ".npz":
            # Save numeric arrays in NPZ and metadata as a JSON string
            meta = self.to_dict()
            dm = np.array(self.density_matrix, dtype=np.float64) if self.density_matrix is not None else np.empty(0)
            np.savez_compressed(
                target,
                metadata_json=json.dumps(meta),
                density_matrix=dm,
            )
        else:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EmbeddedAtomState":
        """Load embedded atom state from JSON or NPZ file."""
        target = Path(filepath).resolve()
        suffix = target.suffix.lower()

        if suffix == ".npz":
            with np.load(target) as npz:
                meta = json.loads(str(npz["metadata_json"]))
                if "density_matrix" in npz and npz["density_matrix"].size > 0:
                    meta["density_matrix"] = npz["density_matrix"].tolist()
                return cls.from_dict(meta)
        else:
            with open(target, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))

    def to_initial_orbitals(self) -> List[InitialOrbital]:
        """
        Convert embedded orbitals into InitialOrbital objects for CalculationDeck &ATORB generation.
        """
        res = []
        for orb in self.orbitals:
            e_a = float(orb.get("energy_alpha", 0.0))
            e_b = float(orb.get("energy_beta", e_a))
            occ_a = float(orb.get("occ_alpha", 0.0))
            occ_b = float(orb.get("occ_beta", occ_a))

            raw_coefs = orb.get("coefficients", [])
            coef_tuples = []
            for c in raw_coefs:
                if len(c) == 3:
                    isto, ml, c1 = c
                    coef_tuples.append((int(isto), int(ml), float(c1), float(c1)))
                elif len(c) >= 4:
                    isto, ml, c1, c2 = c[:4]
                    coef_tuples.append((int(isto), int(ml), float(c1), float(c2)))

            res.append(InitialOrbital(
                eigval=[e_a, e_b],
                occno=[occ_a, occ_b],
                coefficients=coef_tuples,
            ))
        return res

    def to_basis_functions(self) -> List[BasisFunction]:
        """Convert basis functions metadata into BasisFunction instances."""
        return [BasisFunction.from_dict(b) for b in self.basis_functions]


class CalculationState(dict):
    """
    Complete calculation state representing a converged PHATPSY run.
    Subclasses `dict` to preserve 100% backward compatibility with legacy dictionary
    access (e.g. res['total_energy']), while providing object attributes, serialization,
    and warm restart creation.
    """

    def __init__(self, *args, deck: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._deck = deck
        # Ensure virial_ratio is computed if energies exist
        if "virial_ratio" not in self and "potential_energy" in self and "kinetic_energy" in self:
            ke = float(self["kinetic_energy"])
            pe = float(self["potential_energy"])
            self["virial_ratio"] = -pe / ke if abs(ke) > 1e-12 else 0.0

    @property
    def deck(self) -> Optional[Any]:
        return self._deck

    @deck.setter
    def deck(self, val: Any) -> None:
        self._deck = val

    def decompose_subsystems(
        self,
        bulk_references: Optional[Dict[str, float]] = None,
        isolated_references: Optional[Dict[str, float]] = None,
    ) -> Any:
        """
        Decomposes total energy into active cluster, environment, and embedding components
        according to the ONIOM / DFT-in-DFT subsystem energy formalism.
        """
        from phatpsy.subsystem import decompose_subsystem_energies
        return decompose_subsystem_energies(
            self,
            bulk_references=bulk_references,
            isolated_references=isolated_references,
        )

    @property
    def subsystem_decomposition(self) -> Any:
        """Subsystem decomposition of the calculation state."""
        return self.decompose_subsystems()

    @property
    def energy_cluster(self) -> float:
        """Reference isolated atomic energy sum of active cluster atoms."""
        return self.subsystem_decomposition.energy_cluster

    @property
    def energy_environment(self) -> float:
        """Bulk cohesive reference energy sum of frozen environment atoms."""
        return self.subsystem_decomposition.energy_environment

    @property
    def energy_embedding(self) -> float:
        """Embedding energy strictly satisfying E_total - E_cluster - E_env."""
        return self.subsystem_decomposition.energy_embedding

    @property
    def energy_excess(self) -> float:
        """Excess energy of active cluster embedded in bulk (E_total - E_env)."""
        return self.subsystem_decomposition.energy_excess

    @property
    def gradient(self) -> np.ndarray:
        """Cartesian nuclear gradient (dE/dX_A) in Hartree / Bohr with Pulay basis corrections."""
        if "gradient" not in self and self._deck is not None:
            from phatpsy.driver import PhatpsyDriver
            d = PhatpsyDriver(quiet=True)
            self["gradient"] = d.gradient(self._deck)
        return self.get("gradient", np.zeros((len(self.atom_charges), 3)))

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'CalculationState' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_deck",):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def get_embedded_atom(self, atom_id_or_index: Union[str, int]) -> EmbeddedAtomState:
        """
        Retrieves the EmbeddedAtomState for a specific atom by its unique ID or 0-based index.
        """
        embedded = self.get("embedded_atoms", {})
        if isinstance(atom_id_or_index, str):
            if atom_id_or_index in embedded:
                return embedded[atom_id_or_index]
            for atom_key, atom_state in embedded.items():
                if atom_state.atom_id == atom_id_or_index:
                    return atom_state
            raise KeyError(f"Embedded atom '{atom_id_or_index}' not found in calculation state.")
        elif isinstance(atom_id_or_index, int):
            atom_list = list(embedded.values())
            if 0 <= atom_id_or_index < len(atom_list):
                return atom_list[atom_id_or_index]
            raise IndexError(f"Atom index {atom_id_or_index} out of range ({len(atom_list)} atoms).")
        else:
            raise TypeError(f"atom_id_or_index must be str or int, got {type(atom_id_or_index)}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert full calculation state to a serializable dictionary, converting
        all numpy arrays to lists and embedded atoms to dicts.
        """
        out: Dict[str, Any] = {}
        if self._deck is not None and hasattr(self._deck, "to_dict"):
            out["deck"] = self._deck.to_dict()

        for k, v in self.items():
            if k == "deck":
                continue
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, dict):
                if k == "embedded_atoms":
                    out[k] = {
                        atom_k: (atom_v.to_dict() if isinstance(atom_v, EmbeddedAtomState) else atom_v)
                        for atom_k, atom_v in v.items()
                    }
                elif k == "basis_info":
                    out[k] = {
                        bk: (bv.tolist() if isinstance(bv, np.ndarray) else bv)
                        for bk, bv in v.items()
                    }
                else:
                    out[k] = v
            elif isinstance(v, (np.bool_, bool)):
                out[k] = bool(v)
            elif isinstance(v, (np.floating, float)):
                out[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                out[k] = int(v)
            else:
                out[k] = v
        return out

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save calculation state checkpoint to JSON or NPZ.
        """
        target = Path(filepath).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()

        if suffix == ".npz":
            data_dict = {}
            arrays_to_save = {}
            for k, v in self.items():
                if isinstance(v, np.ndarray):
                    arrays_to_save[k] = v
                else:
                    data_dict[k] = v

            serializable_meta = CalculationState(data_dict, deck=self._deck).to_dict()
            np.savez_compressed(
                target,
                checkpoint_json=json.dumps(serializable_meta),
                **arrays_to_save,
            )
        else:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CalculationState":
        """
        Load calculation state checkpoint from JSON or NPZ.
        """
        from phatpsy.schema import CalculationDeck

        target = Path(filepath).resolve()
        suffix = target.suffix.lower()

        if suffix == ".npz":
            with np.load(target) as npz:
                meta = json.loads(str(npz["checkpoint_json"]))
                deck = CalculationDeck.from_dict(meta["deck"]) if "deck" in meta else None
                state_dict = dict(meta)

                # Restore arrays
                for k in npz.files:
                    if k != "checkpoint_json":
                        state_dict[k] = npz[k]

                # Restore embedded atoms
                if "embedded_atoms" in state_dict:
                    restored_atoms = {}
                    for ak, av in state_dict["embedded_atoms"].items():
                        restored_atoms[ak] = EmbeddedAtomState.from_dict(av)
                    state_dict["embedded_atoms"] = restored_atoms

                # Convert basis info lists back to arrays if present
                if "basis_info" in state_dict and isinstance(state_dict["basis_info"], dict):
                    binfo = {}
                    for bk, bv in state_dict["basis_info"].items():
                        binfo[bk] = np.array(bv)
                    state_dict["basis_info"] = binfo

                # Ensure numpy arrays for primary matrix fields
                for key in ("eigenvalues", "occupations", "mo_coefficients", "overlap_packed",
                            "ewmat_packed", "atom_charges", "atom_coordinates", "atom_populations"):
                    if key in state_dict and not isinstance(state_dict[key], np.ndarray):
                        state_dict[key] = np.array(state_dict[key])

                return cls(state_dict, deck=deck)
        else:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)

            deck = CalculationDeck.from_dict(data["deck"]) if "deck" in data else None
            state_dict = dict(data)

            # Convert arrays
            for key in ("eigenvalues", "occupations", "mo_coefficients", "overlap_packed",
                        "ewmat_packed", "atom_charges", "atom_coordinates", "atom_populations"):
                if key in state_dict and not isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = np.array(state_dict[key])

            if "basis_info" in state_dict and isinstance(state_dict["basis_info"], dict):
                binfo = {}
                for bk, bv in state_dict["basis_info"].items():
                    binfo[bk] = np.array(bv)
                state_dict["basis_info"] = binfo

            if "embedded_atoms" in state_dict:
                restored_atoms = {}
                for ak, av in state_dict["embedded_atoms"].items():
                    restored_atoms[ak] = EmbeddedAtomState.from_dict(av)
                state_dict["embedded_atoms"] = restored_atoms

            return cls(state_dict, deck=deck)

    def create_restart_deck(
        self,
        new_geometry: Optional[Any] = None,
        control_overrides: Optional[Dict[str, Any]] = None,
        units: str = "bohr",
        use_embedded_orbitals: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """
        Creates a new ready-to-run CalculationDeck warm-started from this calculation state.
        Pre-converged embedded atom states and orbitals are attached to the atoms,
        allowing rapid convergence across geometry changes, PES scans, or new molecular configurations.
        """
        import copy
        from phatpsy.builder import parse_geometry
        from phatpsy.schema import CalculationDeck, AtomConfig

        if self._deck is not None:
            new_deck = copy.deepcopy(self._deck)
        else:
            # Reconstruct minimal deck from atoms and state
            embedded = self.get("embedded_atoms", {})
            geom = []
            for atom_id, ea in embedded.items():
                geom.append(AtomConfig(
                    element=ea.element,
                    id=atom_id,
                    xyz=list(ea.xyz),
                    charge=ea.net_charge,
                    fixed=ea.fixed,
                    virtual=ea.virtual,
                    spin="down" if ea.net_spin < -0.01 else ("up" if ea.net_spin > 0.01 else None),
                ))
            new_deck = CalculationDeck(
                title=title or "PHATPSY Warm Restart",
                geometry=geom,
            )

        if title is not None:
            new_deck.title = title

        # Update geometry if provided
        if new_geometry is not None:
            new_atoms = parse_geometry(new_geometry, units=units)
            if len(new_atoms) != len(new_deck.geometry):
                raise ValueError(
                    f"New geometry has {len(new_atoms)} atoms, but previous calculation had {len(new_deck.geometry)} atoms."
                )
            for old_a, new_a in zip(new_deck.geometry, new_atoms):
                old_a.xyz = list(new_a.xyz)

        # Attach pre-converged embedded atom states to override initial basis starting orbitals
        if use_embedded_orbitals:
            embedded = self.get("embedded_atoms", {})
            for idx, atom in enumerate(new_deck.geometry):
                # Match by atom ID or positional index
                cand = None
                if atom.id and atom.id in embedded:
                    cand = embedded[atom.id]
                else:
                    cand = self.get_embedded_atom(idx)

                if cand is not None:
                    atom.embedded_state = cand

        # Apply control overrides
        if control_overrides:
            for k, v in control_overrides.items():
                if hasattr(new_deck.control, k):
                    setattr(new_deck.control, k, v)
                else:
                    new_deck.control.extra_namelist[k] = v

        return new_deck


def extract_calculation_state(
    core_state: Any,
    deck: Optional[Any] = None,
    converged: bool = False,
    cycle_history: Optional[List[Dict[str, float]]] = None,
) -> CalculationState:
    """
    Extracts the full CalculationState from the low-level Fortran workspace,
    decomposing the molecular solution into localized EmbeddedAtomStates.
    """
    norbt = int(core_state.norbt)
    mstot = int(core_state.mstot)
    natom = int(core_state.natom)
    ispin = 2 if bool(core_state.qopen) else 1
    n2orbt = (norbt * (norbt + 1)) // 2

    evalx = core_state.get_evalx(norbt, ispin)
    occx = core_state.get_occx(norbt, ispin)
    evecx = core_state.get_evecx(mstot, norbt, ispin)
    ovlp = core_state.get_ovlp(n2orbt, ispin)
    ewmat = core_state.get_ewmat(n2orbt, ispin)
    coord = core_state.get_coord(3, natom)
    charge = core_state.get_charge(3, natom)
    popul = core_state.get_popul(norbt, ispin)

    iatomx = core_state.get_iatomx(mstot)
    nx = core_state.get_nx(mstot)
    lx = core_state.get_lx(mstot)
    mlx = core_state.get_mlx(mstot)
    etax = core_state.get_etax(mstot)
    anormx = core_state.get_anormx(mstot)

    basis_info = {
        "iatomx": iatomx,
        "nx": nx,
        "lx": lx,
        "mlx": mlx,
        "etax": etax,
        "anormx": anormx,
    }

    # Decompose into EmbeddedAtomState for each atom
    embedded_atoms: Dict[str, EmbeddedAtomState] = {}

    # Gather atom IDs and elements from deck if available
    deck_atoms = deck.geometry if (deck is not None and hasattr(deck, "geometry")) else []

    current_orb_offset = 0
    for i in range(1, natom + 1):
        atom_idx = i - 1
        atom_mask = (iatomx == i)
        atom_bf_indices = np.where(atom_mask)[0]
        n_bfs = len(atom_bf_indices)

        # Coordinate in Bohrs
        atom_xyz = coord[:, atom_idx].tolist()

        # Charge & spin analysis:
        # Row 0: nuclear charge Z
        # Row 1: electronic charge q_elec
        # Row 2: net spin q_alpha - q_beta
        nuc_z = float(charge[0, atom_idx])
        elec_q = float(charge[1, atom_idx])
        net_sp = float(charge[2, atom_idx])
        net_ch = nuc_z - elec_q
        alpha_sp = (elec_q + net_sp) / 2.0
        beta_sp = (elec_q - net_sp) / 2.0

        # Atom ID and element
        if atom_idx < len(deck_atoms):
            d_atom = deck_atoms[atom_idx]
            atom_id = d_atom.id or f"{d_atom.element}{i}"
            elem_sym = d_atom.element
            is_fixed = d_atom.fixed
            is_virtual = d_atom.virtual
        else:
            elem_sym = "X"
            atom_id = f"Atom{i}"
            is_fixed = False
            is_virtual = False

        # Extract unique radial basis functions for this atom
        # Group by (n, l, zeta)
        radial_funcs: List[Dict[str, Any]] = []
        bf_to_radial_idx: Dict[int, int] = {}
        for local_bf_idx, global_bf_idx in enumerate(atom_bf_indices):
            n_val = int(nx[global_bf_idx])
            l_val = int(lx[global_bf_idx])
            z_val = float(etax[global_bf_idx])
            func_key = (n_val, l_val, round(z_val, 6))

            matched_idx = None
            for r_idx, r_func in enumerate(radial_funcs):
                if (r_func["n"], r_func["l"], round(r_func["zeta"], 6)) == func_key:
                    matched_idx = r_idx + 1
                    break
            if matched_idx is None:
                radial_funcs.append({"n": n_val, "l": l_val, "zeta": z_val})
                matched_idx = len(radial_funcs)
            bf_to_radial_idx[local_bf_idx] = matched_idx

        # Extract localized embedded orbitals for this atom
        # In PHATPSY, orbitals centered on atom i occupy the slice [current_orb_offset : current_orb_offset + n_bfs]
        atom_orbitals: List[Dict[str, Any]] = []
        atom_norb = n_bfs  # Each basis component has a corresponding local orbital

        for local_orb_idx in range(atom_norb):
            global_orb_idx = current_orb_offset + local_orb_idx
            if global_orb_idx >= norbt:
                break

            e_alpha = float(evalx[global_orb_idx, 0])
            e_beta = float(evalx[global_orb_idx, 1]) if ispin > 1 else e_alpha
            occ_alpha = float(occx[global_orb_idx, 0])
            occ_beta = float(occx[global_orb_idx, 1]) if ispin > 1 else occ_alpha
            pop_alpha = float(popul[global_orb_idx, 0])
            pop_beta = float(popul[global_orb_idx, 1]) if ispin > 1 else pop_alpha

            # Expansion coefficients over this atom's radial STOs
            coef_list: List[Tuple[int, int, float, float]] = []
            for local_bf_idx, global_bf_idx in enumerate(atom_bf_indices):
                c_alpha = float(evecx[global_bf_idx, global_orb_idx, 0])
                c_beta = float(evecx[global_bf_idx, global_orb_idx, 1]) if ispin > 1 else c_alpha

                if abs(c_alpha) > 1e-6 or abs(c_beta) > 1e-6:
                    isto = bf_to_radial_idx[local_bf_idx]
                    ml_val = int(mlx[global_bf_idx])
                    coef_list.append((isto, ml_val, round(c_alpha, 6), round(c_beta, 6)))

            atom_orbitals.append({
                "orbital_index": local_orb_idx + 1,
                "energy_alpha": e_alpha,
                "energy_beta": e_beta,
                "occ_alpha": occ_alpha,
                "occ_beta": occ_beta,
                "population_alpha": pop_alpha,
                "population_beta": pop_beta,
                "coefficients": coef_list,
            })

        current_orb_offset += atom_norb

        emb_state = EmbeddedAtomState(
            atom_id=atom_id,
            element=elem_sym,
            atomic_number=int(round(nuc_z)),
            xyz=atom_xyz,
            nuclear_charge=nuc_z,
            electronic_charge=elec_q,
            net_charge=net_ch,
            alpha_spin=alpha_sp,
            beta_spin=beta_sp,
            net_spin=net_sp,
            basis_functions=radial_funcs,
            orbitals=atom_orbitals,
            fixed=is_fixed,
            virtual=is_virtual,
        )
        embedded_atoms[atom_id] = emb_state

    # Potential / Kinetic energies & Virial ratio
    tot_pe = float(core_state.totpe)
    tot_ke = float(core_state.totke)
    virial = -tot_pe / tot_ke if abs(tot_ke) > 1e-12 else 0.0

    state_dict = {
        "converged": bool(converged),
        "cycles": int(core_state.ncycl),
        "total_energy": float(core_state.tmole),
        "kinetic_energy": tot_ke,
        "potential_energy": tot_pe,
        "nuclear_repulsion": float(core_state.zzpot),
        "delta_energy": float(core_state.deltae),
        "virial_ratio": virial,
        "norbt": norbt,
        "mstot": mstot,
        "ispin": ispin,
        "eigenvalues": evalx,
        "occupations": occx,
        "mo_coefficients": evecx,
        "overlap_packed": ovlp,
        "ewmat_packed": ewmat,
        "atom_charges": charge,
        "atom_coordinates": coord,
        "atom_populations": popul,
        "basis_info": basis_info,
        "embedded_atoms": embedded_atoms,
        "cycle_history": list(cycle_history) if cycle_history is not None else [],
    }

    try:
        from phatpsy.subsystem import decompose_subsystem_energies
        decomp = decompose_subsystem_energies(state_dict, total_energy=state_dict["total_energy"])
        state_dict["subsystem_decomposition"] = decomp.to_dict()
    except Exception:
        pass

    return CalculationState(state_dict, deck=deck)
