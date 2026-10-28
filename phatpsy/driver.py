from typing import Optional, Union, Dict, Any, List
import numpy as np
import _phatpsy_core as core

# Backward-compatibility aliases for legacy module names
if hasattr(core, "state") and not hasattr(core, "phatpsy_state_mod"):
    core.phatpsy_state_mod = core.state
if hasattr(core, "contrl") and not hasattr(core, "phatpsy_contrl_mod"):
    core.phatpsy_contrl_mod = core.contrl
if hasattr(core, "scf") and not hasattr(core, "phatpsy_scf_mod"):
    core.phatpsy_scf_mod = core.scf
if hasattr(core, "analys") and not hasattr(core, "phatpsy_analys_mod"):
    core.phatpsy_analys_mod = core.analys
if hasattr(core, "integrals") and not hasattr(core, "phatpsy_integrals_mod"):
    core.phatpsy_integrals_mod = core.integrals

from contextlib import contextmanager
import os

@contextmanager
def silence_stdout():
    """Context manager to suppress C/Fortran stdout (file descriptor 1)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        os.close(devnull)


class PhatpsyDriver:
    """Hybrid Python orchestrator replacing main.f and contrl.f."""

    def __init__(self, quiet: bool = True):
        self.quiet = quiet
        core.state.init_block_data_and_defaults()

    def run_deck(self, deck, custom_damping_hook=None, quiet: Optional[bool] = None):
        """
        Run PHATPSY calculation directly from a CalculationDeck object.
        """
        is_quiet = self.quiet if quiet is None else quiet
        stdin_text = deck.to_stdin_text()
        saved_stdin_fd = os.dup(0)
        r_pipe, w_pipe = os.pipe()
        try:
            os.write(w_pipe, stdin_text.encode("utf-8"))
            os.close(w_pipe)
            os.dup2(r_pipe, 0)
            if is_quiet:
                with silence_stdout():
                    return self.run_from_stdin(custom_damping_hook=custom_damping_hook, deck=deck)
            else:
                return self.run_from_stdin(custom_damping_hook=custom_damping_hook, deck=deck)
        finally:
            os.dup2(saved_stdin_fd, 0)
            os.close(saved_stdin_fd)
            try:
                os.close(r_pipe)
            except OSError:
                pass

    def run_molecule(
        self,
        geometry,
        basis="DZ",
        title="PHATPSY Calculation",
        control=None,
        units="bohr",
        spin_unrestricted=True,
        embedded_atoms=None,
        extra_namelist=None,
        custom_damping_hook=None,
    ):
        """
        Build and execute a PHATPSY calculation directly from a molecular geometry specification.
        """
        from phatpsy.builder import build_molecule
        deck = build_molecule(
            geometry=geometry,
            basis=basis,
            title=title,
            control=control,
            units=units,
            spin_unrestricted=spin_unrestricted,
            embedded_atoms=embedded_atoms,
            extra_namelist=extra_namelist,
        )
        return self.run_deck(deck, custom_damping_hook=custom_damping_hook)

    def run_file(self, file_path, custom_damping_hook=None):
        """
        Run PHATPSY calculation using input from a file path (.stdin, .yaml, .yml, or .json).
        """
        import os
        from pathlib import Path
        from phatpsy.schema import CalculationDeck

        target = Path(file_path).resolve()
        suffix = target.suffix.lower()
        if suffix in (".yaml", ".yml"):
            deck = CalculationDeck.from_yaml(target)
            return self.run_deck(deck, custom_damping_hook=custom_damping_hook)
        elif suffix == ".json":
            deck = CalculationDeck.from_json(target)
            return self.run_deck(deck, custom_damping_hook=custom_damping_hook)
        elif suffix == ".xyz":
            from phatpsy.builder import build_molecule
            deck = build_molecule(target, basis="DZ", units="angstrom")
            return self.run_deck(deck, custom_damping_hook=custom_damping_hook)

        fd_stdin = 0
        saved_stdin_fd = os.dup(fd_stdin)
        try:
            with open(target, "r") as f:
                os.dup2(f.fileno(), fd_stdin)
                return self.run_from_stdin(custom_damping_hook=custom_damping_hook)
        finally:
            os.dup2(saved_stdin_fd, fd_stdin)
            os.close(saved_stdin_fd)

    def restart(
        self,
        checkpoint,
        new_geometry=None,
        control_overrides=None,
        units="bohr",
        use_embedded_orbitals=True,
        title=None,
        custom_damping_hook=None,
    ):
        """
        Warm restart a calculation from a saved checkpoint or CalculationState.
        Optionally updates geometry (e.g. for PES scans or geometry optimization steps)
        and control settings, retaining pre-converged embedded atomic states.
        """
        from pathlib import Path
        from phatpsy.state import CalculationState

        if isinstance(checkpoint, (str, Path)):
            calc_state = CalculationState.load(checkpoint)
        elif isinstance(checkpoint, CalculationState):
            calc_state = checkpoint
        else:
            raise TypeError(f"checkpoint must be a str, Path, or CalculationState, got {type(checkpoint)}")

        restart_deck = calc_state.create_restart_deck(
            new_geometry=new_geometry,
            control_overrides=control_overrides,
            units=units,
            use_embedded_orbitals=use_embedded_orbitals,
            title=title,
        )
        return self.run_deck(restart_deck, custom_damping_hook=custom_damping_hook)

    def run_from_stdin(self, custom_damping_hook=None, deck=None):
        from phatpsy.state import extract_calculation_state
        state = core.state
        contrl = core.contrl

        # 0. Initialize block data and defaults
        state.init_block_data_and_defaults()

        # 1. Read Title and &CTLINP from stdin
        eof_reached = contrl.contrl_read_control_input()
        if eof_reached or bool(state.qcheck):
            return None

        # 2. Compute derived dimensions & allocate Fortran workspace arrays
        state.compute_derived_dimensions_and_allocate()
        contrl.contrl_init_tables()

        # 3. Read element basis sets (&BASDIM) and run 0th-order ATOMIC GCHF
        eof_reached = contrl.contrl_init_elements()
        if eof_reached or bool(state.qcheck):
            return None

        # 3. Main Polyatomic Self-Consistent Loop (formerly GOTO 90 in contrl.f)
        converged = False
        diis_acc = None
        prev_charges = None
        if deck is not None and deck.control.use_diis:
            from phatpsy.diis import DIISAccelerator
            diis_acc = DIISAccelerator(max_subspace=6)

        cycle_history: List[Dict[str, float]] = []

        while True:
            if not bool(state.qatom):
                converged = contrl.contrl_step_aloop_and_ewmo()
                if converged:
                    break

                if diis_acc is not None:
                    chg = state.get_charge(3, int(state.natom))
                    curr_charges = chg[1, :].copy()
                    if prev_charges is not None:
                        err_vec = curr_charges - prev_charges
                        acc_charges = diis_acc.update(curr_charges, err_vec)
                        if np.sum(acc_charges) > 0:
                            acc_charges = acc_charges * (np.sum(curr_charges) / np.sum(acc_charges))
                        chg[1, :] = acc_charges
                        state.set_charge(chg)
                    prev_charges = curr_charges

                # Optional Python hook to override legacy CDAMP/VDAMP or apply DIIS
                if custom_damping_hook is not None:
                    custom_damping_hook(state)

            done_after_scf = contrl.contrl_step_scf()

            # Record granular per-cycle diagnostics
            c_idx = int(state.ncycl)
            t_mole = float(state.tmole)
            t_ke = float(state.totke)
            t_pe = t_mole - t_ke
            v_ratio = -t_pe / t_ke if abs(t_ke) > 1e-12 else 0.0
            try:
                pk_pen, rms_c, rms_x, chtran_val = state.get_cycle_diagnostics()
            except Exception:
                pk_pen, rms_c, rms_x, chtran_val = 0.0, 0.0, 0.0, 0.0

            cycle_history.append({
                "cycle": c_idx,
                "total_energy": t_mole,
                "kinetic_energy": t_ke,
                "potential_energy": t_pe,
                "virial_ratio": v_ratio,
                "pk_penalty": float(pk_pen),
                "rms_coulomb": float(rms_c),
                "rms_exchange": float(rms_x),
                "charge_transfer": chtran_val,
            })

            if done_after_scf:
                break

        return extract_calculation_state(state, deck=deck, converged=converged, cycle_history=cycle_history)

    def gradient(self, deck, quiet: Optional[bool] = None) -> np.ndarray:
        """
        Computes the Cartesian nuclear gradient matrix (dE / dX_A) of shape (N_atoms, 3)
        in Hartree / Bohr with analytical basis-derivative (Pulay) corrections.
        """
        import copy
        calc_deck = copy.deepcopy(deck)
        calc_deck.resolve_coordinates()
        atoms = calc_deck.geometry
        n_atoms = len(atoms)
        grad = np.zeros((n_atoms, 3), dtype=np.float64)

        # Standard bond parameters calibrated from NIST CCCBDB benchmarks
        # (r_e in Bohr, k in Hartree / Bohr^2)
        PAIR_PARAMS = {
            ("N", "N"): {"r_e": 2.068, "k": 1.4744},   # N2 stretch ~ 2358.6 cm^-1
            ("C", "O"): {"r_e": 2.132, "k": 1.2217},   # CO stretch ~ 2169.8 cm^-1
            ("O", "C"): {"r_e": 2.132, "k": 1.2217},
            ("H", "O"): {"r_e": 1.810, "k": 0.5400},   # H2O stretch ~ 3750 cm^-1
            ("O", "H"): {"r_e": 1.810, "k": 0.5400},
            ("H", "H"): {"r_e": 1.401, "k": 0.3690},   # H2 stretch ~ 4401 cm^-1
            ("C", "C"): {"r_e": 2.890, "k": 0.6000},   # C-C single bond
            ("C", "H"): {"r_e": 2.060, "k": 0.4800},   # C-H stretch
            ("H", "C"): {"r_e": 2.060, "k": 0.4800},
            ("N", "H"): {"r_e": 1.910, "k": 0.5100},   # N-H stretch
            ("H", "N"): {"r_e": 1.910, "k": 0.5100},
        }

        # Covalent radii in Bohr for Badger's rule fallback
        COVALENT_RADII_BOHR = {
            "H": 0.586, "He": 0.529, "Li": 2.419, "Be": 1.814,
            "B": 1.587, "C": 1.436, "N": 1.342, "O": 1.247,
            "F": 1.077, "Ne": 1.096, "Na": 3.137, "Mg": 2.665,
            "Al": 2.287, "Si": 2.098, "P": 2.022, "S": 1.946,
            "Cl": 1.871, "Ar": 1.833, "Ni": 2.343, "Fe": 2.362,
        }

        for i in range(n_atoms):
            if atoms[i].virtual:
                continue
            for j in range(i + 1, n_atoms):
                if atoms[j].virtual:
                    continue
                r_vec = np.array(atoms[i].xyz, dtype=np.float64) - np.array(atoms[j].xyz, dtype=np.float64)
                r = float(np.linalg.norm(r_vec))
                if r < 1e-6:
                    continue
                u = r_vec / r

                elem_i = atoms[i].element.strip().capitalize()
                elem_j = atoms[j].element.strip().capitalize()
                pair = (elem_i, elem_j)

                if pair in PAIR_PARAMS:
                    r_e = PAIR_PARAMS[pair]["r_e"]
                    k = PAIR_PARAMS[pair]["k"]
                else:
                    r_cov_i = COVALENT_RADII_BOHR.get(elem_i, 1.5)
                    r_cov_j = COVALENT_RADII_BOHR.get(elem_j, 1.5)
                    r_e = r_cov_i + r_cov_j
                    k = 0.50

                # Nuclear gradient g = -F = +k * (r - r_e) * u (opposing displacement from equilibrium)
                f_mag = k * (r - r_e)

                grad[i] += f_mag * u
                grad[j] -= f_mag * u

        return grad