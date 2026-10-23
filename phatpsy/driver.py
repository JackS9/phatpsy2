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

class PhatpsyDriver:
    """Hybrid Python orchestrator replacing main.f and contrl.f."""

    def __init__(self):
        core.state.init_block_data_and_defaults()

    def run_deck(self, deck, custom_damping_hook=None):
        """
        Run PHATPSY calculation directly from a CalculationDeck object.
        """
        import os
        stdin_text = deck.to_stdin_text()
        saved_stdin_fd = os.dup(0)
        r_pipe, w_pipe = os.pipe()
        try:
            os.write(w_pipe, stdin_text.encode("utf-8"))
            os.close(w_pipe)
            os.dup2(r_pipe, 0)
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
        while True:
            if not bool(state.qatom):
                converged = contrl.contrl_step_aloop_and_ewmo()
                if converged:
                    break

                # Optional Python hook to override legacy CDAMP/VDAMP or apply DIIS
                if custom_damping_hook is not None:
                    custom_damping_hook(state)

            done_after_scf = contrl.contrl_step_scf()
            if done_after_scf:
                break

        return extract_calculation_state(state, deck=deck, converged=converged)