"""Test-suite configuration.

BASELINE_QUARANTINE (temporary, AGENTS.md section 6, ROADMAP "Baseline Re-establishment")
-----------------------------------------------------------------------------------------
Tests listed here assert converged results or frozen values on the *default* SCF path.
That path (a) reached "convergence" only by hitting the cycle limit or loose tolerances,
(b) is nondeterministic for several systems, and (c) before the first-atom spin fix started
spin-down first atoms as spin-up.  They are marked xfail (non-strict, they still run) so the
suite is green while the baselines are re-established.  A quarantined test that passes shows
up as XPASS, which is the cue to remove its entry.  Do not add entries without a ROADMAP note.
"""
import os
import pytest

BASELINE_QUARANTINE_REASON = (
    "Baseline re-establishment (AGENTS.md section 6): relies on default-path convergence "
    "or frozen values captured under false convergence / ignored first-atom spin."
)

BASELINE_QUARANTINE = {
    "tests/test_basis_adf.py::test_adf_n2_scf_convergence",
    "tests/test_benchmarks.py::test_n2_benchmark",
    "tests/test_chemisorption.py::test_evaluate_no_ni100_benchmark",
    "tests/test_coords_pes_optimize.py::test_scan_1d_n2_rigid_stretch",
    "tests/test_correlation_and_dispersion.py::test_deck_with_pbe_xc_and_dispersion",
    "tests/test_correlation_and_dispersion.py::test_milestone8_diatomic_energy_stability_benchmark",
    "tests/test_exchange_dft.py::test_exchange_dft_deck_run",
    "tests/test_exchange_dft.py::test_pbe_deck_run",
    "tests/test_modelpot.py::test_dynamic_exponents_deck",
    "tests/test_pyscf_bridge.py::test_system_parity_n2",
    "tests/test_python_driver.py::test_driver_run_file_and_json_export",
    "tests/test_python_driver.py::test_driver_run_from_yaml",
    "tests/test_python_driver.py::test_diis_charge_acceleration",
    "tests/test_state_and_restart.py::test_calculation_state_extraction_and_parity",
    "tests/test_state_and_restart.py::test_embedded_atom_state_extraction",
    "tests/test_state_and_restart.py::test_warm_restart_on_modified_geometry",
    "tests/test_subsystem_energy.py::test_subsystem_energy_conservation_pure_cluster",
    "tests/test_subsystem_energy.py::test_cycle_history_and_diagnostics",
    "tests/test_validation_baselines.py::test_n2_physics_invariants",
    "tests/test_validation_baselines.py::test_bulk_ni_unit_pk_scale_convergence",
    "tests/test_validation_baselines.py::test_n2_frozen_baseline",
    "tests/test_validation_baselines.py::test_h2o_frozen_baseline",
    "tests/test_validation_baselines.py::test_bulk_ni_frozen_baseline",
    "tests/test_validation_baselines.py::test_ni_slab_3layer_frozen_baseline",
}


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run @pytest.mark.slow tests in addition to the default fast suite.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running multi-center numerical quadrature or slab embedding tests",
    )


def pytest_collection_modifyitems(config, items):
    markexpr = (getattr(config.option, "markexpr", "") or "").strip()
    explicit_slow_expr = ("slow" in markexpr) and ("not slow" not in markexpr)
    run_slow = (
        bool(config.getoption("--runslow"))
        or os.environ.get("PHATPSY_RUN_SLOW", "") == "1"
        or explicit_slow_expr
    )
    skip_slow = pytest.mark.skip(
        reason="slow numerical quadrature/slab test; pass --runslow or -m slow to run"
    )
    marker = pytest.mark.xfail(reason=BASELINE_QUARANTINE_REASON, strict=False)
    for item in items:
        if item.nodeid in BASELINE_QUARANTINE:
            item.add_marker(marker)
        if not run_slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
