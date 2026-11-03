"""Unit tests for the SCF cycle-limit continuation policy (phatpsy.scf_control)."""

import math

from phatpsy.scf_control import ScfExtensionPolicy, judge_continuation

CONVRG = 1.0e-4


def _geometric(rate, d0=1.0, n=30, e0=-100.0):
    """Energies whose successive changes shrink as d0 * rate**k."""
    e, out = e0, []
    for k in range(n):
        e -= d0 * rate**k
        out.append(e)
    return out


def test_geometric_convergence_is_extended():
    energies = _geometric(0.8, n=30)
    dec = judge_continuation(energies, CONVRG, cycles_used=30, base_max_cycles=30)
    assert dec.should_continue
    # d_last ~ 0.8**29 ~ 1.6e-3; need ~ log(1e-4/1.6e-3)/log(0.8) ~ 12 more cycles
    assert 12 <= dec.extra_cycles <= 60
    assert math.isclose(dec.contraction_rate, 0.8, rel_tol=0.02)


def test_wild_oscillation_aborts():
    energies = [-100.0 + (0.5 if k % 2 else -0.5) for k in range(30)]
    dec = judge_continuation(energies, CONVRG, cycles_used=30, base_max_cycles=30)
    assert not dec.should_continue
    assert "oscillat" in dec.reason


def test_stagnation_aborts():
    energies = [-100.0 - 0.01 * k for k in range(30)]  # constant step, no contraction
    dec = judge_continuation(energies, CONVRG, cycles_used=30, base_max_cycles=30)
    assert not dec.should_continue


def test_divergence_aborts():
    energies = _geometric(1.1, d0=1e-3, n=30)
    dec = judge_continuation(energies, CONVRG, cycles_used=30, base_max_cycles=30)
    assert not dec.should_continue


def test_very_slow_convergence_aborts():
    energies = _geometric(0.995, n=30)
    dec = judge_continuation(energies, CONVRG, cycles_used=30, base_max_cycles=30)
    assert not dec.should_continue
    assert "slow" in dec.reason


def test_prediction_beyond_extension_limit_aborts():
    policy = ScfExtensionPolicy(max_extension_cycles=5)
    energies = _geometric(0.8, n=30)
    dec = judge_continuation(energies, CONVRG, 30, 30, policy)
    assert not dec.should_continue


def test_hard_cap_aborts():
    energies = _geometric(0.8, n=30)
    dec = judge_continuation(energies, CONVRG, cycles_used=150, base_max_cycles=30)
    assert not dec.should_continue
    assert "cap" in dec.reason


def test_non_finite_energy_aborts():
    energies = _geometric(0.8, n=10) + [float("nan")]
    dec = judge_continuation(energies, CONVRG, 11, 11)
    assert not dec.should_continue


def test_disabled_policy_always_aborts():
    energies = _geometric(0.8, n=30)
    dec = judge_continuation(energies, CONVRG, 30, 30, ScfExtensionPolicy(enabled=False))
    assert not dec.should_continue


def test_too_few_cycles_aborts():
    dec = judge_continuation([-1.0, -1.1], CONVRG, 2, 2)
    assert not dec.should_continue



def test_period4_limit_cycle_aborts():
    """CO-like orbit: three small steps then a jump, repeating without decay."""
    steps = [-0.0014, -0.0006, 0.0089, 0.0041, 0.0022, -0.0079, 0.0056, 0.0025]
    e, energies = -112.35, []
    for k in range(60):
        e += steps[k % len(steps)]
        energies.append(e)
    dec = judge_continuation(energies, CONVRG, cycles_used=60, base_max_cycles=60)
    assert not dec.should_continue
    assert "limit cycle" in dec.reason or "oscillat" in dec.reason


def test_damped_oscillation_still_continues():
    """Alternating-sign changes that decay geometrically are converging."""
    e, energies = -100.0, []
    for k in range(40):
        e += 0.3 * ((-0.8) ** k)
        energies.append(e)
    dec = judge_continuation(energies, CONVRG, cycles_used=40, base_max_cycles=40)
    assert dec.should_continue


# --- AdaptiveDamping -------------------------------------------------------

from phatpsy.scf_control import AdaptiveDamping  # noqa: E402


def _drive(ctrl, energies):
    """Feed a growing energy history to the controller, cycle by cycle."""
    hist, v = [], ctrl.vdamp
    for e in energies:
        hist.append(e)
        v = ctrl.update(hist)
    return v


def test_adaptive_damping_raises_vdamp_on_oscillation():
    ctrl = AdaptiveDamping(0.5)
    energies = [-100.0 + (0.2 if k % 2 else -0.2) for k in range(30)]
    v = _drive(ctrl, energies)
    assert v > 0.9
    assert v <= 0.99


def test_adaptive_damping_raises_vdamp_on_growing_changes():
    ctrl = AdaptiveDamping(0.5)
    e, energies = -100.0, []
    for k in range(30):
        e -= 1e-3 * 1.4**k
        energies.append(e)
    assert _drive(ctrl, energies) > 0.5


def test_adaptive_damping_never_exceeds_vmax_or_drops_below_start():
    ctrl = AdaptiveDamping(0.6, vmax=0.95)
    _drive(ctrl, [-100.0 + (0.3 if k % 2 else -0.3) for k in range(60)])
    assert ctrl.vdamp <= 0.95
    ctrl2 = AdaptiveDamping(0.6)
    e, energies = -100.0, []
    for k in range(60):
        e -= 0.5**k
        energies.append(e)
    _drive(ctrl2, energies)
    assert ctrl2.vdamp >= 0.6


def test_adaptive_damping_respects_cooldown():
    ctrl = AdaptiveDamping(0.5, cooldown=5)
    _drive(ctrl, [-100.0 + (0.2 if k % 2 else -0.2) for k in range(12)])
    # At most one adjustment fits in 12 cycles with a 5-cycle cooldown after
    # the first full window, so vdamp can only have been halved a couple of times.
    assert len(ctrl.history) <= 4


def test_adaptive_damping_leaves_steady_state_alone():
    ctrl = AdaptiveDamping(0.7)
    e, energies = -100.0, []
    for k in range(40):
        e -= 0.9 * 0.95**k * 1e-2  # slow smooth contraction: neither growth nor fast decay
        energies.append(e)
    v = _drive(ctrl, energies)
    assert 0.7 <= v <= 0.99
