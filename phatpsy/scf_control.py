"""SCF cycle-limit policy: decide whether to extend or abort at the cycle limit.

Reaching ``max_cycles`` is never convergence. When the Fortran cycle loop stops
for that reason, :func:`judge_continuation` inspects the per-cycle total-energy
history and decides between:

* ``continue``: the energy changes are shrinking geometrically at a rate that
  predicts convergence within a bounded number of additional cycles, or
* ``abort``: the iteration is diverging, wildly oscillating, stagnating, or
  converging too slowly to be worth continuing. The driver then returns a
  result with ``converged=False`` and the reason recorded.

The convergence criterion itself is unchanged: ``|E_n - E_{n-1}| < CONVRG``
(see ``contrl_step_aloop_and_ewmo`` in ``src/contrl.f90``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ScfExtensionPolicy:
    """Tunable limits for extending an SCF run that hit its cycle limit.

    Attributes:
        enabled: If False, hitting the cycle limit always aborts as not converged.
        window: Number of most recent energy differences analysed.
        max_extension_cycles: Largest single extension granted, in cycles.
        max_total_cycles_factor: Hard cap on total cycles, as a multiple of the
            originally requested ``max_cycles``.
        max_contraction_rate: Largest acceptable per-cycle contraction factor
            ``r`` of ``|dE|`` (``|dE_k| ~ r**k``). Values near 1 mean stagnation.
        oscillation_sign_fraction: Fraction of sign changes in successive signed
            energy differences above which the run counts as oscillating.
        safety_factor: Multiplier applied to the predicted remaining cycles.
        limit_cycle_sign_fraction: Minimum fraction of sign changes in successive
            energy differences for the limit-cycle check to apply.
        limit_cycle_range_ratio: The energy range over the last window must fall
            below this fraction of the previous window's range, otherwise the run
            is judged to be in a limit cycle.
        adaptive_damping: If True, the driver adjusts the potential damping
            ``VDAMP`` each cycle with :class:`AdaptiveDamping` (default False).
        damping_max: Upper bound on the adaptive ``VDAMP`` (must be < 1).
    """

    enabled: bool = True
    window: int = 10
    max_extension_cycles: int = 200
    max_total_cycles_factor: float = 5.0
    max_contraction_rate: float = 0.98
    oscillation_sign_fraction: float = 0.6
    safety_factor: float = 1.5
    limit_cycle_sign_fraction: float = 0.15
    limit_cycle_range_ratio: float = 0.5
    adaptive_damping: bool = False
    damping_max: float = 0.99


@dataclass(frozen=True)
class ContinuationDecision:
    """Outcome of :func:`judge_continuation`."""

    action: str  # "continue" or "abort"
    reason: str
    extra_cycles: int = 0
    contraction_rate: float = float("nan")

    @property
    def should_continue(self) -> bool:
        return self.action == "continue"


def _abort(reason: str, rate: float = float("nan")) -> ContinuationDecision:
    return ContinuationDecision("abort", reason, 0, rate)


def judge_continuation(
    energies: Sequence[float],
    convergence: float,
    cycles_used: int,
    base_max_cycles: int,
    policy: ScfExtensionPolicy = ScfExtensionPolicy(),
) -> ContinuationDecision:
    """Decide whether an SCF run that hit its cycle limit should continue.

    Args:
        energies: Total energy after each completed cycle (Hartree), in order.
        convergence: Energy convergence threshold ``CONVRG`` (Hartree).
        cycles_used: Cycles completed so far.
        base_max_cycles: The originally requested cycle limit.
        policy: Extension limits.

    Returns:
        A :class:`ContinuationDecision`.
    """
    if not policy.enabled:
        return _abort("cycle limit reached (extension disabled)")

    hard_cap = int(math.ceil(policy.max_total_cycles_factor * max(base_max_cycles, 1)))
    if cycles_used >= hard_cap:
        return _abort(f"hard cycle cap of {hard_cap} reached")

    e: List[float] = [float(x) for x in energies]
    if len(e) < 4:
        return _abort("too few cycles to judge convergence trend")
    if not all(math.isfinite(x) for x in e):
        return _abort("non-finite total energy")

    signed = [e[i] - e[i - 1] for i in range(1, len(e))]
    signed = signed[-policy.window:]
    mags = [abs(d) for d in signed]
    if len(mags) < 3:
        return _abort("too few cycles to judge convergence trend")

    floor = 1e-14
    log_mags = [math.log(max(m, floor)) for m in mags]
    n = len(log_mags)
    xs = list(range(n))
    xbar = sum(xs) / n
    ybar = sum(log_mags) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, log_mags)) / sxx
    rate = math.exp(slope)

    flips = sum(1 for a, b in zip(signed, signed[1:]) if a * b < 0.0)
    flip_fraction = flips / max(len(signed) - 1, 1)

    # Limit-cycle check: the energy keeps visiting the same range window after
    # window (period-2, period-4, ... orbits such as charge sloshing). A fit of
    # |dE| inside one window can look slightly contracting for such orbits, so
    # compare the energy range of the last window with that of the one before.
    w = policy.window
    if len(e) >= 2 * w + 1 and flip_fraction >= policy.limit_cycle_sign_fraction:
        recent = e[-(w + 1):]
        prev = e[-(2 * w + 1):-w]
        range_recent = max(recent) - min(recent)
        range_prev = max(prev) - min(prev)
        if (
            range_recent > convergence
            and range_recent >= policy.limit_cycle_range_ratio * range_prev
        ):
            if flip_fraction >= policy.oscillation_sign_fraction:
                return _abort("oscillating without decay", rate)
            return _abort(
                f"periodic limit cycle (energy range {range_recent:.2e} Ha not shrinking)", rate
            )

    if rate >= 1.0:
        if flip_fraction >= policy.oscillation_sign_fraction:
            return _abort("oscillating without decay", rate)
        return _abort("energy changes not decreasing (diverging or stagnating)", rate)
    if rate > policy.max_contraction_rate:
        if flip_fraction >= policy.oscillation_sign_fraction:
            return _abort("oscillating with negligible decay", rate)
        return _abort("converging too slowly", rate)

    d_last = max(mags[-1], floor)
    # Smooth the last magnitude with the fitted trend to avoid single-cycle noise.
    d_fit = math.exp(ybar + slope * (n - 1 - xbar))
    d_ref = max(d_last, d_fit)
    if d_ref < convergence:
        remaining = 1
    else:
        remaining = int(math.ceil(math.log(convergence / d_ref) / math.log(rate)))
    extra = int(math.ceil(policy.safety_factor * remaining)) + 1
    extra = min(extra, hard_cap - cycles_used)
    if extra > policy.max_extension_cycles or remaining > policy.max_extension_cycles:
        return _abort(
            f"predicted {remaining} more cycles exceeds extension limit of {policy.max_extension_cycles}",
            rate,
        )
    if extra < 1:
        return _abort(f"hard cycle cap of {hard_cap} reached", rate)
    return ContinuationDecision(
        "continue",
        f"energy changes contracting at rate {rate:.3f}/cycle; ~{remaining} more cycles predicted",
        extra,
        rate,
    )



class AdaptiveDamping:
    """Adaptive potential damping (``VDAMP``) driven by the energy history.

    ``VDAMP`` is the fraction of the previous cycle's embedding potentials and
    Phillips-Kleinman projector kept when the new ones are mixed in (the new
    contribution is weighted by ``1 - VDAMP``). The energy-weighted projector is
    built from the previous cycle's eigenvalues, so the cycle map can have a gain
    well above 1 and a fixed ``VDAMP`` that is too small oscillates in a slow
    limit cycle. This controller raises ``VDAMP`` (shrinks the update fraction)
    when it sees oscillation or growth in ``|dE|``, and relaxes it again when
    convergence is smooth.

    Args:
        vdamp: Starting ``VDAMP`` (normally the deck's ``potential_damp``).
        vmax: Largest ``VDAMP`` allowed (< 1).
        window: Number of recent energy differences examined.
        cooldown: Minimum number of cycles between adjustments.
    """

    def __init__(self, vdamp: float, vmax: float = 0.99, window: int = 6, cooldown: int = 3):
        self.vdamp = float(vdamp)
        self.vmin = float(vdamp)
        self.vmax = float(vmax)
        self.window = int(window)
        self.cooldown = int(cooldown)
        self._since_change = 0
        self.history: List[float] = [self.vdamp]

    def update(self, energies: Sequence[float]) -> float:
        """Return the ``VDAMP`` to use for the next cycle."""
        self._since_change += 1
        n = 2 * (self.window // 2)
        if len(energies) < n + 1 or self._since_change < self.cooldown:
            return self.vdamp
        d = [energies[i] - energies[i - 1] for i in range(len(energies) - n, len(energies))]
        old = sum(abs(x) for x in d[: n // 2]) / (n // 2)
        new = sum(abs(x) for x in d[n // 2:]) / (n // 2)
        flips = sum(1 for a, b in zip(d, d[1:]) if a * b < 0.0)
        flip_fraction = flips / max(len(d) - 1, 1)
        growth = new / old if old > 0.0 else 1.0

        step = 1.0 - self.vdamp  # fraction of the new potentials mixed in
        if flip_fraction >= 0.5 or growth > 1.2:
            step *= 0.5          # oscillating or growing: halve the update fraction
        elif growth < 0.7 and flip_fraction <= 0.2:
            step *= 1.25         # smooth contraction: relax toward faster updates
        else:
            return self.vdamp
        self.vdamp = min(self.vmax, max(self.vmin, 1.0 - step))
        self._since_change = 0
        self.history.append(self.vdamp)
        return self.vdamp
