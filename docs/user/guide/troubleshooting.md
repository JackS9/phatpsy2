# Troubleshooting & Best Practices

This guide provides diagnostic procedures for resolving self-consistent field (SCF) convergence issues, performing physical sanity checks, and avoiding common pitfalls.

---

## SCF Convergence Troubleshooting

### Symptom 1: Energy Oscillates Between Cycles (Charge Sloshing)

In open-shell systems, transition metal clusters (e.g. $\text{Ag}_{13}$), or systems with small HOMO-LUMO gaps, electrons may oscillate back and forth between degenerate or near-degenerate orbitals.

**Remedies:**

1. **Increase Damping:** Reduce `charge_damp` and `potential_damp` from the default `0.5` down to `0.2` or `0.3`:
   ```yaml
   control:
     charge_damp: 0.25
     potential_damp: 0.3
     charge_accel: 1.0
   ```
2. **Enable Isolated Atom Pre-Solving:** Ensure `separate_atoms_first_cycle: true` is set. This seeds the initial cycle with accurate spherical atomic densities rather than an unphysical uniform overlap guess.
3. **Seed Initial Spin Projections:** Set alternating `spin: up` and `spin: down` across adjacent atoms using `deck.assign_alternating_spin()` or `phatpsy build -s`.

---

### Symptom 2: Calculation Reaches `max_cycles` Without Converging

If the energy change $\Delta E$ decreases steadily but does not drop below $1.0 \times 10^{-4}$ within 100 cycles:

**Remedies:**

1. **Increase Cycle Limit:** Set `max_cycles: 150` or `200`.
2. **Tune Convergence Threshold:** For large slabs or cluster geometries undergoing preliminary relaxation, a threshold of `convergence: 1.0e-3` is often sufficient.

---

### Symptom 3: Sudden Energy Divergence or NaN

If the total energy becomes positive or outputs `NaN`:

**Remedies:**

1. **Check Coordinate Units:** Coordinates in YAML decks are in **Bohrs**. If you input Ångströms without converting, atoms will be spaced roughly half their intended physical distance, causing massive nuclear repulsion.
2. **Inspect Overlapping Centers:** Ensure no two atomic centers share identical Cartesian coordinates.

---

## Physical Sanity Checks

Always verify that converged states satisfy physical boundary conditions:

### 1. The Virial Ratio ($-V / T$)

According to the quantum mechanical virial theorem, any bound Coulombic molecular system at its equilibrium geometry satisfies:

$$\eta = -\frac{V}{T} \approx 2.000$$

* **Acceptable Range:** $1.95 \le \eta \le 2.05$.
* **Warning Signs:** If $\eta < 1.80$ or $\eta > 2.20$, the geometry is far from equilibrium, or atoms have unphysical nuclear separations.

### 2. Integrated Electron Count

Compute the numerical volume integral of the total electron density:

```python
total_electrons = density_grid.integrate()
print(f"Computed Electrons: {total_electrons:.3f}")
```

For diatomic nitrogen ($N_2$), the integrated value should equal $14.0 \pm 0.05$ electrons with standard grid padding and spacing.

---

## Known Limitations & Gotchas

1. **One Calculation Per Driver:**
   Because the native Fortran core manages global state per calculation execution, instantiate a new `PhatpsyDriver()` instance for each run, or execute calculations in separate worker processes when running high-throughput loops.
2. **Fortran Output Redirection:**
   The Fortran core writes directly to C standard output (`stdout`, file descriptor 1). In Jupyter notebooks, use the `suppress_stdout()` context manager if you wish to suppress raw terminal output in notebook cells.
3. **Gaussian Cube Precision:**
   Gaussian Cube files format numbers to 5 or 6 decimal places ($1.23456\times 10^{-2}$). When round-tripping Cube files in automated tests, use `np.isclose(rtol=1e-4)` rather than exact floating-point equality.
4. **Memory Usage for Large Grids:**
   Grid memory scales as $\mathcal{O}(N_x N_y N_z)$. A $200 \times 200 \times 200$ grid contains $8 \times 10^6$ points ($\approx 64\text{ MB}$). Evaluating Electrostatic Potentials (ESP) requires calculating distances to every grid point for every atom; for slabs with $> 50$ atoms, use `spacing=0.25` Bohrs or larger.

---

## Frequently Asked Questions (FAQ)

??? question "How do I know whether to use SZ, DZ, or THAKKAR basis?"
    Use **DZ (Double-Zeta)** for standard production calculations and molecular properties. Use **SZ (Single-Zeta)** for initial structural screenings or large cluster explorations where speed is paramount. Use **THAKKAR** when capturing diffuse valence polarization is critical.

??? question "Can I run periodic boundary condition (PBC) calculations?"
    PHATPSY 2.0 uses a cluster/slab approach with non-local projection operators rather than plane-wave PBCs. Slabs are constructed with finite lateral dimensions ($2 \times 2$ or $3 \times 3$ supercells) with optional virtual boundary atoms to eliminate edge boundary dipole artifacts.

??? question "How do I restart a failed or interrupted calculation?"
    If you saved the calculation results (`res.save("checkpoint.json")`), you can perform a warm restart using `driver.restart(checkpoint)` or export a restart deck using `res.create_restart_deck()`.
