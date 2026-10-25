# Restarts, Scans & Geometry Optimization

This chapter explains how to perform warm-restart calculations, run 1D and 2D potential energy surface (PES) scans, configure internal coordinates, and optimize molecular geometries.

---

## Warm Restarts & Checkpoints

Re-converging an SCF calculation from scratch for slightly perturbed geometries is inefficient. PHATPSY supports **warm restarts**, which seed the initial density from a previous `CalculationState`:

```python
# 1. Initial calculation
res1 = driver.run_deck(deck)

# 2. Modify geometry (e.g. stretch bond)
new_geometry = [
    ("N", [0.0, 0.0,  1.100]),
    ("N", [0.0, 0.0, -1.100]),
]

# 3. Warm restart reusing converged molecular orbitals
res2 = driver.restart(res1, new_geometry=new_geometry)
```

You can also generate an independent restart YAML deck containing the converged orbital coefficients:

```python
restart_deck = res1.create_restart_deck()
restart_deck.to_yaml("warm_restart.yaml")
```

---

## Internal Coordinates & Z-Matrices

PHATPSY provides internal coordinate primitives for manipulating geometry without raw Cartesian math:

```python
from phatpsy import InternalCoordinates, BondCoordinate, AngleCoordinate, DihedralCoordinate

# Define internal coordinates
bond = BondCoordinate(atom1=0, atom2=1)
angle = AngleCoordinate(atom1=0, atom2=1, atom3=2)
dihedral = DihedralCoordinate(atom1=0, atom2=1, atom3=2, atom4=3)

coords = InternalCoordinates([bond, angle, dihedral])
```

You can also construct systems using `ZMatrix`:

```python
from phatpsy import ZMatrix, ZMatrixEntry

zmat = ZMatrix([
    ZMatrixEntry("O"),
    ZMatrixEntry("H", bond_to=0, bond_length=0.957),
    ZMatrixEntry("H", bond_to=0, bond_length=0.957, angle_to=1, angle_val=104.5),
])

# Convert to Cartesian CalculationDeck
deck = zmat.to_deck(title="Water Molecule", basis="DZ")
```

---

## Potential Energy Surface (PES) Scans

### 1D Scans (`scan_1d`)

Scan a bond, valence angle, or dihedral angle along a discrete grid with automatic warm restarting:

```python
import numpy as np
import matplotlib.pyplot as plt
from phatpsy import scan_1d

# Define grid of bond distances in Bohrs
r_grid = np.linspace(1.8, 3.2, 15)

# Coordinate specification: ("bond", atom_i, atom_j)
bond_coord = ("bond", 0, 1)

# Execute scan with warm restarts enabled
pes_result = scan_1d(deck, coordinate=bond_coord, values=r_grid, warm_restart=True)

# Plot potential energy curve
plt.figure(figsize=(7, 4.5))
plt.plot(pes_result["values"], pes_result["energies"], marker="o", color="navy")
plt.xlabel("Bond Distance $R$ (Bohrs)")
plt.ylabel("Total Energy (Hartrees)")
plt.title("N2 Bond Dissociation Energy Curve")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
```

### 2D Scans (`scan_2d`)

For studying coupling between multiple degrees of freedom (e.g. stretching two bonds simultaneously):

```python
from phatpsy import scan_2d

r1_grid = np.linspace(1.9, 2.5, 7)
r2_grid = np.linspace(1.9, 2.5, 7)

pes2d = scan_2d(
    deck,
    coord1=("bond", 0, 1),
    values1=r1_grid,
    coord2=("bond", 1, 2),
    values2=r2_grid,
    warm_restart=True,
)
```

---

## Geometry Optimization (`optimize_geometry`)

Automated Cartesian geometry optimization uses the BFGS quasi-Newton algorithm with finite-difference numerical gradients:

```python
from phatpsy import optimize_geometry

opt_result = optimize_geometry(
    deck,
    max_steps=30,
    grad_tol=1.0e-3,     # RMS gradient threshold
    energy_tol=1.0e-5,   # Energy change threshold
    finite_diff_step=0.005,
)

print(f"Optimization Converged: {opt_result.converged}")
print(f"Total Steps Taken     : {len(opt_result.steps)}")
print(f"Final Energy          : {opt_result.final_energy:.6f} Hartrees")

# Relaxed geometry as a new CalculationDeck
relaxed_deck = opt_result.final_deck
relaxed_deck.to_yaml("relaxed_geometry.yaml")
```

### Inspecting Optimization Trajectory

The `OptimizationResult` stores intermediate steps with energies, gradients, and Cartesian coordinates:

```python
for step in opt_result.steps:
    print(f"Step {step.step_num:2d} | Energy: {step.energy:.6f} Eh | Grad Norm: {step.gradient_norm:.3e}")
```

---

## Transition State Searching (Climbing-Image NEB)

To locate reaction transition states (first-order saddle points) and calculate activation energy barriers, PHATPSY 2.0 provides the **Climbing-Image Nudged Elastic Band (CI-NEB)** method via `neb_search()`:

```python
from phatpsy import neb_search, build_diatomic

# 1. Define reactant and product states
deck_reactant = build_diatomic("N", "N", bond_length=2.068, basis="DZ")
deck_product = build_diatomic("N", "N", bond_length=2.800, basis="DZ")

# 2. Run CI-NEB calculation across 7 pathway images
neb_res = neb_search(
    initial_deck=deck_reactant,
    final_deck=deck_product,
    num_images=7,
    k_spring=0.1,        # Spring constant in Hartree/Bohr^2
    max_iterations=40,
    force_tol=0.003,     # Force threshold in Hartree/Bohr
    climb=True,          # Inverts tangent force on highest image to locate exact saddle point
    warm_restart=True,
)

print(f"NEB Converged      : {neb_res.converged}")
print(f"Iterations         : {neb_res.num_iterations}")
print(f"Forward Barrier    : {neb_res.barrier_forward_kcal:.2f} kcal/mol ({neb_res.barrier_forward:.6f} Eh)")
print(f"Reverse Barrier    : {neb_res.barrier_reverse_kcal:.2f} kcal/mol")
print(f"Reaction Energy    : {neb_res.reaction_energy_kcal:.2f} kcal/mol")
print(f"Transition State   : Image {neb_res.transition_state_index}")

# Transition state geometry as a new CalculationDeck
ts_deck = neb_res.transition_state_deck
ts_deck.to_yaml("transition_state.yaml")
```

### Reaction Profile Plotting

Plot the potential energy reaction profile with automatic cubic-spline interpolation:

```python
# Plot in kcal/mol (default) or Hartree
neb_res.plot_pathway("reaction_barrier.png")
```

### Interactive 3D Pathway Playback in Notebooks

Scrub through reaction pathway images and watch animated bond rearrangements directly inside Jupyter cells:

```python
neb_res.show(height=550)
```

You can also export the reaction trajectory to multi-frame XYZ format:

```python
neb_res.save_xyz("reaction_pathway.xyz")
```

---

## Born-Oppenheimer Molecular Dynamics (BOMD)

To investigate thermal vibrations, conformational exploration, and time-dependent dynamics, PHATPSY 2.0 provides **Born-Oppenheimer Molecular Dynamics (BOMD)** via `run_md()`:

```python
from phatpsy import run_md, build_diatomic

# 1. Prepare initial geometry deck
deck = build_diatomic("N", "N", bond_length=2.068, basis="DZ", use_symmetry=False)

# 2. Run NVE dynamics with 0.5 fs timestep at 300 K
md_res = run_md(
    deck=deck,
    time_step_fs=0.5,
    num_steps=100,
    temperature_k=300.0,
    ensemble="nve",           # "nve", "berendsen", "rescale", or "andersen"
    remove_com_drift=True,    # Remove net center-of-mass linear momentum
    seed=42,
)

print(f"Total time simulated: {md_res.total_time_fs:.1f} fs")
print(f"Initial Total Energy: {md_res.initial_energy:.6f} Eh")
print(f"Final Total Energy  : {md_res.final_energy:.6f} Eh")
print(f"Energy Drift        : {md_res.energy_drift:.2e} Eh")
print(f"Average Temperature : {md_res.average_temperature:.1f} ± {md_res.temperature_std:.1f} K")
```

### Thermodynamic Ensembles

* **NVE (Microcanonical):** Standard symplectic Velocity Verlet integration without heat bath coupling. Conserves total Hamiltonian energy $E_{\text{tot}} = E_{\text{pot}} + E_{\text{kin}}$.
* **Berendsen ($NVT$):** Weak-coupling velocity scaling with characteristic relaxation time `tau_fs` (e.g., `tau_fs=20.0`).
* **Velocity Rescaling ($NVT$):** Instantaneous velocity scaling per time step to rigidly maintain target kinetic temperature.
* **Andersen ($NVT$):** Stochastic thermal collisions with frequency `collision_freq` (default: 0.1 / fs).

### Trajectory Analysis & Plotting

Plot time series of potential energy, kinetic energy, total energy conservation, and instantaneous temperature:

```python
# Save dynamics profile plot
md_res.plot_dynamics("md_trajectory.png")
```

### Interactive 3D Trajectory Playback in Jupyter Notebooks

Render an interactive WebGL multi-frame trajectory player in Jupyter or HTML:

```python
# Interactive Three.js timeline player inside notebook cell
md_res.show(height=550)

# Or export standalone HTML viewer
md_res.visualize("trajectory.html", open_browser=True)

# Export standard multi-frame XYZ for external tools (VMD, PyMOL, Avogadro)
md_res.save_xyz("trajectory.xyz")
```

---

## Harmonic Vibrational Frequency Analysis & Thermochemistry

PHATPSY 2.0 provides full second-derivative potential curvature analysis, normal mode evaluation, and ideal-gas statistical thermochemistry via the [`phatpsy.vibrations`](file:///Users/jacksmith/phatpsy2/phatpsy/vibrations.py) module.

### Core Methodology

1. **Numerical Cartesian Hessian Matrix ($\mathbf{H}$):**
   $$\mathbf{H}_{i, j} = \frac{\partial^2 E}{\partial X_i \partial X_j}$$
   Evaluated via numerical finite differences across active coordinate degrees of freedom (preserving constraints for frozen atoms and ghost centers).
   - `method="central"`: Full central difference across all diagonal and off-diagonal cross terms ($O(h^2)$ precision).
   - `method="forward"`: Central differences on diagonal elements and forward cross-difference on off-diagonal elements ($1 + 2M + M(M-1)/2$ SCF calculations, fast).

2. **Eckart Translational & Rotational Projection:**
   Rigid-body translations ($T_x, T_y, T_z$) and infinitesimal rotations ($R_x, R_y, R_z$) are decoupled from the mass-weighted Cartesian Hessian $\mathcal{H}_{i, j}^{\text{mw}} = \frac{\mathbf{H}_{i, j}}{\sqrt{M_A M_B}}$ using the projection operator:
   $$\mathbf{P} = \mathbf{I}_{3N} - \mathbf{D}_{\text{TR}} \mathbf{D}_{\text{TR}}^T, \quad \mathcal{H}_{\text{proj}} = \mathbf{P} \mathcal{H}^{\text{mw}} \mathbf{P}$$
   where $N_{\text{TR}} = 5$ for linear rotors and $N_{\text{TR}} = 6$ for non-linear rotors. (When fixed substrate atoms are present, projection is automatically skipped because the substrate breaks translational and rotational invariance).

3. **Normal Modes & Frequencies:**
   Diagonalization of $\mathcal{H}_{\text{proj}}$ yields eigenvalues $\lambda_k$ (in $\text{Hartree}/(\text{Bohr}^2 \cdot \text{amu})$):
   - **Harmonic Frequency ($\text{cm}^{-1}$):**
     $$\tilde{\nu}_k = 5140.487144 \times \sqrt{\lambda_k}$$
     If $\lambda_k < 0$, the mode is an **imaginary frequency** ($\tilde{\nu}_k < 0$, `is_imaginary=True`), indicating negative potential curvature.
   - **Reduced Mass ($\mu_k$):** $\mu_k = \frac{1}{\sum_{A, \alpha} Q_{A\alpha, k}^2 / M_A}$ in amu.
   - **Force Constant ($k_k$):** $k_k = \lambda_k \cdot \mu_k \times 15.568931$ in $\text{mDyne}/\text{Å}$.
   - **Stationary Point Classification:**
     - $N_{\text{imag}} = 0$: Local Potential Minimum (`vib_res.is_minimum == True`)
     - $N_{\text{imag}} = 1$: First-Order Saddle Point / Transition State (`vib_res.is_transition_state == True`)

4. **Rigid-Rotor Harmonic Oscillator (RRHO) Thermochemistry:**
   Evaluates standard gas-phase thermodynamic partition functions at temperature $T$ and pressure $P$:
   - **Zero-Point Vibrational Energy (ZPVE):** $\text{ZPVE} = \frac{1}{2} \sum_{k, \tilde{\nu}_k > 0} h c \tilde{\nu}_k$.
   - **Internal Thermal Energy ($U_{\text{therm}}$):** $E_{\text{trans}} + E_{\text{rot}} + E_{\text{vib}}$.
   - **Thermal Enthalpy ($H_{\text{therm}}$):** $U_{\text{therm}} + RT$.
   - **Entropy ($S(T)$):** Sackur-Tetrode translational entropy $S_{\text{trans}}$, rotational entropy $S_{\text{rot}}$ (accounting for rotational symmetry number $\sigma$), vibrational entropy $S_{\text{vib}}$, and electronic spin degeneracy $S_{\text{elec}} = R \ln(2S + 1)$.
   - **Thermodynamic Potentials:**
     - Internal Energy at 0 K: $U_0 = E_{\text{elec}} + \text{ZPVE}$
     - Internal Energy at $T$: $U(T) = E_{\text{elec}} + U_{\text{therm}}$
     - Enthalpy at $T$: $H(T) = E_{\text{elec}} + H_{\text{therm}}$
     - Gibbs Free Energy at $T$: $G(T) = H(T) - T \cdot S(T)$
   - **Constant-Volume Heat Capacity ($C_v$):** $C_{v, \text{trans}} + C_{v, \text{rot}} + C_{v, \text{vib}}$.

### Python Workflow Example

```python
from phatpsy import PhatpsyDriver, CalculationDeck, compute_vibrations

# 1. Load calculation deck (relaxed stationary point)
driver = PhatpsyDriver(quiet=True)
deck = CalculationDeck.from_yaml("relaxed_molecule.yaml")

# 2. Run harmonic vibrational frequency and thermochemical analysis
vib_res = compute_vibrations(
    deck=deck,
    driver=driver,
    step_size=0.005,      # Finite-difference step in Bohr
    method="forward",     # "central" (O(h^2)) or "forward" (fast)
    project_tr=True,      # Project out translation and rotation
    temperature_k=298.15, # Reference temperature in K
    pressure_atm=1.0,     # Reference pressure in atm
    quiet=True,
)

# 3. Print formatted report table
print(vib_res.summary())

# 4. Check stationary point type
if vib_res.is_minimum:
    print("Structure is confirmed as a true local energy minimum!")
elif vib_res.is_transition_state:
    imag_mode = vib_res.imaginary_modes[0]
    print(f"Confirmed transition state! Imaginary frequency: {imag_mode.frequency_cm:.2f} cm^-1")

# 5. Inspect thermochemistry
thermo = vib_res.thermochemistry
print(f"Zero-Point Energy (ZPVE)  : {thermo.zero_point_energy_kcal:.4f} kcal/mol")
print(f"Standard Entropy S(298 K) : {thermo.entropy_cal_mol_k['total']:.3f} cal/(mol·K)")
print(f"Gibbs Free Energy G(298 K): {thermo.total_gibbs_free_energy_g_t:.8f} Hartree")

# 6. Save structured results to JSON
vib_res.save_json("vibrations_results.json")
```

### Interactive 3D Normal Mode Animation

Inspect normal mode vibrations in 3D with interactive real-time WebGL harmonic animation in Three.js:

```python
# Render interactive Three.js normal mode widget inside Jupyter notebook
vib_res.show(mode_index=1, amplitude=0.5, height=520)

# Export standalone HTML visualization with timeline and vector cones
vib_res.visualize("normal_modes.html", open_browser=True)
```

---

## Intrinsic Reaction Coordinate (IRC) Pathway Tracing

Once a transition state (first-order saddle point with $N_{\text{imag}} = 1$) has been identified by CI-NEB and verified by vibrational frequency analysis, the **Intrinsic Reaction Coordinate (IRC)** connects the transition state with the corresponding reactant and product energy minima.

The [`phatpsy.irc`](file:///Users/jacksmith/phatpsy2/phatpsy/irc.py) module computes the mass-weighted steepest descent reaction pathway:

$$\mathbf{x}_i = \sqrt{M_i} \mathbf{r}_i, \quad \frac{d\mathbf{x}(s)}{ds} = - \frac{\mathbf{g}_x}{\|\mathbf{g}_x\|}$$

where $s$ is the reaction coordinate arc length (in $\text{amu}^{1/2} \cdot \text{Bohr}$), and $\mathbf{g}_{x, i} = \mathbf{g}_{r, i} / \sqrt{M_i}$ is the mass-weighted gradient.

### Reaction Path Methodology

1. **Transition Vector Initiation:**
   The transition state is a stationary point ($\mathbf{g} = 0$). The path is initiated by displacing along the normalized transition normal mode vector corresponding to the imaginary frequency ($\mathbf{Q}_{\text{imag}}$):
   $$\mathbf{x}_0^\pm = \mathbf{x}_{\text{TS}} \pm \Delta s_0 \frac{\mathbf{q}_{\text{imag}}}{\|\mathbf{q}_{\text{imag}}\|}$$
   where positive ($+$) traces the forward (product) branch and negative ($-$) traces the reverse (reactant) branch.
2. **Mass-Weighted Steepest Descent Integration:**
   Steps along normalized mass-weighted descent direction $\mathbf{u} = -\mathbf{g}_x / \|\mathbf{g}_x\|$ with adaptive backtracking line search to ensure monotonic potential energy reduction down to the local minimum ($\|\mathbf{g}\| \le \text{tol}$ or $\Delta E \le \text{tol}$).
3. **Activation Barriers & Thermochemistry:**
   - **Forward Activation Barrier:** $\Delta E_f^\ddagger = E_{\text{TS}} - E_{\text{reactant}}$ (kcal/mol)
   - **Reverse Activation Barrier:** $\Delta E_r^\ddagger = E_{\text{TS}} - E_{\text{product}}$ (kcal/mol)
   - **Reaction Energy:** $\Delta E_{\text{rxn}} = E_{\text{product}} - E_{\text{reactant}}$ (kcal/mol)

### Python Workflow Example

```python
from phatpsy import PhatpsyDriver, CalculationDeck, compute_irc

driver = PhatpsyDriver(quiet=True)
deck_ts = CalculationDeck.from_yaml("ts_molecule.yaml")

# Trace IRC in both directions (reactant and product branches)
irc_res = compute_irc(
    deck=deck_ts,
    driver=driver,
    step_size=0.1,        # Arc length step size in amu^(1/2) * Bohr
    max_steps=40,         # Maximum integration steps per branch
    direction="both",     # "both", "forward", or "reverse"
    grad_tol=1e-3,        # Convergence gradient norm in Hartree / Bohr
    quiet=True,
)

# Print formatted summary table of pathway points and barriers
print(irc_res.summary())

# Access key reaction energetics
print(f"Forward Barrier (dE_f): {irc_res.barrier_forward_kcal:.2f} kcal/mol")
print(f"Reverse Barrier (dE_r): {irc_res.barrier_reverse_kcal:.2f} kcal/mol")
print(f"Reaction Energy (dE_rxn): {irc_res.reaction_energy_kcal:+.2f} kcal/mol")

# Export structured JSON and multi-frame XYZ trajectory
irc_res.save_json("irc_results.json")
irc_res.save_xyz("irc_trajectory.xyz")

# Plot 2D energy profile along the reaction coordinate
irc_res.plot_pathway("irc_profile.png")
```

### Interactive 3D Reaction Pathway Animation

Scrub through the reaction pathway in 3D WebGL with synchronized energy cards in Three.js:

```python
# Display interactive widget inside Jupyter Notebook
irc_res.show(height=520)

# Export standalone HTML visualization with playback controls
irc_res.visualize("irc_pathway.html", open_browser=True)
```

---

## Transition-State Refinement: Dimer Method (Dimer-over-RFO)

Because PHATPSY calculates numerical finite-difference gradients and does not construct analytic Hessians, traditional Hessian-based partitioned Rational Function Optimization (P-RFO) is computationally prohibitive ($O(3N)$ or $(3N)^2$ SCF evaluations per step) and fragile against numerical SCF noise.

PHATPSY 2.0 adopts the **Dimer method** (Heyden/Bell variant) for transition-state refinement:

### Mathematical Formulation
The dimer consists of two images $\mathbf{R}_1 = \mathbf{R} + \Delta R \cdot \mathbf{N}$ and $\mathbf{R}_2 = \mathbf{R} - \Delta R \cdot \mathbf{N}$ centered at $\mathbf{R}$ with orientation unit vector $\mathbf{N}$ ($\|\mathbf{N}\| = 1$) and half-separation $\Delta R = 0.005\ \text{Bohr}$.

1. **Center Force**: $\mathbf{F}_0 = -\nabla V(\mathbf{R})$.
2. **Rotational Torque**: Evaluating $\mathbf{F}_1 = -\nabla V(\mathbf{R}_1)$ and $\mathbf{F}_2 = -\nabla V(\mathbf{R}_2)$, the rotational force is $\mathbf{F}_R = \frac{1}{2}(\mathbf{F}_1 - \mathbf{F}_2)$. The torque acting perpendicular to $\mathbf{N}$ is:
   $$\mathbf{F}_R^\perp = \mathbf{F}_R - (\mathbf{F}_R \cdot \mathbf{N})\mathbf{N}$$
3. **Analytical Fourier Rotation**: If $\|\mathbf{F}_R^\perp\| > \text{tol}_{\text{rot}}$, a trial rotation by $\Delta \theta$ along $\mathbf{\Theta} = \mathbf{F}_R^\perp / \|\mathbf{F}_R^\perp\|$ requires only one additional gradient evaluation $\mathbf{F}_1^{\text{trial}}$ (with $\mathbf{F}_2^{\text{trial}} \approx 2\mathbf{F}_0 - \mathbf{F}_1^{\text{trial}}$). The optimal rotation angle $\phi = \frac{1}{2}\arctan(B/A)$ is determined analytically with guaranteed minimum curvature $C$.
4. **Effective Translation Force**: When curvature $C < 0$, the force along the minimum-curvature mode is inverted:
   $$\mathbf{F}_{\text{eff}} = \mathbf{F}_0 - 2(\mathbf{F}_0 \cdot \mathbf{N})\mathbf{N}$$
   When $C \ge 0$, the dimer climbs uphill along $\mathbf{N}$ ($\mathbf{F}_{\text{eff}} = -(\mathbf{F}_0 \cdot \mathbf{N})\mathbf{N}$) to escape local minimum basins.
5. **QuickMin Translation**: The center $\mathbf{R}$ steps along $\mathbf{F}_{\text{eff}}$ using projected velocity damping.

### Finite-Difference Step Size vs. Noise Floor Calibration
Numerical tests on $N_2$ characterize the SCF noise floor:
- Finite-difference gradient step size: $h = 0.001\ \text{Bohr}$ yields numerically stable central differences ($g_z = 0.0906$) immune to roundoff noise below $10^{-4}\ \text{Bohr}$ and anharmonic distortion above $0.005\ \text{Bohr}$.
- Dimer separation: $\Delta R = 0.005\ \text{Bohr}$ sits at the center of the curvature plateau ($C = 47.3\ \text{Hartree/Bohr}^2$ across $\Delta R \in [0.001, 0.010]$ with $0.2\%$ variance).

### Python API

```python
from phatpsy.driver import PhatpsyDriver
from phatpsy.schema import CalculationDeck
from phatpsy.neb import neb_search
from phatpsy.dimer import refine_neb_saddle, dimer_search

driver = PhatpsyDriver(quiet=True)

# 1. Handoff from CI-NEB (polishes climbing image to exact saddle precision)
neb_res = neb_search(
    initial_deck=deck_reactant,
    final_deck=deck_product,
    driver=driver,
    num_images=7,
)

dimer_res = refine_neb_saddle(
    neb_result=neb_res,
    driver=driver,
    dimer_dist=0.005,
    max_cycles=30,
    force_tol=1e-3,
)

# 2. Single-ended search from a displaced/perturbed geometry
dimer_single = dimer_search(
    deck=deck_guess,
    driver=driver,
    dimer_dist=0.005,
    max_cycles=40,
    force_tol=1e-3,
)

# View structured report
print(dimer_res.summary())

# Interactive 3D WebGL Three.js trajectory player in Jupyter
dimer_res.show(height=520)

# Export standalone HTML
dimer_res.visualize("dimer_refinement.html", open_browser=True)
```

---

## Electronic, Optical & Magnetic (EOM) Properties

The `phatpsy.eom` module evaluates ground- and excited-state electromagnetic response properties:

### 1. Permanent Electric Dipole Moment ($\boldsymbol{\mu}$)
Computes the nuclear and electronic dipole moments in atomic units ($e \cdot a_0$) and Debyes ($1\text{ a.u.} = 2.541746\text{ Debye}$):
$$\boldsymbol{\mu}_{\text{nuc}} = \sum_A Z_A (\mathbf{R}_A - \mathbf{R}_0), \quad \boldsymbol{\mu}_{\text{elec}} = \int (\mathbf{r} - \mathbf{R}_0) \rho(\mathbf{r}) d^3r$$
where $\mathbf{R}_0 = \sum_A Z_A \mathbf{R}_A / \sum_A Z_A$ is the center of nuclear charge, ensuring origin invariance for neutral systems. For homonuclear diatomics (such as $N_2$), $\boldsymbol{\mu} \equiv \mathbf{0}$ by inversion symmetry.

### 2. Optical Polarizability Tensors ($\boldsymbol{\alpha}(\omega)$)
Evaluates dynamic and static polarizabilities using second-order Sum-Over-States (SOS) perturbation theory:
$$\alpha_{jk}(\omega) = \sum_{i \in \text{occ}} \sum_{a \in \text{vir}} \frac{2 \Delta E_{ia}}{\Delta E_{ia}^2 - \omega^2} \langle \psi_i | r_j | \psi_a \rangle \langle \psi_a | r_k | \psi_i \rangle$$
reporting the full $3 \times 3$ tensor, mean polarizability $\bar{\alpha} = \frac{1}{3}\text{Tr}(\boldsymbol{\alpha})$ in atomic units ($a_0^3$) and $\text{\AA}^3$ ($1\text{ a.u.} = 0.148185\text{ \AA}^3$), and optical anisotropy $\gamma^2$.

### 3. Magnetic Susceptibility Tensors ($\boldsymbol{\chi}$)
Evaluates the three fundamental contributions to magnetic susceptibility:
1. **Langevin-Van Vleck Diamagnetic Susceptibility**:
   $$\chi_{jk}^{\text{dia}} = -\frac{1}{4c^2} \langle r^2 \delta_{jk} - r_j r_k \rangle$$
2. **Van Vleck Temperature-Independent Paramagnetic Susceptibility**:
   $$\chi_{jk}^{\text{para}} = \frac{1}{2c^2} \sum_{i \in \text{occ}} \sum_{a \in \text{vir}} \frac{1}{\Delta E_{ia}} \text{Re}\left( \langle \psi_i | \hat{L}_j | \psi_a \rangle \langle \psi_a | \hat{L}_k | \psi_i \rangle \right)$$
   where $\hat{\mathbf{L}} = -i(\mathbf{r} \times \nabla)$ is the orbital angular momentum operator.
3. **Curie Paramagnetism** for open-shell systems ($S > 0$):
   $$\chi^{\text{Curie}} = \frac{g_e^2 \mu_B^2 S(S+1)}{3 k_B T}$$

### 4. UV-Vis Absorption Spectroscopy & Oscillator Strengths
Computes single-particle excitation energies $\Delta E_{ia} = \varepsilon_a - \varepsilon_i$, wavelengths $\lambda = hc/\Delta E$, transition dipole moments $\boldsymbol{\mu}_{ia} = \langle \psi_i | \mathbf{r} | \psi_a \rangle$, and dimensionless oscillator strengths:
$$f_{ia} = \frac{2}{3} \Delta E_{ia} \|\boldsymbol{\mu}_{ia}\|^2$$
with continuous simulated Gaussian absorption profile generation and stick transition plotting (`plot_uvvis`).

### Python Example

```python
from phatpsy.builder import build_diatomic
from phatpsy.driver import PhatpsyDriver
from phatpsy.eom import compute_eom

driver = PhatpsyDriver(quiet=True)
deck = build_diatomic("N", "N", 2.068, basis="SZ", spin_flip=True, units="bohr")

eom_res = compute_eom(
    deck=deck,
    driver=driver,
    grid_spacing=0.25,
    dynamic_wavelengths=[589.3, 1064.0],
    temperature_k=298.15,
    quiet=True,
)

# Text summary report
print(eom_res.summary())

# Save structured JSON
eom_res.save_json("eom_results.json")

# Plot simulated UV-Vis absorption spectrum with sticks
eom_res.plot_uvvis("uvvis_spectrum.png", wmin_nm=100.0, wmax_nm=700.0)

# Embedded 3D Three.js WebGL visualizer with dipole arrow, polarizability ellipsoid, and UV-Vis HUD
eom_res.show(height=550)
```




