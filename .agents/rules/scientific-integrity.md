# Scientific Integrity Rules for PHATPSY 2.0

These rules apply whenever modifying, inspecting, or testing scientific algorithms in `src/` (Fortran) or `phatpsy/` (Python).

---

## 1. Units and Physical Conventions

- **Internal Representation:** All physical dimensions, coordinates, and energy quantities within the computational core **must** be stored and evaluated in **Atomic Units**:
  - Distance: Bohrs ($a_0 \approx 0.52917721092 \text{ \AA}$)
  - Energy: Hartrees ($E_h \approx 27.211386245988 \text{ eV} \approx 627.509 \text{ kcal/mol}$)
  - Charge: Elementary charge units ($e$)
- **Input Conversions:** If user inputs are specified in \AA ngstr\o ms (e.g. via `units="angstrom"` in `phatpsy.builder` or `QANGST=T`), conversion to Bohrs must use the project standard constant:
  $$\text{BOHR\_PER\_ANGSTROM} = 1.889726125$$

---

## 2. Core Scientific Invariants

1. **Atom-Atom Subspace Partitioning & Phillips-Kleinman Projection:**
   - PHATPSY employs an **atom-in-molecule partitioning**, structuring molecular wavefunctions via localized atomic spin-orbitals centered on each nucleus.
   - The Phillips-Kleinman pseudopotential is applied specifically for **atom-atom partitioning** (not core-valence separation) to enforce interatomic orthogonality and simulate Pauli exclusion between interacting atomic centers ($\langle \phi_i^A | \phi_j^B \rangle \approx 0$).
   - Because the Phillips-Kleinman operator is inherently **energy-dependent**, it cannot be solved as a static potential across all states simultaneously. It must be treated through approximations (e.g., pseudometrics, overlap expansions). Determining the most effective and robust approximation for this energy dependence remains an active scientific work in progress.

2. **LAPACK Linear Algebra Standard:**
   - Standardize all matrix diagonalizations, tridiagonalizations, and linear least-squares on **LAPACK** (`dsyev`, `dgels`, etc.).
   - Legacy algorithms (e.g. Jacobi iterations, EISPAK routines) are deprecated in modern modules in favor of standard LAPACK implementations.

3. **In-Memory Storage Model:**
   - One- and two-electron integrals, overlap matrices, and density projections must be retained in dynamically allocated memory buffers (e.g. `phatpsy_integrals_mod.f90`).
   - Do not re-introduce legacy scratch disk files or tape unit streaming (`IUNIT(1)`, `IUNIT(2)`).

4. **Virial Ratio & Energy Invariants:**
   - Self-consistent ground-state solutions for neutral isolated molecules must approach the physical virial theorem ratio:
     $$-V/T \approx 2.00$$
   - Any modification resulting in significant virial divergence ($|-V/T - 2.0| > 0.1$ for converged closed-shell systems) indicates unphysical potential fitting or projector corruption.

5. **Localized Valence-Bond (LVB) Alternating-Spin States:**
   - Open-shell calculations utilize local spin polarization ($\alpha$ / $\beta$ alternating on adjacent atomic centers).
   - Spin-flip assignments (`QFLIP=T` or `spin="down"`) must preserve zero net total molecular spin for closed-shell singlet radical equivalents unless an explicit net spin state is requested.

6. **Special Atomic Centers & Spatial Symmetry:**
   - **Fixed Centers (`QFIXED=T` / `fixed=True`):** Wavefunctions and model potentials must remain strictly frozen throughout SCF cycles. Do not update or perturb fixed center matrices.
   - **Virtual Centers (`QVIRTL=T` / `virtual=True`):** Used to complete the local coordination neighborhood and external potentials of an atom during its local atomic calculation. They are explicitly excluded at the molecular (EWMO) level to preserve a finite cluster model, and do not contribute to classical molecular nuclear repulsion or final molecular secular dimensions. (They are not used for BSSE).
   - **Symmetry-Equivalent Centers (`NEQAT`, `ISYM`, `IXYZ`):** Must be generated from primary generator atoms via spatial transformation (`SYMOP`, `SYMXYZ`). Orbital and potential representations on equivalent centers must match transformed generator states exactly.
