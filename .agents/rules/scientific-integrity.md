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
 
 1. **Atom-in-Molecule Partitioning & Phillips-Kleinman Projection:**
    - PHATPSY employs an **atom-in-molecule partitioning**, structuring molecular wavefunctions via localized atomic spin-orbitals centered on each nucleus.
    - The Phillips-Kleinman pseudopotential uses an **atom-partitioned projection manifold** (occupied states of neighboring centers, summed atom-by-atom per thesis Eq. III-24); the $(\varepsilon - \varepsilon_i^B)$ energy weighting makes it core-selective in effect, generalizing — not replacing — core-valence separation. Pauli exclusion is enforced energetically through $V_{\text{PK}}$ and metrically through the pseudometric $M = S - P$; the operator does not impose strict $\langle \phi_i^A | \phi_j^B \rangle = 0$ orthogonality.
    - Because the Phillips-Kleinman operator is inherently **energy-dependent**, it cannot be solved as a static potential across all states simultaneously. It is treated through regularized projection, dynamic eigenvalue scaling ($V_{\text{PK}}(E)$), and canonical Lowdin regularization.
    - **Localized VB & the Exact Projector (current architecture):** In the localized VB and embedding layer, the environment enters through the exact joint projector built from $(\mathbf{S}_{\mathcal{BB}})^{-1}$ over the full environment block. With the exact projector, `pk_scale = 1.0` identically and the pseudometric $\tilde{\mathbf{S}}_{\text{eff}}$ is positive definite. Fitted or regularized PK strengths are not permitted in the VB/embedding layer. The legacy regularized PK treatment remains only inside the heritage `phatpsy_scf` engine and as the designated non-bonded environment treatment, pending its replacement by terms derived from the exact two-center integrals (see `ROADMAP.md`, Open Model Items).

 2. **LAPACK Linear Algebra Standard:**
    - Standardize all matrix diagonalizations, tridiagonalizations, and linear least-squares on standard **LAPACK** (`dsyev`, `dgels`, etc.).
    - Legacy algorithms (e.g. Jacobi iterations, EISPAK routines) are deprecated and replaced in favor of standard LAPACK implementations.

 3. **In-Memory Storage Model:**
    - One- and two-electron integrals, overlap matrices, and density projections must be retained in dynamically allocated memory buffers (e.g. `phatpsy_integrals_mod.f90`).
    - Scratch disk files or punch-card tape unit streaming (`IUNIT(1)`, `IUNIT(2)`) and legacy namelist input decks (`*.stdin`) are retired.

 4. **Virial Ratio & Energy Invariants:**
    - Self-consistent ground-state solutions for neutral isolated molecules must approach the physical virial theorem ratio:
      $$-V/T \approx 2.00 \quad (|-V/T - 2.0| \le 0.05)$$
    - Any modification resulting in significant virial divergence ($|-V/T - 2.0| > 0.1$ for converged closed-shell systems) indicates unphysical potential fitting or projector corruption.

 5. **External Scientific Benchmarks:**
    - Validation is performed against experimental physical observables and external ab initio benchmarks (NIST CCCBDB, CRC Handbook, Herzberg, Somorjai & Li, PySCF/Gaussian) in [`phatpsy/benchmarks.py`](file:///Users/jacksmith/phatpsy2/phatpsy/benchmarks.py) rather than frozen legacy 1980 outputs.

 6. **Spin Assignments (explicit, per-atom):**
    - Spin is assigned per atom and per orbital (`AtomConfig.spin`, `valence_config`): each atom carries explicit up/down spin-orbital assignments, and the deck `net_spin` defaults to $(n_\uparrow - n_\downarrow)$ counted over atom labels.
    - Alternating-spin (LVB / perfect-pairing) configurations are the standard closed-shell construction, but ferromagnetic and otherwise frustrated spin configurations are legitimate inputs. The code must treat the configuration it is given — never refuse it and never silently reassign it.
    - `spin_flip` survives only as a deprecated compatibility shim; new code and decks use explicit assignments.

 7. **Special Atomic Centers & Spatial Symmetry:**
    - **Fixed-State Centers (`fixed_state`; legacy `QFIXED=T` in its original electronic-state sense):** The atom's electronic state — its orbitals and occupations — is prescribed and is not variationally updated during SCF/VB cycles. Fixed state is an electronic axis only; it says nothing about the atom's position.
    - **Clamped Centers (`clamped=True`; written `fixed=True` in the current schema until the `ROADMAP.md` deck-vocabulary rename lands):** The atom's nuclear position is held — it is excluded from geometry-optimization steps and from the vibrational Hessian. Clamping is a positional axis only; it says nothing about the atom's electronic state. Do not use the unqualified word "fixed" in new code, decks, or documentation: name the axis (`fixed_state` or `clamped`).
    - **Virtual Centers (`virtual=True` / `QVIRTL=T`):** Used to complete the local coordination neighborhood and external potentials of an atom during its local atomic calculation. They are explicitly excluded at the molecular (EWMO) level to preserve a finite cluster model, and do not contribute to classical molecular nuclear repulsion or final molecular secular dimensions.
    - **Symmetry-Equivalent Centers:** Generated from primary generator atoms via spatial transformation (`SYMOP`, `SYMXYZ`). Orbital and potential representations on equivalent centers must match transformed generator states exactly. An equivalent center inherits its reference center's treatment on every axis (kind, state, position, participation).
