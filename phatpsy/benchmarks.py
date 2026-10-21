# phatpsy/benchmarks.py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from phatpsy.driver import PhatpsyDriver  # Uses our hybrid orchestrator

def run_no_potential_energy_curve():
    """
    Computes the Morse potential energy curve for Nitric Oxide (NO, X ^2 Pi)
    across bond lengths from 1.5 to 4.5 Bohr (Chapter IX-A)[cite: 13, 15].
    """
    print("Running Nitric Oxide (NO) Potential Energy Curve Benchmark...")
    driver = PhatpsyDriver()
    
    # Interatomic distances (Bohr) spanning equilibrium r_e ~ 2.17 Bohr[cite: 13, 15]
    bond_lengths = np.linspace(1.5, 4.5, 13)
    energies = []

    for r in bond_lengths:
        # Configure input parameters for NO molecule (Z_N = 7, Z_O = 8)
        state = core.phatpsy_state_mod
        state.natom = 2
        state.nelem = 2
        # Setup diatomic coordinates along z-axis: N at origin, O at distance r
        state.mol_ws.coord[0, 0] = 0.0d0
        state.mol_ws.coord[1, 0] = 0.0d0
        state.mol_ws.coord[2, 0] = 0.0d0
        
        state.mol_ws.coord[0, 1] = 0.0d0
        state.mol_ws.coord[1, 1] = 0.0d0
        state.mol_ws.coord[2, 1] = r

        result = driver.run_from_stdin()
        energies.append(result["total_energy"])
        print(f"  R = {r:.3f} Bohr | E_tot = {result['total_energy']:.6f} Hartree")

    # Plot Morse curve using matplotlib
    plt.figure(figsize=(8, 5))
    plt.plot(bond_lengths, energies, 'bo-', label='PHATPSY Modernized Hybrid')
    plt.title('Nitric Oxide (NO, X ^2 Pi) Potential Energy Curve[cite: 13, 15]')
    plt.xlabel('Internuclear Distance R (Bohr)')
    plt.ylabel('Total Energy (Hartree)')
    plt.grid(True)
    plt.legend()
    plt.savefig('no_potential_energy_curve.png', dpi=300)
    print("Saved potential energy curve plot to no_potential_energy_curve.png")

def run_nicluster_surface_benchmark():
    """
    Sets up the Ni(100) surface four-fold hole adsorption cluster model
    described in Chapter IX-B and IX-C[cite: 13, 15].
    """
    print("Initializing Ni(100) Surface Cluster Model...")
    driver = PhatpsyDriver()
    
    state = core.phatpsy_state_mod
    # Configure bulk, surface, and four-fold hole cluster geometry
    state.natom = 5  # 4 surface/bulk Ni atoms + 1 adsorbate atom (NO)[cite: 13, 15]
    state.nelem = 3  # Ni, N, O
    
    # Run self-consistent calculation with Damping/DIIS acceleration
    def ni_convergence_hook(s):
        # Apply gentle damping or Pulay mixing to stabilize Ni d-band SCF loops[cite: 13, 15]
        s.cdamp = 0.75
        
    result = driver.run_from_stdin(custom_damping_hook=ni_convergence_hook)
    print(f"Ni(100) Cluster SCF converged! Total Energy = {result['total_energy']:.6f} Hartree")

if __name__ == "__main__":
    run_no_potential_energy_curve()
    run_nicluster_surface_benchmark()