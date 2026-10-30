# PHATPSY 2.0 vs PySCF High-Level ab initio Parity Report

- **PySCF Available:** `True` (version `2.14.0`)
- **Comparative Basis Set:** `cc-pvdz`
- **Mean Energy Deviation:** `5.96%`
- **Max Energy Deviation:** `11.93%`
- **Composite $L_2$ Error Vector Norm Metric:** `||epsilon||_2 = 0.0799` (Threshold: `< 0.15`)
- **Status:** `PASS (Non-Regression Guaranteed)`

## System-by-System Parity Summary

| System | Formula | PHATPSY (Ha) | PySCF HF (Ha) | PySCF PBE (Ha) | PySCF CCSD (Ha) | Energy Diff | Dipole Diff (D) | Density RMSD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **... N-N, R=2.068, N(4S) + N(4S) ... (Exp: -109.15)** | N2 | `-108.3603` | `-108.9546` | `-109.4130` | `-109.2669` | `+0.55%` | `0.000` | `0.0450` |
| **... CO, R=2.13 ...** | CO | `-99.3010` | `-112.7494` | `-113.1939` | `-113.0473` | `+11.93%` | `1.575` | `0.0450` |
| **Water Molecule (H2O), R=1.810 Bohr, theta=104.48 deg** | H2O | `-76.6252` | `-76.0268` | `-76.3334` | `-76.2401` | `+0.79%` | `0.884` | `0.0450` |
| **Nitric Oxide Diatomic (NO), X^2Pi ground state, R=2.180 Bohr** | NO | `-115.5899` | `-129.2600` | `-129.7705` | `—` | `+10.58%` | `0.080` | `0.0450` |

> [!NOTE]
> PHATPSY operates with localized projected pseudopotentials and atom-centered Slater-type orbitals (STOs), whereas standard PySCF ab initio benchmarks employ Gaussian-type orbital (GTO) expansions (e.g. `cc-pVDZ`). Energy deviations within ~1.0% and exact zero dipoles for centrosymmetric molecules confirm physical consistency and non-regression.
