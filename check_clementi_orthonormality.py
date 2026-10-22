#!/usr/bin/env python3
"""
Orthonormality and sanity checker for phatpsy/data/clementi_sto.json
(Clementi & Roetti 1974, Tables 42/43: SZ and DZ best functions, Z=2..54).

For every element and basis set (SZ, DZ), per angular-momentum block:
  - builds the STO overlap matrix S from the stored (n, zeta) pairs,
  - forms C^T S C from the stored orbital coefficients,
  - reports the worst deviation from the identity matrix.
Also checks: electron count vs Z, coefficient function_index references,
suspicious exponents, and placeholder (0.0) orbital energies.

STO overlap (normalized STOs of the same l):
  R(r) = N r^(n-1) exp(-zeta r),  N = (2 zeta)^(n+1/2) / sqrt((2n)!)
  S_ij = N_i N_j (n_i + n_j)! / (zeta_i + zeta_j)^(n_i + n_j + 1)

Usage: python3 check_clementi_orthonormality.py [path/to/clementi_sto.json]
Exit code 0 if every block passes at TOL, 1 otherwise.
"""
import json
import sys
from math import factorial, sqrt

TOL = 1e-3  # tables are printed to 5 decimals; clean parses land ~1e-4 or better


def overlap(n1, z1, n2, z2):
    N1 = (2 * z1) ** (n1 + 0.5) / sqrt(factorial(2 * n1))
    N2 = (2 * z2) ** (n2 + 0.5) / sqrt(factorial(2 * n2))
    return N1 * N2 * factorial(n1 + n2) / (z1 + z2) ** (n1 + n2 + 1)


def check_entry(sym, entry):
    problems = []
    fns = {f["index"]: f for f in entry["basis_functions"]}

    # electron count
    ne = sum(o["occ_total"] for o in entry["orbitals"])
    if abs(ne - entry["atomic_number"]) > 1e-9:
        problems.append(f"electron count {ne} != Z={entry['atomic_number']}")

    # placeholder orbital energies
    for o in entry["orbitals"]:
        if o.get("eigval", None) == 0.0:
            problems.append(f"orbital {o['name']}: eigval is 0.0 (placeholder?)")

    # suspicious exponents (25.0 exactly recurs in the current file for Li/Be)
    for f in entry["basis_functions"]:
        if f["zeta"] == 25.0:
            problems.append(f"function {f['index']}: zeta == 25.0 exactly (suspect)")

    # orthonormality per l-block
    worst = 0.0
    by_l = {}
    for o in entry["orbitals"]:
        by_l.setdefault(o["l"], []).append(o)
    for l, olist in by_l.items():
        for o in olist:
            idxs = [c["function_index"] for c in o["coefficients"]]
            missing = [i for i in idxs if i not in fns]
            if missing:
                problems.append(f"orbital {o['name']}: unknown function_index {missing}")
            wrong_l = [i for i in idxs if i in fns and fns[i]["l"] != l]
            if wrong_l:
                problems.append(f"orbital {o['name']}: coefficients reference l={fns[wrong_l[0]]['l']} functions {wrong_l}")
        for i, a in enumerate(olist):
            ca = {c["function_index"]: c["c"] for c in a["coefficients"]}
            for j, b in enumerate(olist):
                cb = {c["function_index"]: c["c"] for c in b["coefficients"]}
                v = 0.0
                for ia, va in ca.items():
                    for ib, vb in cb.items():
                        if ia in fns and ib in fns:
                            v += va * overlap(fns[ia]["n"], fns[ia]["zeta"],
                                               fns[ib]["n"], fns[ib]["zeta"]) * vb
                target = 1.0 if i == j else 0.0
                worst = max(worst, abs(v - target))
    return worst, problems


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "phatpsy/data/clementi_sto.json"
    data = json.load(open(path))
    n_fail = 0
    for group in ("SZ", "DZ"):
        print(f"== {group} ==")
        for sym, entry in data[group].items():
            worst, problems = check_entry(sym, entry)
            status = "PASS" if worst <= TOL and not problems else "FAIL"
            if status == "FAIL":
                n_fail += 1
            print(f"{sym:>3}  worst|CSC-I| = {worst:9.2e}  {status}")
            for p in problems:
                print(f"      - {p}")
    print(f"\n{ n_fail } element/set combinations failed at tol {TOL:g}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
