#!/usr/bin/env python3
"""Summarize localized-VB scan outputs (tools/localized_vb_scan.py) into R_e, w_e, D_e
for localized VB (covalent+ionic, 'cas'), RHF and seeded CASSCF, versus experiment.

Usage: PYTHONPATH=. .venv/bin/python tools/summarize_localized_vb.py /tmp/lvb2_*.txt
"""
import re
import sys

import numpy as np

HEV = 27.211386245988
AMU = 1822.888486
MU = {"NN": 14.003074 / 2, "CO": 12.0 * 15.994915 / 27.994915, "LiH": 7.016003 * 1.007825 / 8.023828,
      "FH": 18.998403 * 1.007825 / 20.006228}
EXPT = {"NN": (2.0744, 2358.6, 9.91), "CO": (2.1322, 2169.8, 11.23), "LiH": (3.0154, 1405.7, 2.52),
        "FH": (1.7328, 4138.3, 6.12)}


def parse(path):
    txt = open(path).read()
    head = re.search(r"exact integrals\): (\S+) (\S+) ===", txt)
    mol, basis = head.group(1), head.group(2)
    e_at = float(re.search(r"same basis\): (\S+)", txt).group(1))
    block = txt.split("(total energy, Ha)")[1].strip().splitlines()
    cols = txt.split("(total energy, Ha)")[0].splitlines()[-1].split()
    rows = [[float(x) for x in l.split()] for l in block if l.strip() and l.split()[0][0].isdigit()]
    return mol, basis, e_at, np.array(rows)


def fit(R, E, mu):
    c = np.polyfit(R, E, min(3, len(R) - 1))
    r = np.linspace(R.min(), R.max(), 4001)
    e = np.polyval(c, r)
    k = int(np.argmin(e))
    if k in (0, len(r) - 1):
        return np.nan, np.nan, e[k]
    k2 = np.polyval(np.polyder(c, 2), r[k])
    return r[k], np.sqrt(k2 / (mu * AMU)) * 219474.63, e[k]


print(f"{'mol':4s} {'basis':5s} {'method':8s} {'R_e':>7s} {'w_e':>7s} {'D_e(eV)':>8s}")
for p in sys.argv[1:]:
    mol, basis, e_at, d = parse(p)
    for j, name in ((1, "locVB"), (2, "RHF"), (3, "CASSCF")):
        Re, we, emin = fit(d[:, 0], d[:, j], MU[mol])
        print(f"{mol:4s} {basis:5s} {name:8s} {Re:7.4f} {we:7.0f} {(e_at - emin) * HEV:8.3f}")
    x = EXPT[mol]
    print(f"{mol:4s} {basis:5s} {'expt':8s} {x[0]:7.4f} {x[1]:7.0f} {x[2]:8.3f}")

