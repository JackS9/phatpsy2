# Polarization exponents from the literature — Phase 7 follow-up

Date: 2026-10-29. Prepared by Lux at Jack's request ("pull those exponents down from those papers").

## Where Phase 7 stands

Commit `bfb7d08` (Oct 28, verified from pushed bytes) implements Phase 7 (+ Phase 8);
ROADMAP marks both COMPLETED. The polarization augmentation lives in
`phatpsy/basis.py` as `POLARIZATION_PARAMETERS`: 2p on H/He, 3d on Li–Ar, added
to C&R SZ/DZ bases (and applicable to the extracted Thakkar set via
`get_polarized_basis`).

**Caveat:** the dict's values are simple constructed numbers — clear arithmetic
progressions (B–Ne 3d ζ = 1.00, 1.25, 1.50, 1.75, 2.00, 2.25; Na–Ar 3d ζ =
0.50–1.90 in steps of 0.20). The code comment labels them "Standard
Roetti-Clementi / Thakkar polarized exponents" — that label is unverified and
almost certainly wrong. They are placeholders awaiting literature values.

## Confirmed literature values

| Element | Polarization fn | ADF standard ζ (bohr⁻¹) | PHATPSY current | Δ |
|---------|----------------|------------------------|-----------------|---|
| H       | 2p             | **1.25**               | 1.00            | +0.25 |
| O       | 3d             | **2.00**               | 1.75            | +0.25 |

Source: Gasević, Stückrath, Grimme, Bursch, "Optimization of the r²SCAN-3c
Composite Electronic-Structure Method for Use with Slater-Type Orbital Basis
Sets," *J. Phys. Chem. A* **2022**, *126*, 3826–3838 (DOI
10.1021/acs.jpca.2c02951). The paper states the O 3d exponent was changed
2.00 → 2.15 and the H 2p exponent 1.25 → 1.70 for their customized set —
i.e. **2.00 and 1.25 are the published ADF standard (ZORA-contracted)
exponents**. Defining paper for the sets: van Lenthe & Baerends,
*J. Comput. Chem.* **24**, 1142 (2003), DOI 10.1002/jcc.10255.

## Why it matters for the H₂O dipole

`bfb7d08` reports the H₂O dipole going 4.35 → **1.18 D** with the simple
exponents — below the 1.855 D experimental value, i.e. an overcorrection.
The simple exponents are more diffuse (smaller ζ) than the literature values,
so they polarize more strongly. Tighter literature exponents should move the
dipole back up toward experiment. **Re-run the H₂O/CO dipole checks after
swapping in literature values** — this is the highest-value validation of the
swap.

## Getting the full H–Ar table (free, no ADF license)

SCM offers the van Lenthe–Baerends ZORA STO basis sets as a **free download**
(no license needed — stated on their basis-set page):

- Download: <https://www.scm.com/doc/ADF/_downloads/13a32cb06e43fc959219ca9fb75da628/zorabasis.tar.gz>
- Page: <https://www.scm.com/zorabasis/Welcome.html> ("can also be downloaded freely")
- Contents: ZORA STO sets; `ZORA/DZP` holds all-electron files for H–Ar.
  ADF defines **DZP = DZ + 1 polarization function**, so diffing the DZP file
  against the DZ file for the same element isolates the polarization function.
- File format (ASCII, documented): <https://www.scm.com/doc/ADF/Appendices/Basis_set_file_format.html>
  — the `BASIS` section lists one STO per line as `<n><L> <zeta>`,
  e.g. `3D  2.000`; ζ is in bohr⁻¹. All-electron files are named exactly the
  element symbol (frozen-core variants carry suffixes like `.2p`).

Provenance note: these are ZORA-optimized exponents. For H–Ar relativistic
effects are tiny, so they are the right literature values to use; label them
"ADF standard (ZORA-contracted), van Lenthe & Baerends 2003" in the code
comment, replacing the current unverified label.

## Paste-ready task for Antigravity (MacBook)

> Pull the ADF polarization exponents for H–Ar and swap them into
> `phatpsy/basis.py`.
>
> 1. Download <https://www.scm.com/doc/ADF/_downloads/13a32cb06e43fc959219ca9fb75da628/zorabasis.tar.gz>
>    (free download, no ADF license needed) and extract it.
> 2. For each element H–Ar, take the all-electron files `ZORA/DZP/<El>` and
>    `ZORA/DZ/<El>` (bare element name, no frozen-core suffix).
> 3. Parse each file's `BASIS` section (lines between `BASIS` and `END`;
>    records look like `3D  2.000` = n=3, l=d, ζ=2.000 bohr⁻¹).
> 4. The polarization function is the single `(n, l, ζ)` entry present in DZP
>    but not in DZ (expect 2p for H/He, 3d for Li–Ar — exactly one per element).
> 5. Sanity-check: H must come out ζ=1.25, O ζ=2.00 (published ADF standard
>    values, Gasević et al. JPCA 2022). If they don't, stop and report.
> 6. Replace `POLARIZATION_PARAMETERS` in `phatpsy/basis.py` with the extracted
>    values (same `(n, l, zeta)` tuple format) and fix the comment to read
>    "ADF standard polarization exponents (ZORA-contracted), van Lenthe &
>    Baerends, J. Comput. Chem. 24, 1142 (2003)".
> 7. Re-run the H₂O and CO dipole checks (tests/test_basis_polarized.py);
>    expect the H₂O dipole to move up from 1.18 D toward the 1.855 D
>    experimental value.
>
> A starter script implementing steps 2–5 is in this folder
> (`parse_adf_polarization.py`). It was written from the documented file
> format and has not been run against the real files — verify its output
> (one polarization function per element; H=1.25, O=2.00) before using it.

## What Lux could not do

The tarball is a binary download; Lux's fetch tooling handles page text, not
file downloads, so the actual download-and-parse step is handed to Antigravity
on the MacBook (2-minute job, no license). The DOI landing page for
van Lenthe & Baerends 2003 (doi.org/10.1002/jcc.10255) returned 403 from here.
