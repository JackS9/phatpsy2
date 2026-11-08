# Valence configuration for AtomConfig — design spec

**Status:** spec drafted Nov 7, 2026, for Jack's review. Not yet implemented.
**Goal:** record orbital-level electron assignments — occupation and spin per hybrid
orbital, oriented relative to other atoms — in `AtomConfig`, so PHATPSY 2.0 can
describe **non-alternant bonding, mixed spins (O in ethylene oxide), ferromagnetic
bonds, dative/coordinate bonds, and transition-metal complexes** explicitly instead
of inferring them from a per-atom spin label.

**Scope of this step:** record + validate only. `valence_config` does not change any
SCF energy, gradient, or Hamiltonian in this step. A later phase may wire it into
starting-orbital construction; that is out of scope here.

## 1. Data model

New dataclass in `phatpsy/schema.py`, next to `AtomConfig`:

```python
@dataclass
class ValenceOrbital:
    spin: str = "none"
    # 'up' | 'down' | 'none' — same vocabulary as AtomConfig.spin
    orbital_type: str = "sigma"
    # role / symmetry: sigma | pi | delta | lone_pair
    occupation: int = 0
    # electrons in this orbital: 0 (empty), 1 (singly occupied), 2 (doubly occupied)
    connect_to: Optional[str] = None
    # atom label this orbital is directed at / interacts with
    plane: Optional[List[str]] = None
    # None / unknown, or 1-3 atom labels (see section 3)
    orientation: str = "in_plane"
    # in_plane (default) | out_of_plane
```

New optional field on `AtomConfig` (default `None` — fully backward compatible;
every existing deck without it behaves exactly as today):

```python
valence_config: Optional[List[ValenceOrbital]] = None
```

YAML input is a list of dicts under the atom, e.g.:

```yaml
- element: O
  id: O1
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: C1}
    - {spin: down, orbital_type: sigma, occupation: 1, connect_to: C2}
```

### Meaning of `connect_to`, by orbital type

| orbital_type | `connect_to` means |
|---|---|
| sigma / pi / delta | the atom this orbital is directed at: the bond partner (singly-occupied pair partner), or the π-system partner for a singly-occupied radical π orbital |
| lone_pair | dative-donor direction: the atom this pair is donated toward (None = ordinary nonbonding lone pair) |

**Occupation is separate from role.** An empty orbital is `occupation: 0` with a
role recording its symmetry — the vacant p orbital on planar BF₃ is role `pi`,
occupation 0; in the pyramidal H₃N→BF₃ adduct the same orbital is role `sigma`,
occupation 0 (see the BF₃ example in section 7). A vacant orbital with no
`connect_to` is a plain vacant orbital; with `connect_to` it is a vacant acceptor
orbital directed at the donor.

A dative/coordinate bond is therefore **a pair of entries**: a `lone_pair` entry on the
donor with `connect_to` = acceptor, matched by an entry with `occupation: 0` on the
acceptor with `connect_to` = donor (see section 6).

**Migration note.** A temporary `orbital_spins` field exists in an uncommitted version
of `AtomConfig`; it attempted orbital-level spin assignment but is ambiguous and
incomplete. This design supersedes it: the implementation must remove `orbital_spins`
and replace any uses with `valence_config`. No compatibility with `orbital_spins` is
preserved — it was never committed or released.

## 2. Derived quantities (functions, never stored)

```python
NOBLE_CORES = (0, 2, 10, 18, 36, 54, 86)
NOBLE_NEXT  = (2, 10, 18, 36, 54, 86, 118)

core(Z)                  = max(c in NOBLE_CORES if c <= Z)
core_next(Z)             = min(c in NOBLE_NEXT  if c >= Z)
total_valency_available  = core_next - core            # in {0, 2, 8, 18, 32}
n_valence_orbitals       = total_valency_available // 2
occupied_valency         = Z - core - charge           # charge = AtomConfig.charge
```

The noble-core lists are (0, 2, 10, 18, 36, 54, 86) and (2, 10, 18, 36, 54, 86, 118);
`occupied_valency` uses the standard formal-charge sign (`charge` = +1 cation,
−1 anion), so O⁻ gives 8 − 2 − (−1) = 7.

Spot table (charge 0 unless noted):

| atom | Z | core | core_next | total_valency | n_orbitals | occupied |
|---|---|---|---|---|---|---|
| H | 1 | 0 | 2 | 2 | 1 | 1 |
| He | 2 | 2 | 2 | 0 | 0 | 0 |
| C | 6 | 2 | 10 | 8 | 4 | 4 |
| N | 7 | 2 | 10 | 8 | 4 | 5 |
| O | 8 | 2 | 10 | 8 | 4 | 6 |
| Fe | 26 | 18 | 36 | 18 | 9 | 8 |
| Fe(II) | 26 | 18 | 36 | 18 | 9 | 6 |
| Rb | 37 | 36 | 54 | 18 | 9 | 1 |
| Cs | 55 | 54 | 86 | 32 | 16 | 1 |

Derived from a `valence_config` (count, don't store):

```python
orbital_counts   = {t: count of explicitly listed entries with orbital_type == t}  # sigma, pi, delta, lone_pair (roles)
occupancy_counts = (n_empty, n_singly, n_doubly)                  # from the occupation field (0, 1, 2)
net_orbital_spin = (#up) − (#down) over the entries            # per atom
```

The proposal's `hybridization = [sigmas, pis, deltas, lone_pairs, empty]` is exactly
`orbital_counts` plus the empty count — it is **derived, not stored, and not called
`hybridization`**, because `AtomConfig.hybridization` already exists as a string label
(`'sp3'`, `'sp2'`, `'dsp2'`, `'d2sp3'`). The label stays the independent
specification; the counts are validated against it (section 5, check 5). Note the
split the BF₃ case forces: `empty` is an *occupation* (0 electrons), not a role —
an empty orbital still has a symmetry, and the role records it.

## 3. Plane specification

`plane` is `None` (unknown / not applicable) or a list of 1–3 atom labels:

- **1 label:** a direction only — equivalent to `connect_to`. Allowed but redundant;
  prefer `connect_to`.
- **2 labels:** those two atoms plus the current atom define the plane (3 points).
- **3 labels:** those three atoms define a plane that does not include the current atom.

`orientation` is interpreted relative to `plane`: `in_plane` (default) or
`out_of_plane` (perpendicular to the plane). With no plane it is inert and must be
left at the default (check 6 below).

Typical use: a π orbital names `plane: [bond_partner, third_atom]` and
`orientation: out_of_plane`; a σ bond needs only `connect_to`.

## 4. Encoding rules (the contract)

1. `spin` ∈ {'up', 'down', 'none'} — the same vocabulary as `AtomConfig.spin` and
   `build_molecule(spins=…)`. `__post_init__` normalizes 'alpha' → 'up', 'beta' →
   'down', and also accepts the ints +1/−1/0 (mapped to up/down/none) for convenience.
2. `orbital_type` ∈ {sigma, pi, delta, lone_pair} — the orbital's role / symmetry.
   There is no `empty` type: emptiness is an occupation (rule 3).
3. `occupation` ∈ {0, 1, 2} electrons, consistent with `spin`:
   - `occupation == 1` ⟺ `spin` ∈ {'up', 'down'} (singly occupied).
   - `occupation == 0` or `2` ⟺ `spin == 'none'` (empty or doubly occupied).
   Equivalently (`n_alpha`, `n_beta`): 'up' = (1, 0), 'down' = (0, 1), 'none' with
   occupation 2 = (1, 1), 'none' with occupation 0 = (0, 0) — the same convention
   as `starting_orbitals` `occno`.
4. `lone_pair` ⇒ `occupation == 2` (and `spin == 'none'`). A lone pair is doubly occupied
   **by definition**. A singly-occupied nonbonding orbital (e.g. NO's unpaired
   electron on N) is typed by its symmetry — sigma / pi / delta — not called a
   lone pair.
5. `connect_to` must not name the atom itself (checked at deck build, section 5).
6. `orientation == out_of_plane` requires a plane of ≥ 2 labels (a plane needs
   3 points: the current atom + 2, or 3 others).
7. `len(valence_config)` ≤ `n_valence_orbitals`. A shorter list is padded with
   empty entries (role `sigma`, occupation 0). Padded entries count toward the
   orbital total (check 2) but not toward the hybridization-label counts (check 5)
   — only explicitly listed entries count there. `None` means "no valence
   information" (existing behavior).
8. When `valence_config` is present, `charge` must be integer-valued (formal
   charge), so that `occupied_valency` is an integer.

## 5. Sanity checks

### Atom-local (in `AtomConfig.__post_init__` or a `validate()` method)

1. Length: `len(valence_config)` ≤ `n_valence_orbitals`.
2. Orbital count: `n_empty + n_singly + n_doubly == n_valence_orbitals`
   (after padding missing entries as empty).
3. Electron count: `n_singly + 2 * n_doubly == occupied_valency`.
4. Spin sign: `net_orbital_spin` must agree in sign with the atom's `spin` label
   (`up` ⟺ sum > 0, `down` ⟺ sum < 0). **When the sum is 0 there is no constraint
   from `valence_config`** — the label is then fixed by the graph-level alternating
   assignment, as today. (E.g. O in EO has orbital spins 'up', 'down', 'none', 'none' → net 0, yet
   still needs an `up`/`down` label from the solver.)
5. Hybridization-label consistency (only when `AtomConfig.hybridization` is set;
   the `phatpsy/hybrids.py` registry is the single source of truth for
   label → (n_hybrids, n_pi)). Counts are over **explicitly listed** entries only
   (padded empties excluded), and an explicitly listed empty orbital counts under
   its role — the vacant σ on B in H₃N→BF₃ is the fourth sp³ hybrid:
   - `n_sigma + n_lone_pair == n_hybrids` (σ bonds, lone pairs, and vacant σ orbitals occupy the hybrids)
   - `n_pi == n_pi(label)` (sp → 2, sp2 → 1, sp3/sp3d/d2sp3/dsp2 → 0)
   - `n_delta == 0` unless the label contains `d`
6. Encoding rules 1–4, 6–8 from section 4.
7. `starting_orbitals` agreement (same pattern as the existing `spin` vs
   `starting_orbitals` reconciliation in `__post_init__`): when both are present,
   the per-orbital alpha/beta occupations must agree with the `valence_config`
   encoding — (`occupation`, `spin`) maps directly onto `occno` (rule 3). On
   conflict, raise — never silently prefer one.

### Deck-build (cross-atom; labels reference other atoms, so these cannot run in `__post_init__`)

8. Every `connect_to` and `plane` label must name an existing atom id in the molecule.
9. Dative pairing (section 6): a `lone_pair` entry with `connect_to = X` must be
   matched by an entry on X with `occupation == 0` and `connect_to` = this atom
   (the acceptor orbital — role sigma or pi according to the structure). Error if
   X has a `valence_config` and no matching entry; warning if X has no
   `valence_config` (its config may simply be partial).
10. Ferromagnetic declaration: two adjacent atoms whose facing singly-occupied
    orbitals carry the **same** spin (mutual `connect_to`, both 'up' or both 'down') are a
    ferromagnetic coupling — that atom pair must appear in `ferromagnetic_links`
    (cf. `phatpsy/spin.py: detect_ferromagnetic_links`). Same-spin adjacency must
    never be silent.
11. Physical net spin: Σ over all atoms of `net_orbital_spin` equals the molecule's
    physical net spin 2S (e.g. 0 for N2, +1 for NO, +2 per Ni in ferromagnetic bulk).

**Note on check 4 vs the existing `net_spin`.** The proposal's
"sum of valence_config.spin = net_spin" needs a correction: `build_molecule`'s
`net_spin` is `n_up − n_down` counted **over atoms** (one label per atom), not over
electrons — for NO it is 0 while the physical net spin is +1. The two quantities
are different and both stay: per-atom orbital-spin sums constrain the per-atom
labels (check 4), and their molecule-wide sum is the physical 2S (check 11).
The existing deck-level `net_spin` assertion is unchanged.

## 6. Dative / coordinate-bond convention

A dative bond is a **pair of entries**:

- donor: `{spin: none, orbital_type: lone_pair, occupation: 2, connect_to: <acceptor>}`
- acceptor: `{spin: none, orbital_type: sigma, occupation: 0, connect_to: <donor>}` (role sigma or pi according to the structure — pi for planar BF₃, sigma for the pyramidal adduct)

The pair is still counted as donor electrons in check 3: the donor entry is doubly occupied (occupation 2) and the acceptor entry is empty (occupation 0) — the VB picture keeps the pair on the donor. The matching
`connect_to` pair is what makes the bond machine-readable. Unmatched directed
lone pairs are check 9 above.

## 7. Worked examples (YAML)

### EO oxygen — mixed spins (the motivating case)

O: Z=8, core=2, charge 0 → occupied 6, 4 orbitals. Two σ bonds with opposite spins
(1 up to C1, 1 down to C2), two lone pairs:

```yaml
- element: O
  id: O1
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: C1}
    - {spin: down, orbital_type: sigma, occupation: 1, connect_to: C2}
    - {spin: none, orbital_type: lone_pair, occupation: 2, plane: [C1, C2], orientation: in_plane}
    - {spin: none, orbital_type: lone_pair, occupation: 2, plane: [C1, C2], orientation: out_of_plane}
```

Checks: n_singly=2, n_doubly=2 → 2 + 4 = 6 ✓; net orbital spin 0 (label from solver) ✓.
The two carbons carry the opposite spins on their facing σ orbitals (C1: −1 to O,
C2: +1 to O), so each O–C bond is a paired (+1, −1) couple. Along the EO→AA path
the config is re-specified per geometry — the spin narrative (O's spin-down moving
to a hydrogen on C2, etc.) becomes a sequence of `valence_config`s.

### NO radical — non-alternant π, unpaired electron on N

N: occupied 5, 4 orbitals. The σ and π bond orbitals carry opposite spins across
the bond (alternating), and the third singly-occupied π on N is the radical:

```yaml
- element: N
  id: N1
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: O1}
    - {spin: down, orbital_type: pi, occupation: 1, connect_to: O1}
    - {spin: up, orbital_type: pi, occupation: 1, connect_to: O1}
    - {spin: none, orbital_type: lone_pair, occupation: 2}
- element: O
  id: O1
  valence_config:
    - {spin: down, orbital_type: sigma, occupation: 1, connect_to: N1}
    - {spin: up, orbital_type: pi, occupation: 1, connect_to: N1}
    - {spin: none, orbital_type: lone_pair, occupation: 2}
    - {spin: none, orbital_type: lone_pair, occupation: 2}
```

N: net orbital spin +1 → label `up` ✓; O: 0 (label from solver) ✓; molecule Σ = +1 = 2S ✓.
(For a diatomic the π plane is degenerate — any plane containing the bond axis — so no plane is recorded; the two π orbitals on N are distinguished by spin.)

### Dative bond — NH₃→BH₃

```yaml
- element: N
  id: N1
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: H1}
    - {spin: down, orbital_type: sigma, occupation: 1, connect_to: H2}
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: H3}
    - {spin: none, orbital_type: lone_pair, occupation: 2, connect_to: B1}   # donated pair
- element: B
  id: B1
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: H4}
    - {spin: down, orbital_type: sigma, occupation: 1, connect_to: H5}
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: H6}
    - {spin: none, orbital_type: sigma, occupation: 0, connect_to: N1}       # acceptor orbital
```

The (lone_pair → B1, empty → N1) pair is the dative bond (check 9).

### Dative bond with rehybridization — BF₃ + NH₃ → H₃N→BF₃

Free BF₃: B is sp², trigonal planar. The acceptor orbital is a vacant p orbital
perpendicular to the BF₃ plane — role pi, occupation 0:

```yaml
- element: B
  id: B1
  hybridization: sp2
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: F1}
    - {spin: down, orbital_type: sigma, occupation: 1, connect_to: F2}
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: F3}
    - {spin: none, orbital_type: pi, occupation: 0, plane: [F1, F2, F3], orientation: out_of_plane}
```

Checks: n_sigma + n_lone_pair = 3 == n_hybrids(sp2) ✓; n_pi = 1 ✓; 3 + 0 = 3
electrons ✓. The empty π entry names the BF₃ plane with three labels (not
including B) and sits out of it.

In the adduct, B rehybridizes to sp³ (pyramidal) and the vacant orbital becomes the
fourth sp³ hybrid — role sigma, still occupation 0 — while N's lone pair is the
donated pair:

```yaml
- element: B
  id: B1
  hybridization: sp3
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: F1}
    - {spin: down, orbital_type: sigma, occupation: 1, connect_to: F2}
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: F3}
    - {spin: none, orbital_type: sigma, occupation: 0, connect_to: N1}   # acceptor orbital, now sp3
- element: N
  id: N1
  hybridization: sp3
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: H1}
    - {spin: down, orbital_type: sigma, occupation: 1, connect_to: H2}
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: H3}
    - {spin: none, orbital_type: lone_pair, occupation: 2, connect_to: B1}  # donated pair
```

Checks: B: n_sigma = 4 == n_hybrids(sp3) ✓ (the empty σ counts — it occupies the
fourth hybrid); N: 3 σ + 1 lone pair = 4 ✓; dative pairing (check 9) matches N's
lone_pair→B with B's occupation-0 σ→N ✓.

The point: the empty orbital keeps its role (pi → sigma) across the change, and the
hybridization label is a property of the atom **in this structure** — sp² in free
BF₃, sp³ in the adduct. A `valence_config` is a per-geometry snapshot; along a
reaction path it is re-specified per geometry, exactly as for the EO→AA path.

### Transition metal — Fe(II), octahedral, high-spin d⁶

Fe: Z=26, core=18, charge +2 → occupied 6, 9 orbitals. The five d orbitals are typed
relative to the ligand axes: e_g (dz², dx²−y²) are σ-type, pointing at ligands;
t_2g (dxy, dxz, dyz) are π-type, pointing between ligands. 4s/4p stay empty here:

```yaml
- element: Fe
  id: Fe1
  charge: 2.0
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: L1}   # dz2 -> axial ligand
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: L2}   # dx2-y2 -> equatorial ligand
    - {spin: up, orbital_type: pi, occupation: 1, connect_to: L3, plane: [L3, L4], orientation: out_of_plane}
    - {spin: up, orbital_type: pi, occupation: 1, connect_to: L4, plane: [L3, L4], orientation: out_of_plane}
    - {spin: none, orbital_type: pi, occupation: 2, connect_to: L5, plane: [L5, L6], orientation: out_of_plane}
    - {spin: none, orbital_type: sigma, occupation: 0}   # 4s
    - {spin: none, orbital_type: sigma, occupation: 0}   # 4px
    - {spin: none, orbital_type: sigma, occupation: 0}   # 4py
    - {spin: none, orbital_type: sigma, occupation: 0}   # 4pz
```

Checks: n_singly=4, n_doubly=1 → 4 + 2 = 6 ✓ (high-spin d⁶, t_2g⁴e_g², 4 unpaired).
Low-spin d⁶ would be the same five d entries with three doubly-occupied π and no
singly-occupied d orbitals — the spin state becomes data, not inference. The same
machinery covers δ bonds (M–M quadruple bonds: a δ entry is a d-orbital overlap
perpendicular to the M–M axis, `connect_to` = the other metal) and crystal-field
splitting generally, since each d orbital names the ligand it points at or between.

### Ferromagnetic pair — Ni bulk (schematic)

Two adjacent Ni atoms with facing singly-occupied σ orbitals carrying the **same**
spin. Each Ni: Z=28, core=18, occupied 10, 9 orbitals; only the facing orbitals
shown, remainder doubly-occupied/empty as counted:

```yaml
- element: Ni
  id: Ni1
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: Ni2}
      # ... remaining 8 orbitals: 2 more singly-occupied (+1), rest doubly/empty
- element: Ni
  id: Ni2
  valence_config:
    - {spin: up, orbital_type: sigma, occupation: 1, connect_to: Ni1}
      # ... same
```

The (Ni1, Ni2) pair **must** appear in `ferromagnetic_links` (check 10) — the
ferromagnetic coupling is declared, never silent. This is the per-orbital form of
the baseline spec's "Ni bulk: all spins up".

## 8. Relation to existing fields (reconciliation chain)

| existing field | relation to `valence_config` |
|---|---|
| `AtomConfig.spin` ('up'/'down'/None) | one label per atom. Constrained by check 4 (sign of orbital-spin sum when nonzero); otherwise fixed by the graph solver as today. `valence_config` does **not** replace the per-atom label requirement — `build_molecule` still raises on `None` for unrestricted decks. |
| `AtomConfig.hybridization` (string label) | independent specification; validated against the derived counts (check 5). The `hybrids.py` registry is the source of truth for label → (n_hybrids, n_pi). |
| `AtomConfig.starting_orbitals` (per-orbital alpha/beta `occno`) | the computational-level occupation record. When both are present they must agree (check 7). `valence_config` is the chemical-level record; `starting_orbitals` is the basis-level record. |
| `build_molecule(spins=…, net_spin=…)` | unchanged. Deck `net_spin` stays the atom-label balance; physical 2S is check 11. |
| `ferromagnetic_links` / `spin.py` | unchanged, plus check 10: same-spin facing orbitals force the pair into `ferromagnetic_links`. |
| Graphical builder (goal_bee84a65390d) | the builder must be able to construct and edit `valence_config` (atom equivalency, virtual/ghost and fixed atoms included — a ghost atom's `valence_config` is still meaningful as embedding info). Builder work is a separate step; this spec only defines the data it will edit. |

### Automation and the explicit record

Automation proposes; the record disposes.

- The existing automation (`assign_alternating_spin`, hybridization assigned from
  geometry) may **propose** a `valence_config`, but the proposal is materialized as
  data — a `propose_valence_config()` builder helper that returns the explicit list
  of entries — so the user can review and edit it before it becomes the record.
  Nothing automated is silently assumed: if it isn't in the record, it isn't in
  the calculation.
- Precedence: an explicit `valence_config` wins over automation for the entries it
  specifies; automation fills only unspecified orbitals (default empty) and atoms
  with no config at all (which behave exactly as today).
- A dative bond is never invented by automation: it is user-specified (or confirmed
  from a proposal). A distance-based heuristic may propose one, but the record only
  carries it once the user accepts it.
- Rehybridization is geometry-driven: the automated label follows the geometry
  (planar BF₃ → sp²; pyramidal adduct → sp³), and the user may override the label
  and the orbital roles per structure. The config is a per-geometry snapshot.

## 9. Acceptance tests

New `tests/test_valence_config.py` (or equivalent), all passing under `./make_build_test.sh`:

1. **Formula table:** `core` / `core_next` / `total_valency_available` /
   `n_valence_orbitals` for spot Z values (1, 2, 6, 7, 8, 18, 19, 26, 37, 54, 55, 86, 87)
   — including the Rb (Z=37) case that catches the missing-54 bug, and He (0 orbitals).
2. **Occupied valency:** Fe(II) → 6, Fe(III) → 5, O⁻ → 7, N⁺ → 4
   (catches the sign error).
3. **Examples validate:** the EO oxygen, NO pair, NH₃→BH₃ pair, BF₃ free/adduct pair, and Fe(II) high-spin
   configs from section 7 all pass checks 1–7; the NO molecule passes check 11 (Σ=+1).
4. **Dative pairing:** NH₃→BH₃ passes check 9; removing B's `empty` entry makes the
   deck build raise; removing B's whole `valence_config` downgrades it to a warning.
5. **Ferromagnetic declaration:** two Ni atoms with same-spin facing σ orbitals and
   no `ferromagnetic_links` entry → deck build raises; with the entry → passes.
6. **Error cases, each raising `ValueError`:** `spin: 'sideways'`; `lone_pair` with
   `occupation: 1`; `occupation: 1` with `spin: 'none'`; `occupation: 0` with `spin: 'up'`;
   `occupation: 2` with `spin: 'down'`; `out_of_plane` with no plane; list longer than
   `n_valence_orbitals`; `connect_to` naming a nonexistent atom; electron count
   mismatch (e.g. O with 5 counted valence electrons); `spin` label contradicting
   a nonzero orbital-spin sum.
7. **Backward compatibility:** every existing test and example deck (no
   `valence_config` anywhere) passes unchanged; `AtomConfig` round-trips through
   `to_dict` / YAML with the new field present and absent.
8. **Rehybridization:** free BF₃ (sp², empty π) and the H₃N→BF₃ adduct (sp³, empty σ)
   both validate; free BF₃ labeled sp³ fails check 5; the adduct passes the
   dative-pairing check.
9. **Automation:** `propose_valence_config()` returns the automated assignment as an
   explicit list; editing one entry and rebuilding uses the edited value; an explicit
   config overrides the automated spin assignment for the atoms it covers; automation
   never creates a dative entry unprompted.

## 10. Guardrails

- **Record + validate only.** Nothing in this step changes SCF, gradients, energies,
  or the Hamiltonian. Any later wiring of `valence_config` into starting orbitals
  is a separate phase with its own spec.
- **No silent approximations, no widened tolerances.** All checks raise (or warn
  where stated); no check may be weakened to make an example pass.
- **Anti-Circularity Invariant** (`docs/development/validation_standards.md` §9):
  the section-7 examples are acceptance fixtures, not calibration targets — no
  acceptance value may appear as a constant in the implementation.
- **Docs-first:** document `ValenceOrbital` and the encoding rules in the manual
  (MkDocs) alongside the code, per the project's docs-first rule for scientific
  routines.
- **Verification loop:** `./make_build_test.sh` on the final state, output shown.

---

## 11. Paste-ready prompt for Antigravity

> Implement orbital-level valence configuration for PHATPSY 2.0, per the spec in
> `workspace/goals/modernize-phatpsy/files/valence-config-spec.md` (read it first —
> it is the acceptance spec).
>
> 1. In `phatpsy/schema.py`, add a `ValenceOrbital` dataclass (`spin: str = "none"`,
>    `orbital_type: str = "sigma"`, `occupation: int = 0`,
>    `connect_to: Optional[str] = None`,
>    `plane: Optional[List[str]] = None`, `orientation: str = "in_plane"`) and an
>    optional `valence_config: Optional[List[ValenceOrbital]] = None` field on
>    `AtomConfig`. YAML dicts must parse into it (follow the existing
>    `hybridization` dict-parsing pattern in `builder.py`); `to_dict` must
>    round-trip it. Remove the temporary uncommitted
>    `orbital_spins` field from `AtomConfig` as part of this change; do not preserve
>    compatibility with it.
> 2. Add pure helper functions for the section-2 derived quantities
>    (`core`/`core_next`/`total_valency_available`/`n_valence_orbitals`/
>    `occupied_valency`, `orbital_counts`, `occupancy_counts`,
>    `net_orbital_spin`). Do not store derived values on the dataclass.
> 3. Implement the section-4 encoding rules and section-5 atom-local checks
>    (1–7) in `AtomConfig.__post_init__` or a `validate()` method it calls, and
>    the deck-build checks (8–11) in `build_molecule` (cross-atom label checks,
>    dative pairing, ferromagnetic-link enforcement, physical-2S sum). Reuse the
>    `hybrids.py` registry for check 5; reuse the existing `spin` vs
>    `starting_orbitals` reconciliation pattern for check 7. All failures raise
>    `ValueError` (warning only where the spec says warning). Also implement
>    `propose_valence_config()` (section 8): it materializes the automated
>    spin/hybridization assignment as an explicit list of entries for review and
>    editing; explicit configs override automation, and automation never invents a
>    dative entry.
> 4. Add `tests/test_valence_config.py` covering every item in section 9,
>    including the Rb missing-54 case, the O⁻ sign case, the BF₃ rehybridization
>    case, the automation cases, and all error cases.
> 5. Document `ValenceOrbital`, the encoding rules, and the dative convention in
>    the MkDocs manual.
>
> Guardrails: this step is record + validate ONLY — do not change any SCF, gradient,
> energy, or Hamiltonian code path, and do not touch `data/validation_baselines/*.json`.
> Backward compatible: `valence_config=None` must leave every existing deck and test
> behaving exactly as today. No silent approximations, no widened tolerances, no
> acceptance value as a constant (Anti-Circularity Invariant). Run
> `./make_build_test.sh` on the final state and show the output.
