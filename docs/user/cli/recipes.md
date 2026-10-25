# Command-Line Recipes

This chapter provides battle-tested CLI recipes and shell pipelines for common computational chemistry workflows.

---

## Recipe 1: End-to-End Molecular Workflow (XYZ $\to$ View)

Convert an external XYZ geometry, build a calculation deck, solve the SCF equations, generate a 3D orbital cube, and visualize the result in your browser:

```bash
# 1. Build a Double-Zeta calculation deck with alternating LVB spin
python -m phatpsy build molecule.xyz -b DZ -s -o deck.yaml

# 2. Run the calculation and save structured results
python -m phatpsy deck.yaml -o results.json

# 3. Extract the Highest Occupied Molecular Orbital (HOMO) as a Gaussian Cube
# Note: replace 7 with the HOMO index from results.json
python -m phatpsy cube deck.yaml --mo 7 -o homo.cube

# 4. Launch the Three.js isosurface visualizer
python -m phatpsy cube deck.yaml --mo 7 --view --isovalue 0.08
```

---

## Recipe 2: Constructing and Running a Chemisorption Slab

Generate a 3-layer $\text{Ni}(100)$ surface slab with carbon monoxide ($CO$) adsorbed on an on-top site, freeze the bottom layer, and compute the deformation density:

```bash
# 1. Construct the slab deck with fixed bottom layer
python -m phatpsy supercell \
  -e Ni -c fcc -a 3.52 -m 1,0,0 -l 3 -r 2,2 -f 1 \
  --adsorbate CO --site top --height 1.85 \
  -b DZ -o ni100_co.yaml

# 2. Verify geometry in the 3D viewer
python -m phatpsy view ni100_co.yaml --view

# 3. Solve the SCF ground state
python -m phatpsy ni100_co.yaml -o ni100_co_res.json

# 4. Generate the 3D deformation density (electron reorganization upon binding)
python -m phatpsy cube ni100_co.yaml --deformation -o deformation.cube --view --isovalue 0.01
```

---

## Recipe 3: Automated Potential Energy Scan with Plotting

Compute a high-resolution 1D bond dissociation curve for diatomic nitrogen and save the energy plot:

```bash
python -m phatpsy scan examples/n2.yaml \
  -b 0,1 \
  -r 1.8:3.2:0.05 \
  -o n2_dissociation.json \
  -p n2_curve.png
```

PHATPSY uses **warm restarts** across scan steps, reusing the converged density matrix of the preceding geometry as the initial guess for the next point. This typically cuts total compute time by 60–80%.

---

## Recipe 4: Batch Processing with Shell Loops

Run calculations across a collection of YAML files and extract total energies and virial ratios using `jq`:

```bash
# Process all decks in a directory
for deck in decks/*.yaml; do
  out="${deck%.yaml}_res.json"
  echo "Calculating $deck -> $out"
  python -m phatpsy "$deck" -o "$out"
done

# Summarize energies and virial ratios into a clean table
echo "File | Energy (Eh) | Virial (-V/T)"
echo "-------------------------------------"
for res in decks/*_res.json; do
  energy=$(jq '.total_energy' "$res")
  virial=$(jq '.virial_ratio' "$res")
  echo "$(basename "$res") | $energy | $virial"
done
```

## Recipe 5: Pure JSON Pipeline over Stdin/Stdout

Integrate PHATPSY directly into Unix data pipelines without touching disk:

```bash
cat examples/n2.yaml | python -m phatpsy - --json-stdout | jq '.total_energy'
```


---

## Recipe 6: Harmonic Vibrational Frequency & Thermochemical Report

Relax a structure to a stationary point, compute its full Cartesian Hessian and normal vibrational modes, verify whether it is a true local minimum ($N_{\text{imag}} = 0$) or a transition state ($N_{\text{imag}} = 1$), and output Gibbs free energy $G(298.15\text{ K})$:

```bash
# 1. Relax geometry using BFGS
python -m phatpsy opt molecule.yaml -o relaxed.yaml

# 2. Compute harmonic frequencies and standard thermochemistry
python -m phatpsy freq relaxed.yaml -T 298.15 -P 1.0 -o vibrations.json --view

# 3. Extract Zero-Point Energy (ZPVE) and Gibbs Free Energy from JSON
echo "ZPVE (kcal/mol): $(jq '.thermochemistry.zero_point_energy_kcal' vibrations.json)"
echo "Gibbs Free Energy (Hartree): $(jq '.thermochemistry.total_gibbs_free_energy_g_t' vibrations.json)"
```

