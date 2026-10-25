# phatpsy/cli.py
"""
Command line interface for the PHATPSY Python driver and format converter.
Allows executing calculations from YAML, JSON, or legacy .stdin files,
and converting between deck formats.
"""
import argparse
import sys
from pathlib import Path
from typing import Union, List, Optional
from phatpsy.driver import PhatpsyDriver
from phatpsy.io import save_results_to_json, parse_stdin_to_deck
from phatpsy.schema import CalculationDeck


def load_deck_from_arg(input_file: Union[str, Path], auto_spin: bool = False) -> CalculationDeck:
    in_path = Path(input_file)
    suffix = in_path.suffix.lower()

    if suffix == ".xyz":
        from phatpsy.builder import build_molecule
        deck = build_molecule(in_path, auto_spin_flip=auto_spin)
    elif suffix in (".yaml", ".yml"):
        deck = CalculationDeck.from_yaml(in_path)
        if auto_spin:
            deck.assign_alternating_spin()
    elif suffix == ".json":
        deck = CalculationDeck.from_json(in_path)
        if auto_spin:
            deck.assign_alternating_spin()
    elif suffix in (".stdin", ".dat", ".txt") or (not suffix and in_path.exists()):
        deck = parse_stdin_to_deck(in_path)
        if auto_spin:
            deck.assign_alternating_spin()
    else:
        deck = parse_stdin_to_deck(in_path)
        if auto_spin:
            deck.assign_alternating_spin()
    return deck


def handle_convert(args):
    in_path = Path(args.input_file)
    deck = load_deck_from_arg(in_path)

    out_path = Path(args.output) if args.output else None
    out_suffix = out_path.suffix.lower() if out_path else ".yaml"

    if out_suffix in (".yaml", ".yml"):
        text = deck.to_yaml(out_path)
    elif out_suffix == ".json":
        text = deck.to_json(out_path)
    elif out_suffix == ".stdin":
        text = deck.to_stdin_text()
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
    else:
        text = deck.to_yaml(out_path)

    if not out_path:
        sys.stdout.write(text)
    else:
        sys.stderr.write(f"PHATPSY: Converted '{in_path}' -> '{out_path}'\n")
    return 0


def handle_elements(args):
    from phatpsy.basis import list_available_elements, get_element_basis

    if args.symbol:
        sym = args.symbol.strip().capitalize()
        try:
            ed = get_element_basis(sym, basis_name=args.basis)
        except KeyError as e:
            sys.stderr.write(f"Error: {e}\n")
            return 1
        print(f"Element: {ed.symbol} (Z={ed.atomic_number})")
        print(f"Basis Set: {ed.basis_name} - {ed.description}")
        print(f"Functions (NSTO): {ed.nsto} (MSTO={ed.msto})")
        print("  n   l       zeta")
        for f in ed.functions:
            print(f"{f.n:3d} {f.l:3d} {f.zeta:10.5f}")
        if ed.default_orbitals:
            print(f"Default Orbitals (NORB={len(ed.default_orbitals)}):")
            for i, orb in enumerate(ed.default_orbitals, 1):
                eig_str = f"{orb.eigval[0]:.4f}" if orb.eigval else "N/A"
                occ_str = f"up={orb.occno[0]:.2f}, down={orb.occno[1]:.2f}" if orb.occno else "N/A"
                print(f"  Orbital {i:2d}: EIGVAL={eig_str} au, OCCNO={occ_str}")
        return 0

    elems = list_available_elements()
    print(f"PHATPSY Element Library ({len(elems)} elements available):")
    print(f"{'Symbol':<6} {'Available Basis Sets'}")
    print("-" * 35)
    for sym, bases in elems.items():
        print(f"{sym:<6} {', '.join(bases)}")
    return 0


def handle_build(args):
    from phatpsy.builder import build_molecule
    deck = build_molecule(
        args.xyz_file,
        basis=args.basis,
        title=args.title or f"PHATPSY calculation for {Path(args.xyz_file).stem}",
        units=args.units,
        auto_spin_flip=args.auto_spin,
    )
    out_path = Path(args.output) if args.output else None
    out_suffix = out_path.suffix.lower() if out_path else ".yaml"

    if out_suffix in (".yaml", ".yml"):
        text = deck.to_yaml(out_path)
    elif out_suffix == ".json":
        text = deck.to_json(out_path)
    elif out_suffix == ".stdin":
        text = deck.to_stdin_text()
        if out_path:
            out_path.write_text(text, encoding="utf-8")
    else:
        text = deck.to_yaml(out_path)

    if not out_path:
        sys.stdout.write(text)
    else:
        sys.stderr.write(f"PHATPSY: Built deck '{out_path}' from '{args.xyz_file}' (basis={args.basis})\n")
    return 0


def handle_view(args):
    from phatpsy.visualizer import visualize_deck
    deck = load_deck_from_arg(args.input_file, auto_spin=args.auto_spin)
    out_path = Path(args.output) if args.output else None
    auto_open = not args.no_browser
    visualize_deck(deck, output_path=out_path, auto_open=auto_open)
    if out_path:
        sys.stderr.write(f"PHATPSY: Wrote 3D HTML visualization to '{out_path}'\n")
    return 0


def handle_scan(args):
    from phatpsy.pes import scan_1d, scan_2d
    deck = load_deck_from_arg(args.input_file)

    def parse_range(range_str: str) -> List[float]:
        parts = [float(x.strip()) for x in range_str.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 0.1 if stop >= start else -0.1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Invalid range format: '{range_str}'. Expected 'start:stop:step'.")
        n = int(round(abs(stop - start) / abs(step))) + 1
        return [round(start + i * step, 6) for i in range(n)]

    coord1 = None
    if args.bond:
        parts = [int(x.strip()) for x in args.bond.replace("-", ",").split(",")]
        coord1 = ("bond", parts[0], parts[1])
    elif args.angle:
        parts = [int(x.strip()) for x in args.angle.replace("-", ",").split(",")]
        coord1 = ("angle", parts[0], parts[1], parts[2])
    elif args.coord:
        coord1 = args.coord

    if not coord1:
        sys.stderr.write("Error: Must specify coordinate to scan via --bond, --angle, or --coord.\n")
        return 1

    values1 = parse_range(args.range)

    if args.coord2 or args.bond2:
        coord2 = None
        if args.bond2:
            parts = [int(x.strip()) for x in args.bond2.replace("-", ",").split(",")]
            coord2 = ("bond", parts[0], parts[1])
        else:
            coord2 = args.coord2
        if not args.range2:
            sys.stderr.write("Error: Must specify --range2 for 2D scan.\n")
            return 1
        values2 = parse_range(args.range2)
        res = scan_2d(deck, coord1, values1, coord2, values2, warm_restart=not args.no_warm_restart)
        print(f"2D PES Scan Complete: {res.coord1_name} x {res.coord2_name}")
        print(f"  Grid Size:    {len(values1)} x {len(values2)} = {len(values1)*len(values2)} points")
        print(f"  Minimum:      {res.min_values} -> Energy = {res.min_energy:16.8f} Hartree")
        if args.output:
            res.save_json(args.output)
            print(f"  Saved JSON:   {args.output}")
        if args.plot:
            res.plot(output_path=args.plot)
            print(f"  Saved Plot:   {args.plot}")
        return 0

    res = scan_1d(deck, coord1, values1, warm_restart=not args.no_warm_restart)
    print(f"1D PES Scan Complete: {res.coordinate_name}")
    print(f"  Points:       {len(values1)}")
    print(f"  Min Value:    {res.min_value:.4f} Bohr/Rad (Energy = {res.min_energy:16.8f} Hartree)")
    if abs(res.equilibrium_value - res.min_value) > 1e-4:
        print(f"  Equilibrium:  {res.equilibrium_value:.4f} Bohr/Rad (parabolic fit)")
    print(f"{'Value':<10} {'Energy (au)':<18} {'-V/T Ratio':<12} {'Converged'}")
    print("-" * 52)
    for v, e, vr, c in zip(res.values, res.energies, res.virial_ratios, res.converged):
        print(f"{v:<10.4f} {e:<18.8f} {vr:<12.5f} {c}")

    if args.output:
        res.save_json(args.output)
        print(f"Saved JSON to {args.output}")
    if args.plot:
        res.plot(output_path=args.plot)
        print(f"Saved plot to {args.plot}")
    return 0


def handle_opt(args):
    from phatpsy.optimize import optimize_geometry
    deck = load_deck_from_arg(args.input_file)

    opt = optimize_geometry(
        deck,
        max_steps=args.max_steps,
        grad_tol=args.grad_tol,
        energy_tol=args.energy_tol,
        warm_restart=not args.no_warm_restart,
    )

    print(f"Geometry Optimization {'Converged' if opt.converged else 'Incomplete'}:")
    print(f"  Steps:          {opt.num_steps}")
    print(f"  Initial Energy: {opt.initial_energy:18.8f} Hartree")
    print(f"  Final Energy:   {opt.final_energy:18.8f} Hartree")
    print(f"  Energy Delta:   {opt.energy_change:18.8f} Hartree")

    if args.trajectory:
        opt.save_json(args.trajectory)
        print(f"  Saved Trajectory: {args.trajectory}")
    if args.plot:
        opt.plot_convergence(output_path=args.plot)
        print(f"  Saved Plot:       {args.plot}")

    out_path = Path(args.output) if args.output else None
    if out_path:
        out_suffix = out_path.suffix.lower()
        if out_suffix in (".yaml", ".yml"):
            opt.optimized_deck.to_yaml(out_path)
        elif out_suffix == ".json":
            opt.optimized_deck.to_json(out_path)
        elif out_suffix == ".stdin":
            out_path.write_text(opt.optimized_deck.to_stdin_text(), encoding="utf-8")
        else:
            opt.optimized_deck.to_yaml(out_path)
        print(f"  Optimized Deck: {out_path}")
    elif not args.trajectory and not args.plot:
        sys.stdout.write(opt.optimized_deck.to_yaml())

    return 0


def handle_neb(args):
    from phatpsy.neb import neb_search

    initial_deck = load_deck_from_arg(args.initial)
    final_deck = load_deck_from_arg(args.final)

    res = neb_search(
        initial_deck=initial_deck,
        final_deck=final_deck,
        num_images=args.images,
        k_spring=args.spring,
        max_iterations=args.max_iter,
        force_tol=args.force_tol,
        climb=not args.no_climb,
        warm_restart=not args.no_warm_restart,
    )

    print(f"Nudged Elastic Band {'Converged' if res.converged else 'Incomplete'}:")
    print(f"  Iterations:       {res.num_iterations}")
    print(f"  Images:           {len(res.images)}")
    print(f"  Climbing Image:   {'Enabled' if res.climbing_image_enabled else 'Disabled'}")
    print(f"  Transition State: Image {res.transition_state_index}")
    print(f"  TS Energy:        {res.transition_state_energy:18.8f} Hartree")
    print(f"  Forward Barrier:  {res.barrier_forward:18.8f} Hartree ({res.barrier_forward_kcal:8.2f} kcal/mol)")
    print(f"  Reverse Barrier:  {res.barrier_reverse:18.8f} Hartree ({res.barrier_reverse_kcal:8.2f} kcal/mol)")
    print(f"  Reaction Energy:  {res.reaction_energy:18.8f} Hartree ({res.reaction_energy_kcal:8.2f} kcal/mol)")

    if args.output:
        res.save_json(args.output)
        print(f"  Saved JSON:       {args.output}")

    if args.xyz:
        res.save_xyz(args.xyz)
        print(f"  Saved XYZ Path:   {args.xyz}")

    if args.plot:
        res.plot_pathway(output_path=args.plot)
        print(f"  Saved Plot:       {args.plot}")

    if args.view:
        res.visualize(open_browser=True)

    return 0


def handle_md(args):
    from phatpsy.md import run_md

    deck = load_deck_from_arg(args.input)

    res = run_md(
        deck=deck,
        time_step_fs=args.dt,
        num_steps=args.steps,
        temperature_k=args.temp,
        ensemble=args.thermostat,
        tau_fs=args.tau,
        remove_com_drift=not args.no_com_removal,
        warm_restart=not args.no_warm_restart,
    )

    print(f"Born-Oppenheimer Molecular Dynamics ({res.ensemble.upper()}):")
    print(f"  Total Steps:      {res.num_steps} ({res.total_time_fs:.2f} fs)")
    print(f"  Time Step (dt):   {res.time_step_fs:.3f} fs")
    print(f"  Initial Energy:   {res.initial_energy:18.8f} Hartree ({res.initial_energy * 627.509474:.2f} kcal/mol)")
    print(f"  Final Energy:     {res.final_energy:18.8f} Hartree ({res.final_energy * 627.509474:.2f} kcal/mol)")
    print(f"  Energy Drift:     {res.energy_drift:18.8f} Hartree ({res.energy_drift * 627.509474:+.4f} kcal/mol)")
    print(f"  Average Temp:     {res.average_temperature:8.2f} K (± {res.temperature_std:.2f} K)")
    print(f"  Conserved E Std:  {res.conserved_energy_std:18.8f} Hartree ({res.conserved_energy_std * 627.509474:.4f} kcal/mol)")

    if args.output:
        res.save_json(args.output)
        print(f"  Saved JSON:       {args.output}")

    if args.xyz:
        res.save_xyz(args.xyz)
        print(f"  Saved Trajectory: {args.xyz}")

    if args.plot:
        res.plot_dynamics(output_path=args.plot)
        print(f"  Saved Plot:       {args.plot}")

    if args.view:
        res.visualize(open_browser=True)

    return 0


def handle_vibrations(args):
    in_path = Path(args.input)
    deck = load_deck_from_arg(in_path)

    from phatpsy.driver import PhatpsyDriver
    from phatpsy.vibrations import compute_vibrations

    driver = PhatpsyDriver(quiet=True)
    vib_res = compute_vibrations(
        deck=deck,
        driver=driver,
        step_size=args.step,
        method=args.method,
        project_tr=not args.no_project,
        temperature_k=args.temp,
        pressure_atm=args.pressure,
        quiet=True,
    )

    print(vib_res.summary())

    if args.output:
        vib_res.save_json(args.output)
        print(f"\n  Saved Vibrations JSON: {args.output}")

    if args.view:
        vib_res.visualize(open_browser=True)

    return 0


def handle_irc(args):
    in_path = Path(args.input)
    deck = load_deck_from_arg(in_path)

    from phatpsy.driver import PhatpsyDriver
    from phatpsy.irc import compute_irc

    driver = PhatpsyDriver(quiet=True)
    irc_res = compute_irc(
        deck=deck,
        driver=driver,
        step_size=args.step,
        max_steps=args.max_steps,
        direction=args.direction,
        grad_tol=args.grad_tol,
        finite_diff_step=args.diff_step,
        quiet=True,
    )

    print(irc_res.summary())

    if args.output:
        irc_res.save_json(args.output)
        print(f"\n  Saved IRC JSON:       {args.output}")

    if args.xyz:
        irc_res.save_xyz(args.xyz)
        print(f"  Saved IRC Trajectory: {args.xyz}")

    if args.plot:
        irc_res.plot_pathway(output_path=args.plot)
        print(f"  Saved Pathway Plot:   {args.plot}")

    if args.view:
        irc_res.visualize(open_browser=True)

    return 0


def handle_supercell(args):
    from phatpsy.supercell import PeriodicCell, build_slab, attach_adsorbate
    from phatpsy.spin import BOHR_PER_ANGSTROM

    repeat_str = args.repeat or ("2,2,2" if args.bulk else "2,2")
    rep_parts = [int(p) for p in repeat_str.lower().replace("x", " ").replace(",", " ").split() if p]
    if args.bulk:
        if len(rep_parts) == 1:
            repeat_3d = (rep_parts[0], rep_parts[0], rep_parts[0])
        elif len(rep_parts) == 3:
            repeat_3d = (rep_parts[0], rep_parts[1], rep_parts[2])
        else:
            repeat_3d = (2, 2, 2)
    else:
        if len(rep_parts) == 1:
            repeat_2d = (rep_parts[0], rep_parts[0])
        elif len(rep_parts) >= 2:
            repeat_2d = (rep_parts[0], rep_parts[1])
        else:
            repeat_2d = (2, 2)

    if args.bulk:
        cell = PeriodicCell.bulk(
            element=args.element,
            crystal_structure=args.crystal,
            a=args.lattice_constant,
        )
        cell = cell.build_supercell(repeat=repeat_3d, auto_spin=args.auto_spin)
    else:
        m_str = args.miller.replace(",", " ").replace("-", " -").strip()
        m_parts = [int(p) for p in m_str.split() if p]
        if len(m_parts) == 1 and len(m_str) == 3:
            m_parts = [int(c) for c in m_str]
        if len(m_parts) != 3:
            sys.stderr.write(f"Error: Invalid Miller index '{args.miller}'. Expected 3 integers (e.g. '1,0,0' or '111').\n")
            return 1
        miller = (m_parts[0], m_parts[1], m_parts[2])

        cell = build_slab(
            element=args.element,
            crystal_structure=args.crystal,
            a=args.lattice_constant,
            miller_index=miller,
            layers=args.layers,
            vacuum=args.vacuum,
            repeat=repeat_2d,
            fixed_layers=args.fixed_layers,
            add_virtual_perimeter=args.virtual_perimeter,
        )

    if args.adsorbate:
        cell = attach_adsorbate(
            cell,
            adsorbate=args.adsorbate,
            site_type=args.site,
            height=args.height,
            tilt_deg=args.tilt,
        )

    out_path = Path(args.output) if args.output else None
    out_suffix = out_path.suffix.lower() if out_path else None

    fmt = (args.format or (out_suffix[1:] if out_suffix else "yaml")).lower()

    if fmt == "poscar" or out_suffix == ".poscar":
        content = cell.to_poscar()
        if out_path:
            out_path.write_text(content, encoding="utf-8")
        else:
            sys.stdout.write(content)
    elif fmt == "cif" or out_suffix == ".cif":
        content = cell.to_cif()
        if out_path:
            out_path.write_text(content, encoding="utf-8")
        else:
            sys.stdout.write(content)
    elif fmt == "xyz" or out_suffix == ".xyz":
        content = cell.to_xyz()
        if out_path:
            out_path.write_text(content, encoding="utf-8")
        else:
            sys.stdout.write(content)
    else:
        deck = cell.to_deck(basis=args.basis)
        if fmt in ("stdin",) or out_suffix == ".stdin":
            content = deck.to_stdin_text()
            if out_path:
                out_path.write_text(content, encoding="utf-8")
            else:
                sys.stdout.write(content)
        elif fmt in ("json",) or out_suffix == ".json":
            content = deck.to_json(out_path)
            if not out_path:
                sys.stdout.write(content)
        else:
            content = deck.to_yaml(out_path)
            if not out_path:
                sys.stdout.write(content)

    if out_path:
        a_aa = cell.lattice.a / BOHR_PER_ANGSTROM
        b_aa = cell.lattice.b / BOHR_PER_ANGSTROM
        c_aa = cell.lattice.c / BOHR_PER_ANGSTROM
        sys.stderr.write(
            f"PHATPSY: Generated {len(cell.atoms)} atoms ({a_aa:.3f} x {b_aa:.3f} x {c_aa:.3f} A) -> '{out_path}'\n"
        )

    if args.view:
        from phatpsy.visualizer import visualize_deck
        deck = cell.to_deck(basis=args.basis)
        visualize_deck(deck, auto_open=not args.no_browser)

    return 0


def handle_cube(args):
    in_path = Path(args.input_file)
    deck = load_deck_from_arg(in_path)

    driver = PhatpsyDriver()
    res = driver.run_deck(deck)
    if res is None:
        sys.stderr.write("PHATPSY: Calculation failed or returned no results.\n")
        return 1

    from phatpsy.volumetric import compute_grid
    from phatpsy.visualizer import visualize_grid

    # Check property-colored density
    property_grid = None
    if args.esp and (args.density or not (args.mo or args.ao or args.spin_density or args.deformation)):
        grid = compute_grid(res, grid_type="density", spacing=args.spacing, padding=args.padding)
        property_grid = compute_grid(res, grid_type="esp", spacing=args.spacing, padding=args.padding)
    elif args.ao:
        grid = compute_grid(res, grid_type="ao", ao_index=args.ao, spacing=args.spacing, padding=args.padding)
    elif args.mo:
        grid = compute_grid(res, grid_type="mo", mo_index=args.mo, spacing=args.spacing, padding=args.padding)
    elif args.spin_density:
        grid = compute_grid(res, grid_type="spin_density", spacing=args.spacing, padding=args.padding)
    elif args.deformation:
        grid = compute_grid(res, grid_type="deformation", spacing=args.spacing, padding=args.padding)
    elif args.esp:
        grid = compute_grid(res, grid_type="esp", spacing=args.spacing, padding=args.padding)
    else:
        grid = compute_grid(res, grid_type="density", spacing=args.spacing, padding=args.padding)

    out_path = Path(args.output) if args.output else None
    if out_path and out_path.suffix.lower() == ".html":
        visualize_grid(grid, isovalue=args.isovalue, property_grid=property_grid, output_path=out_path, auto_open=not args.no_browser)
        sys.stderr.write(f"PHATPSY: Interactive 3D visualization saved to '{out_path}'\n")
    elif out_path:
        grid.to_cube(out_path)
        sys.stderr.write(f"PHATPSY: Gaussian Cube exported to '{out_path}'\n")
        sys.stderr.write(f"  Field:       {grid.title}\n")
        sys.stderr.write(f"  Dimensions:  {grid.dimensions[0]} x {grid.dimensions[1]} x {grid.dimensions[2]}\n")
        sys.stderr.write(f"  Integral:    {grid.integrate():12.6f}\n")
    else:
        tag = "esp_density" if property_grid else ("mo" + str(args.mo) if args.mo else ("ao" + str(args.ao) if args.ao else ("spin" if args.spin_density else ("deformation" if args.deformation else ("esp" if args.esp else "density")))))
        cube_file = in_path.with_suffix(f".{tag}.cube")
        grid.to_cube(cube_file)
        sys.stderr.write(f"PHATPSY: Gaussian Cube exported to '{cube_file}'\n")
        sys.stderr.write(f"  Field:       {grid.title}\n")
        sys.stderr.write(f"  Dimensions:  {grid.dimensions[0]} x {grid.dimensions[1]} x {grid.dimensions[2]}\n")
        sys.stderr.write(f"  Integral:    {grid.integrate():12.6f}\n")

    if args.view:
        visualize_grid(grid, isovalue=args.isovalue, property_grid=property_grid, auto_open=not args.no_browser)

    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Handle 'cube' command
    if argv and argv[0] in ("cube", "grid", "volumetric"):
        parser = argparse.ArgumentParser(
            prog="phatpsy cube",
            description="Evaluate 3D volumetric fields (orbitals, densities, ESP) and export Gaussian Cube files or interactive 3D visualizations."
        )
        parser.add_argument("input_file", help="Input calculation deck (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("--mo", type=int, default=None, help="Molecular orbital index (1-based) to evaluate")
        parser.add_argument("--ao", type=int, default=None, help="Atomic orbital index (1-based) to evaluate")
        parser.add_argument("--density", action="store_true", help="Evaluate total electron density rho(r)")
        parser.add_argument("--spin-density", action="store_true", help="Evaluate spin density Delta rho_s(r)")
        parser.add_argument("--deformation", action="store_true", help="Evaluate deformation/binding density Delta rho(r)")
        parser.add_argument("--esp", action="store_true", help="Evaluate Molecular Electrostatic Potential (ESP/MEP)")
        parser.add_argument("--isovalue", type=float, default=None, help="Isosurface contour value (default: auto)")
        parser.add_argument("--spacing", type=float, default=0.2, help="Grid voxel spacing in Bohr (default: 0.2)")
        parser.add_argument("--padding", type=float, default=3.0, help="Grid spatial padding around molecule in Bohr (default: 3.0)")
        parser.add_argument("-o", "--output", default=None, help="Output path for Gaussian Cube file (.cube) or HTML viewer (.html)")
        parser.add_argument("--view", action="store_true", help="Open interactive 3D WebGL visualization in browser")
        parser.add_argument("-n", "--no-browser", action="store_true", help="Do not open web browser automatically")
        args = parser.parse_args(argv[1:])
        return handle_cube(args)


    # Handle 'supercell' or 'slab' command
    if argv and argv[0] in ("supercell", "slab"):
        parser = argparse.ArgumentParser(
            prog="phatpsy supercell",
            description="Generate 3D bulk supercells or oriented surface slabs with optional adsorbates."
        )
        parser.add_argument("-e", "--element", default="Ni", help="Chemical element (e.g. Ni, Cu, Ag, Pt, Fe, default: Ni)")
        parser.add_argument("-c", "--crystal", default=None, help="Crystal structure ('fcc', 'bcc', 'diamond')")
        parser.add_argument("-a", "--lattice-constant", type=float, default=None, help="Lattice constant in Angstroms")
        parser.add_argument("--bulk", action="store_true", help="Generate 3D bulk supercell instead of surface slab")
        parser.add_argument("-m", "--miller", default="1,0,0", help="Miller index for surface slab (e.g. '1,0,0', '1,1,1', '1,1,0')")
        parser.add_argument("-l", "--layers", type=int, default=3, help="Number of atomic layers in slab (default: 3)")
        parser.add_argument("-v", "--vacuum", type=float, default=15.0, help="Vacuum spacing above slab in Angstroms (default: 15.0)")
        parser.add_argument("-r", "--repeat", default=None, help="Supercell replication (e.g. '2,2' or '2x2' or '2,2,2')")
        parser.add_argument("-f", "--fixed-layers", type=int, default=1, help="Number of bottom layers frozen with QFIXED=T (default: 1)")
        parser.add_argument("--virtual-perimeter", action="store_true", help="Add ghost/virtual coordination atoms around lateral boundary")
        parser.add_argument("-s", "--auto-spin", action="store_true", help="Assign alternating-spin states (up/down)")
        parser.add_argument("--adsorbate", default=None, help="Adsorbate molecule formula ('CO', 'NO', 'H2') or geometry file")
        parser.add_argument("--site", choices=["top", "bridge", "hollow"], default="top", help="Adsorption site type (default: top)")
        parser.add_argument("--height", type=float, default=2.0, help="Adsorbate vertical clearance above surface in Angstroms (default: 2.0)")
        parser.add_argument("--tilt", type=float, default=0.0, help="Adsorbate tilt angle in degrees (default: 0.0)")
        parser.add_argument("-b", "--basis", default="DZ", help="Basis set name for calculation deck export (default: DZ)")
        parser.add_argument("--format", choices=["yaml", "stdin", "json", "poscar", "cif", "xyz"], default=None, help="Explicit output format")
        parser.add_argument("-o", "--output", default=None, help="Output file path (.yaml, .json, .stdin, .poscar, .cif, .xyz)")
        parser.add_argument("--view", action="store_true", help="Open interactive 3D WebGL visualization")
        parser.add_argument("-n", "--no-browser", action="store_true", help="Do not open web browser automatically when viewing")
        args = parser.parse_args(argv[1:])
        return handle_supercell(args)

    # Handle 'view' command
    if argv and argv[0] == "view":
        parser = argparse.ArgumentParser(
            prog="phatpsy view",
            description="Interactive 3D WebGL visualization of molecules and calculation decks."
        )
        parser.add_argument("input_file", help="Input file path (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("-o", "--output", default=None, help="Output HTML file path")
        parser.add_argument("-s", "--auto-spin", action="store_true", help="Assign alternating-spin states before viewing")
        parser.add_argument("-n", "--no-browser", action="store_true", help="Do not open web browser automatically")
        args = parser.parse_args(argv[1:])
        return handle_view(args)

    # Handle 'elements' command
    if argv and argv[0] == "elements":
        parser = argparse.ArgumentParser(
            prog="phatpsy elements",
            description="Inspect available elements and STO basis sets in the built-in library."
        )
        parser.add_argument("symbol", nargs="?", default=None, help="Element symbol to inspect (e.g. C, Si, Au)")
        parser.add_argument("-b", "--basis", default=None, help="Specific basis name (e.g. SZ, DZ, THAKKAR)")
        args = parser.parse_args(argv[1:])
        return handle_elements(args)

    # Handle 'build' command
    if argv and argv[0] == "build":
        parser = argparse.ArgumentParser(
            prog="phatpsy build",
            description="Build a calculation deck from an XYZ molecular geometry file."
        )
        parser.add_argument("xyz_file", help="Input XYZ file path")
        parser.add_argument("-b", "--basis", default="DZ", help="Basis set name ('DZ', 'SZ', 'THAKKAR', default: DZ)")
        parser.add_argument("-u", "--units", default="angstrom", help="Coordinate units ('angstrom' or 'bohr', default: angstrom)")
        parser.add_argument("-t", "--title", default=None, help="Calculation title")
        parser.add_argument("-s", "--auto-spin", action="store_true", help="Automatically assign alternating spin ('up'/'down') states")
        parser.add_argument("-o", "--output", default=None, help="Output calculation deck file (.yaml, .json, .stdin)")
        args = parser.parse_args(argv[1:])
        return handle_build(args)

    # Handle 'convert' command
    if argv and argv[0] == "convert":
        parser = argparse.ArgumentParser(
            prog="phatpsy convert",
            description="Convert between legacy .stdin, YAML, and JSON calculation decks."
        )
        parser.add_argument("input_file", help="Input file path (.stdin, .yaml, .json)")
        parser.add_argument("-o", "--output", default=None, help="Output file path (.yaml, .json, or .stdin)")
        args = parser.parse_args(argv[1:])
        return handle_convert(args)

    # Handle 'scan' or 'pes' command
    if argv and argv[0] in ("scan", "pes"):
        parser = argparse.ArgumentParser(
            prog="phatpsy scan",
            description="Perform automated 1D or 2D Potential Energy Surface (PES) coordinate scans."
        )
        parser.add_argument("input_file", help="Input calculation deck (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("-b", "--bond", default=None, help="Bond indices to scan (e.g. '0,1' or '0-1')")
        parser.add_argument("-a", "--angle", default=None, help="Angle indices to scan (e.g. '0,1,2')")
        parser.add_argument("-c", "--coord", default=None, help="Custom coordinate specifier")
        parser.add_argument("-r", "--range", required=True, help="Coordinate range (e.g. '1.8:2.5:0.05')")
        parser.add_argument("--bond2", default=None, help="Second bond for 2D scan")
        parser.add_argument("--coord2", default=None, help="Second coordinate for 2D scan")
        parser.add_argument("--range2", default=None, help="Second coordinate range for 2D scan")
        parser.add_argument("-o", "--output", default=None, help="Output path for PES JSON data")
        parser.add_argument("-p", "--plot", default=None, help="Output path for PES plot image (.png, .pdf)")
        parser.add_argument("--no-warm-restart", action="store_true", help="Disable warm-restarting between scan points")
        args = parser.parse_args(argv[1:])
        return handle_scan(args)

    # Handle 'opt' or 'optimize' command
    if argv and argv[0] in ("opt", "optimize"):
        parser = argparse.ArgumentParser(
            prog="phatpsy opt",
            description="Ground-state molecular structure relaxation and geometry optimization (BFGS)."
        )
        parser.add_argument("input_file", help="Input calculation deck (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("-o", "--output", default=None, help="Output path for optimized deck (.yaml, .json, .stdin)")
        parser.add_argument("-t", "--trajectory", default=None, help="Output path for optimization trajectory JSON")
        parser.add_argument("-p", "--plot", default=None, help="Output path for convergence plot (.png, .pdf)")
        parser.add_argument("--max-steps", type=int, default=30, help="Maximum optimization steps (default: 30)")
        parser.add_argument("--grad-tol", type=float, default=1e-3, help="Gradient tolerance in Hartree/Bohr (default: 1e-3)")
        parser.add_argument("--energy-tol", type=float, default=1e-5, help="Energy tolerance in Hartree (default: 1e-5)")
        parser.add_argument("--no-warm-restart", action="store_true", help="Disable warm-restarting between optimization steps")
        args = parser.parse_args(argv[1:])
        return handle_opt(args)

    # Handle 'neb', 'ci-neb', or 'path' command
    if argv and argv[0] in ("neb", "ci-neb", "path"):
        parser = argparse.ArgumentParser(
            prog="phatpsy neb",
            description="Transition state searching and reaction pathway optimization using (Climbing-Image) Nudged Elastic Band."
        )
        parser.add_argument("initial", help="Reactant calculation deck (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("final", help="Product calculation deck (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("-n", "--images", type=int, default=7, help="Number of pathway images including endpoints (default: 7)")
        parser.add_argument("-k", "--spring", type=float, default=0.1, help="Spring constant in Hartree/Bohr^2 (default: 0.1)")
        parser.add_argument("--max-iter", type=int, default=40, help="Maximum NEB optimization iterations (default: 40)")
        parser.add_argument("--force-tol", type=float, default=0.003, help="Force convergence tolerance in Hartree/Bohr (default: 0.003)")
        parser.add_argument("--no-climb", action="store_true", help="Disable climbing image (use standard NEB instead of CI-NEB)")
        parser.add_argument("--no-warm-restart", action="store_true", help="Disable warm-restarting between image SCF calculations")
        parser.add_argument("-o", "--output", default=None, help="Output path for pathway JSON data")
        parser.add_argument("--xyz", default=None, help="Output path for multi-frame XYZ trajectory")
        parser.add_argument("-p", "--plot", default=None, help="Output path for reaction profile plot (.png, .pdf)")
        parser.add_argument("--view", action="store_true", help="Launch interactive 3D WebGL reaction pathway viewer in browser")
        args = parser.parse_args(argv[1:])
        return handle_neb(args)

    # Handle 'md', 'bomd', or 'dynamics' command
    if argv and argv[0] in ("md", "bomd", "dynamics"):
        parser = argparse.ArgumentParser(
            prog="phatpsy md",
            description="Born-Oppenheimer Molecular Dynamics (BOMD) simulation on the PHATPSY potential energy surface."
        )
        parser.add_argument("input", help="Input calculation deck (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("-n", "--steps", type=int, default=50, help="Number of MD integration steps (default: 50)")
        parser.add_argument("--dt", type=float, default=0.5, help="Integration time step in femtoseconds (default: 0.5 fs)")
        parser.add_argument("-T", "--temp", type=float, default=300.0, help="Initial or target temperature in Kelvin (default: 300.0 K)")
        parser.add_argument("--thermostat", default="nve", choices=["nve", "berendsen", "rescale", "andersen"], help="Ensemble / thermostat type (default: nve)")
        parser.add_argument("--tau", type=float, default=50.0, help="Berendsen thermostat coupling time constant in fs (default: 50.0 fs)")
        parser.add_argument("--no-com-removal", action="store_true", help="Disable center-of-mass linear velocity drift removal")
        parser.add_argument("--no-warm-restart", action="store_true", help="Disable warm-restarting between successive time-step SCF calculations")
        parser.add_argument("-o", "--output", default=None, help="Output path for trajectory JSON data")
        parser.add_argument("--xyz", default=None, help="Output path for multi-frame XYZ trajectory")
        parser.add_argument("-p", "--plot", default=None, help="Output path for energy and temperature dynamics plot image")
        parser.add_argument("--view", action="store_true", help="Launch interactive 3D WebGL trajectory playback player in browser")
        args = parser.parse_args(argv[1:])
        return handle_md(args)

    # Handle 'freq', 'vib', or 'vibrations' command
    if argv and argv[0] in ("freq", "vib", "vibrations"):
        parser = argparse.ArgumentParser(
            prog="phatpsy freq",
            description="Harmonic vibrational frequency analysis, normal modes, and RRHO ideal-gas thermochemistry."
        )
        parser.add_argument("input", help="Input calculation deck (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("-s", "--step", type=float, default=0.005, help="Cartesian displacement step in Bohr (default: 0.005)")
        parser.add_argument("-m", "--method", default="central", choices=["central", "forward"], help="Finite difference stencil (default: central)")
        parser.add_argument("-T", "--temp", type=float, default=298.15, help="Thermochemistry temperature in Kelvin (default: 298.15 K)")
        parser.add_argument("-P", "--pressure", type=float, default=1.0, help="Thermochemistry pressure in atmospheres (default: 1.0 atm)")
        parser.add_argument("--no-project", action="store_true", help="Disable translational/rotational Eckart projection")
        parser.add_argument("-o", "--output", default=None, help="Output path for vibration results JSON data")
        parser.add_argument("--view", action="store_true", help="Launch interactive 3D WebGL normal mode visualizer in browser")
        args = parser.parse_args(argv[1:])
        return handle_vibrations(args)

    # Handle 'irc' command
    if argv and argv[0] == "irc":
        parser = argparse.ArgumentParser(
            prog="phatpsy irc",
            description="Intrinsic Reaction Coordinate (IRC) pathway tracing along mass-weighted steepest descent."
        )
        parser.add_argument("input", help="Input calculation deck (.yaml, .json, .stdin, .xyz)")
        parser.add_argument("-s", "--step", type=float, default=0.1, help="Arc length integration step size in amu^(1/2)*Bohr (default: 0.1)")
        parser.add_argument("-n", "--max-steps", type=int, default=40, help="Maximum steps per branch (default: 40)")
        parser.add_argument("--direction", default="both", choices=["both", "forward", "reverse"], help="Integration direction (default: both)")
        parser.add_argument("--grad-tol", type=float, default=1e-3, help="Cartesian gradient norm convergence threshold (default: 1e-3)")
        parser.add_argument("--diff-step", type=float, default=0.005, help="Displacement step for numerical gradient (default: 0.005)")
        parser.add_argument("-o", "--output", default=None, help="Output path for IRC JSON dataset")
        parser.add_argument("--xyz", default=None, help="Output path for multi-frame XYZ trajectory")
        parser.add_argument("-p", "--plot", default=None, help="Output path for reaction coordinate profile plot image")
        parser.add_argument("--view", action="store_true", help="Launch interactive 3D WebGL IRC visualizer in browser")
        args = parser.parse_args(argv[1:])
        return handle_irc(args)

    parser = argparse.ArgumentParser(
        prog="phatpsy",
        description="PHATPSY Python driver - execute calculation and export structured JSON."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input calculation deck (.yaml, .json, .stdin), or '-' to read from stdin (default: -)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to write structured JSON output (e.g. data/n2_python_driver.json)"
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Stream complete JSON to standard output"
    )

    args = parser.parse_args(argv)

    driver = PhatpsyDriver()
    if args.input == "-" or args.input is None:
        results = driver.run_from_stdin()
    else:
        results = driver.run_file(args.input)

    if results is None:
        sys.stderr.write("PHATPSY: Calculation finished (check mode or EOF).\n")
        return 0

    if args.output:
        save_results_to_json(results, args.output)
        sys.stderr.write(f"PHATPSY: Calculation results saved to {args.output}\n")
        sys.stderr.write(f"  Total Energy:   {results['total_energy']:18.8f} Hartree\n")
        sys.stderr.write(f"  Kinetic Energy: {results['kinetic_energy']:18.8f} Hartree\n")
        sys.stderr.write(f"  SCF Cycles:     {results.get('cycles', 'N/A')}\n")

    if args.json_stdout or not args.output:
        import tempfile
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as tf:
            save_results_to_json(results, tf.name)
            tf.seek(0)
            sys.stdout.write(tf.read())

    return 0


if __name__ == "__main__":
    sys.exit(main())
