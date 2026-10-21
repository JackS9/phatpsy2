# phatpsy/cli.py
"""
Command line interface for the PHATPSY Python driver and format converter.
Allows executing calculations from YAML, JSON, or legacy .stdin files,
and converting between deck formats.
"""
import argparse
import sys
from pathlib import Path
from phatpsy.driver import PhatpsyDriver
from phatpsy.io import save_results_to_json, parse_stdin_to_deck
from phatpsy.schema import CalculationDeck


def handle_convert(args):
    in_path = Path(args.input_file)
    suffix = in_path.suffix.lower()

    if suffix in (".stdin", ".dat", ".txt") or (not suffix and in_path.exists()):
        deck = parse_stdin_to_deck(in_path)
    elif suffix in (".yaml", ".yml"):
        deck = CalculationDeck.from_yaml(in_path)
    elif suffix == ".json":
        deck = CalculationDeck.from_json(in_path)
    else:
        # Default try stdin parsing
        deck = parse_stdin_to_deck(in_path)

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
    in_path = Path(args.input_file)
    suffix = in_path.suffix.lower()

    if suffix == ".xyz":
        from phatpsy.builder import build_molecule
        deck = build_molecule(in_path, auto_spin_flip=args.auto_spin)
    elif suffix in (".yaml", ".yml"):
        deck = CalculationDeck.from_yaml(in_path)
        if args.auto_spin:
            deck.assign_alternating_spin()
    elif suffix == ".json":
        deck = CalculationDeck.from_json(in_path)
        if args.auto_spin:
            deck.assign_alternating_spin()
    else:
        deck = parse_stdin_to_deck(in_path)
        if args.auto_spin:
            deck.assign_alternating_spin()

    out_path = Path(args.output) if args.output else None
    auto_open = not args.no_browser
    visualize_deck(deck, output_path=out_path, auto_open=auto_open)
    if out_path:
        sys.stderr.write(f"PHATPSY: Wrote 3D HTML visualization to '{out_path}'\n")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

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
