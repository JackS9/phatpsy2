# phatpsy/visualizer.py
"""
Interactive 3D WebGL visualizer for PHATPSY molecules and calculation decks.
Provides color-coded 3D representations of:
- Spin states: Spin Up (alpha, cyan) vs Spin Down (beta/flipped, orange)
- Virtual (Ghost) atoms: Translucent wireframe spheres
- Fixed (Frozen) atoms: Bounding metallic cages
- Symmetry elements: Mirror planes (ISYM=-1), rotation axes (ISYM=N), inversion centers,
  and generator-to-equivalent connector lines.
"""
from typing import List, Dict, Optional, Union, Any, Tuple

from pathlib import Path
import json
import webbrowser
import tempfile
import copy
import numpy as np

from phatpsy.schema import CalculationDeck, AtomConfig, SymmetrySpec
from phatpsy.spin import build_connectivity_graph, BOHR_PER_ANGSTROM, get_covalent_radius_bohr

CPK_COLORS: Dict[str, str] = {
    "H": "#FFFFFF", "HE": "#D9FFFF",
    "LI": "#CC80FF", "BE": "#C2FF00", "B": "#FFB5B5", "C": "#808080", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "NE": "#B3E3F5",
    "NA": "#AB5CF2", "MG": "#8AFF00", "AL": "#BFA6A6", "SI": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "CL": "#1FF01F", "AR": "#80D1E3",
    "K": "#8F40D4", "CA": "#3DFF00", "SC": "#E6E6E6", "TI": "#BFC2C7", "V": "#A6A6AB", "CR": "#8A99C7", "MN": "#9C7AC7", "FE": "#E06633",
    "CO": "#F090A0", "NI": "#50D050", "CU": "#C88033", "ZN": "#7D80B0", "GA": "#C28F8F", "GE": "#668F8F", "AS": "#BD80E3", "SE": "#FFA100",
    "BR": "#A62929", "KR": "#5CB8D1",
    "RB": "#702EB0", "SR": "#00FF00", "Y": "#94FFFF", "ZR": "#94E0E0", "NB": "#73C2C9", "MO": "#54B5B5", "TC": "#3B9E9E", "RU": "#248F8F",
    "RH": "#0A7D8C", "PD": "#006985", "AG": "#C0C0C0", "CD": "#FFD98F", "IN": "#A67573", "SN": "#668080", "SB": "#9E63B5", "TE": "#D47A00",
    "I": "#940094", "XE": "#429EB0",
    "PT": "#D0D0E0", "AU": "#FFD123", "HG": "#B8B8D0", "PB": "#575961", "BI": "#9E4FB5",
}


def get_atom_color(element: str) -> str:
    """Returns hexadecimal CPK color for an element."""
    return CPK_COLORS.get(element.strip().upper(), "#A0A0A0")


Z_TO_ELEMENT: Dict[int, str] = {
    1: "H", 2: "HE", 3: "LI", 4: "BE", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "NE",
    11: "NA", 12: "MG", 13: "AL", 14: "SI", 15: "P", 16: "S", 17: "CL", 18: "AR",
    19: "K", 20: "CA", 21: "SC", 22: "TI", 23: "V", 24: "CR", 25: "MN", 26: "FE",
    27: "CO", 28: "NI", 29: "CU", 30: "ZN", 31: "GA", 32: "GE", 33: "AS", 34: "SE",
    35: "BR", 36: "KR", 42: "MO", 46: "PD", 47: "AG", 78: "PT", 79: "AU",
}


def map_rwb_colors(values: Any, v_max: Optional[float] = None) -> Tuple[List[float], float]:
    """
    Maps 1D scalar property values onto a smooth Red-White-Blue RGB colormap:
    - Red (< 0, electron-rich / nucleophilic / negative charge)
    - White (~ 0, neutral)
    - Blue (> 0, electron-deficient / electrophilic / positive charge)
    """
    import numpy as np
    vals = np.asarray(values, dtype=np.float64)
    if v_max is None or v_max <= 0:
        if len(vals) > 0:
            v_max = float(np.percentile(np.abs(vals), 98))
            if v_max < 1e-6:
                v_max = float(np.max(np.abs(vals)))
        else:
            v_max = 1.0
        if v_max < 1e-6:
            v_max = 1.0

    norm_vals = np.clip(vals / v_max, -1.0, 1.0)
    flat_colors: List[float] = []
    for val in norm_vals:
        if val < 0.0:
            # Negative: White (1,1,1) -> Red (0.95, 0.15, 0.15)
            u = -val
            r = 1.0
            g = 1.0 - 0.85 * u
            b = 1.0 - 0.85 * u
        else:
            # Positive: White (1,1,1) -> Blue (0.15, 0.40, 0.95)
            u = val
            r = 1.0 - 0.85 * u
            g = 1.0 - 0.60 * u
            b = 1.0
        flat_colors.extend([round(float(r), 4), round(float(g), 4), round(float(b), 4)])
    return flat_colors, float(v_max)


def generate_grid_visualization_data(
    grid: Any,
    isovalue: Optional[float] = None,
    property_grid: Optional[Any] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extracts 3D isosurface meshes and generates visualization data for a VolumetricGrid.
    """
    import numpy as np
    atom_data = []
    atom_configs = []
    atoms = getattr(grid, "atoms", [])
    for idx, a in enumerate(atoms):
        z_nuc = int(round(a.get("z", 1)))
        elem = a.get("element", "X")
        if elem == "X" or elem.upper().startswith("Z"):
            elem = Z_TO_ELEMENT.get(z_nuc, elem)

        xyz_bohr = a.get("xyz", [0.0, 0.0, 0.0])
        x_bohr, y_bohr, z_bohr = xyz_bohr[0], xyz_bohr[1], xyz_bohr[2]
        x_aa = x_bohr / BOHR_PER_ANGSTROM
        y_aa = y_bohr / BOHR_PER_ANGSTROM
        z_aa = z_bohr / BOHR_PER_ANGSTROM
        r_bohr = get_covalent_radius_bohr(elem)
        r_aa = r_bohr / BOHR_PER_ANGSTROM

        atom_configs.append(AtomConfig(id=f"{elem}{idx+1}", element=elem, xyz=[x_bohr, y_bohr, z_bohr]))
        atom_data.append({
            "index": idx,
            "id": f"{elem}{idx+1}",
            "element": elem,
            "equivalent_to": None,
            "x": round(x_aa, 4),
            "y": round(y_aa, 4),
            "z": round(z_aa, 4),
            "x_bohr": round(x_bohr, 4),
            "y_bohr": round(y_bohr, 4),
            "z_bohr": round(z_bohr, 4),
            "radius": round(r_aa, 3),
            "color": get_atom_color(elem),
            "spin": "none",
            "fixed": False,
            "virtual": False,
            "symmetry": None,
        })

    bond_data = []
    if len(atom_configs) > 1:
        adj = build_connectivity_graph(atom_configs, tolerance=1.25)
        for i in range(len(atom_configs)):
            for j in adj[i]:
                if j > i:
                    bond_data.append([i, j])


    # Determine default isovalue if not provided
    max_abs = float(np.max(np.abs(grid.data)))
    if isovalue is None:
        if max_abs > 1e-4:
            if grid.min_value >= -1e-6:
                isovalue = min(0.02, 0.05 * max_abs)
            else:
                isovalue = 0.15 * max_abs
        else:
            isovalue = 0.02

    surfaces = []
    v_max_prop = 0.0
    has_prop = property_grid is not None

    if property_grid is not None:
        iso = abs(isovalue)
        mesh = grid.extract_isosurface(iso, property_grid=property_grid)
        v_bohr = mesh["vertices"]
        if len(v_bohr) > 0:
            v_aa = np.round(v_bohr / BOHR_PER_ANGSTROM, 3).tolist()
            faces = mesh["faces"].tolist()
            normals = np.round(mesh["normals"], 3).tolist()
            prop_vals = mesh.get("property_values", np.zeros(len(v_bohr)))
            colors, v_max_prop = map_rwb_colors(prop_vals)
            surfaces.append({
                "name": f"Isosurface ({iso:.4f}) with {property_grid.title}",
                "isovalue": iso,
                "color": "#ffffff",
                "opacity": 0.80,
                "vertices": v_aa,
                "faces": faces,
                "normals": normals,
                "colors": colors,
            })
    elif grid.min_value < -1e-5 and grid.max_value > 1e-5:
        # Dual-phase orbital / deformation (+ and - lobes)
        iso = abs(isovalue)
        mesh_pos = grid.extract_isosurface(+iso)
        if len(mesh_pos["vertices"]) > 0:
            surfaces.append({
                "name": f"Positive Phase (+{iso:.4f})",
                "isovalue": +iso,
                "color": "#00e5ff",
                "opacity": 0.75,
                "vertices": np.round(mesh_pos["vertices"] / BOHR_PER_ANGSTROM, 3).tolist(),
                "faces": mesh_pos["faces"].tolist(),
                "normals": np.round(mesh_pos["normals"], 3).tolist(),
                "colors": None,
            })
        mesh_neg = grid.extract_isosurface(-iso)
        if len(mesh_neg["vertices"]) > 0:
            surfaces.append({
                "name": f"Negative Phase (-{iso:.4f})",
                "isovalue": -iso,
                "color": "#ff9100",
                "opacity": 0.75,
                "vertices": np.round(mesh_neg["vertices"] / BOHR_PER_ANGSTROM, 3).tolist(),
                "faces": mesh_neg["faces"].tolist(),
                "normals": np.round(mesh_neg["normals"], 3).tolist(),
                "colors": None,
            })
    else:
        # Single envelope
        iso = abs(isovalue)
        mesh = grid.extract_isosurface(iso)
        if len(mesh["vertices"]) > 0:
            surfaces.append({
                "name": f"Isosurface ({iso:.4f})",
                "isovalue": iso,
                "color": "#38bdf8",
                "opacity": 0.65,
                "vertices": np.round(mesh["vertices"] / BOHR_PER_ANGSTROM, 3).tolist(),
                "faces": mesh["faces"].tolist(),
                "normals": np.round(mesh["normals"], 3).tolist(),
                "colors": None,
            })

    stats = {
        "total_atoms": len(atom_data),
        "spin_up": 0,
        "spin_down": 0,
        "virtual": 0,
        "fixed": 0,
        "grid_shape": list(grid.dimensions),
        "voxel_volume": grid.voxel_volume,
        "surfaces": len(surfaces),
    }

    return {
        "title": title or getattr(grid, "title", "PHATPSY Volumetric Field"),
        "field_type": "orbital" if (grid.min_value < -1e-5 and grid.max_value > 1e-5 and not has_prop) else "density",
        "isovalue": isovalue,
        "has_property": has_prop,
        "property_title": getattr(property_grid, "title", "Property Field") if property_grid else None,
        "v_max": v_max_prop,
        "atoms": atom_data,
        "bonds": bond_data,
        "surfaces": surfaces,
        "symmetry_planes": [],
        "symmetry_axes": [],
        "inversion_centers": [],
        "stats": stats,
    }


def generate_visualization_data(deck_or_atoms: Any) -> Dict[str, Any]:
    """Prepares structured JSON data for 3D rendering."""
    if isinstance(deck_or_atoms, dict) and "atoms" in deck_or_atoms:
        return deck_or_atoms
    if hasattr(deck_or_atoms, "data") and hasattr(deck_or_atoms, "voxel_steps"):
        return generate_grid_visualization_data(deck_or_atoms)

    if hasattr(deck_or_atoms, "resolve_coordinates"):
        deck_or_atoms.resolve_coordinates()
        atoms = deck_or_atoms.geometry
        title = getattr(deck_or_atoms, "title", "PHATPSY Molecule")
    elif hasattr(deck_or_atoms, "geometry"):
        atoms = deck_or_atoms.geometry
        title = getattr(deck_or_atoms, "title", "PHATPSY Molecule")
    else:
        atoms = deck_or_atoms
        title = "PHATPSY Molecule"

    from phatpsy.schema import resolve_atom_coordinates
    resolve_atom_coordinates(atoms)


    atom_data = []
    # Identify symmetry generator mapping using explicit equivalent_to or fallback
    id_to_idx = {a.id: idx for idx, a in enumerate(atoms) if a.id}
    generator_map: Dict[int, int] = {}
    current_primary_idx: Optional[int] = None
    for idx, a in enumerate(atoms):
        if a.equivalent_to and a.equivalent_to in id_to_idx:
            generator_map[idx] = id_to_idx[a.equivalent_to]
        elif a.symmetry is None:
            current_primary_idx = idx
        else:
            if current_primary_idx is not None:
                generator_map[idx] = current_primary_idx

    for idx, a in enumerate(atoms):
        x_bohr, y_bohr, z_bohr = a.xyz
        x_aa = x_bohr / BOHR_PER_ANGSTROM
        y_aa = y_bohr / BOHR_PER_ANGSTROM
        z_aa = z_bohr / BOHR_PER_ANGSTROM
        r_bohr = get_covalent_radius_bohr(a.element)
        r_aa = r_bohr / BOHR_PER_ANGSTROM

        sym_dict = None
        if a.symmetry:
            sym_dict = {
                "ixyz": a.symmetry.ixyz,
                "isym": a.symmetry.isym,
                "qflip": a.symmetry.qflip,
                "generator_idx": generator_map.get(idx),
            }

        atom_data.append({
            "index": idx,
            "id": a.id or f"{a.element}{idx+1}",
            "element": a.element,
            "equivalent_to": a.equivalent_to,
            "x": x_aa,
            "y": y_aa,
            "z": z_aa,
            "x_bohr": x_bohr,
            "y_bohr": y_bohr,
            "z_bohr": z_bohr,
            "radius": round(r_aa, 3),
            "color": get_atom_color(a.element),
            "spin": a.spin or ("down" if a.spin_flip else "none"),
            "fixed": a.fixed,
            "virtual": a.virtual,
            "symmetry": sym_dict,
        })

    # Bonds
    adj = build_connectivity_graph(atoms, tolerance=1.25)
    bonds = []
    for i in range(len(atoms)):
        for j in adj[i]:
            if j > i:
                bonds.append([i, j])

    # Summary statistics
    n_up = sum(1 for a in atoms if a.is_spin_up)
    n_down = sum(1 for a in atoms if a.is_spin_down)
    n_virt = sum(1 for a in atoms if a.virtual)
    n_fixed = sum(1 for a in atoms if a.fixed)

    return {
        "title": title,
        "atoms": atom_data,
        "bonds": bonds,
        "stats": {
            "total_atoms": len(atoms),
            "spin_up": n_up,
            "spin_down": n_down,
            "virtual": n_virt,
            "fixed": n_fixed,
        }
    }


def generate_html(
    deck_or_atoms: Any,
    title: Optional[str] = None,
    isovalue: Optional[float] = None,
    property_grid: Optional[Any] = None,
) -> str:
    """Generates a complete, interactive standalone HTML5 WebGL viewer using Three.js."""
    if isinstance(deck_or_atoms, dict) and "atoms" in deck_or_atoms:
        data = deck_or_atoms
    elif hasattr(deck_or_atoms, "data") and hasattr(deck_or_atoms, "voxel_steps"):
        data = generate_grid_visualization_data(deck_or_atoms, isovalue=isovalue, property_grid=property_grid, title=title)
    else:
        data = generate_visualization_data(deck_or_atoms)
    if title:
        data["title"] = title

    json_payload = json.dumps(data)


    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{data['title']} - PHATPSY 3D Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
    }}
    .ui-panel {{
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px 20px;
      max-width: 380px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
      z-index: 100;
    }}
    .ui-panel h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #38bdf8;
      margin-bottom: 6px;
    }}
    .badge-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
      margin-bottom: 12px;
    }}
    .badge {{
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }}
    .badge-blue {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}
    .badge-orange {{ background: rgba(251, 146, 60, 0.2); color: #fb923c; border: 1px solid rgba(251, 146, 60, 0.4); }}
    .badge-violet {{ background: rgba(192, 132, 252, 0.2); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.4); }}
    .badge-gray {{ background: rgba(148, 163, 184, 0.2); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.4); }}
    
    .controls {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 0.82rem;
      margin-top: 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
      padding-top: 10px;
    }}
    .control-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .control-row label {{
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      user-select: none;
    }}
    input[type="checkbox"] {{
      accent-color: #38bdf8;
      cursor: pointer;
    }}
    .select-input {{
      background: rgba(30, 41, 59, 0.85);
      color: #f1f5f9;
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 0.78rem;
      outline: none;
      cursor: pointer;
    }}
    .select-input:focus {{
      border-color: #38bdf8;
    }}
    .btn {{
      background: #0284c7;
      color: white;
      border: none;
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      margin-top: 8px;
      transition: background 0.15s;
    }}
    .btn:hover {{ background: #0369a1; }}
    
    .legend-box {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 0.78rem;
      display: flex;
      flex-direction: column;
      gap: 6px;
      z-index: 100;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
    }}
    
    #tooltip {{
      position: absolute;
      display: none;
      background: rgba(30, 41, 59, 0.95);
      border: 1px solid rgba(56, 189, 248, 0.4);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.82rem;
      color: #f1f5f9;
      pointer-events: none;
      box-shadow: 0 10px 20px rgba(0,0,0,0.4);
      z-index: 1000;
      min-width: 180px;
    }}
    #tooltip b {{ color: #38bdf8; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container">
    <div id="loading-msg" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #94a3b8; font-size: 0.9rem; text-align: center; pointer-events: none; z-index: 10;">
      Loading 3D WebGL Scene...
    </div>
  </div>
  
  <div class="ui-panel">
    <h1>{data['title']}</h1>
    <div class="badge-bar">
      <span class="badge badge-blue">Total: {data['stats']['total_atoms']}</span>
      {f'<span class="badge badge-blue">Field: {data.get("field_type", "3D Grid").capitalize()}</span>' if data.get('surfaces') else ''}
      {f'<span class="badge badge-violet">Iso: &plusmn;{data.get("isovalue", 0.02):.4f}</span>' if data.get('surfaces') else ''}
      {f'<span class="badge badge-orange">Property: {data.get("property_title", "ESP")}</span>' if data.get('has_property') else ''}
      {f'<span class="badge badge-blue">Spin Up &uarr;: {data["stats"]["spin_up"]}</span>' if data['stats'].get('spin_up', 0) > 0 else ''}
      {f'<span class="badge badge-orange">Spin Down &darr;: {data["stats"]["spin_down"]}</span>' if data['stats'].get('spin_down', 0) > 0 else ''}
      {f'<span class="badge badge-violet">Virtual: {data["stats"]["virtual"]}</span>' if data['stats'].get('virtual', 0) > 0 else ''}
      {f'<span class="badge badge-gray">Fixed: {data["stats"]["fixed"]}</span>' if data['stats'].get('fixed', 0) > 0 else ''}
    </div>
    
    <div class="controls">
      {'''
      <div class="control-row">
        <label><input type="checkbox" id="toggle-surfaces" checked> 3D Isosurface Envelope</label>
      </div>
      <div class="control-row">
        <label for="surface-opacity">Mesh Opacity:</label>
        <input type="range" id="surface-opacity" min="0.1" max="1.0" step="0.05" value="0.75" style="width: 120px; accent-color: #38bdf8; cursor: pointer;">
      </div>
      <div class="control-row">
        <label><input type="checkbox" id="toggle-wireframe"> Wireframe Mode</label>
      </div>
      ''' if data.get('surfaces') else ''}
      {f'''
      <div style="margin-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 8px;">
        <div style="font-size: 0.74rem; color: #94a3b8; margin-bottom: 4px; display: flex; justify-content: space-between;">
          <span>Mapped Property: <b style="color:#f1f5f9;">{data.get('property_title', 'ESP')}</b></span>
          <span style="color:#38bdf8;">&plusmn;{data.get('v_max', 0.0):.3f} au</span>
        </div>
        <div style="height: 12px; border-radius: 4px; background: linear-gradient(90deg, #ef4444 0%, #ffffff 50%, #3b82f6 100%); border: 1px solid rgba(255,255,255,0.2);"></div>
        <div style="display: flex; justify-content: space-between; font-size: 0.68rem; color: #cbd5e1; margin-top: 3px;">
          <span style="color: #f87171;">&minus;{data.get('v_max', 0.0):.2f} (Red/Nucleophilic)</span>
          <span style="color: #e2e8f0;">0.0 (Neutral)</span>
          <span style="color: #60a5fa;">&plus;{data.get('v_max', 0.0):.2f} (Blue/Electrophilic)</span>
        </div>
      </div>
      ''' if data.get('has_property') else ''}
      <div class="control-row">
        <label for="color-mode-select">Color Scheme:</label>
        <select id="color-mode-select" class="select-input">
          <option value="hybrid" selected>Hybrid (CPK + Spin Glow)</option>
          <option value="spin">Color by Spin Channel</option>
          <option value="cpk">Pure Element CPK</option>
        </select>
      </div>
      <div class="control-row">
        <label><input type="checkbox" id="toggle-spin" checked> Localized VB Spin Halos &amp; Glow (&uarr; / &darr;)<span style="display:none">Localized VB Spin Arrows</span></label>
      </div>
      <div class="control-row">
        <label><input type="checkbox" id="toggle-bonds" checked> Covalent Bonds</label>
      </div>
      <div class="control-row">
        <label><input type="checkbox" id="toggle-symmetry" checked> Symmetry Operations & Links</label>
      </div>
      <div class="control-row">
        <label><input type="checkbox" id="toggle-ghosts" checked> Virtual (Ghost) Centers</label>
      </div>
      <div class="control-row">
        <label><input type="checkbox" id="toggle-cages" checked> Fixed Atom Cages</label>
      </div>
      <button class="btn" id="btn-reset">Reset Camera</button>
    </div>
  </div>

  <div class="legend-box">
    {f'''
    <div class="legend-item"><span class="dot" style="background: #ef4444; box-shadow: 0 0 6px #ef4444;"></span> <b>Red</b>: Negative / Electron-Rich (Nucleophilic)</div>
    <div class="legend-item"><span class="dot" style="background: #ffffff; box-shadow: 0 0 6px #ffffff;"></span> <b>White</b>: Neutral Potential</div>
    <div class="legend-item"><span class="dot" style="background: #3b82f6; box-shadow: 0 0 6px #3b82f6;"></span> <b>Blue</b>: Positive / Electron-Deficient (Electrophilic)</div>
    ''' if data.get('has_property') else (
      f'''
      <div class="legend-item"><span class="dot" style="background: #00e5ff; box-shadow: 0 0 6px #00e5ff;"></span> <b>Positive Phase (+)</b>: +{data.get('isovalue', 0.02):.4f} au (Cyan)</div>
      <div class="legend-item"><span class="dot" style="background: #ff9100; box-shadow: 0 0 6px #ff9100;"></span> <b>Negative Phase (-)</b>: -{data.get('isovalue', 0.02):.4f} au (Orange)</div>
      ''' if data.get('field_type') == 'orbital' and data.get('surfaces') else ''
    )}
    {'''
    <div class="legend-item"><span class="dot" style="background: #38bdf8; box-shadow: 0 0 8px #38bdf8;"></span> <b>Spin Up (&uarr;)</b>: Alpha-dominant (Cyan glow &amp; halo ring)</div>
    <div class="legend-item"><span class="dot" style="background: #fb923c; box-shadow: 0 0 8px #fb923c;"></span> <b>Spin Down (&darr;)</b>: Beta-dominant (Orange glow &amp; halo ring, QFLIP=T)</div>
    ''' if data['stats'].get('spin_up', 0) > 0 or data['stats'].get('spin_down', 0) > 0 else ''}
    {f'<div class="legend-item"><span class="dot" style="background: transparent; border: 2px dashed #c084fc;"></span> <b>Virtual Center</b>: Ghost basis / external probe</div>' if data['stats'].get('virtual', 0) > 0 else ''}
    {f'<div class="legend-item"><span class="dot" style="background: transparent; border: 2px solid #94a3b8;"></span> <b>Fixed Atom</b>: Frozen orbitals & potential</div>' if data['stats'].get('fixed', 0) > 0 else ''}
  </div>

  <div id="tooltip"></div>

  <script>
    const MOL_DATA = {json_payload};

    const container = document.getElementById('canvas-container');
    const tooltip = document.getElementById('tooltip');
    const loadingMsg = document.getElementById('loading-msg');

    if (typeof THREE === 'undefined') {{
      if (loadingMsg) {{
        loadingMsg.innerHTML = '<b style="color:#f87171;font-size:1rem;">Three.js WebGL library not loaded</b><br><span style="font-size:0.8rem;color:#94a3b8;margin-top:6px;display:inline-block;">Ensure the notebook is marked <b>Trusted</b> in VS Code, or check network access to CDN.</span>';
      }}
    }} else {{
      if (loadingMsg) loadingMsg.style.display = 'none';

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0f172a);

      function getW() {{ return container.clientWidth || window.innerWidth || 800; }}
      function getH() {{ return container.clientHeight || window.innerHeight || 600; }}

      let curW = getW();
      let curH = getH();
      const camera = new THREE.PerspectiveCamera(45, (curW / (curH || 1)) || 1.33, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(curW, curH);
      renderer.shadowMap.enabled = true;
      container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(10, 20, 15);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.35);
    dirLight2.position.set(-15, -10, -10);
    scene.add(dirLight2);

    // Groups
    const atomGroup = new THREE.Group();
    const bondGroup = new THREE.Group();
    const spinGroup = new THREE.Group();
    const symmetryGroup = new THREE.Group();
    const ghostGroup = new THREE.Group();
    const cageGroup = new THREE.Group();
    const surfaceGroup = new THREE.Group();

    scene.add(atomGroup);
    scene.add(bondGroup);
    scene.add(spinGroup);
    scene.add(symmetryGroup);
    scene.add(ghostGroup);
    scene.add(cageGroup);
    scene.add(surfaceGroup);


    // Compute center of mass
    let cx = 0, cy = 0, cz = 0;
    const atoms = MOL_DATA.atoms;
    const n = atoms.length;
    atoms.forEach(a => {{ cx += a.x; cy += a.y; cz += a.z; }});
    if (n > 0) {{ cx /= n; cy /= n; cz /= n; }}

    const atomMeshes = [];
    const spinRings = [];

    // Sphere geometry cache
    const sphereGeo = new THREE.SphereGeometry(1, 32, 32);

    atoms.forEach(a => {{
      const pos = new THREE.Vector3(a.x - cx, a.y - cy, a.z - cz);
      const r = Math.max(0.35, a.radius * 0.45);

      // Material
      let mat;
      if (a.virtual) {{
        mat = new THREE.MeshStandardMaterial({{
          color: 0xc084fc,
          wireframe: true,
          transparent: true,
          opacity: 0.45,
        }});
      }} else {{
        const isUp = (a.spin === 'up');
        const isDown = (a.spin === 'down');
        const emissiveColor = isUp ? 0x0284c7 : (isDown ? 0xea580c : 0x000000);
        const emissiveInt = (isUp || isDown) ? 0.45 : 0.0;
        mat = new THREE.MeshStandardMaterial({{
          color: a.color,
          roughness: 0.35,
          metalness: 0.15,
          emissive: emissiveColor,
          emissiveIntensity: emissiveInt,
        }});
      }}

      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.scale.set(r, r, r);
      mesh.position.copy(pos);
      mesh.userData = a;

      if (a.virtual) {{
        ghostGroup.add(mesh);
      }} else {{
        atomGroup.add(mesh);
      }}
      atomMeshes.push(mesh);

      // Localized VB Spin Visual Properties (Orbital Halo Torus & Atmospheric Corona)
      if (a.spin === 'up' || a.spin === 'down') {{
        const isUp = (a.spin === 'up');
        const spinHex = isUp ? 0x38bdf8 : 0xfb923c;

        // 1. Luminous orbital halo ring encircling the atom sphere
        const ringGeo = new THREE.TorusGeometry(r * 1.35, r * 0.07, 16, 64);
        const ringMat = new THREE.MeshStandardMaterial({{
          color: spinHex,
          emissive: spinHex,
          emissiveIntensity: 0.55,
          roughness: 0.25,
          metalness: 0.7,
        }});
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.position.copy(pos);
        ringMesh.rotation.x = Math.PI / 3;
        ringMesh.rotation.y = isUp ? Math.PI / 5 : -Math.PI / 5;
        spinGroup.add(ringMesh);
        spinRings.push(ringMesh);

        // 2. Concentric outer aura / corona shell
        const coronaGeo = new THREE.SphereGeometry(r * 1.18, 24, 24);
        const coronaMat = new THREE.MeshBasicMaterial({{
          color: spinHex,
          transparent: true,
          opacity: 0.22,
          blending: THREE.AdditiveBlending,
          side: THREE.BackSide,
          depthWrite: false,
        }});
        const coronaMesh = new THREE.Mesh(coronaGeo, coronaMat);
        coronaMesh.position.copy(pos);
        spinGroup.add(coronaMesh);
      }}

      // Fixed Atom Cages
      if (a.fixed) {{
        const cageGeo = new THREE.BoxGeometry(r * 2.6, r * 2.6, r * 2.6);
        const cageEdges = new THREE.EdgesGeometry(cageGeo);
        const cageMat = new THREE.LineBasicMaterial({{ color: 0x94a3b8, linewidth: 2 }});
        const cage = new THREE.LineSegments(cageEdges, cageMat);
        cage.position.copy(pos);
        cageGroup.add(cage);
      }}

      // Symmetry connector lines
      if (a.symmetry && a.symmetry.generator_idx !== null && a.symmetry.generator_idx !== undefined) {{
        const genAtom = atoms[a.symmetry.generator_idx];
        if (genAtom) {{
          const genPos = new THREE.Vector3(genAtom.x - cx, genAtom.y - cy, genAtom.z - cz);
          const points = [pos, genPos];
          const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
          const lineMat = new THREE.LineDashedMaterial({{
            color: 0xfacc15,
            dashSize: 0.15,
            gapSize: 0.1,
          }});
          const line = new THREE.Line(lineGeo, lineMat);
          line.computeLineDistances();
          symmetryGroup.add(line);
        }}
      }}
    }});

    // Covalent Bonds
    MOL_DATA.bonds.forEach(([i, j]) => {{
      const p1 = new THREE.Vector3(atoms[i].x - cx, atoms[i].y - cy, atoms[i].z - cz);
      const p2 = new THREE.Vector3(atoms[j].x - cx, atoms[j].y - cy, atoms[j].z - cz);
      const dist = p1.distanceTo(p2);
      const dir = p2.clone().sub(p1).normalize();
      const mid = p1.clone().add(p2).multiplyScalar(0.5);

      const cylGeo = new THREE.CylinderGeometry(0.08, 0.08, dist, 16);
      const cylMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, roughness: 0.5 }});
      const cyl = new THREE.Mesh(cylGeo, cylMat);

      cyl.position.copy(mid);
      cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      bondGroup.add(cyl);
    }});

    // Symmetry operations: Reflection planes and rotation axes
    const seenSymmetries = new Set();
    atoms.forEach(a => {{
      if (a.symmetry && !seenSymmetries.has(a.symmetry.ixyz + "_" + a.symmetry.isym)) {{
        seenSymmetries.add(a.symmetry.ixyz + "_" + a.symmetry.isym);
        const isym = a.symmetry.isym;
        const ixyz = a.symmetry.ixyz;

        // Reflection plane (ISYM=-1)
        if (isym === -1) {{
          const planeGeo = new THREE.CircleGeometry(5.0, 32);
          const planeMat = new THREE.MeshBasicMaterial({{
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.12,
            side: THREE.DoubleSide,
            depthWrite: false,
          }});
          const plane = new THREE.Mesh(planeGeo, planeMat);
          if (ixyz === 1) {{
            // XY plane (normal Z)
            plane.position.set(0, 0, 0);
          }} else if (ixyz === 100) {{
            // YZ plane (normal X)
            plane.rotation.y = Math.PI / 2;
          }} else if (ixyz === 10) {{
            // XZ plane (normal Y)
            plane.rotation.x = Math.PI / 2;
          }}
          symmetryGroup.add(plane);
        }}
        // Rotation axis (ISYM > 0)
        else if (isym > 1) {{
          const axisGeo = new THREE.CylinderGeometry(0.04, 0.04, 12, 16);
          const axisMat = new THREE.MeshBasicMaterial({{ color: 0xfacc15 }});
          const axisMesh = new THREE.Mesh(axisGeo, axisMat);
          if (ixyz === 1) {{
            axisMesh.rotation.x = Math.PI / 2;
          }} else if (ixyz === 100) {{
            axisMesh.rotation.z = Math.PI / 2;
          }}
          symmetryGroup.add(axisMesh);
        }}
      }}
    }});

    // 3D Isosurface Meshes
    if (MOL_DATA.surfaces && MOL_DATA.surfaces.length > 0) {{
      MOL_DATA.surfaces.forEach(s => {{
        if (!s.vertices || s.vertices.length === 0 || !s.faces || s.faces.length === 0) return;

        const geom = new THREE.BufferGeometry();
        const posArray = new Float32Array(s.vertices.length * 3);
        for (let i = 0; i < s.vertices.length; i++) {{
          posArray[i * 3]     = s.vertices[i][0] - cx;
          posArray[i * 3 + 1] = s.vertices[i][1] - cy;
          posArray[i * 3 + 2] = s.vertices[i][2] - cz;
        }}
        geom.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

        if (s.normals && s.normals.length === s.vertices.length) {{
          const normArray = new Float32Array(s.normals.length * 3);
          for (let i = 0; i < s.normals.length; i++) {{
            normArray[i * 3]     = s.normals[i][0];
            normArray[i * 3 + 1] = s.normals[i][1];
            normArray[i * 3 + 2] = s.normals[i][2];
          }}
          geom.setAttribute('normal', new THREE.BufferAttribute(normArray, 3));
        }} else {{
          geom.computeVertexNormals();
        }}

        const hasColors = (s.colors && s.colors.length === s.vertices.length * 3);
        if (hasColors) {{
          geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(s.colors), 3));
        }}

        const indexArray = new Uint32Array(s.faces.length * 3);
        for (let i = 0; i < s.faces.length; i++) {{
          indexArray[i * 3]     = s.faces[i][0];
          indexArray[i * 3 + 1] = s.faces[i][1];
          indexArray[i * 3 + 2] = s.faces[i][2];
        }}
        geom.setIndex(new THREE.BufferAttribute(indexArray, 1));

        const mat = new THREE.MeshStandardMaterial({{
          color: hasColors ? 0xffffff : (s.color || 0x38bdf8),
          vertexColors: hasColors,
          transparent: true,
          opacity: s.opacity || 0.75,
          roughness: 0.35,
          metalness: 0.05,
          side: THREE.DoubleSide,
          depthWrite: false,
        }});

        const mesh = new THREE.Mesh(geom, mat);
        mesh.name = s.name;
        surfaceGroup.add(mesh);
      }});
    }}

    // Initial camera position
    let maxDist = 3.0;
    atoms.forEach(a => {{
      const d = Math.sqrt((a.x - cx)**2 + (a.y - cy)**2 + (a.z - cz)**2);
      if (d > maxDist) maxDist = d;
    }});
    if (MOL_DATA.surfaces) {{
      MOL_DATA.surfaces.forEach(s => {{
        if (s.vertices && s.vertices.length > 0) {{
          for (let i = 0; i < s.vertices.length; i += 10) {{
            const d = Math.sqrt((s.vertices[i][0] - cx)**2 + (s.vertices[i][1] - cy)**2 + (s.vertices[i][2] - cz)**2);
            if (d > maxDist) maxDist = d;
          }}
        }}
      }});
    }}
    camera.position.set(maxDist * 2.2, maxDist * 1.8, maxDist * 2.5);
    camera.lookAt(0, 0, 0);

    // Interaction & Raycasting for Tooltip
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    window.addEventListener('mousemove', (e) => {{
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(atomMeshes);

      if (intersects.length > 0) {{
        const d = intersects[0].object.userData;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 14) + 'px';
        tooltip.style.top = (e.clientY + 14) + 'px';
        
        let symStr = 'None';
        if (d.symmetry) {{
          symStr = 'ISYM=' + d.symmetry.isym + ', IXYZ=' + d.symmetry.ixyz + ' (from Atom #' + d.symmetry.generator_idx + ')';
        }}

        let spinStr = d.spin.toUpperCase();
        if (d.spin === 'up') spinStr += ' (&uarr; Alpha-dominant, Cyan glow &amp; halo ring)';
        else if (d.spin === 'down') spinStr += ' (&darr; Beta-dominant, Orange glow &amp; halo ring, QFLIP=T)';

        tooltip.innerHTML = `
          <b>${{d.element}} #${{d.index + 1}}</b><br>
          <b>Coordinates (&Aring;):</b> ${{d.x.toFixed(4)}}, ${{d.y.toFixed(4)}}, ${{d.z.toFixed(4)}}<br>
          <b>Coordinates (Bohr):</b> ${{d.x_bohr.toFixed(4)}}, ${{d.y_bohr.toFixed(4)}}, ${{d.z_bohr.toFixed(4)}}<br>
          <b>Spin Channel:</b> ${{spinStr}}<br>
          <b>State:</b> ${{d.virtual ? 'Virtual (Ghost)' : 'Real'}}${{d.fixed ? ' | Fixed (Frozen)' : ''}}<br>
          <b>Symmetry:</b> ${{symStr}}
        `;
      }} else {{
        tooltip.style.display = 'none';
      }}
    }});

    function applyColorMode(mode) {{
      const spinActive = document.getElementById('toggle-spin').checked;
      atomMeshes.forEach(mesh => {{
        const a = mesh.userData;
        if (a.virtual) return;
        const isUp = (a.spin === 'up');
        const isDown = (a.spin === 'down');
        const hasSpin = isUp || isDown;

        if (mode === 'spin') {{
          if (isUp) {{
            mesh.material.color.setHex(0x38bdf8);
            mesh.material.emissive.setHex(0x0284c7);
            mesh.material.emissiveIntensity = 0.45;
          }} else if (isDown) {{
            mesh.material.color.setHex(0xfb923c);
            mesh.material.emissive.setHex(0xea580c);
            mesh.material.emissiveIntensity = 0.45;
          }} else {{
            mesh.material.color.setHex(0x64748b);
            mesh.material.emissive.setHex(0x000000);
            mesh.material.emissiveIntensity = 0.0;
          }}
        }} else if (mode === 'cpk') {{
          mesh.material.color.set(a.color);
          mesh.material.emissive.setHex(0x000000);
          mesh.material.emissiveIntensity = 0.0;
        }} else {{ // 'hybrid'
          mesh.material.color.set(a.color);
          if (spinActive && hasSpin) {{
            mesh.material.emissive.setHex(isUp ? 0x0284c7 : 0xea580c);
            mesh.material.emissiveIntensity = 0.45;
          }} else {{
            mesh.material.emissive.setHex(0x000000);
            mesh.material.emissiveIntensity = 0.0;
          }}
        }}
      }});
    }}

    // Controls listeners
    const elColorMode = document.getElementById('color-mode-select');
    if (elColorMode) elColorMode.addEventListener('change', (e) => applyColorMode(e.target.value));

    const elToggleSpin = document.getElementById('toggle-spin');
    if (elToggleSpin) elToggleSpin.addEventListener('change', (e) => {{
      spinGroup.visible = e.target.checked;
      if (elColorMode && elColorMode.value === 'hybrid') applyColorMode('hybrid');
    }});

    const elToggleBonds = document.getElementById('toggle-bonds');
    if (elToggleBonds) elToggleBonds.addEventListener('change', (e) => {{ bondGroup.visible = e.target.checked; }});

    const elToggleSymmetry = document.getElementById('toggle-symmetry');
    if (elToggleSymmetry) elToggleSymmetry.addEventListener('change', (e) => {{ symmetryGroup.visible = e.target.checked; }});

    const elToggleGhosts = document.getElementById('toggle-ghosts');
    if (elToggleGhosts) elToggleGhosts.addEventListener('change', (e) => {{ ghostGroup.visible = e.target.checked; }});

    const elToggleCages = document.getElementById('toggle-cages');
    if (elToggleCages) elToggleCages.addEventListener('change', (e) => {{ cageGroup.visible = e.target.checked; }});

    const elToggleSurfaces = document.getElementById('toggle-surfaces');
    if (elToggleSurfaces) elToggleSurfaces.addEventListener('change', (e) => {{ surfaceGroup.visible = e.target.checked; }});

    const elSurfaceOpacity = document.getElementById('surface-opacity');
    if (elSurfaceOpacity) elSurfaceOpacity.addEventListener('input', (e) => {{
      const val = parseFloat(e.target.value);
      surfaceGroup.traverse(child => {{
        if (child.isMesh && child.material) child.material.opacity = val;
      }});
    }});

    const elToggleWireframe = document.getElementById('toggle-wireframe');
    if (elToggleWireframe) elToggleWireframe.addEventListener('change', (e) => {{
      surfaceGroup.traverse(child => {{
        if (child.isMesh && child.material) child.material.wireframe = e.target.checked;
      }});
    }});

    const elBtnReset = document.getElementById('btn-reset');
    if (elBtnReset) elBtnReset.addEventListener('click', () => {{
      camera.position.set(maxDist * 2.2, maxDist * 1.8, maxDist * 2.5);
      controls.target.set(0, 0, 0);
      controls.update();
    }});

      function updateSize() {{
        const w = getW();
        const h = getH();
        if (w > 0 && h > 0) {{
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }}
      }}

      window.addEventListener('resize', updateSize);
      if (window.ResizeObserver) {{
        new ResizeObserver(updateSize).observe(container);
      }}

      // Animation Loop
      function animate() {{
        requestAnimationFrame(animate);
        if (renderer.domElement.width === 0 || renderer.domElement.height === 0) {{
          updateSize();
        }}
        if (spinGroup.visible) {{
          for (let i = 0; i < spinRings.length; i++) {{
            spinRings[i].rotation.z += 0.008;
          }}
        }}
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();
    }}
  </script>
</body>
</html>
"""
    return html_template


def visualize_deck(
    deck_or_atoms: Any,
    output_path: Optional[Union[str, Path]] = None,
    auto_open: bool = False,
    isovalue: Optional[float] = None,
    property_grid: Optional[Any] = None,
    title: Optional[str] = None,
) -> str:
    """
    Renders an interactive 3D WebGL visualization of the molecule, calculation deck, or volumetric grid.

    Args:
        deck_or_atoms: CalculationDeck, list of AtomConfig objects, or VolumetricGrid.
        output_path: Optional path to save the generated HTML file.
        auto_open: If True, opens the HTML file in the default web browser.
        isovalue: Optional isosurface contour value if visualizing a VolumetricGrid.
        property_grid: Optional property grid for vertex-color mapping (e.g. ESP).
        title: Optional custom visualization title.

    Returns:
        The generated HTML content string.
    """
    html_content = generate_html(deck_or_atoms, title=title, isovalue=isovalue, property_grid=property_grid)

    if output_path:
        out_file = Path(output_path)
    elif auto_open:
        temp_dir = Path(tempfile.gettempdir())
        out_file = temp_dir / "phatpsy_molecule_view.html"
    else:
        out_file = None

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html_content, encoding="utf-8")
        if auto_open:
            webbrowser.open(out_file.as_uri())

    return html_content


def visualize_grid(
    grid: Any,
    isovalue: Optional[float] = None,
    property_grid: Optional[Any] = None,
    output_path: Optional[Union[str, Path]] = None,
    auto_open: bool = False,
    title: Optional[str] = None,
) -> str:
    """
    Convenience helper to render an interactive 3D WebGL visualization of a VolumetricGrid.
    """
    return visualize_deck(
        grid,
        output_path=output_path,
        auto_open=auto_open,
        isovalue=isovalue,
        property_grid=property_grid,
        title=title,
    )


def show_deck(
    deck_or_atoms: Any,
    height: int = 600,
    isovalue: Optional[float] = None,
    property_grid: Optional[Any] = None,
    title: Optional[str] = None,
):
    """
    Displays the 3D interactive viewer inline inside a Jupyter notebook.
    """
    try:
        import html
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="IPython.*")
            from IPython.display import HTML
            html_code = generate_html(deck_or_atoms, title=title, isovalue=isovalue, property_grid=property_grid)
            escaped_html = html.escape(html_code, quote=True)
            iframe_html = (
                f'<div style="width:100%; border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.15); background:#0f172a;">'
                f'<iframe srcdoc="{escaped_html}" width="100%" height="{height}" '
                f'sandbox="allow-scripts allow-same-origin allow-popups" '
                f'style="border:none; width:100%; height:{height}px; display:block; background:#0f172a;"></iframe>'
                f'</div>'
            )
            return HTML(iframe_html)
    except ImportError:
        raise RuntimeError("IPython is required to use show_deck() in Jupyter notebooks.")


def show_grid(
    grid: Any,
    isovalue: Optional[float] = None,
    property_grid: Optional[Any] = None,
    height: int = 600,
    title: Optional[str] = None,
):
    """
    Convenience helper to display a 3D volumetric isosurface inline inside a Jupyter notebook.
    """
    return show_deck(
        grid,
        height=height,
        isovalue=isovalue,
        property_grid=property_grid,
        title=title,
    )


def generate_pathway_visualization_data(neb_result: Any) -> Dict[str, Any]:
    """Extracts multi-frame pathway coordinates, bonds, and energies for 3D animation."""
    images = neb_result.images
    energies = neb_result.energies
    ts_idx = getattr(neb_result, "transition_state_index", int(np.argmax(energies)))
    e0 = energies[0]

    frames = []
    for i, deck in enumerate(images):
        frame_data = generate_visualization_data(deck)
        rel_kcal = (energies[i] - e0) * 627.509474
        frames.append({
            "index": i,
            "energy": round(float(energies[i]), 8),
            "rel_kcal": round(float(rel_kcal), 2),
            "is_ts": (i == ts_idx),
            "atoms": frame_data["atoms"],
            "bonds": frame_data["bonds"],
        })

    barrier_fwd = getattr(neb_result, "barrier_forward_kcal", (energies[ts_idx] - e0) * 627.509474)
    rxn_e = getattr(neb_result, "reaction_energy_kcal", (energies[-1] - e0) * 627.509474)

    return {
        "title": "NEB Reaction Pathway",
        "num_frames": len(frames),
        "ts_index": ts_idx,
        "barrier_fwd_kcal": round(float(barrier_fwd), 2),
        "reaction_e_kcal": round(float(rxn_e), 2),
        "frames": frames,
    }


def generate_pathway_html(neb_result: Any, title: Optional[str] = None) -> str:
    """Generates standalone HTML with interactive Three.js reaction pathway playback controls."""
    data = generate_pathway_visualization_data(neb_result)
    if title:
        data["title"] = title
    payload = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{data['title']} - PHATPSY 3D Pathway</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
    }}
    .ui-panel {{
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 16px 20px;
      max-width: 380px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
      z-index: 100;
    }}
    .ui-panel h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 6px;
    }}
    .badge-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }}
    .badge {{
      font-size: 0.75rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }}
    .badge-blue {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}
    .badge-red {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .badge-green {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }}
    
    .timeline-panel {{
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(15, 23, 42, 0.90);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 14px;
      padding: 14px 24px;
      min-width: 480px;
      max-width: 90vw;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
      display: flex;
      flex-direction: column;
      gap: 10px;
      z-index: 100;
    }}
    .timeline-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
      font-weight: 600;
    }}
    .timeline-controls {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .btn {{
      background: #1e293b;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #f8fafc;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .btn:hover {{ background: #334155; border-color: rgba(255, 255, 255, 0.3); }}
    .btn-primary {{ background: #2563eb; border-color: #3b82f6; }}
    .btn-primary:hover {{ background: #1d4ed8; }}
    .slider {{
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #38bdf8;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container"></div>

  <div class="ui-panel">
    <h1>{data['title']}</h1>
    <div class="badge-bar">
      <span class="badge badge-red">Ea: +{data['barrier_fwd_kcal']} kcal/mol</span>
      <span class="badge badge-blue">ΔE: {data['reaction_e_kcal']:+.2f} kcal/mol</span>
      <span class="badge badge-green">Images: {data['num_frames']}</span>
    </div>
  </div>

  <div class="timeline-panel">
    <div class="timeline-header">
      <span id="frame-label" style="color:#38bdf8;">Image 0 / {data['num_frames'] - 1} (Reactant)</span>
      <span id="energy-label" style="color:#94a3b8;">E = 0.000000 Hartree | ΔE = 0.00 kcal/mol</span>
    </div>
    <div class="timeline-controls">
      <button class="btn btn-primary" id="play-btn">▶ Play</button>
      <button class="btn" id="prev-btn">⏮</button>
      <input type="range" min="0" max="{data['num_frames'] - 1}" value="0" class="slider" id="frame-slider">
      <button class="btn" id="next-btn">⏭</button>
    </div>
  </div>

  <script>
    const data = {payload};
    const frames = data.frames;
    let currentFrameIdx = 0;
    let isPlaying = false;
    let playInterval = null;

    const container = document.getElementById("canvas-container");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.7);
    dirLight1.position.set(20, 30, 20);
    scene.add(dirLight1);
    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight2.position.set(-20, -30, -20);
    scene.add(dirLight2);

    const sphereGeom = new THREE.SphereGeometry(1, 32, 32);
    const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 16);

    const atomMeshes = [];
    const bondMeshes = [];
    const moleculeGroup = new THREE.Group();
    scene.add(moleculeGroup);

    // Build initial meshes using Frame 0
    const frame0 = frames[0];
    frame0.atoms.forEach(a => {{
      const mat = new THREE.MeshStandardMaterial({{
        color: new THREE.Color(a.color),
        roughness: 0.3,
        metalness: 0.2
      }});
      const mesh = new THREE.Mesh(sphereGeom, mat);
      mesh.scale.setScalar(a.radius * 0.7);
      mesh.position.set(a.x, a.y, a.z);
      moleculeGroup.add(mesh);
      atomMeshes.push(mesh);
    }});

    // Center camera on molecule bounding box
    const box = new THREE.Box3().setFromObject(moleculeGroup);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 4);
    camera.position.set(center.x, center.y, center.z + maxDim * 2.5);
    controls.target.copy(center);
    controls.update();

    function updateBonds(bondList) {{
      while (bondMeshes.length > 0) {{
        const b = bondMeshes.pop();
        moleculeGroup.remove(b);
      }}
      bondList.forEach(b => {{
        const m1 = atomMeshes[b.atom1];
        const m2 = atomMeshes[b.atom2];
        if (!m1 || !m2) return;
        const v1 = m1.position;
        const v2 = m2.position;
        const dist = v1.distanceTo(v2);
        const cylMat = new THREE.MeshStandardMaterial({{ color: 0x94a3b8, roughness: 0.4 }});
        const cyl = new THREE.Mesh(cylGeom, cylMat);
        cyl.scale.set(b.radius || 0.12, dist, b.radius || 0.12);
        cyl.position.copy(v1).add(v2).multiplyScalar(0.5);
        cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), v2.clone().sub(v1).normalize());
        moleculeGroup.add(cyl);
        bondMeshes.push(cyl);
      }});
    }}

    function applyFrame(idx) {{
      currentFrameIdx = idx;
      const f = frames[idx];
      f.atoms.forEach((a, i) => {{
        if (atomMeshes[i]) {{
          atomMeshes[i].position.set(a.x, a.y, a.z);
        }}
      }});
      updateBonds(f.bonds);

      const frameLabel = document.getElementById("frame-label");
      const energyLabel = document.getElementById("energy-label");
      const slider = document.getElementById("frame-slider");
      slider.value = idx;

      let tag = "";
      if (idx === 0) tag = " (Reactant)";
      else if (idx === frames.length - 1) tag = " (Product)";
      else if (f.is_ts) tag = " (★ Transition State)";

      frameLabel.textContent = `Image ${{idx}} / ${{frames.length - 1}}${{tag}}`;
      frameLabel.style.color = f.is_ts ? "#ef4444" : (idx === 0 ? "#10b981" : "#38bdf8");
      energyLabel.textContent = `E = ${{f.energy.toFixed(6)}} Hartree | ΔE = ${{f.rel_kcal >= 0 ? "+" : ""}}${{f.rel_kcal.toFixed(2)}} kcal/mol`;
    }}

    applyFrame(0);

    const slider = document.getElementById("frame-slider");
    slider.addEventListener("input", (e) => applyFrame(parseInt(e.target.value)));

    const playBtn = document.getElementById("play-btn");
    playBtn.addEventListener("click", () => {{
      isPlaying = !isPlaying;
      playBtn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
      if (isPlaying) {{
        playInterval = setInterval(() => {{
          let next = (currentFrameIdx + 1) % frames.length;
          applyFrame(next);
        }}, 400);
      }} else {{
        clearInterval(playInterval);
      }}
    }});

    document.getElementById("prev-btn").addEventListener("click", () => {{
      if (currentFrameIdx > 0) applyFrame(currentFrameIdx - 1);
    }});
    document.getElementById("next-btn").addEventListener("click", () => {{
      if (currentFrameIdx < frames.length - 1) applyFrame(currentFrameIdx + 1);
    }});

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""
    return html


def visualize_pathway(
    neb_result: Any,
    output_path: Optional[Union[str, Path]] = None,
    open_browser: bool = False,
    title: Optional[str] = None,
) -> str:
    """Renders standalone HTML with interactive 3D WebGL reaction pathway slider and playback."""
    html_content = generate_pathway_html(neb_result, title=title)
    if output_path:
        out_file = Path(output_path)
    elif open_browser:
        temp_dir = Path(tempfile.gettempdir())
        out_file = temp_dir / "phatpsy_reaction_pathway.html"
    else:
        out_file = None

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html_content, encoding="utf-8")
        if open_browser:
            webbrowser.open(out_file.as_uri())

    return html_content


def show_pathway(neb_result: Any, height: int = 550, title: Optional[str] = None):
    """Displays interactive 3D reaction pathway slider inside Jupyter notebooks."""
    try:
        import html
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="IPython.*")
            from IPython.display import HTML
            html_code = generate_pathway_html(neb_result, title=title)
            escaped_html = html.escape(html_code, quote=True)
            iframe_html = (
                f'<div style="width:100%; border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.15); background:#0f172a;">'
                f'<iframe srcdoc="{escaped_html}" width="100%" height="{height}" '
                f'sandbox="allow-scripts allow-same-origin allow-popups" '
                f'style="border:none; width:100%; height:{height}px; display:block; background:#0f172a;"></iframe>'
                f'</div>'
            )
            return HTML(iframe_html)
    except ImportError:
        raise RuntimeError("IPython is required to use show_pathway() in Jupyter notebooks.")


def generate_trajectory_visualization_data(md_result: Any) -> Dict[str, Any]:
    """Extracts multi-frame MD trajectory coordinates, bonds, energies, and temperatures for 3D animation."""
    trajectory = md_result.trajectory
    final_deck = md_result.final_deck
    atoms_template = getattr(final_deck, "geometry", None) or getattr(final_deck, "atoms", None) or []

    frames = []
    for s in trajectory:
        frame_atoms = copy.deepcopy(atoms_template)
        for i, pt in enumerate(s.coordinates):
            frame_atoms[i].xyz = pt
        adj = build_connectivity_graph(frame_atoms, tolerance=1.25)
        bonds = []
        for i in range(len(frame_atoms)):
            for j in adj[i]:
                if j > i:
                    bonds.append([i, j])

        atom_data = []
        for a in frame_atoms:
            atom_data.append({
                "element": a.element,
                "x": round(float(a.xyz[0]), 5),
                "y": round(float(a.xyz[1]), 5),
                "z": round(float(a.xyz[2]), 5),
                "spin": a.spin,
                "is_spin_up": a.is_spin_up,
                "is_spin_down": a.is_spin_down,
                "virtual": a.virtual,
                "fixed": a.fixed,
            })

        frames.append({
            "step": s.step,
            "time_fs": round(float(s.time_fs), 3),
            "potential_energy": round(float(s.potential_energy), 8),
            "kinetic_energy": round(float(s.kinetic_energy), 8),
            "total_energy": round(float(s.total_energy), 8),
            "temperature": round(float(s.temperature), 2),
            "virial_ratio": round(float(s.virial_ratio), 4),
            "atoms": atom_data,
            "bonds": bonds,
        })

    return {
        "title": f"BOMD Trajectory ({md_result.ensemble.upper()})",
        "ensemble": md_result.ensemble.upper(),
        "num_frames": len(frames),
        "time_step_fs": round(float(md_result.time_step_fs), 4),
        "total_time_fs": round(float(md_result.total_time_fs), 4),
        "avg_temperature": round(float(md_result.average_temperature), 2),
        "energy_drift_kcal": round(float(md_result.energy_drift * 627.509474), 4),
        "frames": frames,
    }


def generate_trajectory_html(md_result: Any, title: Optional[str] = None) -> str:
    """Generates standalone HTML with interactive Three.js molecular dynamics trajectory playback controls."""
    data = generate_trajectory_visualization_data(md_result)
    if title:
        data["title"] = title
    payload = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{data['title']} - PHATPSY 3D Trajectory</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
    }}
    .ui-panel {{
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 16px 20px;
      max-width: 400px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
      z-index: 100;
    }}
    .ui-panel h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 6px;
    }}
    .badge-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }}
    .badge {{
      font-size: 0.75rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }}
    .badge-blue {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}
    .badge-orange {{ background: rgba(251, 146, 60, 0.2); color: #fb923c; border: 1px solid rgba(251, 146, 60, 0.4); }}
    .badge-green {{ background: rgba(52, 211, 153, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4); }}
    
    .timeline-panel {{
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      width: min(720px, calc(100vw - 40px));
      background: rgba(15, 23, 42, 0.90);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 14px;
      padding: 14px 20px;
      box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.6);
      z-index: 100;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .timeline-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
      font-weight: 600;
    }}
    .timeline-controls {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .btn {{
      background: #1e293b;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #f8fafc;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .btn:hover {{ background: #334155; border-color: rgba(255, 255, 255, 0.3); }}
    .btn-primary {{ background: #2563eb; border-color: #3b82f6; }}
    .btn-primary:hover {{ background: #1d4ed8; }}
    .slider {{
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #38bdf8;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container"></div>

  <div class="ui-panel">
    <h1>{data['title']}</h1>
    <div class="badge-bar">
      <span class="badge badge-orange">Avg T: {data['avg_temperature']} K</span>
      <span class="badge badge-blue">ΔE: {data['energy_drift_kcal']:+.3f} kcal/mol</span>
      <span class="badge badge-green">Steps: {data['num_frames'] - 1} ({data['total_time_fs']} fs)</span>
    </div>
  </div>

  <div class="timeline-panel">
    <div class="timeline-header">
      <span id="frame-label" style="color:#38bdf8;">Step 0 / {data['num_frames'] - 1} (t = 0.00 fs)</span>
      <span id="metrics-label" style="color:#94a3b8;">T = 0.0 K | E_tot = 0.000000 Ha</span>
    </div>
    <div class="timeline-controls">
      <button class="btn btn-primary" id="play-btn">▶ Play</button>
      <button class="btn" id="prev-btn">⏮</button>
      <input type="range" min="0" max="{data['num_frames'] - 1}" value="0" class="slider" id="frame-slider">
      <button class="btn" id="next-btn">⏭</button>
    </div>
  </div>

  <script>
    const data = {payload};
    const frames = data.frames;
    let currentFrameIdx = 0;
    let isPlaying = false;
    let playInterval = null;

    const container = document.getElementById("canvas-container");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 10);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambLight);
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(10, 20, 15);
    scene.add(dirLight1);
    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.3);
    dirLight2.position.set(-10, -10, -10);
    scene.add(dirLight2);

    const moleculeGroup = new THREE.Group();
    scene.add(moleculeGroup);

    const sphereGeom = new THREE.SphereGeometry(1, 32, 32);
    const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 16);

    const elemColors = {{
      "H": 0xf8fafc, "C": 0x475569, "N": 0x3b82f6, "O": 0xef4444, "F": 0x22c55e,
      "P": 0xf97316, "S": 0xeab308, "CL": 0x10b981, "NI": 0x10b981, "PT": 0xa855f7, "AG": 0xc084fc,
      "AU": 0xf59e0b, "FE": 0xe11d48, "CU": 0xd97706
    }};
    const elemRadii = {{
      "H": 0.35, "C": 0.65, "N": 0.60, "O": 0.58, "F": 0.55,
      "P": 0.75, "S": 0.75, "CL": 0.72, "NI": 0.85, "PT": 0.90, "AG": 0.90
    }};

    const atomMeshes = [];
    const bondMeshes = [];

    const f0 = frames[0];
    f0.atoms.forEach((a, idx) => {{
      const el = a.element.toUpperCase();
      const col = elemColors[el] || 0x38bdf8;
      const rad = elemRadii[el] || 0.65;
      const mat = new THREE.MeshStandardMaterial({{ color: col, roughness: 0.3, metalness: 0.2 }});
      const mesh = new THREE.Mesh(sphereGeom, mat);
      mesh.scale.set(rad, rad, rad);
      mesh.position.set(a.x, a.y, a.z);
      moleculeGroup.add(mesh);
      atomMeshes.push(mesh);
    }});

    const box = new THREE.Box3().setFromObject(moleculeGroup);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 2.0);
    camera.position.set(center.x, center.y, center.z + maxDim * 2.8);
    controls.target.copy(center);
    controls.update();

    function updateBonds(bondList) {{
      while (bondMeshes.length > 0) {{
        const b = bondMeshes.pop();
        moleculeGroup.remove(b);
      }}
      bondList.forEach(b => {{
        const idx1 = Array.isArray(b) ? b[0] : b.atom1;
        const idx2 = Array.isArray(b) ? b[1] : b.atom2;
        const m1 = atomMeshes[idx1];
        const m2 = atomMeshes[idx2];
        if (!m1 || !m2) return;
        const v1 = m1.position;
        const v2 = m2.position;
        const dist = v1.distanceTo(v2);
        const cylMat = new THREE.MeshStandardMaterial({{ color: 0x94a3b8, roughness: 0.4 }});
        const cyl = new THREE.Mesh(cylGeom, cylMat);
        cyl.scale.set(0.12, dist, 0.12);
        cyl.position.copy(v1).add(v2).multiplyScalar(0.5);
        cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), v2.clone().sub(v1).normalize());
        moleculeGroup.add(cyl);
        bondMeshes.push(cyl);
      }});
    }}

    function applyFrame(idx) {{
      currentFrameIdx = idx;
      const f = frames[idx];
      f.atoms.forEach((a, i) => {{
        if (atomMeshes[i]) {{
          atomMeshes[i].position.set(a.x, a.y, a.z);
        }}
      }});
      updateBonds(f.bonds);

      const frameLabel = document.getElementById("frame-label");
      const metricsLabel = document.getElementById("metrics-label");
      const slider = document.getElementById("frame-slider");
      slider.value = idx;

      frameLabel.textContent = `Step ${{f.step}} / ${{frames.length - 1}} (t = ${{f.time_fs.toFixed(2)}} fs)`;
      metricsLabel.textContent = `T = ${{f.temperature.toFixed(1)}} K | E_tot = ${{f.total_energy.toFixed(6)}} Ha | -V/T = ${{f.virial_ratio.toFixed(3)}}`;
    }}

    applyFrame(0);

    const slider = document.getElementById("frame-slider");
    slider.addEventListener("input", (e) => applyFrame(parseInt(e.target.value)));

    const playBtn = document.getElementById("play-btn");
    playBtn.addEventListener("click", () => {{
      isPlaying = !isPlaying;
      playBtn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
      if (isPlaying) {{
        playInterval = setInterval(() => {{
          let next = (currentFrameIdx + 1) % frames.length;
          applyFrame(next);
        }}, 80);
      }} else {{
        clearInterval(playInterval);
      }}
    }});

    document.getElementById("prev-btn").addEventListener("click", () => {{
      if (currentFrameIdx > 0) applyFrame(currentFrameIdx - 1);
    }});
    document.getElementById("next-btn").addEventListener("click", () => {{
      if (currentFrameIdx < frames.length - 1) applyFrame(currentFrameIdx + 1);
    }});

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""
    return html


def visualize_trajectory(
    md_result: Any,
    output_path: Optional[Union[str, Path]] = None,
    open_browser: bool = False,
    title: Optional[str] = None,
) -> str:
    """Renders standalone HTML with interactive 3D WebGL molecular dynamics trajectory player."""
    html_content = generate_trajectory_html(md_result, title=title)
    if output_path:
        out_file = Path(output_path)
    elif open_browser:
        temp_dir = Path(tempfile.gettempdir())
        out_file = temp_dir / "phatpsy_md_trajectory.html"
    else:
        out_file = None

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html_content, encoding="utf-8")
        if open_browser:
            webbrowser.open(out_file.as_uri())

    return html_content


def show_trajectory(md_result: Any, height: int = 550, title: Optional[str] = None):
    """Displays interactive 3D molecular dynamics trajectory player inside Jupyter notebooks."""
    try:
        import html
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="IPython.*")
            from IPython.display import HTML
            html_code = generate_trajectory_html(md_result, title=title)
            escaped_html = html.escape(html_code, quote=True)
            iframe_html = (
                f'<div style="width:100%; border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.15); background:#0f172a;">'
                f'<iframe srcdoc="{escaped_html}" width="100%" height="{height}" '
                f'sandbox="allow-scripts allow-same-origin allow-popups" '
                f'style="border:none; width:100%; height:{height}px; display:block; background:#0f172a;"></iframe>'
                f'</div>'
            )
            return HTML(iframe_html)
    except ImportError:
        raise RuntimeError("IPython is required to use show_trajectory() in Jupyter notebooks.")


# ==============================================================================
# Harmonic Vibrations & Normal Mode 3D WebGL Visualizer
# ==============================================================================

def generate_vibrations_visualization_data(vib_result: Any) -> Dict[str, Any]:
    """
    Serializes a VibrationResult into structured JSON data for 3D WebGL Three.js visualization.
    """
    res_dict = vib_result.to_dict() if hasattr(vib_result, "to_dict") else vib_result

    raw_atoms = res_dict.get("atoms", [])
    raw_modes = res_dict.get("modes", [])

    bohr_to_ang = 0.529177210903

    atoms_data = []
    coords_ang = []
    for idx, a in enumerate(raw_atoms):
        elem = a.get("element", "X").strip().upper()
        xyz_bohr = a.get("xyz", [0.0, 0.0, 0.0])
        xyz_ang = [round(float(c) * bohr_to_ang, 4) for c in xyz_bohr]
        coords_ang.append(xyz_ang)

        cov_r_bohr = get_covalent_radius_bohr(elem)
        radius_ang = max(0.25, round(cov_r_bohr * bohr_to_ang * 0.45, 3))

        atoms_data.append({
            "index": idx,
            "element": elem,
            "id": a.get("id", f"{elem}{idx + 1}"),
            "xyz": xyz_ang,
            "color": get_atom_color(elem),
            "radius": radius_ang,
            "mass": a.get("mass_amu", 12.0),
            "fixed": a.get("fixed", False),
            "virtual": a.get("virtual", False),
        })

    # Molecular bonds via covalent radii threshold
    bonds = []
    n_atoms = len(raw_atoms)
    for i in range(n_atoms):
        elem_i = raw_atoms[i].get("element", "X")
        r_i = get_covalent_radius_bohr(elem_i) * bohr_to_ang
        p_i = np.array(coords_ang[i])
        for j in range(i + 1, n_atoms):
            elem_j = raw_atoms[j].get("element", "X")
            r_j = get_covalent_radius_bohr(elem_j) * bohr_to_ang
            p_j = np.array(coords_ang[j])
            dist = float(np.linalg.norm(p_i - p_j))
            if 0.4 < dist < 1.25 * (r_i + r_j):
                bonds.append([i, j])

    # Convert normal modes displacements to Ångström
    modes_data = []
    for m in raw_modes:
        disp_bohr = m.get("displacements", [])
        disp_ang = [
            [round(float(c) * bohr_to_ang, 5) for c in atom_disp]
            for atom_disp in disp_bohr
        ]
        modes_data.append({
            "index": m.get("index", 1),
            "frequency_cm": round(float(m.get("frequency_cm", 0.0)), 2),
            "is_imaginary": bool(m.get("is_imaginary", False)),
            "reduced_mass_amu": round(float(m.get("reduced_mass_amu", 1.0)), 4),
            "force_constant_mdyne_a": round(float(m.get("force_constant_mdyne_a", 0.0)), 4),
            "is_tr_mode": bool(m.get("is_tr_mode", False)),
            "displacements": disp_ang,
        })

    return {
        "title": "PHATPSY 2.0 Normal Mode Vibrations",
        "num_atoms": n_atoms,
        "num_modes": len(modes_data),
        "num_imaginary": res_dict.get("num_imaginary", 0),
        "is_transition_state": res_dict.get("is_transition_state", False),
        "is_minimum": res_dict.get("is_minimum", True),
        "atoms": atoms_data,
        "bonds": bonds,
        "modes": modes_data,
        "thermochemistry": res_dict.get("thermochemistry"),
    }


def generate_vibrations_html(
    vib_result: Any,
    mode_index: Optional[int] = None,
    amplitude: float = 0.5,
    title: Optional[str] = None,
) -> str:
    """
    Renders standalone modern HTML with interactive 3D WebGL normal mode animation and vector cones.
    """
    data = generate_vibrations_visualization_data(vib_result)
    if title:
        data["title"] = title

    payload = json.dumps(data)

    # Initial default mode: first non-TR mode or mode_index
    default_mode_idx = 0
    if mode_index is not None:
        default_mode_idx = max(0, min(len(data["modes"]) - 1, mode_index - 1))
    else:
        # Find first genuine vibrational mode
        for idx, m in enumerate(data["modes"]):
            if not m.get("is_tr_mode", False):
                default_mode_idx = idx
                break

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{data['title']}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{ width: 100%; height: 100%; }}
    .ui-panel {{
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 14px 18px;
      z-index: 10;
      max-width: 380px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }}
    .ui-panel h1 {{
      font-size: 1.05rem;
      font-weight: 700;
      margin-bottom: 6px;
      background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .badge-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    .badge-green {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }}
    .badge-red {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .badge-blue {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}
    .badge-purple {{ background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }}

    .info-card {{
      position: absolute;
      top: 16px;
      right: 16px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 14px 18px;
      z-index: 10;
      min-width: 260px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }}
    .info-card h2 {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
      margin-bottom: 8px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 5px;
      font-size: 0.82rem;
    }}
    .metric-label {{ color: #94a3b8; }}
    .metric-val {{ font-weight: 600; color: #f1f5f9; }}

    .controls-panel {{
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 16px;
      padding: 12px 20px;
      z-index: 10;
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 580px;
      max-width: 90vw;
      box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.7);
    }}
    .controls-row {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .select-dropdown {{
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      outline: none;
      cursor: pointer;
      flex: 1;
    }}
    .select-dropdown:hover {{ border-color: #38bdf8; }}
    .btn {{
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid rgba(255, 255, 255, 0.18);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .btn:hover {{ background: #334155; border-color: rgba(255, 255, 255, 0.3); }}
    .btn-primary {{ background: #2563eb; border-color: #3b82f6; }}
    .btn-primary:hover {{ background: #1d4ed8; }}
    .btn-active {{ background: #0284c7; border-color: #38bdf8; }}
    .slider {{
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #38bdf8;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
    }}
    .slider-label {{
      font-size: 0.75rem;
      color: #94a3b8;
      min-width: 95px;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container"></div>

  <div class="ui-panel">
    <h1>{data['title']}</h1>
    <div class="badge-bar">
      <span class="badge badge-blue">{data['num_atoms']} Atoms</span>
      <span class="badge badge-purple">{data['num_modes']} Normal Modes</span>
      <span class="badge {'badge-green' if data['num_imaginary'] == 0 else 'badge-red'}">
        {data['num_imaginary']} Imaginary ({'Local Minimum' if data['num_imaginary'] == 0 else ('Transition State' if data['num_imaginary'] == 1 else 'Higher-Order Saddle')})
      </span>
    </div>
  </div>

  <div class="info-card">
    <h2>Active Mode Details</h2>
    <div class="metric-row">
      <span class="metric-label">Frequency:</span>
      <span class="metric-val" id="mode-freq" style="color:#38bdf8;">-- cm⁻¹</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Force Constant:</span>
      <span class="metric-val" id="mode-force">-- mDyne/Å</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Reduced Mass:</span>
      <span class="metric-val" id="mode-mass">-- amu</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Mode Character:</span>
      <span class="metric-val" id="mode-char" style="color:#4ade80;">--</span>
    </div>
  </div>

  <div class="controls-panel">
    <div class="controls-row">
      <label for="mode-select" style="font-size:0.8rem; font-weight:600; color:#94a3b8;">Normal Mode:</label>
      <select id="mode-select" class="select-dropdown"></select>
      <button class="btn btn-primary" id="play-btn">⏸ Pause</button>
      <button class="btn btn-active" id="vectors-btn">🏹 Vectors: ON</button>
    </div>
    <div class="controls-row">
      <span class="slider-label" id="amp-label">Amplitude: {amplitude:.2f} Å</span>
      <input type="range" min="0" max="150" value="{int(amplitude * 100)}" class="slider" id="amp-slider">
      <span class="slider-label" id="speed-label" style="min-width:70px; margin-left:10px;">Speed: 1.0x</span>
      <input type="range" min="20" max="300" value="100" class="slider" id="speed-slider">
    </div>
  </div>

  <script>
    const data = {payload};
    const atoms = data.atoms;
    const bonds = data.bonds;
    const modes = data.modes;

    let activeModeIdx = {default_mode_idx};
    let isPlaying = true;
    let showVectors = true;
    let amplitude = {amplitude};
    let speed = 1.0;
    let animPhase = 0.0;

    const container = document.getElementById("canvas-container");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 8);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const ambLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambLight);
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(10, 20, 15);
    scene.add(dirLight1);
    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.35);
    dirLight2.position.set(-10, -10, -10);
    scene.add(dirLight2);

    const moleculeGroup = new THREE.Group();
    scene.add(moleculeGroup);
    const vectorsGroup = new THREE.Group();
    scene.add(vectorsGroup);

    // Compute center of mass to center the molecule in view
    let com = new THREE.Vector3(0, 0, 0);
    atoms.forEach(a => {{
      com.add(new THREE.Vector3(a.xyz[0], a.xyz[1], a.xyz[2]));
    }});
    com.divideScalar(atoms.length || 1);

    // Build atom sphere meshes
    const atomMeshes = [];
    const sphereGeo = new THREE.SphereGeometry(1, 32, 32);

    atoms.forEach(a => {{
      const mat = new THREE.MeshPhongMaterial({{
        color: new THREE.Color(a.color),
        shininess: 90,
        specular: 0x444444,
      }});
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.scale.set(a.radius, a.radius, a.radius);
      mesh.position.set(a.xyz[0] - com.x, a.xyz[1] - com.y, a.xyz[2] - com.z);
      moleculeGroup.add(mesh);
      atomMeshes.push(mesh);
    }});

    // Build bond cylinders
    const bondMeshes = [];
    const cylGeo = new THREE.CylinderGeometry(0.1, 0.1, 1, 16);

    bonds.forEach(([i, j]) => {{
      const mat = new THREE.MeshPhongMaterial({{
        color: 0x94a3b8,
        shininess: 70,
        specular: 0x222222,
      }});
      const mesh = new THREE.Mesh(cylGeo, mat);
      moleculeGroup.add(mesh);
      bondMeshes.push({{ mesh, i, j }});
    }});

    function updateBonds() {{
      bondMeshes.forEach(b => {{
        const p1 = atomMeshes[b.i].position;
        const p2 = atomMeshes[b.j].position;
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        const len = p1.distanceTo(p2);
        b.mesh.position.copy(mid);
        b.mesh.scale.set(1, len, 1);
        b.mesh.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          new THREE.Vector3().subVectors(p2, p1).normalize()
        );
      }});
    }}

    // Vector arrows for active normal mode
    let arrowHelpers = [];

    function rebuildVectorArrows() {{
      // Remove old arrows
      while (vectorsGroup.children.length > 0) {{
        vectorsGroup.remove(vectorsGroup.children[0]);
      }}
      arrowHelpers = [];

      if (!showVectors || activeModeIdx < 0 || activeModeIdx >= modes.length) return;

      const m = modes[activeModeIdx];
      const disps = m.displacements;

      atoms.forEach((a, idx) => {{
        const d = disps[idx];
        const dir = new THREE.Vector3(d[0], d[1], d[2]);
        const len = dir.length();
        if (len > 1e-4) {{
          dir.normalize();
          const arrowLen = Math.max(0.2, len * amplitude * 2.2);
          const arrowColor = m.is_imaginary ? 0xf87171 : 0xfacc15;
          const headLen = Math.min(arrowLen * 0.35, 0.35);
          const headWidth = headLen * 0.55;
          const arrow = new THREE.ArrowHelper(dir, atomMeshes[idx].position, arrowLen, arrowColor, headLen, headWidth);
          vectorsGroup.add(arrow);
          arrowHelpers.push({{ arrow, idx, dir, len }});
        }}
      }});
    }}

    function updateVectorArrowPositions() {{
      arrowHelpers.forEach(item => {{
        item.arrow.position.copy(atomMeshes[item.idx].position);
      }});
    }}

    // UI elements & Mode selection
    const modeSelect = document.getElementById("mode-select");
    const modeFreq = document.getElementById("mode-freq");
    const modeForce = document.getElementById("mode-force");
    const modeMass = document.getElementById("mode-mass");
    const modeChar = document.getElementById("mode-char");

    modes.forEach((m, idx) => {{
      const opt = document.createElement("option");
      opt.value = idx;
      let label = `Mode ${{m.index}}: ${{m.frequency_cm.toFixed(1)}} cm⁻¹`;
      if (m.is_tr_mode) {{
        label += " (Translation/Rotation)";
      }} else if (m.is_imaginary) {{
        label += " (Imaginary / Saddle)";
      }}
      opt.textContent = label;
      if (idx === activeModeIdx) opt.selected = true;
      modeSelect.appendChild(opt);
    }});

    function updateModeCard() {{
      if (activeModeIdx < 0 || activeModeIdx >= modes.length) return;
      const m = modes[activeModeIdx];
      modeFreq.textContent = `${{m.frequency_cm.toFixed(2)}} cm⁻¹${{m.is_imaginary ? " (i)" : ""}}`;
      modeFreq.style.color = m.is_imaginary ? "#f87171" : "#38bdf8";

      modeForce.textContent = `${{m.force_constant_mdyne_a.toFixed(4)}} mDyne/Å`;
      modeMass.textContent = `${{m.reduced_mass_amu.toFixed(4)}} amu`;

      if (m.is_tr_mode) {{
        modeChar.textContent = "Translation / Rotation";
        modeChar.style.color = "#94a3b8";
      }} else if (m.is_imaginary) {{
        modeChar.textContent = "Transition State (Saddle Point)";
        modeChar.style.color = "#f87171";
      }} else {{
        modeChar.textContent = "Harmonic Vibration (Bound Minimum)";
        modeChar.style.color = "#4ade80";
      }}
    }}

    modeSelect.addEventListener("change", (e) => {{
      activeModeIdx = parseInt(e.target.value);
      animPhase = 0.0;
      updateModeCard();
      rebuildVectorArrows();
    }});

    const playBtn = document.getElementById("play-btn");
    playBtn.addEventListener("click", () => {{
      isPlaying = !isPlaying;
      playBtn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
      if (!isPlaying) {{
        // Reset atoms to equilibrium position
        atoms.forEach((a, idx) => {{
          atomMeshes[idx].position.set(a.xyz[0] - com.x, a.xyz[1] - com.y, a.xyz[2] - com.z);
        }});
        updateBonds();
        updateVectorArrowPositions();
      }}
    }});

    const vectorsBtn = document.getElementById("vectors-btn");
    vectorsBtn.addEventListener("click", () => {{
      showVectors = !showVectors;
      vectorsBtn.textContent = `🏹 Vectors: ${{showVectors ? "ON" : "OFF"}}`;
      vectorsBtn.className = showVectors ? "btn btn-active" : "btn";
      rebuildVectorArrows();
    }});

    const ampSlider = document.getElementById("amp-slider");
    const ampLabel = document.getElementById("amp-label");
    ampSlider.addEventListener("input", (e) => {{
      amplitude = parseFloat(e.target.value) / 100.0;
      ampLabel.textContent = `Amplitude: ${{amplitude.toFixed(2)}} Å`;
      rebuildVectorArrows();
    }});

    const speedSlider = document.getElementById("speed-slider");
    const speedLabel = document.getElementById("speed-label");
    speedSlider.addEventListener("input", (e) => {{
      speed = parseFloat(e.target.value) / 100.0;
      speedLabel.textContent = `Speed: ${{speed.toFixed(1)}}x`;
    }});

    // Initialize display
    updateModeCard();
    updateBonds();
    rebuildVectorArrows();

    // Adjust camera distance to fit molecule
    let maxDist = 1.0;
    atoms.forEach(a => {{
      const d = new THREE.Vector3(a.xyz[0] - com.x, a.xyz[1] - com.y, a.xyz[2] - com.z).length();
      if (d > maxDist) maxDist = d;
    }});
    camera.position.set(0, 0, Math.max(5, maxDist * 3.2));
    controls.update();

    // Animation loop
    function animate() {{
      requestAnimationFrame(animate);

      if (isPlaying && activeModeIdx >= 0 && activeModeIdx < modes.length) {{
        animPhase += 0.06 * speed;
        const sinFactor = Math.sin(animPhase);
        const disps = modes[activeModeIdx].displacements;

        atoms.forEach((a, idx) => {{
          const d = disps[idx];
          const x = (a.xyz[0] - com.x) + sinFactor * amplitude * d[0];
          const y = (a.xyz[1] - com.y) + sinFactor * amplitude * d[1];
          const z = (a.xyz[2] - com.z) + sinFactor * amplitude * d[2];
          atomMeshes[idx].position.set(x, y, z);
        }});

        updateBonds();
        updateVectorArrowPositions();
      }}

      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""
    return html


def visualize_vibrations(
    vib_result: Any,
    output_path: Optional[Union[str, Path]] = None,
    open_browser: bool = False,
    mode_index: Optional[int] = None,
    amplitude: float = 0.5,
    title: Optional[str] = None,
) -> str:
    """Renders standalone HTML with interactive 3D WebGL normal mode player."""
    html_content = generate_vibrations_html(vib_result, mode_index=mode_index, amplitude=amplitude, title=title)
    if output_path:
        out_file = Path(output_path)
    elif open_browser:
        temp_dir = Path(tempfile.gettempdir())
        out_file = temp_dir / "phatpsy_vibrations.html"
    else:
        out_file = None

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html_content, encoding="utf-8")
        if open_browser:
            webbrowser.open(out_file.as_uri())

    return html_content


def show_vibrations(
    vib_result: Any,
    mode_index: Optional[int] = None,
    amplitude: float = 0.5,
    height: int = 550,
    title: Optional[str] = None,
) -> Any:
    """Displays interactive 3D normal mode vibrational visualizer inside Jupyter notebooks."""
    try:
        import html
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="IPython.*")
            from IPython.display import HTML
            html_code = generate_vibrations_html(vib_result, mode_index=mode_index, amplitude=amplitude, title=title)
            escaped_html = html.escape(html_code, quote=True)
            iframe_html = (
                f'<div style="width:100%; border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.15); background:#0f172a;">'
                f'<iframe srcdoc="{escaped_html}" width="100%" height="{height}" '
                f'sandbox="allow-scripts allow-same-origin allow-popups" '
                f'style="border:none; width:100%; height:{height}px; display:block; background:#0f172a;"></iframe>'
                f'</div>'
            )
            return HTML(iframe_html)
    except ImportError:
        raise RuntimeError("IPython is required to use show_vibrations() in Jupyter notebooks.")



