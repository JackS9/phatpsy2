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
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import json
import webbrowser
import tempfile

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


def generate_visualization_data(deck_or_atoms: Union[CalculationDeck, List[AtomConfig]]) -> Dict[str, Any]:
    """Prepares structured JSON data for 3D rendering."""
    if hasattr(deck_or_atoms, "geometry"):
        atoms = deck_or_atoms.geometry
        title = getattr(deck_or_atoms, "title", "PHATPSY Molecule")
    else:
        atoms = deck_or_atoms
        title = "PHATPSY Molecule"

    atom_data = []
    # Identify symmetry generator mapping
    generator_map: Dict[int, int] = {}
    current_primary_idx: Optional[int] = None
    for idx, a in enumerate(atoms):
        if a.symmetry is None:
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
            "element": a.element,
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


def generate_html(deck_or_atoms: Union[CalculationDeck, List[AtomConfig]], title: Optional[str] = None) -> str:
    """Generates a complete, interactive standalone HTML5 WebGL viewer using Three.js."""
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
      <span class="badge badge-blue">Spin Up &uarr;: {data['stats']['spin_up']}</span>
      <span class="badge badge-orange">Spin Down &darr;: {data['stats']['spin_down']}</span>
      {f'<span class="badge badge-violet">Virtual: {data["stats"]["virtual"]}</span>' if data['stats']['virtual'] > 0 else ''}
      {f'<span class="badge badge-gray">Fixed: {data["stats"]["fixed"]}</span>' if data['stats']['fixed'] > 0 else ''}
    </div>
    
    <div class="controls">
      <div class="control-row">
        <label><input type="checkbox" id="toggle-spin" checked> Localized VB Spin Arrows (&uarr; / &darr;)</label>
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
    <div class="legend-item"><span class="dot" style="background: #38bdf8; box-shadow: 0 0 6px #38bdf8;"></span> <b>Spin Up (&uarr;)</b>: Alpha-dominant</div>
    <div class="legend-item"><span class="dot" style="background: #fb923c; box-shadow: 0 0 6px #fb923c;"></span> <b>Spin Down (&darr;)</b>: Beta-dominant (QFLIP=T)</div>
    <div class="legend-item"><span class="dot" style="background: transparent; border: 2px dashed #c084fc;"></span> <b>Virtual Center</b>: Ghost basis / external probe</div>
    <div class="legend-item"><span class="dot" style="background: transparent; border: 2px solid #94a3b8;"></span> <b>Fixed Atom</b>: Frozen orbitals & potential</div>
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

    scene.add(atomGroup);
    scene.add(bondGroup);
    scene.add(spinGroup);
    scene.add(symmetryGroup);
    scene.add(ghostGroup);
    scene.add(cageGroup);

    // Compute center of mass
    let cx = 0, cy = 0, cz = 0;
    const atoms = MOL_DATA.atoms;
    const n = atoms.length;
    atoms.forEach(a => {{ cx += a.x; cy += a.y; cz += a.z; }});
    if (n > 0) {{ cx /= n; cy /= n; cz /= n; }}

    const atomMeshes = [];

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
        mat = new THREE.MeshStandardMaterial({{
          color: a.color,
          roughness: 0.35,
          metalness: 0.15,
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

      // Spin vector arrows
      if (a.spin === 'up' || a.spin === 'down') {{
        const isUp = (a.spin === 'up');
        const arrowDir = isUp ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 0, -1);
        const arrowColor = isUp ? 0x38bdf8 : 0xfb923c;
        const arrowLen = r * 2.2;
        const headLen = arrowLen * 0.4;
        const headWidth = headLen * 0.6;
        
        const arrowOrigin = isUp
          ? pos.clone().add(new THREE.Vector3(0, 0, r * 0.2))
          : pos.clone().sub(new THREE.Vector3(0, 0, r * 0.2));
          
        const arrow = new THREE.ArrowHelper(arrowDir, arrowOrigin, arrowLen, arrowColor, headLen, headWidth);
        spinGroup.add(arrow);
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

    // Initial camera position
    let maxDist = 3.0;
    atoms.forEach(a => {{
      const d = Math.sqrt((a.x - cx)**2 + (a.y - cy)**2 + (a.z - cz)**2);
      if (d > maxDist) maxDist = d;
    }});
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
        if (d.spin === 'up') spinStr += ' (&uarr; Alpha-dominant)';
        else if (d.spin === 'down') spinStr += ' (&darr; Beta-dominant, QFLIP=T)';

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

    // Controls listeners
    document.getElementById('toggle-spin').addEventListener('change', (e) => {{
      spinGroup.visible = e.target.checked;
    }});
    document.getElementById('toggle-bonds').addEventListener('change', (e) => {{
      bondGroup.visible = e.target.checked;
    }});
    document.getElementById('toggle-symmetry').addEventListener('change', (e) => {{
      symmetryGroup.visible = e.target.checked;
    }});
    document.getElementById('toggle-ghosts').addEventListener('change', (e) => {{
      ghostGroup.visible = e.target.checked;
    }});
    document.getElementById('toggle-cages').addEventListener('change', (e) => {{
      cageGroup.visible = e.target.checked;
    }});
    document.getElementById('btn-reset').addEventListener('click', () => {{
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
    deck_or_atoms: Union[CalculationDeck, List[AtomConfig]],
    output_path: Optional[Union[str, Path]] = None,
    auto_open: bool = False,
) -> str:
    """
    Renders an interactive 3D WebGL visualization of the molecule or calculation deck.

    Args:
        deck_or_atoms: CalculationDeck or list of AtomConfig objects.
        output_path: Optional path to save the generated HTML file.
        auto_open: If True, opens the HTML file in the default web browser.

    Returns:
        The generated HTML content string.
    """
    html_content = generate_html(deck_or_atoms)

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


def show_deck(deck_or_atoms: Union[CalculationDeck, List[AtomConfig]], height: int = 600):
    """
    Displays the 3D interactive viewer inline inside a Jupyter notebook.
    """
    try:
        import html
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="IPython.*")
            from IPython.display import HTML
            html_code = generate_html(deck_or_atoms)
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
