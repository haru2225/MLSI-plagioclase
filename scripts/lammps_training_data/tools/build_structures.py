#!/usr/bin/env python3
"""Build the LAMMPS data files (structures/crystal/*.data, structures/glass_init/*.data).

Crystals: supercells of the COD CIF templates in structures/cif (see CRYSTALS below for the COD ids and
the substitutions used for phases without an ordered CIF).  Glass/melt: random packings of the target
compositions that are melted in LAMMPS.  Charges are written for the potential of each system
(BKS for pure SiO2, SHIK otherwise).  The generated data files are part of the package, so this script
only has to be rerun to change the structures.  Needs numpy and ase (CIF parsing only).

Usage:  python tools/build_structures.py
"""
import re
from pathlib import Path

import numpy as np
from ase.geometry import cellpar_to_cell
from ase.io.cif import parse_cif

ROOT = Path(__file__).resolve().parents[1]
CIF_DIR = ROOT / "structures" / "cif"
SEED = 20261018
L_MIN = 24.0  # minimum perpendicular width of a crystal supercell (A)

TYPES = {"BKS": ["Si", "O"], "SHIK": ["Si", "O", "Al", "Na", "Ca"]}
MASS = {"Si": 28.0855, "O": 15.9994, "Al": 26.981538, "Na": 22.98977, "Ca": 40.078}
Q_BKS = {"Si": 2.4, "O": -1.2}
Q_SHIK = {"Si": 1.7755, "Al": 1.6334, "Na": 0.6018, "Ca": 1.3967}  # O: charge neutrality

# Al-rich T sites of the anorthite-type Al/Si order (Al1ozo, Al1moo, Al2ooo, Al2mzo in COD 9001258),
# used to place Al in labradorite without Al-O-Al linkages.
LAB_AL_SITES = {"T1oz", "T1mo", "T2oo", "T2mz"}
LAB_AN = 0.6  # labradorite composition Ca0.6Na0.4Al1.6Si2.4O8 (COD 2108244)

CRYSTALS = {
    # name: (output phase folder, COD id, potential, recipe)
    "albite": ("plagioclase", "9001630", "SHIK", "plain"),
    "anorthite": ("plagioclase", "9001258", "SHIK", "plain"),
    "labradorite": ("plagioclase", "2108244", "SHIK", "labradorite"),
    "jadeite": ("jadeite", "9000342", "SHIK", "plain"),
    "lingunite": ("lingunite", "9001521", "SHIK", "lingunite"),
    "grossular": ("grossular", "9001532", "SHIK", "plain"),
    "kyanite": ("kyanite", "9000720", "SHIK", "plain"),
    "quartz": ("quartz", "9000775", "SHIK", "plain"),
    "coesite": ("coesite", "9000802", "SHIK", "plain"),
    "cristobalite": ("cristobalilte", "9001578", "SHIK", "plain"),
    "tridymite": ("tridymite", "9000520", "SHIK", "plain"),
    "stishovite": ("stishovite", "9007530", "SHIK", "plain"),
    "seifertite": ("seifertite", "9002782", "SHIK", "plain"),
    # beta forms in SHIK: alpha-quartz turns into beta-quartz on heating, alpha-cristobalite into beta-cristobalite (as in Yuan & Huang,
    # PRB 85, 134114); HP-tridymite from COD.  (Alpha-cristobalite itself collapses at 0-400 K in SHIK, so the cristobalite class is the beta form.)
    "beta_quartz": ("beta_quartz", "9000775", "SHIK", "plain"),
    "beta_tridymite": ("beta_tridymite", "8104539", "SHIK", "plain"),
    # high-pressure products of plagioclase (SHIK)
    "corundum": ("corundum", "1000017", "SHIK", "plain"),
    "casio3_perovskite": ("casio3_perovskite", "1545170", "SHIK", "plain"),
    "cas_phase": ("cas_phase", "9006693", "SHIK", "cas"),
    "cf_naalsio4": ("cf_naalsio4", "9009463", "SHIK", "cf"),
    # polymorphs of phases above (high-temperature forms need their own temperatures, see run_all.sh)
    "andalusite": ("andalusite", "9000715", "SHIK", "plain"),
    "sillimanite": ("sillimanite", "9003985", "SHIK", "plain"),
    "monalbite": ("plagioclase", "9000707", "SHIK", "monalbite"),
    "anorthite_I1": ("plagioclase", "9016764", "SHIK", "anorthiteI1"),
}

# supercell matrices (in unit-cell vectors) that replace the diagonal one: a cubic 2x2x2 of grossular is only 23.7 A wide
# and 3x3x3 has 4320 atoms, this det-12 cell has 1920 atoms and all perpendicular widths >= 24 A
SMAT = {"grossular": [[-2, -1, -1], [-1, 0, 2], [-1, 2, 0]]}

# glass/melt compositions: formula units and atoms per formula unit (all 1872 atoms)
GLASSES = {
    "SiO2": ("SHIK", 624, {"Si": 1, "O": 2}),
    "albite": ("SHIK", 144, {"Na": 1, "Al": 1, "Si": 3, "O": 8}),
    "anorthite": ("SHIK", 144, {"Ca": 1, "Al": 2, "Si": 2, "O": 8}),
    "labradorite": ("SHIK", 144, None),  # An60, counts set in glass_counts()
}
GLASS_DENSITY = 2.3  # g/cm^3, random starting configuration (melted afterwards)


# ----------------------------------------------------------------------------- CIF handling
def element_of(label, type_symbol):
    s = type_symbol if isinstance(type_symbol, str) and type_symbol not in ("?", ".") else label
    for el in ("Si", "Al", "Na", "Ca", "K"):
        if s.startswith(el):
            return el
    if s.startswith("O"):
        return "O"
    raise ValueError(f"unknown element for site {label!r} / {type_symbol!r}")


def parse_op(op):
    rot, tr = np.zeros((3, 3)), np.zeros(3)
    for i, expr in enumerate(op.lower().replace(" ", "").split(",")):
        for sign, term in re.findall(r"([+-]?)([^+-]+)", expr):
            s = -1.0 if sign == "-" else 1.0
            if term in ("x", "y", "z"):
                rot[i, "xyz".index(term)] += s
            elif "/" in term:
                n, d = term.split("/")
                tr[i] += s * float(n) / float(d)
            else:
                tr[i] += s * float(term)
    return rot, tr


def read_cif(cod_id):
    blk = next(parse_cif(str(CIF_DIR / f"{cod_id}.cif")))
    ops = blk.get("_space_group_symop_operation_xyz") or blk.get("_symmetry_equiv_pos_as_xyz")
    labels = blk.get("_atom_site_label")
    types = blk.get("_atom_site_type_symbol") or [None] * len(labels)
    occ = blk.get("_atom_site_occupancy") or [1.0] * len(labels)
    frac = np.array([blk.get(f"_atom_site_fract_{c}") for c in "xyz"], float).T
    sites = [dict(label=l, el=element_of(l, t), frac=f, occ=float(o)) for l, t, f, o in zip(labels, types, frac, occ)]
    return cellpar_to_cell(blk.get_cellpar()), [parse_op(o) for o in ops], sites


def expand(sites, ops, tol=1e-3):
    """Apply all symmetry operations; returns labels, elements, fractional coordinates of the unit cell."""
    labels, els, fr = [], [], []
    for s in sites:
        pts = []
        for rot, tr in ops:
            p = (rot @ s["frac"] + tr) % 1.0
            if all(np.linalg.norm((p - q + 0.5) % 1.0 - 0.5) > tol for q in pts):
                pts.append(p)
        labels += [s["label"]] * len(pts)
        els += [s["el"]] * len(pts)
        fr += pts
    return np.array(labels), np.array(els, dtype=object), np.array(fr)


def merge_same_position(cell, labels, els, fr, placeholder, tol=0.05):
    """Mixed sites (e.g. Si/Al on one position) -> one atom with a placeholder element."""
    keep, out_els = [], els.copy()
    for i in range(len(fr)):
        dup = [j for j in keep if np.linalg.norm(((fr[i] - fr[j] + 0.5) % 1.0 - 0.5) @ cell) < tol]
        if dup:
            out_els[dup[0]] = placeholder
        else:
            keep.append(i)
    return labels[keep], out_els[keep], fr[keep]


# ----------------------------------------------------------------------------- recipes
def unit_cell(name):
    phase, cod, pot, recipe = CRYSTALS[name]
    cell, ops, sites = read_cif(cod)
    if recipe == "labradorite":
        # split M sites (Moi/Moo, Mzo/Mzi): keep the main position of each pair
        sites = [s for s in sites if s["label"] not in ("Moo", "Mzi")]
    elif recipe == "plain" and any(abs(s["occ"] - 1.0) > 0.02 for s in sites):
        raise ValueError(f"{name}: CIF {cod} has partial occupancies")
    labels, els, fr = expand(sites, ops)
    if recipe == "labradorite":
        els = np.array(["M" if l.startswith("M") else ("TA" if l in LAB_AL_SITES else ("Si" if l.startswith("T") else e))
                        for l, e in zip(labels, els)], dtype=object)
    elif recipe == "lingunite":
        labels, els, fr = merge_same_position(cell, labels, els, fr, "T")
        els[els == "K"] = "Na"
    elif recipe == "monalbite":  # high-temperature C2/m albite: Al 1/4, Si 3/4 on both T sites
        labels, els, fr = merge_same_position(cell, labels, els, fr, "T")
    elif recipe == "anorthiteI1":  # I-1 anorthite: Ca on split positions with occupancy 1/2
        els[els == "Ca"] = "CaS"
    elif recipe == "cf":  # NaAlSiO4 (CaFe2O4 type): Al/Si 1:1 disordered over the two octahedral sites
        labels, els, fr = merge_same_position(cell, labels, els, fr, "M")
    elif recipe == "cas":  # CaAl4Si2O11: M1 = Si2/3 Al1/3, AlT = pair of half-occupied positions 0.64 A apart
        labels, els, fr = merge_same_position(cell, labels, els, fr, "M")
        els[labels == "AlT"] = "TA"
    return cell, labels, els, fr


def assign(rng, els, placeholder, counts):
    idx = rng.permutation(np.where(els == placeholder)[0])
    assert sum(counts.values()) == len(idx), (placeholder, counts, len(idx))
    start = 0
    for el, n in counts.items():
        els[idx[start:start + n]] = el
        start += n


def thin_split_sites(h, els, fr, rng, label, keep, dmax, size):
    """Split sites: of each group of `size` positions closer than dmax keep one (element `keep`) at random, drop the others."""
    idx = np.where(els == label)[0]
    done = set()
    for i in rng.permutation(idx):
        if i in done:
            continue
        d = ((fr[idx] - fr[i] + 0.5) % 1.0 - 0.5) @ h
        mates = idx[np.linalg.norm(d, axis=1) < dmax]
        assert len(mates) == size, f"{label}: split site with {len(mates)} positions, expected {size}"
        done.update(mates.tolist())
        els[mates] = "X"
        els[i] = keep
    return els


def order_loewenstein(h, els, fr, rng, dmax=3.5, steps=200000):
    """Reshuffle the Al/Si of the T sites (same Al count) until no two Al T sites share an O (Loewenstein rule)."""
    t = np.where((els == "Al") | (els == "Si"))[0]
    d = (fr[t][:, None, :] - fr[t][None, :, :] + 0.5) % 1.0 - 0.5
    near = np.linalg.norm(d @ h, axis=2) < dmax
    np.fill_diagonal(near, False)
    al = els[t] == "Al"
    n_bad = int((near & al[:, None] & al[None, :]).sum() // 2)
    for _ in range(steps):
        if n_bad == 0:
            break
        i = rng.choice(np.where(al)[0])
        j = rng.choice(np.where(~al)[0])
        # moving Al from i to j: remove i's Al-Al pairs, add j's (j is not a neighbor-pair partner of itself)
        delta = int((near[j] & al).sum()) - int((near[i] & al).sum()) - int(near[i, j])
        if delta <= 0:
            al[i], al[j] = False, True
            n_bad += delta
    els[t] = np.where(al, "Al", "Si")
    print(f"  Al-O-Al pairs after ordering: {n_bad}")
    return els


def finalize_species(name, els, rng):
    recipe = CRYSTALS[name][3]
    if recipe == "labradorite":
        n_m = int(np.sum(els == "M"))
        n_ca = int(round(LAB_AN * n_m))
        n_na = n_m - n_ca
        n_al = n_na + 2 * n_ca
        assign(rng, els, "M", {"Ca": n_ca, "Na": n_na})
        assign(rng, els, "TA", {"Al": n_al, "Si": int(np.sum(els == "TA")) - n_al})
    elif recipe == "lingunite":
        n_t = int(np.sum(els == "T"))
        assign(rng, els, "T", {"Al": n_t // 4, "Si": n_t - n_t // 4})
    elif recipe == "monalbite":
        n_t = int(np.sum(els == "T"))
        assign(rng, els, "T", {"Al": n_t // 4, "Si": n_t - n_t // 4})
    elif recipe == "cf":
        n_m = int(np.sum(els == "M"))
        assign(rng, els, "M", {"Al": n_m // 2, "Si": n_m - n_m // 2})
    elif recipe == "cas":
        n_m = int(np.sum(els == "M"))
        n_si = int(round(2 * n_m / 3))
        assign(rng, els, "M", {"Si": n_si, "Al": n_m - n_si})
    return els


# ----------------------------------------------------------------------------- cell handling
def supercell(cell, els, fr, smat):
    smat = np.array(smat, int)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ smat
    rng_ = [range(int(np.floor(corners[:, d].min())) - 1, int(np.ceil(corners[:, d].max())) + 1) for d in range(3)]
    inv = np.linalg.inv(smat)
    new_els, new_fr = [], []
    for n in np.array(np.meshgrid(*rng_, indexing="ij")).reshape(3, -1).T:
        f = (fr + n) @ inv
        m = np.all((f > -1e-8) & (f < 1 - 1e-8), axis=1)
        new_els += list(els[m])
        new_fr.append(f[m])
    new_fr = np.vstack(new_fr) % 1.0
    assert len(new_els) == len(els) * round(abs(np.linalg.det(smat))), "supercell atom count mismatch"
    return smat @ cell, np.array(new_els, dtype=object), new_fr


def diagonal_smat(cell, l_min=L_MIN):
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    return np.diag([int(np.ceil(l_min / w)) for w in widths])


def lammps_box(h):
    a, b, c = h
    lx = np.linalg.norm(a)
    xy = b @ a / lx
    ly = np.sqrt(b @ b - xy**2)
    xz = c @ a / lx
    yz = (b @ c - xy * xz) / ly
    lz = np.sqrt(c @ c - xz**2 - yz**2)
    return lx, ly, lz, xy, xz, yz


def reduce_tilt(h):
    """Change to an equivalent basis of the same lattice so that |xy|<=lx/2, |xz|<=lx/2, |yz|<=ly/2."""
    h = h.copy()
    for _ in range(10):
        lx, ly, lz, xy, xz, yz = lammps_box(h)
        if abs(xy) > lx / 2 + 1e-6:
            h[1] -= np.round(xy / lx) * h[0]
        elif abs(yz) > ly / 2 + 1e-6:
            h[2] -= np.round(yz / ly) * h[1]
        elif abs(xz) > lx / 2 + 1e-6:
            h[2] -= np.round(xz / lx) * h[0]
        else:
            return h
    raise RuntimeError("tilt reduction did not converge")


# ----------------------------------------------------------------------------- output
def charges(pot, els):
    if pot == "BKS":
        return np.array([Q_BKS[e] for e in els])
    q = np.array([Q_SHIK.get(e, 0.0) for e in els])
    q[els == "O"] = -q.sum() / np.sum(els == "O")
    return q


def write_data(path, title, pot, els, box, frac):
    lx, ly, lz, xy, xz, yz = box
    hl = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    order = np.argsort([TYPES[pot].index(e) for e in els], kind="stable")
    els, frac = els[order], frac[order]
    pos, q = frac @ hl, charges(pot, els)
    assert abs(q.sum()) < 1e-6
    with open(path, "w") as fh:
        fh.write(f"# {title}\n\n{len(els)} atoms\n{len(TYPES[pot])} atom types\n\n")
        fh.write(f"0.0 {lx:.8f} xlo xhi\n0.0 {ly:.8f} ylo yhi\n0.0 {lz:.8f} zlo zhi\n{xy:.8f} {xz:.8f} {yz:.8f} xy xz yz\n\nMasses\n\n")
        for i, e in enumerate(TYPES[pot], 1):
            fh.write(f"{i} {MASS[e]} # {e}\n")
        fh.write("\nAtoms # charge\n\n")
        for i, (e, qi, p) in enumerate(zip(els, q, pos), 1):
            fh.write(f"{i} {TYPES[pot].index(e) + 1} {qi:.8f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return els, pos, hl


def min_distance(pos, hl):
    inv = np.linalg.inv(hl)
    dmin = np.inf
    for i in range(len(pos) - 1):
        d = (pos[i + 1:] - pos[i]) @ inv
        d = (d - np.round(d)) @ hl
        dmin = min(dmin, np.sqrt((d**2).sum(1)).min())
    return dmin


def composition(els):
    u, n = np.unique(els, return_counts=True)
    return " ".join(f"{e}{c}" for e, c in zip(u, n))


def density(els, box):
    lx, ly, lz = box[:3]
    return sum(MASS[e] for e in els) / 6.02214076e23 / (lx * ly * lz * 1e-24)


def build_crystal(name, rng):
    phase, cod, pot, recipe = CRYSTALS[name]
    cell, labels, els, fr = unit_cell(name)
    if name in ("quartz", "beta_quartz"):  # orthogonal supercell of the hexagonal cell: a, a+2b, c
        smat = [[6, 0, 0], [3, 6, 0], [0, 0, 5]]
    else:
        smat = SMAT.get(name) or diagonal_smat(cell)
    h, els, fr = supercell(cell, els, fr, smat)
    if recipe == "cas":
        els = thin_split_sites(h, els, fr, rng, "TA", "Al", 1.0, 2)
    elif recipe == "anorthiteI1":
        els = thin_split_sites(h, els, fr, rng, "CaS", "Ca", 1.0, 2)
    els = finalize_species(name, els, rng)
    if recipe == "monalbite":  # separate generator: leaves the random numbers of the other crystals unchanged
        els = order_loewenstein(h, els, fr, np.random.default_rng(SEED + 1))
    keep = els != "X"
    els, fr = els[keep], fr[keep]
    h_red = reduce_tilt(h)
    fr = (fr @ h @ np.linalg.inv(h_red)) % 1.0
    box = lammps_box(h_red)
    note = f"{name} ({phase}), COD {cod}, recipe {recipe}, supercell {np.array(smat).tolist()}, {pot}"
    els, pos, hl = write_data(ROOT / "structures" / "crystal" / f"{name}.data", note, pot, els, box, fr)
    print(f"{name:13s} {pot:4s} N={len(els):5d} rho={density(els, box):.3f} g/cc  "
          f"box={box[0]:.1f}x{box[1]:.1f}x{box[2]:.1f} tilt=({box[3]:.2f},{box[4]:.2f},{box[5]:.2f})  "
          f"dmin={min_distance(pos, hl):.3f} A  {composition(els)}")
    return els, pos, hl


def glass_counts(name):
    pot, nfu, per_fu = GLASSES[name]
    if per_fu is not None:
        return pot, {e: n * nfu for e, n in per_fu.items()}
    n_ca = int(round(LAB_AN * nfu))
    n_na = nfu - n_ca
    n_al = n_na + 2 * n_ca
    return pot, {"Ca": n_ca, "Na": n_na, "Al": n_al, "Si": 4 * nfu - n_al, "O": 8 * nfu}


def build_glass_init(name, rng, dmin=1.6):
    pot, counts = glass_counts(name)
    els = np.array([e for e, n in counts.items() for _ in range(n)], dtype=object)
    mass = sum(MASS[e] for e in els)
    lbox = (mass / 6.02214076e23 / GLASS_DENSITY * 1e24) ** (1 / 3)
    pos = np.zeros((len(els), 3))
    for i in range(len(els)):
        while True:
            p = rng.random(3) * lbox
            d = pos[:i] - p
            d -= np.round(d / lbox) * lbox
            if i == 0 or np.min((d**2).sum(1)) > dmin**2:
                pos[i] = p
                break
    box = (lbox, lbox, lbox, 0.0, 0.0, 0.0)
    write_data(ROOT / "structures" / "glass_init" / f"{name}.data",
               f"random {name} composition for melt-quench, {pot}, rho={GLASS_DENSITY}", pot, els, box, pos / lbox)
    print(f"glass_init {name:12s} {pot:4s} N={len(els)} L={lbox:.2f} A  {composition(els)}")


def check_labradorite_order(els, pos, hl):
    """Al-O-Al linkages in the labradorite supercell (should be zero)."""
    inv = np.linalg.inv(hl)
    t = np.where((els == "Al") | (els == "Si"))[0]
    n_alalo = 0
    for o in np.where(els == "O")[0]:
        d = (pos[t] - pos[o]) @ inv
        d = (d - np.round(d)) @ hl
        nb = t[np.sqrt((d**2).sum(1)) < 2.0]
        n_alalo += int(np.sum(els[nb] == "Al") >= 2)
    print(f"  labradorite: O bridging two Al = {n_alalo}")


def main():
    for d in ("crystal", "glass_init"):
        (ROOT / "structures" / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for name in CRYSTALS:
        els, pos, hl = build_crystal(name, rng)
        if name == "labradorite":
            check_labradorite_order(els, pos, hl)
    for name in GLASSES:
        build_glass_init(name, rng)


if __name__ == "__main__":
    main()
