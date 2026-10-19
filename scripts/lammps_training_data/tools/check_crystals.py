"""Check that crystal MD trajectories are still the crystal they started from.

    python tools/check_crystals.py <md_training_data dir> [more dirs ...] > check.tsv

For every crystal trajectory (all folders except melt/ and amorphous/) the first and last frame are compared with the
starting structure structures/crystal/<name>.data:
  rho        density (g/cm3) of the last frame, and the ratio to the starting structure
  cnSi, cnAl average number of O within 2.3 A of Si and Al (4 = tetrahedral, 6 = octahedral)
  rdf        Pearson correlation of the total g(r) (2.4-7 A, distances scaled to the starting density) between the
             last frame and the starting structure; amorphization or a reconstructive transition lowers it
  disp       displacement (A) of the atoms from their starting sites: the mean of the first and last frame is taken in
             fractional coordinates of each frame's cell (so cell changes do not count), the common shift is removed,
             and the 90th percentile over the atoms is reported; vibrations give ~0.3 A, a collapse > 1 A
  flag       "BROKEN?" when the coordination changed by > 0.3, the density by > 8 % or disp > 1 A
             (rdf is printed for reference only: low-symmetry crystals at 300-500 K already give 0.75-0.85)
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRUC = os.path.join(HERE, "..", "structures", "crystal")
SKIP = {"melt", "amorphous"}
MASS_SHIK = {1: 28.0855, 2: 15.9994, 3: 26.981538, 4: 22.98977, 5: 40.078}
EL_SHIK = {1: "Si", 2: "O", 3: "Al", 4: "Na", 5: "Ca"}
RBINS = np.arange(0.0, 7.05, 0.05)


def read_frames(path, which):
    """Yield (types, positions, h) for the frames listed in `which` ('first', 'last')."""
    with open(path) as f:
        starts = []
        pos = 0
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                starts.append(pos)
            pos += len(line.encode())
    out = {}
    with open(path, "rb") as f:  # rows sorted by atom id
        for key, idx in (("first", 0), ("last", len(starts) - 1)):
            if key not in which:
                continue
            f.seek(starts[idx])
            lines = [f.readline().decode() for _ in range(9)]
            n = int(lines[3])
            b = np.array([list(map(float, l.split())) for l in lines[5:8]])
            a = np.loadtxt([f.readline().decode() for _ in range(n)])
            a = a[np.argsort(a[:, 0])]
            out[key] = (a[:, 1].astype(int), a[:, 2:5], box_to_h(b, "xy" in lines[4]))
    return out, len(starts)


def box_to_h(b, tilted):
    if tilted:
        (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo, zhi, yz) = b
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo, yhi = ylo_b - min(0.0, yz), yhi_b - max(0.0, yz)
    else:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = b[:, :2]
        xy = xz = yz = 0.0
    return np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])


def read_data(path):
    txt = open(path).read().split("\n")
    h = np.zeros((3, 3))
    xy = xz = yz = 0.0
    for l in txt[:30]:
        s = l.split()
        if l.endswith("xlo xhi"):
            h[0, 0] = float(s[1]) - float(s[0])
        elif l.endswith("ylo yhi"):
            h[1, 1] = float(s[1]) - float(s[0])
        elif l.endswith("zlo zhi"):
            h[2, 2] = float(s[1]) - float(s[0])
        elif l.endswith("xy xz yz"):
            xy, xz, yz = map(float, s[:3])
    h[1, 0], h[2, 0], h[2, 1] = xy, xz, yz
    i = next(k for k, l in enumerate(txt) if l.startswith("Atoms"))
    rows = [l.split() for l in txt[i + 1:] if l.strip()]
    rows = [r for r in rows if len(r) >= 6 and r[0].isdigit()]
    a = np.array([[float(x) for x in r[:6]] for r in rows])
    a = a[np.argsort(a[:, 0])]
    return a[:, 1].astype(int), a[:, 3:6], h


def pair_dists(pos, h, sel_a=None, sel_b=None, rmax=7.0):
    inv = np.linalg.inv(h)
    fr = pos @ inv
    A = np.arange(len(pos)) if sel_a is None else sel_a
    B = np.arange(len(pos)) if sel_b is None else sel_b
    out = []
    for c in range(0, len(A), 256):
        d = fr[B][None, :, :] - fr[A[c:c + 256]][:, None, :]
        d -= np.round(d)
        r = np.linalg.norm(d @ h, axis=2)
        out.append(r)
    return np.vstack(out)


def analyse(types, pos, h, scale=1.0):
    vol = abs(np.linalg.det(h))
    nat = len(types)
    mass = sum(MASS_SHIK.get(t, 0.0) for t in types) if types.max() > 2 else \
        np.sum(np.where(types == 1, MASS_SHIK[1], MASS_SHIK[2]))
    rho = mass / vol * 1.66053907
    o = np.where(types == 2)[0]
    cn = {}
    for t, el in ((1, "Si"), (3, "Al")):
        idx = np.where(types == t)[0]
        if len(idx) and not (t == 3 and types.max() <= 2):
            r = pair_dists(pos, h, idx, o)
            cn[el] = float(((r > 0.1) & (r < 2.3)).sum(1).mean())
    sub = np.arange(0, nat, max(1, nat // 600))
    r = pair_dists(pos, h, sub)[:, :] * scale
    hist, _ = np.histogram(r[(r > 0.1)], bins=RBINS)
    rc = 0.5 * (RBINS[1:] + RBINS[:-1])
    g = hist / (rc ** 2)
    k = np.exp(-0.5 * (np.arange(-4, 5) * 0.05 / 0.08) ** 2)
    g = np.convolve(g, k / k.sum(), mode="same")
    return rho, cn, g


def site_displacement(frames, pos0, h0):
    f0 = pos0 @ np.linalg.inv(h0)
    acc = np.zeros_like(f0)
    for _, p, h in frames:
        d = p @ np.linalg.inv(h) - f0
        acc += d - np.round(d)
    d = acc / len(frames)
    d -= np.median(d, axis=0)
    d -= np.round(d)
    hm = np.mean([h for _, _, h in frames], axis=0)
    return float(np.percentile(np.linalg.norm(d @ hm, axis=1), 90))


def main(dirs):
    rc = 0.5 * (RBINS[1:] + RBINS[:-1])
    m = (rc > 2.4) & (rc < 6.9)
    refs = {}
    print("\t".join(["label", "file", "frames", "rho", "rho/rho0", "cnSi", "cnSi0", "cnAl", "cnAl0", "disp", "rdf", "flag"]))
    for d in dirs:
        for label in sorted(os.listdir(d)):
            if label in SKIP or not os.path.isdir(os.path.join(d, label)):
                continue
            for fn in sorted(os.listdir(os.path.join(d, label))):
                if not fn.endswith(".lammpstrj"):
                    continue
                name = re.sub(r"_T\d+K_P\d+GPa\.lammpstrj$", "", fn)
                fr, nframes = read_frames(os.path.join(d, label, fn), ("first", "last"))
                t, p, h = fr["last"]
                if name not in refs:
                    t0, p0, h0 = read_data(os.path.join(STRUC, name + ".data"))
                    refs[name] = analyse(t0, p0, h0) + (p0, h0)
                rho0, cn0, g0, p0, h0 = refs[name]
                disp = site_displacement([fr["first"], fr["last"]], p0, h0) if len(p0) == len(p) else float("nan")
                rho, cn, _ = analyse(t, p, h)
                _, _, g = analyse(t, p, h, scale=(rho / rho0) ** (1 / 3))
                corr = float(np.corrcoef(g[m], g0[m])[0, 1])
                bad = abs(rho / rho0 - 1) > 0.08 or not disp <= 1.0 or any(
                    abs(cn.get(k, 0) - cn0.get(k, 0)) > 0.3 for k in cn0)
                print("\t".join([label, fn, str(nframes), f"{rho:.3f}", f"{rho / rho0:.3f}",
                                 f"{cn.get('Si', float('nan')):.2f}", f"{cn0.get('Si', float('nan')):.2f}",
                                 f"{cn.get('Al', float('nan')):.2f}", f"{cn0.get('Al', float('nan')):.2f}",
                                 f"{disp:.2f}", f"{corr:.3f}", "BROKEN?" if bad else "ok"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
