"""Compare the exactly-commensurate RigidTTG continuum model against the
rigid-structure TB bands (ML-predicted Hamiltonian, GPU-Lanczos, OpenMX
.Band format) from rack:~/ttg_organized.

Usage:
  python compare_rigid.py TAG [--pairing derived|legacy] [--mirror]
        [--w 0.11] [--vfc 0.80] [--offset auto|<eV>] [--nshells 12]
        [--stride 1] [--out PREFIX] [--no-plot]

The TB bands along Gamma-M-K-Gamma contain BOTH graphene valleys
(E_K'(k) = E_K(-k) by time reversal), so the model is evaluated at +k and -k
and the union is compared/overlaid.

Outputs: metrics on stdout (last line = one JSON dict) + overlay PNG + NPZ.
"""
import sys, json, argparse, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttg_continuum.rigid_ttg import RigidTTG

HARTREE = 27.211386245988
DATA = ROOT / "data_rigid"


def parse_openmx_band(path):
    """Return dict: efermi_eV, kfrac (nk,2), E (nk, nb) in eV relative to E_F,
    seg_labels, seg_bounds (indices into kfrac where segments start/end)."""
    tok = open(path, encoding="utf-8").read().split("\n")
    nb, _spin, ef = tok[0].split()[0], tok[0].split()[1], float(tok[0].split()[2])
    nb = int(nb)
    nseg = int(tok[2].split()[0])
    segs = []
    for s in range(nseg):
        p = tok[3 + s].split()
        segs.append((int(p[0]), [float(x) for x in p[1:4]], [float(x) for x in p[4:7]],
                     p[7], p[8]))
    kfrac, E = [], []
    line = 3 + nseg
    while line < len(tok) and tok[line].strip():
        h = tok[line].split()
        kfrac.append([float(h[1]), float(h[2])])
        ev = [float(x) for x in tok[line + 1].split()]
        assert len(ev) == nb, f"expected {nb} eigenvalues, got {len(ev)}"
        E.append(ev)
        line += 2
    kfrac = np.array(kfrac)
    E = (np.array(E) - ef) * HARTREE
    # segment boundaries for plotting labels
    seg_bounds, idx = [0], 0
    for (nk, *_ ) in segs:
        idx += nk
        seg_bounds.append(idx - 1)
    labels = [segs[0][3]] + [s[4] for s in segs]
    return dict(efermi_eV=ef * HARTREE, kfrac=kfrac, E=E,
                labels=labels, seg_bounds=seg_bounds)


def chamfer(A, B):
    """Mean over points in A of distance to nearest point in B (1D energies,
    per k-point). A, B: lists of arrays per k.

    WEIGHTING CAVEAT. All per-k distances are concatenated into a SINGLE global
    mean, so every (band, k) pair carries equal weight -- not every k-point.
    Where the band density is uniform along the path (33-42 bands/k for most
    configs here) this is immaterial: k-point-weighted and band-weighted
    evaluations agree to <0.1 meV. It matters badly when the density varies:
    TTG_2_7 carries ~2 bands in the window over most of the path but 8 near
    Gamma, so the crowded Gamma k-points dominate and this metric rated a fit
    as improved (12.70 -> 8.26 meV) whose Dirac-cone branches visibly degraded
    over the rest of the path (k-point-weighted: 10.63 -> 15.24 meV).

    For sparse or non-uniform band densities, cross-check with the k-point
    weighting, i.e. average 0.5*(mean_a min_b + mean_b min_a) per k first:
        np.mean([0.5*(np.abs(a[:,None]-b[None,:]).min(1).mean()
                    + np.abs(b[:,None]-a[None,:]).min(1).mean())
                 for a, b in zip(A, B) if len(a) and len(b)])
    Related pitfall: being assignment-free, Chamfer also tolerates a splitting
    reproduced only halfway (see fit_ph2.stageA, which uses sorted-level
    matching instead for exactly this reason).
    """
    d = []
    for a, b in zip(A, B):
        if len(a) == 0 or len(b) == 0:
            continue
        d.append(np.abs(a[:, None] - b[None, :]).min(axis=1))
    return float(np.mean(np.concatenate(d))) if d else np.nan


def run(tag, pairing="derived", mirror=False, w=0.11, vfc=0.80, offset="auto",
        nshells=12, stride=1, out=None, plot=True, num_eigs=70, v_mid=0.0,
        beta_ph=0.0, lam_nl=0.0, dw=0.0):
    i, j = int(tag.split("_")[1]), int(tag.split("_")[2])
    band = parse_openmx_band(DATA / tag / "openmx.Band")
    mdl = RigidTTG(i, j, n_shells=nshells, mirror=mirror, pairing=pairing)
    print(mdl.summary())

    kf = band["kfrac"][::stride]
    E_tb = [band["E"][ik] for ik in range(0, len(band["E"]), stride)]
    kc = mdl.frac_to_cart(kf)
    VL = [0.0, v_mid, 0.0] if v_mid != 0.0 else None
    waa, wab = w + dw / 2.0, w - dw / 2.0

    # model bands, both valleys
    Emod = []
    for ik, k in enumerate(kc):
        e1 = mdl.eigs_at(k, waa, wab, vfc, num_eigs=num_eigs, V_layer=VL,
                         beta_ph=beta_ph, lam_nl=lam_nl)
        e2 = mdl.eigs_at(-k, waa, wab, vfc, num_eigs=num_eigs, V_layer=VL,
                         beta_ph=beta_ph, lam_nl=lam_nl)
        Emod.append((np.sort(e1), np.sort(e2)))
        if ik % 25 == 0:
            print(f"  k {ik}/{len(kc)}", flush=True)
    Eu = [np.sort(np.concatenate(e)) for e in Emod]

    # comparison window: TB stores only nb bands around E_F -> the reliable
    # window is the min over k of the TB extremes, shrunk slightly
    lo = max(np.max([e.min() for e in E_tb]), -0.35) * 0.9
    hi = min(np.min([e.max() for e in E_tb]), 0.35) * 0.9

    def in_win(arrs, off=0.0):
        return [a[(a >= lo) & (a <= hi)] - off for a in arrs]

    def sym_chamfer(off):
        A = in_win(E_tb)
        Bm = [e + off for e in Eu]
        B = [b[(b >= lo) & (b <= hi)] for b in Bm]
        return 0.5 * (chamfer(A, B) + chamfer(B, A))

    if offset == "auto":
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(sym_chamfer, bounds=(-0.05, 0.05), method="bounded",
                              options={"xatol": 1e-4})
        off = float(res.x)
    else:
        off = float(offset)
    cham = sym_chamfer(off) * 1e3  # meV

    # cone diagnostics at Gamma and K path points
    def point_metric(kfrac_pt):
        ik = int(np.argmin(np.linalg.norm(kf - np.asarray(kfrac_pt), axis=1)))
        tb_min = float(np.min(np.abs(E_tb[ik])))
        md_min = float(np.min(np.abs(Eu[ik] + off)))
        return tb_min * 1e3, md_min * 1e3, ik

    tbG, mdG, _ = point_metric([0.0, 0.0])
    tbK, mdK, _ = point_metric([2.0 / 3.0, 1.0 / 3.0])

    metrics = dict(tag=tag, pairing=pairing, mirror=bool(mirror), w=w, vfc=vfc,
                   v_mid_meV=v_mid * 1e3, beta_ph=beta_ph, lam_nl=lam_nl,
                   dw_meV=dw * 1e3,
                   offset_meV=off * 1e3, chamfer_meV=cham,
                   minE_G_tb=tbG, minE_G_model=mdG,
                   minE_K_tb=tbK, minE_K_model=mdK,
                   window=(float(lo), float(hi)), ndof=mdl.N)

    if out is None:
        out = f"{tag}_{pairing}{'_mir' if mirror else ''}"
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    np.savez(outdir / f"{out}.npz", kfrac=kf, E_tb=np.array(E_tb, dtype=object),
             E_model=np.array(Eu, dtype=object), offset=off)

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for ik in range(len(x)):
            ax.plot([x[ik]] * len(E_tb[ik]), E_tb[ik], ".", color="k", ms=2.5,
                    label="TB (ML rigid)" if ik == 0 else None)
            em = Eu[ik] + off
            ax.plot([x[ik]] * len(em), em, ".", color="crimson", ms=1.5, alpha=0.7,
                    label=f"continuum (w={w}, vfc={vfc})" if ik == 0 else None)
        sb = [b // stride for b in band["seg_bounds"]]
        for b in sb:
            if b < len(x):
                ax.axvline(x[min(b, len(x) - 1)], color="gray", lw=0.5)
        ax.set_xticks([x[min(b, len(x) - 1)] for b in sb])
        ax.set_xticklabels(band["labels"])
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylim(lo * 1.15, hi * 1.15)
        ax.set_ylabel("E - E_F (eV)")
        ax.set_title(f"{tag}  rigid TB vs exact-commensurate continuum "
                     f"({pairing}{', mirror' if mirror else ''})\n"
                     f"Chamfer {cham:.1f} meV   K-cone TB {tbK:.1f} / model {mdK:.1f} meV")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"{out}.png", dpi=160)
        print(f"saved {outdir / (out + '.png')}")

    print("METRICS " + json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--pairing", default="derived", choices=["derived", "legacy"])
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--w", type=float, default=0.11)
    ap.add_argument("--vfc", type=float, default=0.80)
    ap.add_argument("--offset", default="auto")
    ap.add_argument("--nshells", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    run(a.tag, a.pairing, a.mirror, a.w, a.vfc, a.offset, a.nshells, a.stride,
        a.out, not a.no_plot)
