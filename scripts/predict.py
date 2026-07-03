"""ZERO-SHOT prediction test: run the model on configs it has never been
fitted to, with FROZEN family-universal parameters (pre-registered 2026-07-04
as medians of the clean ph2 fits over TTG_{6_19..14_43, 2_12}). Only a global
energy offset (chemical-potential alignment, physics-free) is optimized.

  tier A (plain BM):     w = 0.11 eV, vfc = 0.80
  tier B (full eff. H):  w = 0.112, vfc = 0.82, V2 = -50 meV,
                         beta = -1.9 eV*A^2, lam = 3.0 A

Usage: python predict.py TAG i j m n [--nshells N] [--stride S]
       [--data SUBDIR] [--tiers AB|A|B]
Writes results/<tag>_pred.{json,png}
"""
import sys, json, argparse, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ttg_continuum.rigid_ttg import RigidTTG
from compare_rigid import parse_openmx_band, chamfer

RESULTS = ROOT / "results"
TIER_A = dict(w=0.11, vfc=0.80, v2=0.0, beta=0.0, lam=0.0)
TIER_B = dict(w=0.112, vfc=0.82, v2=-0.050, beta=-1.9, lam=3.0)


def predict(mdl, band, prm, stride, num_eigs=60, label=""):
    kf = band["kfrac"][::stride]
    E_tb = [band["E"][ik] for ik in range(0, len(band["E"]), stride)]
    kc = mdl.frac_to_cart(kf)
    lo = max(np.max([e.min() for e in E_tb]), -0.35) * 0.9
    hi = min(np.min([e.max() for e in E_tb]), 0.35) * 0.9
    A = [e[(e >= lo) & (e <= hi)] for e in E_tb]
    VL = [0.0, prm["v2"], 0.0] if prm["v2"] else None
    Eu = []
    for ik, k in enumerate(kc):
        e1 = mdl.eigs_at(k, prm["w"], prm["w"], prm["vfc"], num_eigs=num_eigs,
                         V_layer=VL, beta_ph=prm["beta"], lam_nl=prm["lam"])
        e2 = mdl.eigs_at(-k, prm["w"], prm["w"], prm["vfc"], num_eigs=num_eigs,
                         V_layer=VL, beta_ph=prm["beta"], lam_nl=prm["lam"])
        Eu.append(np.sort(np.concatenate([e1, e2])))
        if ik % 10 == 0:
            print(f"  [{label}] k {ik}/{len(kc)}", flush=True)

    from scipy.optimize import minimize_scalar

    def obj(off):
        B = [(e + off)[((e + off) >= lo) & ((e + off) <= hi)] for e in Eu]
        return 0.5 * (chamfer(A, B) + chamfer(B, A))
    r = minimize_scalar(obj, bounds=(-0.05, 0.05), method="bounded",
                        options={"xatol": 1e-4})
    off = float(r.x)

    def point_metric(pt):
        ik = int(np.argmin(np.linalg.norm(kf - np.asarray(pt), axis=1)))
        return (float(np.min(np.abs(E_tb[ik]))) * 1e3,
                float(np.min(np.abs(Eu[ik] + off))) * 1e3)
    gtb, gmd = point_metric([0, 0])
    ktb, kmd = point_metric([2 / 3, 1 / 3])
    return dict(chamfer_meV=float(r.fun * 1e3), offset_meV=off * 1e3,
                minE_G_tb=gtb, minE_G_model=gmd,
                minE_K_tb=ktb, minE_K_model=kmd,
                window=(float(lo), float(hi)), stride=stride,
                params=prm), kf, E_tb, Eu, off, (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("i", type=int)
    ap.add_argument("j", type=int); ap.add_argument("m", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("--nshells", type=int, default=0)   # 0 = auto
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--data", default=None)             # data_rigid subdir
    ap.add_argument("--tiers", default="AB")
    a = ap.parse_args()

    band = parse_openmx_band(ROOT / "data_rigid" / (a.data or a.tag) / "openmx.Band")

    # auto basis size: >= 3.5 shells of the largest moire q (each q23 shell
    # costs ~hv*q23 in energy, so beyond ~3 shells the dressing of the
    # +-0.1 eV window is < (w/hv q23)^2 ~ 1e-2 per shell — checked vs +6)
    probe = RigidTTG(a.i, a.j, m=a.m, n=a.n, n_shells=4)
    qmax = max(np.linalg.norm((np.asarray(q, float) / 3.0) @ probe.B.T)
               for qs in probe.q_int.values() for q in qs[:1])
    b1 = np.linalg.norm(probe.b1s)
    ns = a.nshells or max(12, int(np.ceil(3.5 * qmax / b1)) + 2)
    print(f"{a.tag}: |q_max|/|b1| = {qmax/b1:.2f} -> n_shells = {ns}", flush=True)
    mdl = RigidTTG(a.i, a.j, m=a.m, n=a.n, n_shells=ns)
    print(mdl.summary(), flush=True)
    # cross-check analytic lattice vs the .Band header (columns = b vectors)
    # (header can be ~0.3% off the POSCAR for some configs — warning only)
    out = {"tag": a.tag, "n_shells": ns, "ndof": mdl.N}

    def overlay(tier, kf, E_tb, Eu, off, win, cham):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        kc = mdl.frac_to_cart(kf)
        x = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for ik in range(len(x)):
            ax.plot([x[ik]] * len(E_tb[ik]), E_tb[ik], ".", c="k", ms=3,
                    label="TB (fresh)" if ik == 0 else None)
            em = Eu[ik] + off
            ax.plot([x[ik]] * len(em), em, ".", c="crimson", ms=1.8, alpha=0.75,
                    label=f"zero-shot prediction (tier {tier})" if ik == 0 else None)
        ax.set_ylim(win[0] * 1.1, win[1] * 1.1)
        ax.axhline(0, c="gray", lw=0.5, ls="--")
        ax.set_ylabel("E - E_F (eV)")
        nseg = (len(kf) - 1) // 3
        ax.set_xticks([x[0], x[nseg], x[2 * nseg], x[-1]])
        ax.set_xticklabels(["Γ", "M", "K", "Γ"])
        ax.set_title(f"{a.tag}  ZERO-SHOT (frozen universal params)\n"
                     f"tier {tier} Chamfer {cham:.1f} meV")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{a.tag}_pred_{tier}.png", dpi=160)
        print(f"saved {RESULTS / (a.tag + '_pred_' + tier + '.png')}", flush=True)

    tiers = {}
    if "B" in a.tiers:
        tiers["B"], kf, E_tb, Eu, off, win = predict(mdl, band, TIER_B,
                                                     a.stride, label="tierB")
        print(f"[{a.tag}] tier B: {tiers['B']['chamfer_meV']:.2f} meV "
              f"(off {tiers['B']['offset_meV']:+.1f})", flush=True)
        overlay("B", kf, E_tb, Eu, off, win, tiers["B"]["chamfer_meV"])
    if "A" in a.tiers:
        sA = a.stride * (2 if mdl.N > 6000 and "B" in a.tiers else 1)
        tiers["A"], kf, E_tb, Eu, off, win = predict(mdl, band, TIER_A, sA,
                                                     label="tierA")
        print(f"[{a.tag}] tier A: {tiers['A']['chamfer_meV']:.2f} meV", flush=True)
        overlay("A", kf, E_tb, Eu, off, win, tiers["A"]["chamfer_meV"])
    out["tiers"] = tiers
    (RESULTS / f"{a.tag}_pred.json").write_text(json.dumps(out, indent=1))
    print("PRED " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
