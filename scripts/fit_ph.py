"""Two-stage fit including middle-layer onsite V2 and the particle-hole
beta*|k-K_l|^2 term (continuum image of graphene's NNN hopping).

Stage A: fit (V2, beta, off) to the Gamma-point level structure only
         (fixed w=0.11, vfc=0.80; one eigsh per evaluation -> fast).
Stage B: global symmetric-Chamfer refinement of (w, vfc, off, V2, beta)
         on a subsampled path.
Final:   full-resolution overlay + Gamma-zoom figure + metrics json.

Usage: python fit_ph.py TAG [TAG ...]
Writes results/<tag>_ph.json / _ph.png / _ph_gzoom.png
"""
import sys, json, pathlib
import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttg_continuum.rigid_ttg import RigidTTG
from compare_rigid import parse_openmx_band, chamfer, run, DATA

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
W0, VFC0 = 0.11, 0.80


def gamma_chamfer_factory(mdl, band, num_eigs=40):
    """Symmetric Chamfer between model and TB level sets at the Gamma point."""
    kf = band["kfrac"]
    iG = len(kf) - 1                       # last path point is Gamma exactly
    assert np.linalg.norm(kf[iG]) < 1e-8
    lo = max(band["E"][iG].min(), -0.35) * 0.9
    hi = min(band["E"][iG].max(), 0.35) * 0.9
    tb = band["E"][iG]
    tb = tb[(tb >= lo) & (tb <= hi)]

    def gcham(p):
        v2, beta, off = p
        if not (abs(v2) <= 0.12 and abs(beta) <= 6.0 and abs(off) <= 0.06):
            return 1.0
        e = mdl.eigs_at(np.zeros(2), W0, W0, VFC0, num_eigs=num_eigs,
                        V_layer=[0.0, v2, 0.0], beta_ph=beta) + off
        e = e[(e >= lo) & (e <= hi)]
        if len(e) == 0:
            return 1.0
        return 0.5 * (chamfer([tb], [e]) + chamfer([e], [tb]))
    return gcham


def fit_one(tag, stride=8, n_shells=9, num_eigs=50):
    i, j = int(tag.split("_")[1]), int(tag.split("_")[2])
    band = parse_openmx_band(DATA / tag / "openmx.Band")
    mdl = RigidTTG(i, j, n_shells=n_shells)

    # ---- stage A: Gamma levels -> (v2, beta, off)
    gcham = gamma_chamfer_factory(mdl, band)
    x0 = np.array([-0.045, 1.5, 0.005])
    rA = minimize(gcham, x0, method="Nelder-Mead",
                  options={"maxfev": 150, "xatol": 1e-4, "fatol": 1e-7})
    v2A, betaA, offA = rA.x
    print(f"[{tag}] stage A (Gamma): v2={v2A*1e3:.1f} meV beta={betaA:.2f} eV*A^2 "
          f"off={offA*1e3:.1f} meV  Gamma-chamfer {gcham(x0)*1e3:.2f} -> "
          f"{rA.fun*1e3:.2f} meV (nfev={rA.nfev})", flush=True)

    # ---- stage B: global Chamfer -> (w, vfc, off, v2, beta)
    kf = band["kfrac"][::stride]
    E_tb = [band["E"][ik] for ik in range(0, len(band["E"]), stride)]
    kc = mdl.frac_to_cart(kf)
    lo = max(np.max([e.min() for e in E_tb]), -0.35) * 0.9
    hi = min(np.min([e.max() for e in E_tb]), 0.35) * 0.9
    A = [e[(e >= lo) & (e <= hi)] for e in E_tb]

    def objective(p):
        w, vfc, off, v2, beta = p
        if not (0.05 <= w <= 0.2 and 0.5 <= vfc <= 1.1 and abs(off) <= 0.06
                and abs(v2) <= 0.12 and abs(beta) <= 6.0):
            return 1.0
        Eu = []
        for k in kc:
            e1 = mdl.eigs_at(k, w, w, vfc, num_eigs=num_eigs,
                             V_layer=[0.0, v2, 0.0], beta_ph=beta)
            e2 = mdl.eigs_at(-k, w, w, vfc, num_eigs=num_eigs,
                             V_layer=[0.0, v2, 0.0], beta_ph=beta)
            Eu.append(np.sort(np.concatenate([e1, e2])) + off)
        B = [e[(e >= lo) & (e <= hi)] for e in Eu]
        return 0.5 * (chamfer(A, B) + chamfer(B, A))

    xB0 = np.array([W0, VFC0, offA, v2A, betaA])
    rB = minimize(objective, xB0, method="Nelder-Mead",
                  options={"maxfev": 90, "xatol": 1e-4, "fatol": 1e-6})
    w, vfc, off, v2, beta = rB.x
    print(f"[{tag}] stage B (global): w={w:.4f} vfc={vfc:.4f} off={off*1e3:.1f} "
          f"v2={v2*1e3:.1f} meV beta={beta:.2f}  chamfer {objective(xB0)*1e3:.2f}"
          f" -> {rB.fun*1e3:.2f} meV (nfev={rB.nfev})", flush=True)
    return dict(w=float(w), vfc=float(vfc), offset=float(off), v_mid=float(v2),
                beta_ph=float(beta),
                stageA=dict(v_mid=float(v2A), beta_ph=float(betaA),
                            gamma_chamfer_meV=float(rA.fun * 1e3)),
                chamfer_fit_meV=float(rB.fun * 1e3))


def gamma_zoom_plot(tag, npz_path, out_png, ylim=0.11):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = np.load(npz_path, allow_pickle=True)
    kf, E_tb, E_md, off = Z["kfrac"], Z["E_tb"], Z["E_model"], float(Z["offset"])
    n = len(kf)
    sel = range(2 * n // 3, n)             # K -> Gamma segment
    fig, ax = plt.subplots(figsize=(5, 5))
    for ii, ik in enumerate(sel):
        ax.plot([ii] * len(E_tb[ik]), E_tb[ik], "o", color="k", ms=4,
                label="TB" if ii == 0 else None)
        em = E_md[ik] + off
        ax.plot([ii] * len(em), em, "o", color="crimson", ms=2.5, alpha=0.8,
                label="continuum + V2 + beta" if ii == 0 else None)
    ax.set_ylim(-ylim, ylim)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xticks([0, len(list(sel)) - 1]); ax.set_xticklabels(["K", "Γ"])
    ax.set_ylabel("E - E_F (eV)")
    ax.set_title(f"{tag}  K→Γ zoom")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=160)
    print(f"saved {out_png}", flush=True)


def main(tags):
    for tag in tags:
        try:
            fit = fit_one(tag)
            m = run(tag, pairing="derived", mirror=False, w=fit["w"],
                    vfc=fit["vfc"], offset=str(fit["offset"]), nshells=12,
                    stride=1, out=f"{tag}_ph", plot=True, v_mid=fit["v_mid"],
                    beta_ph=fit["beta_ph"])
            m["fit"] = fit
            (RESULTS / f"{tag}_ph.json").write_text(json.dumps(m, indent=1))
            gamma_zoom_plot(tag, RESULTS / f"{tag}_ph.npz",
                            RESULTS / f"{tag}_ph_gzoom.png")
            print(f"[{tag}] DONE chamfer_full={m['chamfer_meV']:.2f} meV", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{tag}] FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
