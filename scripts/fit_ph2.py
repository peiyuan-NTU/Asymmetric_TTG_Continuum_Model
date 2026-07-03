"""Full effective-Hamiltonian fit: (w, vfc, offset) + middle-layer onsite V2
+ particle-hole beta*k^2 + NON-LOCAL interlayer tunneling lam_nl (+ small
w_aa - w_ab = dw).

Stage A': sorted-level matching at the Gamma point (assignment-free image of
          the level STRUCTURE, unlike Chamfer which tolerates half-captured
          splittings) over (v2, beta, lam, dw); offset closed-form;
          w = 0.11, vfc = 0.80 fixed.
Stage B:  global symmetric-Chamfer NM over (w, vfc, off, v2, beta, lam),
          dw frozen from stage A'.
Final:    full-resolution overlay + Gamma zoom + json (suffix _ph2).

Usage: python fit_ph2.py TAG [TAG ...]
"""
import sys, json, pathlib
import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttg_continuum.rigid_ttg import RigidTTG
from compare_rigid import parse_openmx_band, chamfer, run, DATA
from fit_ph import gamma_zoom_plot

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
W0, VFC0 = 0.11, 0.80


def gamma_sorted_target(band, win=0.09):
    """TB levels at Gamma, valley-deduped (pair-averaged), within +-win."""
    kf = band["kfrac"]
    iG = len(kf) - 1
    assert np.linalg.norm(kf[iG]) < 1e-8
    tb = np.sort(band["E"][iG])
    # TB contains both valleys; at Gamma they are degenerate -> average pairs
    if len(tb) % 2:
        tb = tb[:-1]
    tb = 0.5 * (tb[0::2] + tb[1::2])
    w = min(win, 0.88 * min(-tb.min(), tb.max()))
    return tb[(tb >= -w) & (tb <= w)], w


def stageA(mdl, band):
    tb, win = gamma_sorted_target(band)

    def levels(p):
        v2, beta, lam, dw = p
        e = mdl.eigs_at(np.zeros(2), W0 + dw / 2, W0 - dw / 2, VFC0,
                        num_eigs=max(24, 2 * len(tb) + 8),
                        V_layer=[0.0, v2, 0.0], beta_ph=beta, lam_nl=lam)
        return np.sort(e[(e >= -1.6 * win) & (e <= 1.6 * win)])

    def obj(p):
        if not (-0.09 <= p[0] <= 0.0 and -2 <= p[1] <= 3 and 0 <= p[2] <= 6
                and abs(p[3]) <= 0.015):
            return 1e6
        e = levels(p)
        n = min(len(e), len(tb))
        if n < 4:
            return 1e6
        # align the two sorted lists around their |E|-nearest-zero centers,
        # scanning a few registration shifts (dense spectra can misregister
        # by one state at the window edge)
        ie0 = int(np.argmin(np.abs(e))); it = int(np.argmin(np.abs(tb)))
        best = 1e6
        for r in (-2, -1, 0, 1, 2):
            ie = ie0 + r
            if not (0 <= ie < len(e)):
                continue
            lo = min(ie, it); hi = min(len(e) - ie, len(tb) - it)
            if lo + hi < 4:
                continue
            ee, tt = e[ie - lo:ie + hi], tb[it - lo:it + hi]
            off = float(np.mean(tt - ee))
            best = min(best, float(np.mean((ee + off - tt) ** 2)))
        return best

    x0 = np.array([-0.050, 0.5, 3.0, 0.0])
    r = minimize(obj, x0, method="Nelder-Mead",
                 options={"maxfev": 300, "xatol": 2e-4, "fatol": 1e-8})
    return r


def fit_one(tag, stride=8, n_shells_fit=9, num_eigs=50):
    i, j = int(tag.split("_")[1]), int(tag.split("_")[2])
    band = parse_openmx_band(DATA / tag / "openmx.Band")
    mdlG = RigidTTG(i, j, n_shells=12)          # converged levels for stage A'
    rA = stageA(mdlG, band)
    v2A, betaA, lamA, dwA = rA.x
    rmsA = np.sqrt(rA.fun) * 1e3
    print(f"[{tag}] stage A' (Gamma sorted-match): v2={v2A*1e3:.1f} meV "
          f"beta={betaA:.2f} lam={lamA:.2f} A dw={dwA*1e3:.1f} meV  "
          f"rms={rmsA:.2f} meV (nfev={rA.nfev})", flush=True)
    if rmsA > 6.0:      # misregistered / inapplicable -> neutral, sane start
        v2A, betaA, lamA, dwA = -0.050, 0.5, 3.0, 0.0
        print(f"[{tag}] stage A' unreliable (rms {rmsA:.1f}) -> "
              f"falling back to neutral start", flush=True)

    mdl = RigidTTG(i, j, n_shells=n_shells_fit)
    kf = band["kfrac"][::stride]
    E_tb = [band["E"][ik] for ik in range(0, len(band["E"]), stride)]
    kc = mdl.frac_to_cart(kf)
    lo = max(np.max([e.min() for e in E_tb]), -0.35) * 0.9
    hi = min(np.min([e.max() for e in E_tb]), 0.35) * 0.9
    A = [e[(e >= lo) & (e <= hi)] for e in E_tb]

    def objective(p):
        w, vfc, off, v2, beta, lam = p
        if not (0.05 <= w <= 0.2 and 0.5 <= vfc <= 1.1 and abs(off) <= 0.06
                and -0.12 <= v2 <= 0.02 and -2 <= beta <= 6 and -2 <= lam <= 8):
            return 1.0
        Eu = []
        for k in kc:
            e1 = mdl.eigs_at(k, w + dwA / 2, w - dwA / 2, vfc, num_eigs=num_eigs,
                             V_layer=[0.0, v2, 0.0], beta_ph=beta, lam_nl=lam)
            e2 = mdl.eigs_at(-k, w + dwA / 2, w - dwA / 2, vfc, num_eigs=num_eigs,
                             V_layer=[0.0, v2, 0.0], beta_ph=beta, lam_nl=lam)
            Eu.append(np.sort(np.concatenate([e1, e2])) + off)
        B = [e[(e >= lo) & (e <= hi)] for e in Eu]
        return 0.5 * (chamfer(A, B) + chamfer(B, A))

    xB0 = np.array([W0, VFC0, 0.0, v2A, betaA, lamA])
    rB = minimize(objective, xB0, method="Nelder-Mead",
                  options={"maxfev": 120, "xatol": 1e-4, "fatol": 1e-6})
    w, vfc, off, v2, beta, lam = rB.x
    print(f"[{tag}] stage B (global): w={w:.4f} vfc={vfc:.4f} off={off*1e3:.1f} "
          f"v2={v2*1e3:.1f} beta={beta:.2f} lam={lam:.2f}  chamfer "
          f"{objective(xB0)*1e3:.2f} -> {rB.fun*1e3:.2f} meV (nfev={rB.nfev})",
          flush=True)
    return dict(w=float(w), vfc=float(vfc), offset=float(off), v_mid=float(v2),
                beta_ph=float(beta), lam_nl=float(lam), dw=float(dwA),
                stageA=dict(v_mid=float(v2A), beta_ph=float(betaA),
                            lam_nl=float(lamA), dw=float(dwA),
                            gamma_rms_meV=float(np.sqrt(rA.fun) * 1e3)),
                chamfer_fit_meV=float(rB.fun * 1e3))


def main(tags):
    for tag in tags:
        try:
            fit = fit_one(tag)
            m = run(tag, pairing="derived", mirror=False, w=fit["w"],
                    vfc=fit["vfc"], offset=str(fit["offset"]), nshells=12,
                    stride=1, out=f"{tag}_ph2", plot=True, v_mid=fit["v_mid"],
                    beta_ph=fit["beta_ph"], lam_nl=fit["lam_nl"], dw=fit["dw"])
            m["fit"] = fit
            (RESULTS / f"{tag}_ph2.json").write_text(json.dumps(m, indent=1))
            gamma_zoom_plot(tag, RESULTS / f"{tag}_ph2.npz",
                            RESULTS / f"{tag}_ph2_gzoom.png")
            print(f"[{tag}] DONE chamfer_full={m['chamfer_meV']:.2f} meV", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{tag}] FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
