"""Hygiene refit: Delta_w = w_AA - w_AB PINNED TO ZERO by symmetry.

For rigid, flat-layer structures the two-center interlayer form factor gives
w_AA = w_AB identically; only relaxation (corrugation + in-plane relaxation)
splits them. In the earlier `fit_ph2` protocol Delta_w was left as a slack
variable in stage A' and reached +14.6 meV (8_25), +10.3 (12_37), +7.3 (7_22)
— i.e. the optimizer was absorbing unmodelled physics through a channel that
must vanish. This script repeats the identical two-stage protocol with
Delta_w == 0 throughout, so that the shift of the remaining five parameters
quantifies how much was being absorbed.

Protocol (identical to fit_ph2 except for the pinned Delta_w):
  stage A' : Nelder-Mead over (V2, beta, lambda) on the sorted Gamma-level
             match, with (w, vfc) frozen at the plain-BM values.
  stage B  : Nelder-Mead over (w, vfc, offset, V2, beta, lambda) minimising the
             path-averaged symmetric Chamfer, started from stage A'.

Usage: python fit_ph3.py TAG [TAG ...]
Writes results/<tag>_ph3.json (+ a per-config comparison against _ph2.json).
"""
import sys, json, pathlib
import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ttg_continuum.rigid_ttg import RigidTTG
from compare_rigid import parse_openmx_band, chamfer, run, DATA
from fit_ph2 import gamma_sorted_target

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
W0, VFC0 = 0.11, 0.80


def stageA(mdl, band):
    """Sorted-Gamma-level match over (V2, beta, lambda); Delta_w == 0."""
    tb, win = gamma_sorted_target(band)

    def levels(p):
        v2, beta, lam = p
        e = mdl.eigs_at(np.zeros(2), W0, W0, VFC0,          # w_AA == w_AB
                        num_eigs=max(24, 2 * len(tb) + 8),
                        V_layer=[0.0, v2, 0.0], beta_ph=beta, lam_nl=lam)
        return np.sort(e[(e >= -1.6 * win) & (e <= 1.6 * win)])

    def obj(p):
        if not (-0.09 <= p[0] <= 0.0 and -2 <= p[1] <= 3 and 0 <= p[2] <= 6):
            return 1e6
        e = levels(p)
        if min(len(e), len(tb)) < 4:
            return 1e6
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

    return minimize(obj, np.array([-0.050, 0.5, 3.0]), method="Nelder-Mead",
                    options={"maxfev": 250, "xatol": 2e-4, "fatol": 1e-8})


def fit_one(tag, stride=8, n_shells_fit=9, num_eigs=50):
    i, j = int(tag.split("_")[1]), int(tag.split("_")[2])
    band = parse_openmx_band(DATA / tag / "openmx.Band")
    mdlG = RigidTTG(i, j, n_shells=12)
    rA = stageA(mdlG, band)
    v2A, betaA, lamA = rA.x
    rmsA = np.sqrt(rA.fun) * 1e3
    print(f"[{tag}] stage A' (dw=0): v2={v2A*1e3:.1f} meV beta={betaA:.2f} "
          f"lam={lamA:.2f} A  rms={rmsA:.2f} meV (nfev={rA.nfev})", flush=True)
    if rmsA > 6.0:
        v2A, betaA, lamA = -0.050, 0.5, 3.0
        print(f"[{tag}] stage A' unreliable (rms {rmsA:.1f}) -> neutral start",
              flush=True)

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
                and -0.12 <= v2 <= 0.02 and -2 <= beta <= 3 and -2 <= lam <= 8):
            return 1.0
        Eu = []
        for k in kc:
            e1 = mdl.eigs_at(k, w, w, vfc, num_eigs=num_eigs,
                             V_layer=[0.0, v2, 0.0], beta_ph=beta, lam_nl=lam)
            e2 = mdl.eigs_at(-k, w, w, vfc, num_eigs=num_eigs,
                             V_layer=[0.0, v2, 0.0], beta_ph=beta, lam_nl=lam)
            Eu.append(np.sort(np.concatenate([e1, e2])) + off)
        B = [e[(e >= lo) & (e <= hi)] for e in Eu]
        return 0.5 * (chamfer(A, B) + chamfer(B, A))

    xB0 = np.array([W0, VFC0, 0.0, v2A, betaA, lamA])
    rB = minimize(objective, xB0, method="Nelder-Mead",
                  options={"maxfev": 120, "xatol": 1e-4, "fatol": 1e-6})
    w, vfc, off, v2, beta, lam = rB.x
    print(f"[{tag}] stage B (dw=0): w={w:.4f} vfc={vfc:.4f} off={off*1e3:.1f} "
          f"v2={v2*1e3:.1f} beta={beta:.2f} lam={lam:.2f}  chamfer "
          f"{objective(xB0)*1e3:.2f} -> {rB.fun*1e3:.2f} meV (nfev={rB.nfev})",
          flush=True)
    return dict(w=float(w), vfc=float(vfc), offset=float(off), v_mid=float(v2),
                beta_ph=float(beta), lam_nl=float(lam), dw=0.0,
                stageA=dict(v_mid=float(v2A), beta_ph=float(betaA),
                            lam_nl=float(lamA), dw=0.0,
                            gamma_rms_meV=float(rmsA)),
                chamfer_fit_meV=float(rB.fun * 1e3))


def main(tags):
    for tag in tags:
        try:
            fit = fit_one(tag)
            m = run(tag, pairing="derived", mirror=False, w=fit["w"],
                    vfc=fit["vfc"], offset=str(fit["offset"]), nshells=12,
                    stride=1, out=f"{tag}_ph3", plot=True, v_mid=fit["v_mid"],
                    beta_ph=fit["beta_ph"], lam_nl=fit["lam_nl"], dw=0.0)
            m["fit"] = fit
            (RESULTS / f"{tag}_ph3.json").write_text(json.dumps(m, indent=1))
            # contamination report vs the Delta_w-slack fit
            old_p = RESULTS / f"{tag}_ph2.json"
            if old_p.exists():
                o = json.loads(old_p.read_text())["fit"]
                print(f"[{tag}] SHIFT vs ph2 (dw was {o.get('dw',0)*1e3:+.1f} meV): "
                      f"w {o['w']:.4f}->{fit['w']:.4f}  "
                      f"vfc {o['vfc']:.3f}->{fit['vfc']:.3f}  "
                      f"V2 {o['v_mid']*1e3:+.0f}->{fit['v_mid']*1e3:+.0f}  "
                      f"beta {o['beta_ph']:+.2f}->{fit['beta_ph']:+.2f}  "
                      f"lam {o['lam_nl']:.2f}->{fit['lam_nl']:.2f}  |  "
                      f"chamfer {json.loads(old_p.read_text())['chamfer_meV']:.2f}"
                      f"->{m['chamfer_meV']:.2f} meV", flush=True)
            print(f"[{tag}] DONE chamfer_full={m['chamfer_meV']:.2f} meV", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{tag}] FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
