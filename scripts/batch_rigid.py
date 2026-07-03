"""Batch fit + final compare for the rigid TTG dataset.

For each tag: Nelder-Mead fit of (w, vfc, offset) to the TB bands (symmetric
Chamfer in the reliable window, subsampled k-path, small converged basis),
then a full-resolution compare/overlay at the fitted parameters.

Usage: python batch_rigid.py TAG [TAG ...]
Writes results/<tag>_fit.json and results/<tag>_fit.png
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


def fit_one(tag, stride=6, n_shells=9, num_eigs=50, maxfev=70, onsite=False):
    i, j = int(tag.split("_")[1]), int(tag.split("_")[2])
    band = parse_openmx_band(DATA / tag / "openmx.Band")
    mdl = RigidTTG(i, j, n_shells=n_shells)
    kf = band["kfrac"][::stride]
    E_tb = [band["E"][ik] for ik in range(0, len(band["E"]), stride)]
    kc = mdl.frac_to_cart(kf)
    lo = max(np.max([e.min() for e in E_tb]), -0.35) * 0.9
    hi = min(np.min([e.max() for e in E_tb]), 0.35) * 0.9
    A = [e[(e >= lo) & (e <= hi)] for e in E_tb]

    cache = {}

    def bands_at(w, vfc, v2):
        key = (round(w, 6), round(vfc, 6), round(v2, 6))
        if key not in cache:
            VL = [0.0, v2, 0.0] if v2 != 0.0 else None
            Eu = []
            for k in kc:
                e1 = mdl.eigs_at(k, w, w, vfc, num_eigs=num_eigs, V_layer=VL)
                e2 = mdl.eigs_at(-k, w, w, vfc, num_eigs=num_eigs, V_layer=VL)
                Eu.append(np.sort(np.concatenate([e1, e2])))
            cache[key] = Eu
        return cache[key]

    def objective(p):
        w, vfc, off = p[0], p[1], p[2]
        v2 = p[3] if len(p) > 3 else 0.0
        if not (0.02 <= w <= 0.2 and 0.5 <= vfc <= 1.1 and abs(off) <= 0.06
                and abs(v2) <= 0.12):
            return 1.0
        Eu = bands_at(w, vfc, v2)
        B = [e + off for e in Eu]
        B = [b[(b >= lo) & (b <= hi)] for b in B]
        return 0.5 * (chamfer(A, B) + chamfer(B, A))

    x0 = np.array([0.11, 0.80, 0.005] + ([-0.055] if onsite else []))
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-6})
    w, vfc, off = res.x[0], res.x[1], res.x[2]
    v2 = float(res.x[3]) if onsite else 0.0
    print(f"[{tag}] fit: w={w:.4f} vfc={vfc:.4f} off={off*1e3:.2f} meV "
          f"v_mid={v2*1e3:.1f} meV chamfer={res.fun*1e3:.2f} meV  "
          f"(start chamfer={objective(x0)*1e3:.2f} meV, nfev={res.nfev})", flush=True)
    return dict(w=float(w), vfc=float(vfc), offset=float(off), v_mid=v2,
                chamfer_fit_meV=float(res.fun * 1e3),
                chamfer_start_meV=float(objective(x0) * 1e3))


def main(tags, onsite=False):
    sfx = "_os" if onsite else ""
    for tag in tags:
        try:
            fit = fit_one(tag, onsite=onsite,
                          maxfev=110 if onsite else 70)
            m = run(tag, pairing="derived", mirror=False, w=fit["w"],
                    vfc=fit["vfc"], offset=str(fit["offset"]), nshells=12,
                    stride=1, out=f"{tag}_fit{sfx}", plot=True, v_mid=fit["v_mid"])
            m["fit"] = fit
            (RESULTS / f"{tag}_fit{sfx}.json").write_text(json.dumps(m, indent=1))
            print(f"[{tag}] DONE chamfer_full={m['chamfer_meV']:.2f} meV", flush=True)
        except Exception as e:
            print(f"[{tag}] FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--onsite"]
    main(args, onsite="--onsite" in sys.argv[1:])
