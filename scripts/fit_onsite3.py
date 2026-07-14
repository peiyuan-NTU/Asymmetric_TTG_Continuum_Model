"""Per-config free fit of the three layer on-sites (V1, V2, V3) + offset,
with all other parameters FROZEN at the family medians (w=0.112, vfc=0.82,
beta=-1.9, lam=3.0). Purpose: let the data reveal the layer-onsite pattern of
the residual band errors (deformation-potential diagnosis), config by config.

Usage: python fit_onsite3.py TAG i j m n n_shells [stride]
Writes results/<tag>_vfit.json
"""
import sys, json, pathlib
import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ttg_continuum.rigid_ttg import RigidTTG
from compare_rigid import parse_openmx_band, chamfer, DATA

W, VFC = 0.11, 0.80
BETA, LAM = 0.0, 0.0     # diagnosis mode: onsite pattern only (beta/lam are
                         # invalid at 13-deg small cells and orthogonal to this)


def main():
    tag = sys.argv[1]
    i, j, m, n, ns = map(int, sys.argv[2:7])
    stride = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    band = parse_openmx_band(DATA / tag / "openmx.Band")
    mdl = RigidTTG(i, j, m=m, n=n, n_shells=ns)
    tre = [2.0 * (s - 1.0) for s in mdl.scales]      # tr(eps_l)
    print(f"{tag}: tr(eps) = {[f'{t:+.4%}' for t in tre]}  ndof={mdl.N}", flush=True)

    kf = band["kfrac"][::stride]
    E_tb = [band["E"][ik] for ik in range(0, len(band["E"]), stride)]
    kc = mdl.frac_to_cart(kf)
    lo = max(np.max([e.min() for e in E_tb]), -0.35) * 0.9
    hi = min(np.min([e.max() for e in E_tb]), 0.35) * 0.9
    A = [e[(e >= lo) & (e <= hi)] for e in E_tb]

    from scipy.optimize import minimize_scalar

    def cham(p):
        """Chamfer with the offset optimized EXACTLY (inner 1D bounded min)."""
        v1, v2, v3 = p
        if max(abs(v1), abs(v2), abs(v3)) > 0.15:
            return 1.0
        Eu = []
        for k in kc:
            e1 = mdl.eigs_at(k, W, W, VFC, num_eigs=55,
                             V_layer=[v1, v2, v3], beta_ph=BETA, lam_nl=LAM)
            e2 = mdl.eigs_at(-k, W, W, VFC, num_eigs=55,
                             V_layer=[v1, v2, v3], beta_ph=BETA, lam_nl=LAM)
            Eu.append(np.sort(np.concatenate([e1, e2])))

        def obj(off):
            B = [(e + off)[((e + off) >= lo) & ((e + off) <= hi)] for e in Eu]
            return 0.5 * (chamfer(A, B) + chamfer(B, A))
        r1 = minimize_scalar(obj, bounds=(-0.08, 0.08), method="bounded",
                             options={"xatol": 1e-4})
        cham.last_off = float(r1.x)
        return float(r1.fun)

    x0 = np.array([0.0, -0.050, 0.0])
    c0 = cham(x0); off0 = cham.last_off
    r = minimize(cham, x0, method="Nelder-Mead",
                 options={"maxfev": 120, "xatol": 2e-4, "fatol": 1e-6})
    v1, v2, v3 = r.x
    cfin = cham(r.x); off = cham.last_off
    out = dict(tag=tag, tr_eps=tre,
               V1_meV=v1 * 1e3, V2_meV=v2 * 1e3, V3_meV=v3 * 1e3,
               off_meV=off * 1e3,
               chamfer_baseline_meV=c0 * 1e3, baseline_off_meV=off0 * 1e3,
               chamfer_after_meV=cfin * 1e3,
               nfev=r.nfev, frozen=dict(w=W, vfc=VFC, beta=BETA, lam=LAM),
               n_shells=ns, stride=stride)
    # gauge note: only V differences matter (offset absorbs the mean)
    vbar = (v1 + v2 + v3) / 3.0
    out["V_rel_outermean"] = [(v1 - (v1 + v3) / 2) * 1e3,
                              (v2 - (v1 + v3) / 2) * 1e3,
                              (v3 - (v1 + v3) / 2) * 1e3]
    (ROOT / "results" / f"{tag}_vfit.json").write_text(json.dumps(out, indent=1))
    print(f"RESULT {tag}: V=[{v1*1e3:+.1f},{v2*1e3:+.1f},{v3*1e3:+.1f}] meV "
          f"off={off*1e3:+.1f}  chamfer {c0*1e3:.2f} (off {off0*1e3:+.1f}) -> "
          f"{cfin*1e3:.2f} meV (nfev={r.nfev})", flush=True)


if __name__ == "__main__":
    main()
