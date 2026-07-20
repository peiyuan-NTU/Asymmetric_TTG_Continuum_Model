"""DOS over the supercell BZ for one commensurate TTG config.

Total DOS = 4 x single-valley BZ average (spin x2; valley K' at k equals
valley K at -k, and the Gamma-offset MP grid is inversion-symmetric).

Parameter sets:
  full : our validated rigid set  w=0.112 (AA=AB), vc=0.82, V2=-50 meV,
         beta=-1.9 eV*A^2, lam=3.0 A
  zhu  : Zhu et al. PRL 125,116404 values  w_AA=0.07, w_AB=0.11, vc from
         vF=0.8e6 m/s -> hbar*vF = 5.266 eV*A -> vc=0.800; no V2/beta/lam
  bm   : plain BM rigid  w=0.11 (AA=AB), vc=0.80; no extras

Usage: python dos.py TAG i j m n --params full|zhu|bm [--grid 18]
       [--nshells 12] [--kappa 1.0] [--numeigs 50]
Writes results/dos_<tag>_<params>.npz (+ metrics in json line "DOSRESULT ...").
"""
import sys, json, argparse, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttg_continuum.rigid_ttg import RigidTTG

PSETS = {
    "full": dict(waa=0.112, wab=0.112, vfc=0.82, v2=-0.050, beta=-1.9, lam=3.0),
    "zhu":  dict(waa=0.070, wab=0.110, vfc=0.800, v2=0.0, beta=0.0, lam=0.0),
    "bm":   dict(waa=0.110, wab=0.110, vfc=0.800, v2=0.0, beta=0.0, lam=0.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("i", type=int)
    ap.add_argument("j", type=int); ap.add_argument("m", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("--params", default="full", choices=list(PSETS))
    ap.add_argument("--grid", type=int, default=18)
    ap.add_argument("--nshells", type=int, default=12)
    ap.add_argument("--kappa", type=float, default=1.0)   # meV smearing
    ap.add_argument("--numeigs", type=int, default=50)
    ap.add_argument("--win", type=float, default=0.20)    # eV histogram window
    a = ap.parse_args()
    p = PSETS[a.params]

    mdl = RigidTTG(a.i, a.j, m=a.m, n=a.n, n_shells=a.nshells)
    VL = [0.0, p["v2"], 0.0] if p["v2"] else None
    print(f"{a.tag} [{a.params}]: th12={mdl.theta12_deg:.3f} th23="
          f"{mdl.theta23_deg:.3f} ndof={mdl.N} grid={a.grid}", flush=True)

    N = a.grid
    Es = []
    for ix in range(N):
        for iy in range(N):
            kf = [(ix + 0.5) / N, (iy + 0.5) / N]
            k = mdl.frac_to_cart(kf)
            e = mdl.eigs_at(k, p["waa"], p["wab"], p["vfc"],
                            num_eigs=a.numeigs, V_layer=VL,
                            beta_ph=p["beta"], lam_nl=p["lam"])
            Es.append(e)
        print(f"  row {ix+1}/{N}", flush=True)
    Es = np.concatenate(Es)

    # Gaussian-smeared DOS per unit area (states/eV/nm^2), spin+valley x4
    A_nm2 = (mdl.A_shared ** 2 * np.sqrt(3) / 2) / 100.0   # cell area in nm^2
    grid_E = np.arange(-a.win, a.win, 2e-4)                # 0.2 meV bins
    kap = a.kappa * 1e-3
    w = 4.0 / (N * N) / A_nm2 / (kap * np.sqrt(2 * np.pi))
    dos = np.zeros_like(grid_E)
    for e in Es[np.abs(Es) < a.win + 6 * kap]:
        dos += w * np.exp(-0.5 * ((grid_E - e) / kap) ** 2)

    # VHS metrics: highest peak below and above E=0 within +-80 meV
    msk = np.abs(grid_E) < 0.080
    gE, gD = grid_E[msk], dos[msk]
    neg, pos = gE < 0, gE >= 0
    Em = gE[neg][np.argmax(gD[neg])] if neg.any() else np.nan
    Ep = gE[pos][np.argmax(gD[pos])] if pos.any() else np.nan
    out = dict(tag=a.tag, params=a.params,
               th12=mdl.theta12_deg, th23=mdl.theta23_deg,
               eps1=float(mdl.scales[0] - 1), grid=N, kappa_meV=a.kappa,
               dos_max=float(np.max(gD)),
               vhs_minus_meV=float(Em * 1e3), vhs_plus_meV=float(Ep * 1e3),
               vhs_gap_meV=float((Ep - Em) * 1e3))
    np.savez(ROOT / "results" / f"dos_{a.tag}_{a.params}.npz",
             grid_E=grid_E, dos=dos, eigs=Es, **{k: v for k, v in out.items()
                                                 if not isinstance(v, str)})
    print("DOSRESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
