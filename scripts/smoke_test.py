"""Smoke test: build RigidTTG for several configs, verify certificates & timing."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ttg_continuum.rigid_ttg import RigidTTG, A0, f_com

# analytic vs rack POSCAR lattice constants (from data_rigid/poscar_header.txt)
POSCAR_A = {(2, 7): 32.204720585, (2, 12): 53.66186464, (6, 19): 83.469913525,
            (7, 22): 96.298322915, (8, 25): 109.12816327, (10, 31): 134.79050295,
            (12, 37): 160.454970745, (13, 40): 173.28770726, (14, 43): 186.12068646}

for (i, j) in [(8, 25), (2, 7), (2, 12), (10, 31), (14, 43)]:
    t0 = time.time()
    mdl = RigidTTG(i, j, n_shells=12)
    dt = time.time() - t0
    print(mdl.summary())
    da = mdl.A_shared - POSCAR_A[(i, j)]
    print(f"  A_shared vs rack POSCAR: {da:+.3e} A {'OK' if abs(da) < 1e-6 else 'MISMATCH!'}")
    print(f"  build {dt:.1f}s  nnz(M_aa)={mdl.M_aa.nnz}")
    # K path point of the .Band files is frac (2/3, 1/3)
    Kpath = mdl.frac_to_cart([2 / 3, 1 / 3])
    # pure Dirac check: per-layer min |k_rel| must vanish exactly at the
    # layer's folded cone position and nowhere else
    for name, kf in [("Gamma", [0, 0]), ("K", [2 / 3, 1 / 3])]:
        k = mdl.frac_to_cart(kf)
        mins = [np.min(np.linalg.norm(mdl.pos[mdl.lay == l] + k, axis=1))
                for l in (1, 2, 3)]
        print(f"  min|k-K_l| at {name}: " +
              "  ".join(f"L{l}={v:.3e}" for l, v in zip((1, 2, 3), mins)))
    t0 = time.time()
    e = mdl.eigs_at(Kpath, 0.11, 0.11, 0.8, num_eigs=60)
    print(f"  full solve at K: {time.time()-t0:.2f}s  min|E|={np.min(np.abs(e))*1e3:.2f} meV")
