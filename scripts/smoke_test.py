"""Smoke test: build RigidTTG for several configs, verify certificates & timing."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ttg_continuum.rigid_ttg import RigidTTG, A0, f_com, best_mn

# analytic vs rack POSCAR lattice constants (from data_rigid/poscar_header.txt)
POSCAR_A = {(2, 7): 32.204720585, (2, 12): 53.66186464, (6, 19): 83.469913525,
            (7, 22): 96.298322915, (8, 25): 109.12816327, (10, 31): 134.79050295,
            (12, 37): 160.454970745, (13, 40): 173.28770726, (14, 43): 186.12068646}

# Includes catalogue entries that the historical m,n <= 10 search got wrong.
EXPECTED_TILINGS = {
    (8, 25): (3, 1),
    (3, 6): (13, 7),
    (2, 45): (18, 1),
    (2, 47): (19, 1),
}
for ij, expected in EXPECTED_TILINGS.items():
    got = best_mn(*ij)[:2]
    assert got == expected, (ij, got, expected)

for (i, j) in [(8, 25), (2, 7), (2, 12), (10, 31), (14, 43)]:
    t0 = time.time()
    mdl = RigidTTG(i, j, n_shells=12)
    dt = time.time() - t0
    alpha_deg = np.degrees(mdl.alphas)
    assert np.isclose(abs(alpha_deg[1] - alpha_deg[0]), mdl.theta12_deg, atol=1e-12)
    assert np.isclose(abs(alpha_deg[2] - alpha_deg[1]), mdl.theta23_deg, atol=1e-12)
    assert np.isclose(abs(alpha_deg[2] - alpha_deg[0]), mdl.theta13_deg, atol=1e-12)
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

# The batch wrapper must preserve the full-H non-local tunnelling operator.
probe = RigidTTG(8, 25, n_shells=3)
k = probe.frac_to_cart([0.1, 0.2])
kwargs = dict(num_eigs=12, V_layer=[0.0, -0.050, 0.0],
              beta_ph=-1.8, lam_nl=5.8)
via_bands = probe.bands([k], 0.11, 0.11, 0.8, **kwargs)[0]
via_eigs = probe.eigs_at(k, 0.11, 0.11, 0.8, **kwargs)
without_lambda = probe.eigs_at(
    k, 0.11, 0.11, 0.8, num_eigs=12,
    V_layer=kwargs["V_layer"], beta_ph=kwargs["beta_ph"], lam_nl=0.0,
)
assert np.allclose(via_bands, via_eigs, atol=1e-10)
assert np.max(np.abs(via_bands - without_lambda)) > 1e-6
print("bands() forwards lam_nl: OK")
