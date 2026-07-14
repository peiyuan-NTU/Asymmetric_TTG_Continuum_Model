"""Three-way comparison: continuum model vs ML vs true DFT.

The only two configs with a DFT band are TTG_2_7 (full diagonalization, 13286
bands) and TTG_2_12 (100 bands near E_F, DFT Hamiltonian). Both are the large-
angle stress cases (theta23 = 13.17 deg). This is the ONLY place the continuum
model can be checked against ground truth rather than against the ML it was
fit to; it also directly exposes the ML's own error (ML vs DFT).

DFT/ML references are pre-filtered npz (data_rigid/dft/): kfrac (nk,2),
E (object array of per-k eigenvalue arrays, eV rel. E_F).

Usage: python compare_dft.py
Writes results/dft_three_way.png and prints the Chamfer matrix.
"""
import sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ttg_continuum.rigid_ttg import RigidTTG
from compare_rigid import chamfer

DFT = ROOT / "data_rigid" / "dft"
# plain-BM params fit to the ML bands (from batch_rigid results)
CFG = {
    "TTG_2_7":  dict(i=2, j=7,  m=3, n=1, w=0.108, vfc=0.852,
                     dft="dft_2_7.npz", ml=None),        # ml = ttg_organized 50-band
    "TTG_2_12": dict(i=2, j=12, m=5, n=1, w=0.117, vfc=0.848,
                     dft="dft_2_12dft.npz", ml="dft_2_12ml.npz"),
}


def load_npz(name):
    z = np.load(DFT / name, allow_pickle=True)
    return z["kfrac"], list(z["E"])


def win_clip(E, lo, hi, off=0.0):
    return [e[((e + off) >= lo) & ((e + off) <= hi)] + off for e in E]


def sym_chamfer(A, B):
    return 0.5 * (chamfer(A, B) + chamfer(B, A)) * 1e3


def continuum_bands(cfg, kf, num_eigs=40):
    mdl = RigidTTG(cfg["i"], cfg["j"], m=cfg["m"], n=cfg["n"], n_shells=12)
    kc = mdl.frac_to_cart(kf)
    E = []
    for ik, k in enumerate(kc):
        e1 = mdl.eigs_at(k, cfg["w"], cfg["w"], cfg["vfc"], num_eigs=num_eigs)
        e2 = mdl.eigs_at(-k, cfg["w"], cfg["w"], cfg["vfc"], num_eigs=num_eigs)
        E.append(np.sort(np.concatenate([e1, e2])))
        if ik % 30 == 0:
            print(f"  [{cfg['i']}_{cfg['j']}] k {ik}/{len(kc)}", flush=True)
    return E


def fit_offset(A_ref, B_model, lo, hi):
    from scipy.optimize import minimize_scalar
    A = win_clip(A_ref, lo, hi)

    def obj(off):
        B = win_clip(B_model, lo, hi, off)
        return chamfer(A, B) + chamfer(B, A)
    r = minimize_scalar(obj, bounds=(-0.06, 0.06), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    report = {}
    WIN = 0.30
    for ax, (tag, cfg) in zip(axes, CFG.items()):
        kf, E_dft = load_npz(cfg["dft"])
        if cfg["ml"]:
            _, E_ml = load_npz(cfg["ml"])
        else:                                    # 2_7: use the 50-band ML
            from compare_rigid import parse_openmx_band, DATA
            b = parse_openmx_band(DATA / tag / "openmx.Band")
            E_ml = [b["E"][ik] for ik in range(len(b["E"]))]
        E_cont = continuum_bands(cfg, kf)

        lo, hi = -WIN, WIN
        off_c = fit_offset(E_dft, E_cont, lo, hi)
        off_m = fit_offset(E_dft, E_ml, lo, hi)
        E_cont_a = [e + off_c for e in E_cont]
        E_ml_a = [e + off_m for e in E_ml]

        A = win_clip(E_dft, lo, hi)
        report[tag] = dict(
            cont_vs_dft=sym_chamfer(A, win_clip(E_cont_a, lo, hi)),
            ml_vs_dft=sym_chamfer(A, win_clip(E_ml_a, lo, hi)),
            cont_vs_ml=sym_chamfer(win_clip(E_ml_a, lo, hi),
                                   win_clip(E_cont_a, lo, hi)),
            off_cont_meV=off_c * 1e3, off_ml_meV=off_m * 1e3)

        np.savez(ROOT / "results" / f"dft3way_{tag}.npz",
                 kfrac=kf, E_dft=np.array(E_dft, dtype=object),
                 E_ml=np.array(E_ml_a, dtype=object),
                 E_cont=np.array(E_cont_a, dtype=object),
                 off_cont=off_c, off_ml=off_m)
        kc = np.arange(len(kf))
        for ik in range(len(kf)):
            ax.plot([ik] * len(E_dft[ik]), E_dft[ik], "_", c="k", ms=6, mew=1.6,
                    label="DFT (truth)" if ik == 0 else None)
            em = E_ml_a[ik]; em = em[(em >= lo) & (em <= hi)]
            ax.plot([ik] * len(em), em, ".", c="tab:blue", ms=4,
                    label="ML" if ik == 0 else None)
            ec = E_cont_a[ik]; ec = ec[(ec >= lo) & (ec <= hi)]
            ax.plot([ik] * len(ec), ec, ".", c="crimson", ms=3, alpha=0.8,
                    label="continuum" if ik == 0 else None)
        ax.set_ylim(-WIN, WIN); ax.axhline(0, c="gray", lw=0.5, ls="--")
        nseg = (len(kf) - 1) // 3 if (len(kf) - 1) % 3 == 0 else len(kf) // 3
        ax.set_xticks([0, 50, 100, 149]); ax.set_xticklabels(["Γ", "M", "K", "Γ"])
        ax.set_ylabel("E - E_F (eV)")
        r = report[tag]
        ax.set_title(f"{tag} (θ₂₃=13.2°)\ncont-vs-DFT {r['cont_vs_dft']:.1f} | "
                     f"ML-vs-DFT {r['ml_vs_dft']:.1f} | cont-vs-ML "
                     f"{r['cont_vs_ml']:.1f} meV")
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "dft_three_way.png", dpi=160)
    print("\n=== Chamfer matrix (meV, window +-0.30 eV) ===")
    for tag, r in report.items():
        print(f"{tag}: continuum-vs-DFT {r['cont_vs_dft']:.2f} | "
              f"ML-vs-DFT {r['ml_vs_dft']:.2f} | continuum-vs-ML "
              f"{r['cont_vs_ml']:.2f}  (offsets c {r['off_cont_meV']:+.1f}, "
              f"ml {r['off_ml_meV']:+.1f} meV)")
    import json
    (ROOT / "results" / "dft_three_way.json").write_text(json.dumps(report, indent=1))
    print("saved results/dft_three_way.png")


if __name__ == "__main__":
    main()
