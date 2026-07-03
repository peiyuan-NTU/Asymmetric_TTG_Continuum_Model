"""Assemble the results table from results/*_fit.json and *_fit_os.json."""
import json, pathlib
import numpy as np
import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttg_continuum.rigid_ttg import theta_com

RES = ROOT / "results"
TAGS = ["TTG_2_7", "TTG_2_12", "TTG_6_19", "TTG_7_22", "TTG_8_25",
        "TTG_10_31", "TTG_12_37", "TTG_13_40", "TTG_14_43"]

rows = []
for tag in TAGS:
    i, j = int(tag.split("_")[1]), int(tag.split("_")[2])
    t12 = np.degrees(0.5 * (theta_com(i) + theta_com(j)))
    t23 = np.degrees(theta_com(i))
    base = RES / f"{tag}_fit.json"
    ph = RES / f"{tag}_ph.json"
    ph2 = RES / f"{tag}_ph2.json"
    b = json.loads(base.read_text()) if base.exists() else None
    p = json.loads(ph.read_text()) if ph.exists() else None
    q = json.loads(ph2.read_text()) if ph2.exists() else None
    row = dict(tag=tag, t12=t12, t23=t23)
    if b:
        row.update(cham_recipe=b["fit"].get("chamfer_start_meV",
                                            b["fit"].get("chamfer_recipe_meV")),
                   w=b["fit"]["w"], vfc=b["fit"]["vfc"],
                   cham_fit=b["chamfer_meV"],
                   K_tb=b["minE_K_tb"], K_md=b["minE_K_model"])
    if p:
        row.update(cham_ph=p["chamfer_meV"])
    if q:
        row.update(w2=q["fit"]["w"], vfc2=q["fit"]["vfc"],
                   vmid2=q["fit"]["v_mid"] * 1e3, beta2=q["fit"]["beta_ph"],
                   lam2=q["fit"]["lam_nl"], cham_ph2=q["chamfer_meV"],
                   grms=q["fit"]["stageA"].get("gamma_rms_meV"))
    rows.append(row)

hdr = ("| config | θ12 | θ23 | recipe | fit (w,vfc) | +V₂β (abl.) "
       "| +V₂βλ: (w, vfc, V₂, β, λ) | Chamfer ph2 | Γ-rms | K TB/md |")
print(hdr)
print("|" + "---|" * 10)
for r in rows:
    def g(k, fmt="{:.3f}", alt="—"):
        v = r.get(k)
        return fmt.format(v) if v is not None else alt
    fit = f"{g('cham_fit', '{:.1f}')} ({g('w')}, {g('vfc')})" if "w" in r else "—"
    ph2f = (f"({g('w2')}, {g('vfc2')}, {g('vmid2', '{:+.0f}')}meV, "
            f"{g('beta2', '{:.1f}')}, {g('lam2', '{:.1f}')}Å)" if "w2" in r else "—")
    kk = f"{g('K_tb', '{:.1f}')} / {g('K_md', '{:.1f}')}" if "K_tb" in r else "—"
    print(f"| {r['tag']} | {r['t12']:.2f}° | {r['t23']:.2f}° "
          f"| {g('cham_recipe', '{:.1f}')} | {fit} | {g('cham_ph', '{:.1f}')} "
          f"| {ph2f} | {g('cham_ph2', '{:.1f}')} | {g('grms', '{:.1f}')} | {kk} |")
