# Exactly-commensurate continuum model for the rigid TTG dataset

Target data: `rack:~/ttg_organized/TTG_<i>_<j>/band/openmx.Band` — bands of the
ML-predicted TB Hamiltonian for **rigid** (unrelaxed) asymmetric-TTG
supercells built by `moire_structure/supercell/graphene/asy_ttg_poscar_gen.py`.
DFT cross-checks exist for TTG_2_7 and TTG_2_12.

## The problem this solves

The generated structures are commensurate **only because the generator strains
them**: TBG_i (tiled m×m) and TBG_j (tiled n×n) are forced onto a shared hex
lattice `a_shared = (m·L_i + n·L_j)/2`, giving each TBG a biaxial strain of
±½ the lattice mismatch. A continuum model built from the *ideal unstrained*
angles is incommensurate with that supercell — its moiré vectors do not lie on
the supercell reciprocal lattice, so folded high-symmetry points never line up
with the TB/DFT band path. **Fix: put the generator's exact per-layer strain
and rotation into the continuum model.** Then both moiré patterns are exactly
commensurate with the supercell by construction, and Γ/M/K of the model
coincide exactly with the TB path. No approximation is involved beyond BM
itself — the strain is a property of the simulated structure, not a fudge.

## Exact geometry (derived from the generator, verified against POSCARs)

For `TTG_i_j` with tiling (m, n), `f(k) = √(3k²+3k+1)`, `L(k) = a0·f(k)`,
**a0 = 2.47 Å** (from `graphene_60_hex_orgin.vasp` — not 2.46), and
`θ(k) = arccos[(3k²+3k+½)/(3k²+3k+1)]`:

| layer (z-order) | origin | rotation vs shared a1 axis | biaxial scale |
|---|---|---|---|
| L1 (z=0) | TBG_j top | `+θ_j/2 − 30°` | `s1 = a_shared/(n·L_j)` |
| L2 (z=3.35 Å) | TBG_i bottom | `−θ_i/2 − 30°` | `s23 = a_shared/(m·L_i)` |
| L3 (z=6.7 Å) | TBG_i top | `+θ_i/2 − 30°` | `s23` |

Interface twists: **θ12 = (θ_i+θ_j)/2** (not θ_j — see "errata" below),
θ23 = θ_i. The −30° common offset comes from transplanting fractional
coordinates from the TBG supercell (whose a1 sits at 30°+θ/2 from the graphene
a1) into the standard hex cell. Verified: analytic `a_shared(2,7) =
(3√19+13)/2·2.47 = 32.2047206 Å` = rack POSCAR to all digits; per-layer bond
angles/lengths of `TTG_2_7.vasp`, `TTG_10_31.vasp` match to 1e-5.

All three layers share a zero-displacement hexagon center at the origin ⇒ both
interfaces are **exactly AA-registered at r=0**, so the standard BM tunneling
phases apply with no registry correction, and the structure has exact C6z.

## Commensuration certificate (machine precision, not a fit)

Each layer is an exact sublattice of the supercell ⇒ its Dirac momentum obeys
`3·B⁻¹·K_l ∈ ℤ²` (B = supercell reciprocal basis). This is checked at model
construction to ~1e-13 and **fails loudly** if the geometry is wrong (the old
`θ12 = θ_j` convention fails it at the 1e-2–1e-3 level). Consequences, all
exact:

* folded cone positions = `(3·B⁻¹·K_l) mod 3`: for the 3:1 family (j = 3i+1),
  **L1 → supercell K, L2, L3 → Γ**; for TTG_2_12: K, K, K′.
* the three BM transfers per interface are, in absolute momentum, `D_0 = 0`
  (direct) plus two integer-supercell-vector umklapps `D_s = (Q_s − Q_0)/3` —
  supercell Bloch momentum is conserved exactly; the Hamiltonian at the TB
  path's fractional k is directly comparable, high-symmetry points included.

## Model (`rigid_ttg.py`)

Single valley K; basis = supercell reciprocal-lattice points within
`(n_shells+½)|b1s|` of each layer's cone (integer arithmetic throughout;
converged below 0.1 µeV at n_shells = 9 for the ±0.25 eV window). Per-layer
Dirac `−ħv_F·vfc·(k−K_l)·e^{−iα_l}` with HBAR_VF = 6.582 eV·Å; BM tunneling
`T_s = w_aa·I + w_ab·P_s`, `ph_s = (0, −2π/3, +2π/3)`. Rigid structures ⇒
`w_aa = w_ab = w`. Two free parameters (w, vfc) + a small energy offset.

**Tunneling-pairing convention matters**: the transfer direction q_s and phase
P_s must be paired as derived (bra layer at `k+q_s`, `q_s = R(−120°)^s
(K_ket − K_bra)`, ⟨bra|H|ket⟩ = T_s). The opposite ("legacy") pairing—which the
older `ttlg_continuum.build_interlayer` used—doubles the Chamfer error and
opens a spurious ~14 meV gap at the supercell-K Dirac cone of TTG_8_25.

Both graphene valleys appear in the TB bands along Γ-M-K-Γ (TRS:
E_K′(k) = E_K(−k)), so all comparisons use the union eigs(+k) ∪ eigs(−k).
A mirrored model changes nothing after this union (checked — as it must).

## Results

Symmetric Chamfer (meV) between TB and model bands in the reliable window
(≈ ±0.2 eV for 8_25, shrinking to ≈ ±0.1 eV for the largest cells; full path,
150 k-points, both valleys). Three model tiers:

* **recipe** — plain BM, fixed standard parameters w = 0.11 eV, vfc = 0.80,
  only a global offset optimized;
* **fit** — free (w, vfc, offset) (`batch_rigid.py`);
* **ph2** — the full effective Hamiltonian: (w, vfc, offset) + middle-layer
  on-site V₂ + particle-hole β·k²·𝟙 + **non-local interlayer tunneling λ**
  (`fit_ph2.py`; the `+V₂β` column is the λ-less ablation, `fit_ph.py`).

| config | θ12 | θ23 | recipe | fit | +V₂β (abl.) | **ph2** | ph2 params (w, vfc, V₂ meV, β, λ Å) | K-cone TB/md |
|---|---|---|---|---|---|---|---|---|
| TTG_2_7 | 8.79° | 13.17° | 28.5 | 11.1 | **7.6** | 12.7* | (0.135, 0.835, −2, −0.3, 3.2) | 15.1 / 0.4 |
| TTG_2_12 | 7.91° | 13.17° | 19.9 | 12.9 | 5.3 | **5.2** | (0.119, 0.842, −49, +0.9, 3.0) | 1.9 / 0.4 |
| TTG_6_19 | 3.39° | 5.09° | 6.1 | 5.7 | 5.2 | **4.0** | (0.107, 0.804, −53, +1.0, 3.4) | 1.3 / 0.1 |
| TTG_7_22 | 2.94° | 4.41° | 5.3 | 5.1 | 4.3 | **4.2** | (0.111, 0.812, −39, −2.0, 5.9) | 1.0 / 0.2 |
| TTG_8_25 | 2.59° | 3.89° | 4.2 | 4.3 | 3.9 | **3.5** | (0.112, 0.808, −47, −1.8, 5.8) | 3.1 / 1.0 |
| TTG_10_31 | 2.10° | 3.15° | 3.4 | 3.5 | 2.6 | **1.9** | (0.116, 0.840, −44, −2.0, 3.7) | 9.3 / 0.6 |
| TTG_12_37 | 1.76° | 2.65° | 3.1 | 2.4 | 1.7 | **1.0** | (0.119, 0.855, −47, −2.0, 2.8) | 1.5 / 1.4 |
| TTG_13_40 | 1.63° | 2.45° | 4.0 | 1.8 | 1.6 | **0.8** | (0.114, 0.828, −58, −1.9, 2.8) | 5.3 / 1.8 |
| TTG_14_43 | 1.52° | 2.28° | 3.9 | 2.3 | 1.7 | **1.0** | (0.110, 0.811, −63, −1.9, 2.4) | 1.2 / 0.1 |

(*TTG_2_7's ph2 stage-B wandered — its Γ window contains only the two cone
doublets, so V₂ is unconstrained; quote the β-only ablation for it. TTG_11_34
/ 2_45 / 4_45 / 6_32 have no TB bands on rack. Free 2-parameter fits at the
smallest angles slide along the α = w/(ħv·kθ) degeneracy valley — the recipe
column is the honest plain-BM statement, e.g. 12_37's "fit" (0.164, 1.061) is
a valley artifact.)

Headlines:

* Plain BM at standard parameters: **3–6 meV** across the small-angle family
  (old wrong-geometry approach: 13–33 meV at 8_25, 85–125 meV at 13°).
* Full effective Hamiltonian: **0.8–4.2 meV**, reaching ~1 meV for the
  smallest angles; even θ23 = 13.2° (2_12) lands at 5.2 meV.
* The supercell-K Dirac cone survives at all angles in both TB and model,
  with no strain fudge — the exact geometry does it.

## The Γ fine structure and the three extra operators

At supercell Γ (3:1 family) the TB levels per valley form four clusters —
8_25: shoulder −60.9×3 | cone doublets −20.3×2, −0.7×2 | shoulder
+72.8/+76.5×3. Plain BM produces the shoulder clusters (moiré interlayer
hybridization: [−69.6×3, −4.8×4, +57.3×3]) but (a) leaves the two folded cone
apexes degenerate (TB: split 19.6 meV) and (b) is approximately particle-hole
symmetric about the cone energy, so the shoulder centers sit ~17 meV wrong.
The mechanisms, each pinned by an independent fact:

* **V₂ (middle-layer on-site)** — the apex splitting cannot come from
  tunneling: in the two-center approximation the apex-apex transfer K₂−K₃ is
  not in the difference set of the two layers' reciprocal lattices, so the
  coupling is exactly zero (model quartet degenerate to 0.9 meV). Fitted
  **V₂ = −49 ± 7 meV across all clean configs**, consistent with the pz
  on-site measured directly in the ML LCAO Hamiltonian (middle − outer =
  −59 ± 2 meV, rack:/tmp/pz_onsite.py; outer−outer asymmetry only 1–3 meV).
  An earlier "effective V₂ ≈ −10 meV" conclusion was an artifact of fitting
  V₂ without the λ operator: Chamfer then compromises at V₂ ≈ −20 with half
  the splitting, because V₂ alone drags the shoulders the wrong way.
* **β·|k−K_l|²·𝟙** (graphene NNN image) — gapless diagonal term shifting each
  state by β⟨k̃²⟩; actuates p-h asymmetry. Graphene t′ ≈ 0.27 eV predicts
  |β| ≈ 1.2 eV·Å², the right order.
* **λ (non-local interlayer tunneling)** — t → w·(1 + λ·d̂_s·(k̃+k)): the
  momentum dependence of the interlayer form factor around the first shell.
  Acts only on tunneling-hybridized states (bare apexes untouched), with 3×
  the shoulder-vs-apex leverage of β — the missing actuator. Without it the
  operator set is under-actuated for "shoulders up +17 meV relative to
  apexes"; with it all four Γ clusters land within ±3 meV and V₂ returns to
  the measured LCAO value.

Caveats: (β, λ) share a soft degeneracy (band energies constrain mostly their
combination; β often pins at a bound while λ = 2.4–6 Å carries the load) —
quote the pair, do not interpret them separately. Keep |β| ≲ 3 at
n_shells = 12: larger β bends remote basis states into the low-energy window
(spurious bands — a basis artifact seen when β reached 6). λ's fitted sign is
opposite to the naive two-center form-factor slope — treat it as an effective
operator absorbing multi-orbital/overlap non-locality of the ML
Hamiltonian.

Fitting protocol (`fit_ph2.py`): stage A′ matches the *sorted* Γ level lists
(assignment-free — Chamfer-type metrics tolerate half-captured splittings)
over (V₂, β, λ, Δw) at frozen w, vfc, with multi-anchor registration and
physical bounds (dense-Γ configs misregister otherwise); stage B refines
(w, vfc, off, V₂, β, λ) on the global Chamfer.

### Bottom line — the effective low-energy Hamiltonian

For any generated TTG_i_j (both twists ≲ 5°): `RigidTTG(i, j)` with

* geometry: **zero free parameters** (exact generator geometry, certificate-checked);
* plain-BM tier: w_aa = w_ab = **0.11 eV**, ħv_F = **0.80 × 6.582 eV·Å**,
  global offset ≈ +5 meV → 3–6 meV accuracy;
* full tier: + **V₂ ≈ −50 meV** (transferable, matches the measured LCAO
  shift) + effective p-h pair (β, λ) ≈ (−2 eV·Å², 2.4–6 Å) → 0.8–4 meV and
  the complete Γ fine structure (cone-doublet splitting, shoulder positions).

## Errata for the older work in ttg_fit/

* `RESULT_TTG_8_25.md` used θ12 = θ(25) = 1.297°; the actual generator gives
  θ12 = (θ(8)+θ(25))/2 = 2.594°. The folding topology it found (L0→K, L1/L2→Γ)
  was right, but the interface-12 coupling scale |q12| was ~½ the true value,
  and the "near-magic 1.297°" interpretation does not describe these
  structures. The commensuration certificate rejects that convention outright.
* The old README table (θ12 = (θi+θj)/2) was correct on this point; the
  per-layer heterostrain it fitted (−0.4%) belongs to the *older* scale_results
  dataset, not to these `ttg_organized` rigid structures (whose built-in strain
  is only ±0.29% for 2_x and ≤±0.04% for the 3:1 small-angle family).

## Files

| file | purpose |
|---|---|
| `rigid_ttg.py` | geometry + certificate + exactly-commensurate BM model (+V_layer, β·k², λ non-local tunneling) |
| `compare_rigid.py` | openmx.Band parser, two-valley overlay + metrics CLI |
| `batch_rigid.py` | plain-BM per-config (w, vfc, offset) fit + final overlay |
| `fit_ph.py` | + V₂, β fit (λ-less ablation) + Γ-zoom plot |
| `fit_ph2.py` | full fit: Γ sorted-level stage A′ + global stage B over (w, vfc, off, V₂, β, λ) |
| `make_table.py` | assemble the results table from results/*.json |
| `smoke_test.py` | construction checks (certificate, cone folding, timing) |
| `../data_rigid/<tag>/` | openmx.Band, meta.json, poscar_header.txt copies |
