# Asymmetric TTG Continuum Model

Exactly-commensurate Bistritzer–MacDonald continuum model for **asymmetric
twisted trilayer graphene** (θ₁₂ ≠ θ₂₃), built for and validated against
large-scale rigid-structure tight-binding band structures (DeepH-predicted
Hamiltonians, OpenMX `.Band` format) of generator-built commensurate
supercells.

Accuracy against the TB reference along the full Γ–M–K–Γ supercell path
(symmetric Chamfer, ±0.1–0.2 eV window):

| tier | parameters | accuracy |
|---|---|---|
| plain BM | w = 0.11 eV, ħv_F = 0.80×6.582 eV·Å | 3–6 meV |
| full effective H | + V₂ ≈ −50 meV, (β, λ) p-h pair | **0.8–4.2 meV**, incl. the complete Γ fine structure |

Geometry contains **zero free parameters** — layer rotations, per-layer biaxial
strains, Dirac-point positions, both moiré lattices, and all Brillouin-zone
folding are derived analytically from the structure generator
(`moire_structure/supercell/graphene/asy_ttg_poscar_gen.py` conventions) and
locked by a machine-precision **commensuration certificate**
(3·B⁻¹·K_l ∈ ℤ², checked at construction). High-symmetry points of the model
coincide *exactly* with the TB supercell BZ — the historical
"incommensurate ⇒ high-symmetry points don't line up" problem is solved by
putting the generator's own strain into the model, not by approximation.

## The model in one paragraph

Single valley K. For config `TTG_i_j` (tiling m·L(i) ≈ n·L(j)) the three layers
sit at rotations (+θⱼ/2−30°, −θᵢ/2−30°, +θᵢ/2−30°) with biaxial scales
(a_sh/nLⱼ, a_sh/mLᵢ, a_sh/mLᵢ), a₀ = 2.47 Å. The plane-wave basis lives on
supercell reciprocal-lattice points near each layer's Dirac cone (exact
integer arithmetic on the (b₁,b₂)/3 lattice); interlayer BM tunneling has
three channels per interface whose absolute-momentum transfers are the direct
term plus two integer-supercell umklapps — supercell Bloch momentum is
conserved exactly. Optional operators: per-layer on-site `V_layer`
(middle-layer chemical shift V₂ ≈ −50 meV, matching the pz on-site measured
directly in the DeepH LCAO Hamiltonian), particle-hole `beta_ph·|k−K_l|²·𝟙`
(graphene NNN image), and non-local interlayer tunneling
`lam_nl` (t → w(1+λ·d̂_s·k̃), the dominant shoulders-vs-apex actuator).

```python
from ttg_continuum import RigidTTG

mdl = RigidTTG(8, 25)              # TTG_8_25: θ12=2.594°, θ23=3.890°
print(mdl.summary())               # certificate, folded cones, umklapps
k = mdl.frac_to_cart([2/3, 1/3])   # supercell K point (TB path convention)
E = mdl.eigs_at(k, w_aa=0.11, w_ab=0.11, vfc=0.80,
                V_layer=[0, -0.050, 0], beta_ph=-1.8, lam_nl=5.8)
```

TB band files along Γ(0,0)–M(½,0)–K(⅔,⅓)–Γ contain **both** graphene valleys
(TRS: E_K′(k) = E_K(−k)); all comparisons use `eigs(+k) ∪ eigs(−k)`.

## Repository layout

| path | content |
|---|---|
| `ttg_continuum/rigid_ttg.py` | the model: geometry, certificate, basis, BM + V₂/β/λ operators |
| `scripts/compare_rigid.py` | `.Band` parser, two-valley overlay + metrics (CLI) |
| `scripts/batch_rigid.py` | plain-BM (w, vfc, offset) fits |
| `scripts/fit_ph.py` | + V₂, β (λ-less ablation), Γ-zoom plots |
| `scripts/fit_ph2.py` | full fit: Γ sorted-level stage A′ → global stage B over (w, vfc, off, V₂, β, λ) |
| `scripts/make_table.py` | results table from `results/*.json` |
| `scripts/smoke_test.py` | construction checks: certificate, cone folding, A_shared vs POSCAR |
| `data_rigid/<tag>/` | TB reference bands (`openmx.Band`), POSCAR headers (from `rack:~/ttg_organized`) |
| `results/` | per-config overlays (`*_ph2.png`), Γ zooms (`*_ph2_gzoom.png`), fitted params + metrics (`*.json`), eigenvalues (`*.npz`) |
| `docs/DETAILS.md` | full write-up: geometry derivation, certificate, Γ fine-structure physics, fitting protocol, errata for the older ttg_fit work |

Run scripts from anywhere (`python scripts/smoke_test.py`); paths are
repo-relative. Requires numpy, scipy, matplotlib.

## Per-config results (full path, both valleys)

| config | θ₁₂ | θ₂₃ | plain BM | full H | ph2 params (w, vfc, V₂ meV, β, λ Å) |
|---|---|---|---|---|---|
| TTG_2_7 | 8.79° | 13.17° | 28.5 | 7.6* | — (β-only; Γ window has no shoulder constraint) |
| TTG_2_12 | 7.91° | 13.17° | 19.9 | 5.2 | (0.119, 0.842, −49, +0.9, 3.0) |
| TTG_6_19 | 3.39° | 5.09° | 6.1 | 4.0 | (0.107, 0.804, −53, +1.0, 3.4) |
| TTG_7_22 | 2.94° | 4.41° | 5.3 | 4.2 | (0.111, 0.812, −39, −2.0, 5.9) |
| TTG_8_25 | 2.59° | 3.89° | 4.2 | 3.5 | (0.112, 0.808, −47, −1.8, 5.8) |
| TTG_10_31 | 2.10° | 3.15° | 3.4 | 1.9 | (0.116, 0.840, −44, −2.0, 3.7) |
| TTG_12_37 | 1.76° | 2.65° | 3.1 | 1.0 | (0.119, 0.855, −47, −2.0, 2.8) |
| TTG_13_40 | 1.63° | 2.45° | 4.0 | 0.8 | (0.114, 0.828, −58, −1.9, 2.8) |
| TTG_14_43 | 1.52° | 2.28° | 3.9 | 1.0 | (0.110, 0.811, −63, −1.9, 2.4) |

(meV. The θ₂₃ = 13.17° rows are outside formal BM validity and kept as stress
tests. V₂ is transferable: −49 ± 7 meV across configs vs −59 ± 2 meV measured
in the LCAO Hamiltonian. (β, λ) share a soft degeneracy — quote the pair.)

## Zero-shot predictions (blind test)

Seven configurations computed *after* the model was frozen
(`scripts/predict.py`, frozen family-median parameters, only the energy offset
optimized). Spans twist $2.3^\circ$–$13.2^\circ$, tiling $m=3$–$19$
(dim 3400–37000, reduced by the large-angle fast convergence), and both
folding classes (Γ: cones fold to Γ; K/K′/K: $m\!\equiv\!1\bmod 3$, the two
outer cones fold to the same supercell K and split by an L2-mediated gap):

| config | θ₁₂/θ₂₃ | folding | m | tier A (plain BM) | tier B (full) |
|---|---|---|---|---|---|
| TTG_14_43 | 1.52°/2.28° | Γ | 3 | 2.02 | **0.67** |
| TTG_11_34 | 1.92°/2.88° | Γ | 3 | 2.85 | **1.22** |
| TTG_4_45 | 4.03°/7.34° | K/K′/K | 10 | **2.02** | 3.40 |
| TTG_3_45 | 5.08°/9.43° | K/K′/K | 13 | 4.31 | **2.21** |
| TTG_3_6 | 7.26°/9.43° | K/K′/K | 13 | 4.47 | **2.85** |
| TTG_2_45 | 6.95°/13.17° | Γ | 18 | 6.01 | **3.60** |
| TTG_2_47 | 6.93°/13.17° | K/K′/K | 19 | 4.77 | **3.57** |

Take-aways: (i) zero-shot accuracy **0.7–3.6 meV** (full) / 2–6 meV (plain BM)
across all angles, tilings, and folding classes — nothing was fit to these.
(ii) TTG_11_34 (Γ class, in-family angle) predicts to 1.2 meV, confirming the
frozen parameters transfer perfectly within the family. (iii) The K/K′/K-class
$K$-gap is the sensitive probe of $(\beta,\lambda)$: the frozen values match it
at 13° (2_47: TB 10.7 → model 10.5 meV) but miss at 7–9° (4_45, 3_45), because
$\beta$ and $\lambda$ share a soft degeneracy only broken by fitting across
folding classes — a joint refit including these configs would pin them.
(iv) The x_4x family carries a real CNP-vs-dataset-E_F offset ≈ −30 meV (all
tiers agree). (v) TTG_14_43's GPU-Lanczos-v9 rerun matches the old pipeline to
< 0.1 µeV and is predicted at 0.67 meV frozen — better than its own per-config
fit, i.e. universalizing the small-angle parameters lost nothing.

## Relation to earlier work

Supersedes the `Asymmetric_GPU_Continuum_Model` snapshot (old `ttlg_continuum`
lineage: measured-b_super basis, per-layer vF compensation, best ~9 meV on
TTG_8_25). Key upgrades here: exact generator geometry with the certificate
(θ₁₂ = (θᵢ+θⱼ)/2 — see the errata in `docs/DETAILS.md`), integer-arithmetic
momentum bookkeeping (exact high-symmetry alignment), derived tunneling-phase
pairing, two-valley comparison, and the V₂/β/λ operator set with a
level-structure-faithful fitting protocol.
