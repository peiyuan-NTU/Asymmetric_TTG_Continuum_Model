"""
Rigid-structure asymmetric-TTG continuum model, EXACTLY commensurate with the
generated supercell.

Geometry is derived analytically from the generator chain
(moire_structure: cal_theta.py -> lib.py -> tbg_gen.py -> asy_ttg_poscar_gen.py),
NOT fitted and NOT extracted from atomic positions:

  config TTG_i_j, tiling (m, n) with  m*L(i) ~ n*L(j),   L(n) = a0*f(n),
  f(n) = sqrt(3n^2+3n+1),  theta(n) = arccos[(3n^2+3n+1/2)/(3n^2+3n+1)]

  shared supercell = standard 60-deg hex, a1 along +x, edge
      A = (m*a0*f(i) + n*a0*f(j)) / 2                       [a0 = 2.47 A]
  per-layer orientation (relative to the shared a1 axis) and biaxial scale:
      layer 1 (TBG_j top):    alpha1 = +theta_j/2 - 30deg,  s1  = A/(n*a0*f(j))
      layer 2 (TBG_i bottom): alpha2 = -theta_i/2 - 30deg,  s23 = A/(m*a0*f(i))
      layer 3 (TBG_i top):    alpha3 = +theta_i/2 - 30deg,  s23
  (the -30deg: the TBG supercell vector sits at 30+theta/2 from the graphene a1
   axis; transplanting fractional coords into the standard hex cell rotates each
   TBG by -(30+theta/2).)

Because every layer is an exact sublattice of the shared supercell, each layer's
Dirac momentum K_l has coordinates = (integers)/3 in the supercell reciprocal
basis (b1s, b2s) at machine precision -- verified at construction time (the
"commensuration certificate"). The plane-wave basis and all interlayer momentum
transfers are then handled in exact integer arithmetic on the (1/3)-reciprocal
lattice, so the model is block-diagonal in the supercell Bloch momentum and its
high-symmetry points coincide EXACTLY with the supercell BZ of the DFT/TB data.

Checked against rack data: A(2,7) = (3*sqrt(19)+13)/2*2.47 = 32.2047206 A
matches ~/ttg_organized/TTG_2_7/POSCAR line 3 to all printed digits.

Hamiltonian (single valley K; sublattice basis (A,B) per momentum DOF):
  intralayer: h_l(k) off-diag  hAB = -HBAR_VF*vfc_l*(kx + i ky)*e^{-i alpha_l},
              k measured from the layer's own (strained, rotated) K_l
  interlayer: BM tunneling on the three transfers q_s = R(-120)^s (K_l - K_l'),
              T_s = w_aa*I + w_ab*[[0, e^{i ph_s}], [e^{-i ph_s}, 0]],
              ph = (0, -2pi/3, +2pi/3)   (same convention as ttlg_continuum.py,
              validated against the MATLAB ttlg model and ML bands)
Registry: all three layers carry the same sublattice offset u=(a1+a2)/3 in their
own frames (no atom at the rotation origin), which shifts BOTH moire patterns by
the same ~|u| and leaves their relative alignment at O(theta*u) ~ 0.1 A over
50-200 A moire periods -> AAA-aligned tunneling phases are exact to <0.1 meV.

Units: momenta 1/Angstrom, energies eV.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

A0 = 2.47          # graphene lattice constant of the generator (graphene_60_hex_orgin.vasp)
HBAR_VF = 6.582    # eV*Angstrom at vfc = 1


# ----------------------------------------------------------------- helpers
def f_com(n):
    return np.sqrt(3.0 * n * n + 3.0 * n + 1.0)


def theta_com(n):
    """Commensurate (n, n+1) twist angle in radians."""
    return np.arccos((3.0 * n * n + 3.0 * n + 0.5) / (3.0 * n * n + 3.0 * n + 1.0))


def hex_lattice(a):
    """Standard 60-deg hex lattice, a1 along +x. Columns = a1, a2."""
    return a * np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])


def recip(A):
    """Columns a1,a2 -> columns b1,b2 (= 2 pi inv(A)^T)."""
    return 2.0 * np.pi * np.linalg.inv(A).T


def rotmat(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def K_of(G):
    """Valley-K corner (2 g1 + g2)/3 of a reciprocal lattice (columns g1,g2)."""
    return (2.0 * G[:, 0] + G[:, 1]) / 3.0


def best_mn(i, j, mmax=64, tol=0.01):
    """Smallest retained cell with mismatch <= ``tol``.

    The structure generator first applies the mismatch gate and then minimizes
    the exact trilayer atom count, rather than minimizing mismatch alone.  The
    default search range covers every tiling in the 94-structure catalogue.
    """
    best = None
    ni = 3 * i * i + 3 * i + 1
    nj = 3 * j * j + 3 * j + 1
    for m in range(1, mmax + 1):
        for n in range(1, mmax + 1):
            r = abs(m * f_com(i) / (n * f_com(j)) - 1.0)
            if r > tol + 1e-12:
                continue
            n_atoms = 4 * ni * m * m + 2 * nj * n * n
            candidate = (n_atoms, r, m, n)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError(
            f"No tiling for TTG_{i}_{j} has mismatch <= {tol:.3%} "
            f"within m,n <= {mmax}; pass m,n explicitly or enlarge mmax."
        )
    _, r, m, n = best
    return m, n, r


def int_matrix(M, tol=1e-9, what="matrix"):
    Mi = np.round(M).astype(np.int64)
    dev = np.max(np.abs(M - Mi))
    if dev > tol:
        raise ValueError(f"{what} not integer: max dev {dev:.3e}\n{M}")
    return Mi


# ----------------------------------------------------------------- model
class RigidTTG:
    """Exactly-commensurate continuum model of one generated TTG_i_j supercell.

    mirror=True builds the x-axis mirror image (opposite handedness of all
    twists) -- used once to pin the generator's absolute chirality against the
    TB bands / POSCAR; the commensuration is identical for both.
    """

    def __init__(self, i, j, m=None, n=None, n_shells=12, mirror=False,
                 vf_strain_beta=0.0, pairing="derived"):
        self.i, self.j = i, j
        self.pairing = pairing
        if m is None or n is None:
            m, n, _ = best_mn(i, j)
        self.m, self.n = m, n

        ti, tj = theta_com(i), theta_com(j)
        fi, fj = f_com(i), f_com(j)
        self.theta12_deg = np.degrees(0.5 * (ti + tj))
        self.theta23_deg = np.degrees(ti)
        self.A_shared = 0.5 * (m * A0 * fi + n * A0 * fj)
        s1 = self.A_shared / (n * A0 * fj)
        s23 = self.A_shared / (m * A0 * fi)
        self.scales = np.array([s1, s23, s23])

        off = -np.pi / 6.0
        sgn = -1.0 if mirror else 1.0
        self.alphas = sgn * np.array([tj / 2.0 + off, -ti / 2.0 + off, ti / 2.0 + off])

        # supercell reciprocal basis (columns b1s, b2s)
        self.B = recip(hex_lattice(self.A_shared))
        self.b1s, self.b2s = self.B[:, 0].copy(), self.B[:, 1].copy()

        # per-layer Dirac K and the commensuration certificate: 3*B^-1 K integer
        self.K = [K_of(recip(rotmat(a) @ hex_lattice(A0 * s)))
                  for a, s in zip(self.alphas, self.scales)]
        self.cK = [int_matrix(3.0 * np.linalg.solve(self.B, K), 1e-6,
                              f"3*coords of K_layer{l+1} (commensuration broken)")
                   for l, K in enumerate(self.K)]
        # folded cone position of each layer in the supercell BZ, units of (b1s,b2s)/3
        self.folded = [tuple(c % 3) for c in self.cK]

        # R(-120deg) as an integer matrix on (1/3)-lattice coordinates
        # (matches rot120 = R(-2pi/3) used by the validated ttlg_continuum port)
        R120c = rotmat(-2.0 * np.pi / 3.0)
        self.M120 = int_matrix(np.linalg.solve(self.B, R120c @ self.B),
                               1e-9, "R(-120) in b-basis")

        # velocity per layer: v_l = vfc * (1 + (1 - beta)*(s_l - 1)), beta ~ 3.14 if used
        self.vf_layer = 1.0 + (1.0 - vf_strain_beta) * (self.scales - 1.0) \
            if vf_strain_beta != 0.0 else np.ones(3)

        self._build_basis(n_shells)
        self._build_interlayer()
        self.prefac = np.exp(-1j * self.alphas[self.lay - 1])

    # ---------------- basis: supercell reciprocal-lattice points near each cone
    def _build_basis(self, n_shells):
        """Basis state (l, n): plane wave of layer l at absolute momentum
        p = q + B @ n  (q = supercell Bloch momentum, n integer) -- one common
        Bloch class for ALL layers, so supercell momentum is conserved exactly.
        Keep n with |B@n - K_l| <= rcut (disk around the layer's own cone).
        Dirac-relative momentum: p - K_l = q + B @ (3n - cK_l)/3, exact thirds."""
        self.n_shells = n_shells
        rcut = (n_shells + 0.5) * np.linalg.norm(self.b1s)
        span = int(np.ceil(rcut / np.linalg.norm(self.b1s) * 2.0)) + 2
        nvecs, lay = [], []
        for l in range(3):
            n0 = np.round(self.cK[l] / 3.0).astype(np.int64)
            for m1 in range(-span, span + 1):
                for m2 in range(-span, span + 1):
                    n = n0 + np.array([m1, m2], np.int64)
                    if np.linalg.norm(self.B @ (n - self.cK[l] / 3.0)) <= rcut:
                        nvecs.append(n)
                        lay.append(l + 1)
        self.nvecs = np.array(nvecs, dtype=np.int64)       # (N,2) integer b-coords
        self.lay = np.array(lay)
        self.N = len(self.nvecs)
        rel = np.array([3 * self.nvecs[a] - self.cK[self.lay[a] - 1]
                        for a in range(self.N)], dtype=np.int64)
        self.pos = (rel / 3.0) @ self.B.T                  # (N,2) = p - K_l at q=0
        self.index = {(int(l), int(n1), int(n2)): a
                      for a, (l, (n1, n2)) in enumerate(zip(self.lay, self.nvecs))}

    # ---------------- interlayer BM tunneling in integer arithmetic
    def _build_interlayer(self):
        """BM tunneling, three channels s = 0,1,2 per interface.

        Derivation (two-center, first t-shell, AA registry at origin -- exact
        for these structures since all layers share a zero-displacement hexagon
        center at the origin): the bra state of layer l' at cone-relative
        momentum k' couples to the ket state of layer l at k when
            k' = k + q_s,   q_s = R(-120deg)^s (K_l - K_l'),
            <l', k+q_s| H |l, k> = w_aa*I + w_ab*P_s,
            P_s = [[0, e^{i ph_s}], [e^{-i ph_s}, 0]],  ph = (0, -2pi/3, +2pi/3)
        (sublattice order (A,B); phases from G^(s).tau_A with tau = (a1+a2)/3,
        identical for both interfaces because all layers carry the same tau in
        their own frames).

        In absolute momentum the transfer is an integer supercell vector:
        n_bra = n_ket + D_s,  D_s = (Q_s - Q_0)/3,  Q_s = M120^s (cK_l - cK_l');
        D_0 = 0 is the direct (momentum-conserving) term, D_1, D_2 umklapps.
        (D_s is integer because (M120^s - I) cK ≡ 0 mod 3 for every layer.)

        pairing="legacy" reproduces the older ttlg_continuum.build_interlayer
        bookkeeping (P_s attached to the opposite transfer direction, i.e. the
        hermitian conjugate assignment); kept as an A/B switch.
        """
        ph = [0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0]
        Pab = [np.array([[0, np.exp(1j * p)], [np.exp(-1j * p), 0]]) for p in ph]
        I2 = np.eye(2)
        raa, caa, daa = [], [], []
        rab, cab, dab = [], [], []
        # channel-resolved hop record for the non-local tunneling correction:
        # amplitude of every hop is t(|u|) with u = R(-120)^s K_lk + k~_ket
        # (the extended-zone momentum of the tunneling plane wave), so
        # t ~ w * (1 + lam * d_s . (k~_ket + k)),  d_s = unit(R^s K_lk).
        h_raa, h_caa, h_daa = [], [], []       # per-hop AA entries
        h_rab, h_cab, h_dab = [], [], []       # per-hop AB entries
        h_ket_aa, h_dir_aa = [], []            # ket DOF index + channel dir
        h_ket_ab, h_dir_ab = [], []
        self.q_int, self.D_int = {}, {}
        R120c = rotmat(-2.0 * np.pi / 3.0)
        for (lk, lb) in [(1, 2), (2, 3)]:      # ket layer lk -> bra layer lb
            Q0 = self.cK[lk - 1] - self.cK[lb - 1]
            Qs = [Q0, self.M120 @ Q0, self.M120 @ (self.M120 @ Q0)]
            Ds = [int_matrix((Qi - Q0) / 3.0, 1e-9, "umklapp D_s") for Qi in Qs]
            self.q_int[(lk, lb)] = Qs
            self.D_int[(lk, lb)] = Ds
            Kk = self.K[lk - 1]
            dirs = [Kk / np.linalg.norm(Kk)]
            dirs.append(R120c @ dirs[0]); dirs.append(R120c @ dirs[1])
            kets = np.nonzero(self.lay == lk)[0]
            for b in kets:
                n = self.nvecs[b]
                for s, D in enumerate(Ds):
                    if self.pairing == "derived":
                        tgt = n + D                     # bra at n_ket + D_s, T_s
                        P = Pab[s]
                    else:                               # legacy: transfer and
                        tgt = n - D                     # phase both conjugated
                        P = Pab[s].conj().T
                    a = self.index.get((lb, int(tgt[0]), int(tgt[1])))
                    if a is None:
                        continue
                    ai, bj = 2 * a, 2 * b
                    for r in range(2):
                        for c in range(2):
                            if I2[r, c]:
                                raa.append(ai + r); caa.append(bj + c); daa.append(I2[r, c])
                                h_raa.append(ai + r); h_caa.append(bj + c)
                                h_daa.append(I2[r, c])
                                h_ket_aa.append(b); h_dir_aa.append(dirs[s])
                            if P[r, c] != 0:
                                rab.append(ai + r); cab.append(bj + c); dab.append(P[r, c])
                                h_rab.append(ai + r); h_cab.append(bj + c)
                                h_dab.append(P[r, c])
                                h_ket_ab.append(b); h_dir_ab.append(dirs[s])
        sh = (2 * self.N, 2 * self.N)
        M_aa = coo_matrix((daa, (raa, caa)), shape=sh, dtype=complex).tocsr()
        M_ab = coo_matrix((dab, (rab, cab)), shape=sh, dtype=complex).tocsr()
        self.M_aa = M_aa + M_aa.getH()
        self.M_ab = M_ab + M_ab.getH()
        self._sh = sh
        self._haa = (np.array(h_raa), np.array(h_caa), np.array(h_daa, complex),
                     np.array(h_ket_aa), np.array(h_dir_aa))
        self._hab = (np.array(h_rab), np.array(h_cab), np.array(h_dab, complex),
                     np.array(h_ket_ab), np.array(h_dir_ab))

    def interlayer_nl(self, k, w_aa, w_ab, lam_nl):
        """Interlayer matrix with the non-local (momentum-dependent) tunneling
        factor  t -> t * (1 + lam_nl * d_s . (k~_ket + k)),  lam_nl in Angstrom
        (= d ln t / dq at the first shell). lam_nl = 0 reproduces w_aa*M_aa +
        w_ab*M_ab exactly."""
        k = np.asarray(k, float)
        out = None
        for w, (rr, cc, dd, ket, dirs) in ((w_aa, self._haa), (w_ab, self._hab)):
            if w == 0.0 or len(rr) == 0:
                continue
            fac = 1.0 + lam_nl * np.einsum("ij,ij->i", dirs, self.pos[ket] + k)
            M = coo_matrix((w * dd * fac, (rr, cc)), shape=self._sh,
                           dtype=complex).tocsr()
            out = M if out is None else out + M
        if out is None:
            return coo_matrix(self._sh, dtype=complex).tocsr()
        return out + out.getH()

    # ---------------- Hamiltonian / bands
    def dirac(self, k):
        """Sparse Dirac part at supercell Bloch momentum k (cartesian, 1/A)."""
        kx = self.pos[:, 0] + k[0]
        ky = self.pos[:, 1] + k[1]
        hAB = -HBAR_VF * (kx + 1j * ky) * self.prefac * self.vf_layer[self.lay - 1]
        N = self.N
        rows = np.empty(2 * N, np.int64); cols = np.empty(2 * N, np.int64)
        dat = np.empty(2 * N, complex); idx = np.arange(N)
        rows[0::2] = 2 * idx;     cols[0::2] = 2 * idx + 1; dat[0::2] = hAB
        rows[1::2] = 2 * idx + 1; cols[1::2] = 2 * idx;     dat[1::2] = np.conj(hAB)
        return coo_matrix((dat, (rows, cols)), shape=(2 * N, 2 * N), dtype=complex).tocsr()

    def H(self, k, w_aa, w_ab, vfc, V_layer=None, beta_ph=0.0, lam_nl=0.0):
        """beta_ph (eV*A^2): particle-hole-asymmetry term beta*|k - K_l|^2 * identity
        on the sublattice — the continuum image of graphene's next-nearest-neighbor
        hopping. Gapless (does not touch the Dirac crossings), shifts states in
        proportion to their plane-wave momentum squared, so it moves the moire
        hybridization shoulders relative to the cone apexes.
        lam_nl (A): non-local interlayer tunneling, t -> t(1 + lam*d_s.(k~+k)) —
        the momentum dependence of the interlayer form factor around the first
        shell; shifts tunneling-hybridized states asymmetrically while leaving
        the bare cone apexes untouched (the dominant e-h asymmetry mechanism of
        twisted bilayers beyond beta)."""
        if lam_nl != 0.0:
            Hm = self.interlayer_nl(k, w_aa, w_ab, lam_nl) + vfc * self.dirac(k)
        else:
            Hm = w_aa * self.M_aa + w_ab * self.M_ab + vfc * self.dirac(k)
        diag = None
        if V_layer is not None:
            diag = np.asarray(V_layer, float)[self.lay - 1]
        if beta_ph != 0.0:
            k2 = np.sum((self.pos + np.asarray(k, float)) ** 2, axis=1)
            diag = beta_ph * k2 if diag is None else diag + beta_ph * k2
        if diag is not None:
            dv = np.repeat(diag, 2)
            Hm = Hm + coo_matrix((dv, (np.arange(2 * self.N), np.arange(2 * self.N))),
                                 shape=(2 * self.N, 2 * self.N)).tocsr()
        return Hm.tocsc()

    def eigs_at(self, k, w_aa, w_ab, vfc, num_eigs=60, sigma=1e-4, V_layer=None,
                beta_ph=0.0, lam_nl=0.0):
        H = self.H(k, w_aa, w_ab, vfc, V_layer, beta_ph, lam_nl)
        e = eigsh(H, k=num_eigs, sigma=sigma, which="LM", return_eigenvectors=False)
        return np.sort(e.real)

    def bands(self, k_list, w_aa, w_ab, vfc, num_eigs=60, sigma=1e-4,
              V_layer=None, beta_ph=0.0, lam_nl=0.0, verbose=False):
        """k_list: (nk, 2) cartesian supercell Bloch momenta (1/A)."""
        k_list = np.asarray(k_list, float)
        vals = np.zeros((len(k_list), num_eigs))
        for ik, k in enumerate(k_list):
            vals[ik] = self.eigs_at(
                k, w_aa, w_ab, vfc, num_eigs, sigma, V_layer,
                beta_ph, lam_nl,
            )
            if verbose and ik % 10 == 0:
                print(f"  k {ik}/{len(k_list)}  ndof={self.N}", flush=True)
        return vals

    def frac_to_cart(self, kfrac):
        """Supercell-BZ fractional (2,) or (nk,2) -> cartesian 1/A."""
        return np.asarray(kfrac, float) @ self.B.T

    def summary(self):
        s = [f"TTG_{self.i}_{self.j}: m={self.m} n={self.n}  "
             f"A={self.A_shared:.6f} A  ndof={self.N} (dim {2*self.N})",
             f"  theta12={self.theta12_deg:.4f} deg  theta23={self.theta23_deg:.4f} deg",
             f"  strain eps1={self.scales[0]-1:+.5%}  eps23={self.scales[1]-1:+.5%}",
             f"  alphas(deg)={np.degrees(self.alphas).round(4).tolist()}",
             f"  folded cones (units b/3, mod 3): {self.folded}",
             f"  q_int L1-L2: {[q.tolist() for q in self.q_int[(1,2)]]}",
             f"  q_int L2-L3: {[q.tolist() for q in self.q_int[(2,3)]]}"]
        return "\n".join(s)
