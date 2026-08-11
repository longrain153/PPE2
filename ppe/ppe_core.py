"""Power profile estimation (PPE) core algorithms.

Implements:
  * RP1 perturbation matrix G (Sasai's linear least squares PPE model)
  * plain LS-PPE
  * differential sparse anomaly localization: generalized Lasso
    (attenuation-weighted total variation, Fujitsu OECC'24 J-matrix
    style) solved with ADMM on the *differential* signal
    r = y_fault - G @ x_ref
  * matched-filter refinement of the fault position on a fine grid
"""

import numpy as np

from .link_sim import LinkConfig, SignalConfig, dispersion_op


def rx_to_perturbation(a_rx: np.ndarray, u_tx: np.ndarray, link: LinkConfig,
                       sig: SignalConfig) -> np.ndarray:
    """CDC + normalization + common-phase alignment; return y = u_rx - u_tx.

    y approximates the first-order nonlinear perturbation A1 in the
    CD-compensated domain, up to receiver noise.
    """
    cdc = np.conj(dispersion_op(a_rx.size, sig.dt_ps, link.beta2_ps2_km,
                                link.total_length_km))
    u_rx = np.fft.ifft(np.fft.fft(a_rx) * cdc)
    u_rx = u_rx / np.sqrt(np.mean(np.abs(u_rx) ** 2))
    # note: no common-phase alignment here -- the mean nonlinear phase
    # rotation IS part of the RP1 model (each column of G contains a
    # j*u_tx component), so removing it would bias the LS fit. The
    # simulation has no laser phase noise, so no CPR is needed.
    return u_rx - u_tx


def build_g_matrix(u_tx: np.ndarray, z_grid_km: np.ndarray, link: LinkConfig,
                   sig: SignalConfig) -> np.ndarray:
    """Columns g_k = D(-z_k)[ j |u_k|^2 u_k ], u_k = D(z_k) u_tx.

    Expressed in the CD-compensated domain so that
    y ≈ G x with x_k = gamma * P(z_k) * dz (real, >= 0).
    """
    n = u_tx.size
    spec0 = np.fft.fft(u_tx)
    g = np.empty((n, z_grid_km.size), dtype=complex)
    for k, z in enumerate(z_grid_km):
        d = dispersion_op(n, sig.dt_ps, link.beta2_ps2_km, z)
        uk = np.fft.ifft(spec0 * d)
        pert = 1j * np.abs(uk) ** 2 * uk
        g[:, k] = np.fft.ifft(np.fft.fft(pert) * np.conj(d))
    return g


def normal_eqs(g: np.ndarray, y: np.ndarray):
    """Real-constrained normal equations: M = Re(G'G), c = Re(G'y)."""
    return np.real(g.conj().T @ g), np.real(g.conj().T @ y)


def ls_ppe(g: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain (real-constrained) least squares: x = Re(G'G)^-1 Re(G'y)."""
    m, c = normal_eqs(g, y)
    return np.linalg.solve(m, c)


def d2_matrix(m: int) -> np.ndarray:
    """Second-difference operator ((m-2) x m), the NTT ECOC'25 D^(2)."""
    d = np.zeros((m - 2, m))
    idx = np.arange(m - 2)
    d[idx, idx] = 1.0
    d[idx, idx + 1] = -2.0
    d[idx, idx + 2] = 1.0
    return d


def sparse_absolute_profile(m_mat: np.ndarray, c_vec: np.ndarray,
                            target_knots: int = 12) -> np.ndarray:
    """NTT-style link-parameter-free sparse regularization of an
    ABSOLUTE power profile:

        min_x 0.5||G x - y||^2 + lam ||D2 x||_1

    (piecewise-linear prior: D2 x is sparse, nonzero at amplifiers and
    anomalies).  lam descends along a warm-started path until the knot
    support reaches target_knots, i.e. just enough degrees of freedom
    for the amplifier steps, the within-span curvature, and a possible
    anomaly.  Pass the (possibly multi-capture-summed) normal equations.
    """
    d = d2_matrix(c_vec.size)
    lam = 10.0 * np.max(np.abs(c_vec))
    warm = None
    x = np.linalg.solve(m_mat, c_vec)
    for _ in range(60):
        x, knots, dual = admm_gen_lasso(m_mat, c_vec, d, lam, n_iter=1500,
                                        warm=warm)
        warm = (knots, dual)
        if int(np.sum(np.abs(knots) > 1e-8)) >= target_knots:
            break
        lam *= 0.7
    return x


def sparse_profile_subtraction_localize(x_sp_fault: np.ndarray,
                                        x_sp_ref: np.ndarray):
    """Method (d): sparse-regularize reference and fault profiles
    independently (absolute domain), then subtract and pick the largest
    drop of the relative change. Unlike the differential scheme, the
    sparsity prior here never sees the difference, so the two
    regularizations place their knots independently."""
    u = (x_sp_fault - x_sp_ref) / floor_ref(x_sp_ref)
    jump = np.diff(u)
    return jump, int(np.argmin(jump))


def floor_ref(x_ref: np.ndarray, rel_floor: float = 0.02) -> np.ndarray:
    """Clip the reference profile to a small positive floor.

    LS noise can push low-power bins negative; dividing by (or scaling
    columns with) a negative/near-zero reference corrupts the
    relative-change domain, so clip at rel_floor * max (well below any
    physical span power).
    """
    return np.maximum(x_ref, rel_floor * float(np.max(x_ref)))


def diff_matrix(m: int) -> np.ndarray:
    """Plain first-difference operator D ((m-1) x m)."""
    d = np.zeros((m - 1, m))
    idx = np.arange(m - 1)
    d[idx, idx] = -1.0
    d[idx, idx + 1] = 1.0
    return d


def admm_gen_lasso(m_mat: np.ndarray, c_vec: np.ndarray, f: np.ndarray,
                   lam: float, rho: float | None = None, n_iter: int = 500,
                   tol: float = 1e-6, warm=None):
    """Solve min_v 0.5 v'Mv - c'v + lam*||F v||_1 via ADMM.

    (Equivalent to 0.5||G v - r||^2 + lam||F v||_1 with M = Re(G'G),
    c = Re(G'r).)  Returns (v, z, u) where z is the sparse copy of
    F v -- use z for support detection, it is exactly sparse.  Pass
    warm=(z, u) to warm-start along a lambda path.
    """
    if rho is None:
        rho = float(np.mean(np.diag(m_mat)))
    a = m_mat + rho * (f.T @ f)
    chol = np.linalg.cholesky(a)

    def solve(b):
        return np.linalg.solve(chol.T, np.linalg.solve(chol, b))

    mdim = f.shape[0]
    z = np.zeros(mdim) if warm is None else warm[0]
    u = np.zeros(mdim) if warm is None else warm[1]
    v = solve(c_vec)
    for _ in range(n_iter):
        v = solve(c_vec + rho * f.T @ (z - u))
        fv = f @ v
        z = np.sign(fv + u) * np.maximum(np.abs(fv + u) - lam / rho, 0.0)
        u = u + fv - z
        # converged when the primal residual (split constraint) is met
        if np.linalg.norm(fv - z) <= tol * max(1.0, np.linalg.norm(fv)):
            break
    return v, z, u


def differential_sparse_localize(g: np.ndarray, y_fault: np.ndarray,
                                 x_ref: np.ndarray, z_grid_km: np.ndarray,
                                 max_jumps: int = 3):
    """Gemini-style differential sparse PPE (corrected formulation).

    Works in the *relative-change* domain u = dx / x_ref: a lumped loss
    with transmittance T gives u = -(1-T) * step(z > z_fault), so plain
    total variation ||D u||_1 is the exact sparsity basis, and EDFA gain
    steps are automatically normalized out.  Solves

        min_u 0.5|| G diag(x_ref) u - r ||^2 + lam ||D u||_1

    with r = y_fault - G x_ref (generalized Lasso via ADMM).  lam is
    chosen by a descending path: the largest value whose solution has a
    non-empty but small (<= max_jumps) jump support -- i.e. the
    sparsest non-trivial change-point set.  Returns (dx, jump, k_fault).
    """
    x_ref_pos = floor_ref(x_ref)
    r = y_fault - g @ x_ref
    b = g * x_ref_pos[None, :]
    m_mat = np.real(b.conj().T @ b)
    c_vec = np.real(b.conj().T @ r)
    d = diff_matrix(x_ref.size)
    # generous upper bound for lambda: scale of the correlation vector
    lam = 10.0 * np.max(np.abs(c_vec))
    warm = None
    best = None
    for _ in range(60):
        u, jump, dual = admm_gen_lasso(m_mat, c_vec, d, lam, n_iter=1500,
                                       warm=warm)
        warm = (jump, dual)
        nnz = int(np.sum(np.abs(jump) > 1e-8))
        if nnz >= 1 and best is None:
            best = (u, jump)  # sparsest non-trivial support
        if nnz >= max_jumps:
            break
        lam *= 0.7
    if best is None:
        best = (u, jump)
    u, jump = best
    k_fault = int(np.argmin(jump))  # most negative jump = loss onset
    return u * x_ref_pos, jump, k_fault


def naive_differential_localize(x_fault_ls: np.ndarray, x_ref: np.ndarray):
    """Baseline: subtract two estimated profiles, pick the biggest drop
    of the relative change. No sparsity prior."""
    u = (x_fault_ls - x_ref) / floor_ref(x_ref)
    jump = np.diff(u)
    return jump, int(np.argmin(jump))


def _step_signatures_scan(g_cols: np.ndarray, weights: np.ndarray,
                          s_tail: np.ndarray, r: np.ndarray):
    """Correlate r with negative-step signatures s_c = -(suffix + tail)."""
    weighted = g_cols * weights[None, :]
    suffix = np.cumsum(weighted[:, ::-1], axis=1)[:, ::-1]
    n = weights.size
    corr = np.empty(n)
    amp = np.empty(n)
    r_norm = np.linalg.norm(r)
    for i in range(n):
        s = -(suffix[:, i] + s_tail)
        num = np.real(np.vdot(s, r))
        den = np.real(np.vdot(s, s))
        corr[i] = num / (np.sqrt(den) * r_norm + 1e-30)
        amp[i] = max(num / (den + 1e-30), 0.0)
    return corr, amp


def matched_filter_localize(u_tx: np.ndarray, y_fault: np.ndarray,
                            x_ref: np.ndarray, z_grid_km: np.ndarray,
                            g_coarse: np.ndarray, link: LinkConfig,
                            sig: SignalConfig, half_window_km: float = 3.0,
                            fine_step_km: float = 0.1,
                            z_center_km: float | None = None):
    """Two-stage matched-filter scan of candidate fault positions.

    The differential signature of a unit-depth lumped loss at z_c is
    s(z_c) = -G @ [x_ref * step(z > z_c)] (RP1 model).  Stage 1 scans
    all coarse grid positions (skipped when z_center_km is given, e.g.
    the bin found by the sparse detector); stage 2 rebuilds signatures
    on a fine_step_km grid within +-half_window_km of that optimum.
    Returns (z_fine, corr_fine, z_hat, loss_db).
    """
    r = y_fault - g_coarse @ x_ref
    x_ref = floor_ref(x_ref)

    if z_center_km is None:
        # stage 1: global coarse scan on the existing grid
        corr_c, _ = _step_signatures_scan(g_coarse, x_ref,
                                          np.zeros(r.size, complex), r)
        z_c0 = float(z_grid_km[int(np.argmax(corr_c))])
    else:
        z_c0 = z_center_km

    # stage 2: fine scan around the coarse optimum
    dz = float(z_grid_km[1] - z_grid_km[0])
    z_lo = max(fine_step_km, z_c0 - half_window_km)
    z_hi = min(link.total_length_km - fine_step_km, z_c0 + half_window_km)
    z_win = np.arange(z_lo, z_hi + 1e-9, fine_step_km)
    g_win = build_g_matrix(u_tx, z_win, link, sig)
    x_ref_win = np.interp(z_win, z_grid_km, x_ref) * (fine_step_km / dz)
    tail_mask = z_grid_km > z_hi
    s_tail = g_coarse[:, tail_mask] @ x_ref[tail_mask]
    corr, amp = _step_signatures_scan(g_win, x_ref_win, s_tail, r)
    i_best = int(np.argmax(corr))
    loss_db = -10 * np.log10(max(1e-6, 1 - amp[i_best]))
    return z_win, corr, float(z_win[i_best]), loss_db
