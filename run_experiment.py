"""Reproduction of differential sparse PPE anomaly localization.

Scenario (mirrors the Gemini dialogue, Q4-Q6):
  1. Healthy link: many captures -> low-noise reference profile x_ref.
  2. A lumped loss appears mid-span; only a few captures available.
  3. Localize the fault from the differential signal
     r = y_fault - G x_ref with a sparsity prior on the
     attenuation-weighted first difference (generalized Lasso / ADMM),
     then refine on a 100-m grid with a matched filter.

Four methods are compared, on both a 1-dB and a 3-dB lumped loss
(shared healthy reference):
  (a) naive LS profile subtraction
  (b) sparse PPE profile subtraction: NTT-style D2 sparse
      regularization applied independently to the reference and fault
      absolute profiles, then subtracted
  (c) differential sparse (generalized Lasso on the differential)
  (d) matched-filter refinement of (c) on a 100-m grid
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppe.link_sim import (LinkConfig, SignalConfig, gen_qam_waveform,
                          propagate, true_power_profile)
from ppe.ppe_core import (build_g_matrix, differential_sparse_localize,
                          floor_ref, matched_filter_localize,
                          naive_differential_localize, normal_eqs,
                          rx_to_perturbation, sparse_absolute_profile,
                          sparse_profile_subtraction_localize)

RESULTS = "results"

FAULT_POS_KM = 72.35
FAULT_LOSSES_DB = (1.0, 3.0)
N_REF_CAPTURES = 12
N_FAULT_TRIALS = 4
RX_SNR_DB = 18.0
DZ_KM = 1.0
N_SYMBOLS = 81920  # 10x the original 8192 for a smoother reference


def capture(link, sig, rng, z_grid):
    """One capture: fresh data pattern, propagate, extract y, G, and
    the real-constrained normal equations."""
    u_tx = gen_qam_waveform(sig, rng)
    a_rx = propagate(u_tx, link, sig, rng, rx_snr_db=RX_SNR_DB)
    y = rx_to_perturbation(a_rx, u_tx, link, sig)
    g = build_g_matrix(u_tx, z_grid, link, sig)
    m_mat, c_vec = normal_eqs(g, y)
    return u_tx, y, g, m_mat, c_vec


def main():
    import os

    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(2026)
    sig = SignalConfig(n_symbols=N_SYMBOLS)
    link_ok = LinkConfig()
    z_grid = np.arange(DZ_KM / 2, link_ok.total_length_km, DZ_KM)
    nbins = z_grid.size

    # ---- stage 1: reference profiles from many healthy captures ----
    t0 = time.time()
    x_list = []
    m_sum = np.zeros((nbins, nbins))
    c_sum = np.zeros(nbins)
    for i in range(N_REF_CAPTURES):
        _, _, _, m_mat, c_vec = capture(link_ok, sig, rng, z_grid)
        x_list.append(np.linalg.solve(m_mat, c_vec))
        m_sum += m_mat
        c_sum += c_vec
        print(f"ref capture {i + 1}/{N_REF_CAPTURES} ({time.time() - t0:.0f}s)")
    x_ref = np.mean(x_list, axis=0)
    x_single = x_list[0]
    xs_ref = sparse_absolute_profile(m_sum, c_sum)  # sparse-regularized ref
    print(f"sparse reference done ({time.time() - t0:.0f}s)")

    to_db = lambda x: 10 * np.log10(np.maximum(np.abs(x), 1e-12))
    p_true_ok = true_power_profile(z_grid, link_ok, sig)
    ref_db_off = to_db(x_ref)[0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(z_grid, to_db(p_true_ok), "k--", lw=1, label="true profile (healthy)")
    ax.plot(z_grid, to_db(x_single) - ref_db_off, color="tab:blue", alpha=0.5,
            label="LS-PPE, 1 capture")
    ax.plot(z_grid, to_db(x_ref) - ref_db_off, color="tab:red",
            label=f"LS-PPE, {N_REF_CAPTURES}-capture reference")
    ax.plot(z_grid, to_db(xs_ref) - ref_db_off, color="tab:green", lw=1.2,
            label="sparse-regularized reference (D2)")
    ax.set_xlabel("distance [km]"), ax.set_ylabel("relative power [dB]")
    ax.set_title("Reference power profile (healthy link)")
    ax.legend(), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(f"{RESULTS}/fig1_reference_profile.png", dpi=150)

    # ---- stage 2: fault trials for each loss depth ----
    summary = {}
    for loss_db in FAULT_LOSSES_DB:
        tag = f"{loss_db:.0f}dB"
        link_bad = LinkConfig(fault_pos_km=FAULT_POS_KM, fault_loss_db=loss_db)
        rows = []
        keep = None
        for t in range(N_FAULT_TRIALS):
            u_tx, y_f, g_f, m_f, c_f = capture(link_bad, sig, rng, z_grid)
            x_f_ls = np.linalg.solve(m_f, c_f)

            jump_naive, k_naive = naive_differential_localize(x_f_ls, x_ref)
            xs_f = sparse_absolute_profile(m_f, c_f)
            jump_ps, k_ps = sparse_profile_subtraction_localize(xs_f, xs_ref)
            dx, jump_sp, k_sp = differential_sparse_localize(
                g_f, y_f, x_ref, z_grid)
            z_fine, corr, z_mf, loss_mf = matched_filter_localize(
                u_tx, y_f, x_ref, z_grid, g_f, link_bad, sig,
                z_center_km=float(z_grid[k_sp]))

            rows.append(dict(
                naive=z_grid[k_naive] - FAULT_POS_KM,
                psub=z_grid[k_ps] - FAULT_POS_KM,
                sparse=z_grid[k_sp] - FAULT_POS_KM,
                mf=z_mf - FAULT_POS_KM,
                loss_mf=loss_mf,
            ))
            print(f"[{tag}] trial {t + 1}: naive {rows[-1]['naive']:+.2f} km | "
                  f"sparse-prof-sub {rows[-1]['psub']:+.2f} km | "
                  f"diff-sparse {rows[-1]['sparse']:+.2f} km | "
                  f"MF {rows[-1]['mf'] * 1e3:+.0f} m | "
                  f"loss {loss_mf:.2f} dB ({time.time() - t0:.0f}s)")
            if t == 0:
                keep = (x_f_ls, xs_f, dx, jump_naive, jump_ps, jump_sp,
                        z_fine, corr, z_mf)

        err = {k: np.array([r[k] for r in rows])
               for k in ("naive", "psub", "sparse", "mf")}
        loss_est = np.array([r["loss_mf"] for r in rows])
        summary[tag] = (err, loss_est)
        print(f"\n=== [{tag}] localization error "
              f"(mean abs / max abs, {N_FAULT_TRIALS} trials) ===")
        for k, label in (
                ("naive", "naive LS profile subtraction"),
                ("psub", "sparse PPE profile subtraction (D2)"),
                ("sparse", "differential sparse (gen-Lasso, 1-km grid)"),
                ("mf", "matched-filter refinement (100-m grid)")):
            print(f"{label:45s}: {np.mean(np.abs(err[k])):.3f} km / "
                  f"{np.max(np.abs(err[k])):.3f} km")
        print(f"{'loss estimate (matched filter)':45s}: "
              f"{loss_est.min():.2f}-{loss_est.max():.2f} dB (true {loss_db})\n")

        # ------------- figures for this loss depth -------------
        (x_f_ls, xs_f, dx, jump_naive, jump_ps, jump_sp,
         z_fine, corr, z_mf) = keep
        link_bad_p = true_power_profile(z_grid, link_bad, sig)

        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axs[0].plot(z_grid, to_db(link_bad_p) - to_db(p_true_ok), "k--", lw=1.2,
                    label="true differential (dB)")
        x_ref_pos = floor_ref(x_ref)
        xs_ref_pos = floor_ref(xs_ref)
        naive_rel = (x_f_ls - x_ref) / x_ref_pos
        psub_rel = (xs_f - xs_ref) / xs_ref_pos
        sparse_rel = dx / x_ref_pos
        axs[0].plot(z_grid, 10 * np.log10(np.maximum(1 + naive_rel, 1e-3)),
                    color="tab:blue", alpha=0.45, label="naive: LS(fault) - reference")
        axs[0].plot(z_grid, 10 * np.log10(np.maximum(1 + psub_rel, 1e-3)),
                    color="tab:orange", lw=1.5,
                    label="sparse profile subtraction (D2 each, then subtract)")
        axs[0].plot(z_grid, 10 * np.log10(np.maximum(1 + sparse_rel, 1e-3)),
                    color="tab:red", lw=2, label="differential sparse (gen-Lasso)")
        axs[0].axvline(FAULT_POS_KM, color="g", ls=":",
                       label=f"true fault {FAULT_POS_KM} km")
        axs[0].set_ylabel("power change [dB]"), axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.3), axs[0].set_ylim(-2 * loss_db - 3, 6)
        axs[0].set_title(f"Differential profile, {loss_db}-dB lumped loss, "
                         "single post-fault capture")
        axs[1].plot(z_grid[:-1], jump_naive, color="tab:blue", alpha=0.35,
                    label="jump indicator, naive (left axis)")
        axs[1].plot(z_grid[:-1], jump_ps, color="tab:orange", lw=1.5,
                    label="jump indicator, sparse profile subtraction (left axis)")
        axs[1].axvline(FAULT_POS_KM, color="g", ls=":")
        axs[1].set_xlabel("distance [km]")
        axs[1].set_ylabel("naive / prof-sub: diff of rel. change",
                          color="tab:blue")
        axs[1].tick_params(axis="y", labelcolor="tab:blue")
        ax_sp = axs[1].twinx()
        nz = np.abs(jump_sp) > 1e-8
        ml, sl, bl = ax_sp.stem(z_grid[:-1][nz], jump_sp[nz], basefmt=" ",
                                label="jump indicator, diff sparse (right axis)")
        plt.setp(ml, color="tab:red", markersize=9)
        plt.setp(sl, color="tab:red", lw=2.5)
        ax_sp.axhline(0, color="tab:red", lw=0.8, alpha=0.5)
        lim = max(1.5 * np.max(np.abs(jump_sp)), 1e-3)
        ax_sp.set_ylim(-lim, lim)
        ax_sp.set_ylabel("diff sparse: jump of Δx/x_ref", color="tab:red")
        ax_sp.tick_params(axis="y", labelcolor="tab:red")
        k_nz = np.argmin(jump_sp)
        ax_sp.annotate(
            f"detected fault (argmin):\n{z_grid[k_nz]:.1f} km, {jump_sp[k_nz]:.3f}\n"
            f"({int(np.sum(nz))} nonzero of {jump_sp.size} bins)",
            xy=(z_grid[k_nz], jump_sp[k_nz]),
            xytext=(z_grid[k_nz] + 12, jump_sp[k_nz] * 0.55),
            color="tab:red", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="tab:red"))
        h1, l1 = axs[1].get_legend_handles_labels()
        h2, l2 = ax_sp.get_legend_handles_labels()
        axs[1].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        axs[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{RESULTS}/fig2_differential_localization_{tag}.png", dpi=150)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(z_fine, corr, color="tab:purple")
        ax.axvline(FAULT_POS_KM, color="g", ls=":",
                   label=f"true fault {FAULT_POS_KM} km")
        ax.axvline(z_mf, color="tab:purple", ls="--",
                   label=f"matched-filter estimate {z_mf:.2f} km")
        ax.set_xlabel("candidate fault position [km]")
        ax.set_ylabel("normalized correlation")
        ax.set_title(f"Matched-filter refinement on 100-m grid "
                     f"({loss_db}-dB loss, trial 1)")
        ax.legend(), ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{RESULTS}/fig3_matched_filter_{tag}.png", dpi=150)

    print(f"figures saved in {RESULTS}/, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
