"""DAS: effect of a LOCALIZED SOP rotation (at 250 m) vs rotation rate.

A pure polarization rotator (unitary, no common phase -> no real path
length change) is inserted at 250 m, modelling a dynamically twisted /
disturbed cable section or a scrambler. All light serving channels
beyond 250 m passes it twice (reciprocal round trip):

    F_k(t) = J_k Q(t),  Q(t) = J250^-1 R(t) J250,  M_k(t) = F^T F

Channels before 250 m are untouched. Waterfalls for three rotation
rates x two receivers, a per-channel std profile, and a rate sweep of
the downstream phase-noise for single-pol vs pol-diversity RX.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---------------- parameters (match exp_das_sop.py) ----------------
NCH, DX = 1000, 2.0
NT, PRF = 600, 2000.0
GAUGE = 5
BIREF_STEP_RAD = 0.25
SNR_DB = 20.0
EVENTS = [(700.0, 6.0, 80.0, 2.0), (1400.0, 6.0, 160.0, 1.5)]
ROT_POS_M = 250.0
RATES_WF = [2.0, 20.0, 200.0]          # Poincare rev/s shown as waterfalls
RATES_SWEEP = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
OUT_PNG = "results/fig_das_sop_rotation_rate.png"

# optional CLI override: exp_das_sop_rotation.py fast -> high-rate /
# aliasing study (rates up to 2.5x PRF, output *_fast.png)
import sys

if len(sys.argv) > 1 and sys.argv[1] == "fast":
    RATES_WF = [500.0, 1000.0, 2000.0]
    RATES_SWEEP = [50, 100, 200, 500, 800, 1000, 1500, 1900, 2000, 2100,
                   3000, 5000]
    OUT_PNG = "results/fig_das_sop_rotation_rate_fast.png"
elif len(sys.argv) > 1 and sys.argv[1] == "all":
    RATES_WF = [2.0, 20.0, 200.0, 500.0, 1000.0, 2000.0]
    RATES_SWEEP = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 800, 1000,
                   1500, 1900, 2000, 2100, 3000, 5000]
    OUT_PNG = "results/fig_das_sop_rotation_rate_all.png"

t = np.arange(NT) / PRF
z = np.arange(NCH) * DX
K_ROT = int(ROT_POS_M / DX)

# ---------------- fiber realization ----------------
r_scat = (rng.standard_normal(NCH) + 1j * rng.standard_normal(NCH)) / np.sqrt(2)

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)


def su2(delta, axis):
    h = axis[0] * SX + axis[1] * SY + axis[2] * SZ
    return np.cos(delta / 2) * np.eye(2) - 1j * np.sin(delta / 2) * h


jones = np.empty((NCH, 2, 2), complex)
acc = np.eye(2, dtype=complex)
for k in range(NCH):
    ax = rng.standard_normal(3)
    ax /= np.linalg.norm(ax)
    acc = su2(rng.normal(0.0, BIREF_STEP_RAD), ax) @ acc
    jones[k] = acc
m_rt = jones.transpose(0, 2, 1) @ jones
j250 = jones[K_ROT - 1]
j250_inv = np.linalg.inv(j250)

phi_local = np.zeros((NT, NCH))
for z0, width, freq, amp in EVENTS:
    prof = np.exp(-0.5 * ((z - z0) / width) ** 2)
    phi_local += np.sin(2 * np.pi * freq * t)[:, None] * (prof / prof.sum() * amp)[None, :]
phi_cum = np.cumsum(phi_local, axis=1)

ROT_AXIS = np.array([0.36, 0.48, 0.80])  # fixed oblique axis, unit norm
P_IN = np.array([1.0, 0.0], complex)


def received_field(rate_rev_s: float) -> np.ndarray:
    """E[t, k, pol] with the rotator at 250 m spinning at rate_rev_s."""
    theta = 2 * np.pi * rate_rev_s * t
    q = np.empty((NT, 2, 2), complex)
    for i, th in enumerate(theta):
        q[i] = j250_inv @ su2(th, ROT_AXIS) @ j250
    v = np.einsum("tij,j->ti", q, P_IN)                     # Q p
    x = np.einsum("kij,tj->tki", m_rt, v)                   # M Q p
    e_down = np.einsum("tji,tkj->tki", q, x)                # Q^T M Q p
    e_up = np.einsum("kij,j->ki", m_rt, P_IN)[None, :, :] * np.ones((NT, 1, 1))
    e = np.where((np.arange(NCH) >= K_ROT)[None, :, None], e_down, e_up)
    e = e * (r_scat[None, :, None] * np.exp(2j * phi_cum)[:, :, None])
    sig_rms = np.sqrt(np.mean(np.abs(e) ** 2))
    sigma = sig_rms * 10 ** (-SNR_DB / 20)
    e = e + sigma / np.sqrt(2) * (rng.standard_normal(e.shape)
                                  + 1j * rng.standard_normal(e.shape))
    return e


def wf_single_pol(e):
    s = e[:, :, 0]
    h = s[:, GAUGE:] * np.conj(s[:, :-GAUGE])
    return np.angle(h[1:] * np.conj(h[:-1]))


def wf_pol_div(e):
    h = np.sum(e[:, GAUGE:, :] * np.conj(e[:, :-GAUGE, :]), axis=2)
    return np.angle(h[1:] * np.conj(h[:-1]))


z_mid = z[: NCH - GAUGE] + GAUGE * DX / 2
down_mask = z_mid > ROT_POS_M + 30
for z0, *_ in EVENTS:
    down_mask &= np.abs(z_mid - z0) > 40

# ---------------- figure ----------------
fig = plt.figure(figsize=(13, 4.2 * len(RATES_WF) + 3))
gs = fig.add_gridspec(len(RATES_WF) + 1, 2,
                      height_ratios=[2] * len(RATES_WF) + [1.6],
                      hspace=0.42, wspace=0.18)
stds = {}
im = None
for row, rate in enumerate(RATES_WF):
    e = received_field(rate)
    for col, (name, fn) in enumerate([("Single-pol RX", wf_single_pol),
                                      ("Pol-diversity RX (RVS)", wf_pol_div)]):
        wf = fn(e)
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(wf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-1.0, vmax=1.0,
                       extent=[t[1] * 1e3, t[-1] * 1e3, z[0], z[-GAUGE - 1]])
        ax.axhline(ROT_POS_M, color="m", ls="--", lw=1)
        ax.text(6, ROT_POS_M + 25, "SOP rotator 250 m", color="m", fontsize=8)
        for z0, _w, freq, _a in EVENTS:
            ax.annotate(f"{z0:.0f} m / {freq:.0f} Hz",
                        xy=(t[-1] * 1e3 * 0.985, z0),
                        xytext=(t[-1] * 1e3 * 0.74, z0 + 95), fontsize=8,
                        color="darkgreen",
                        bbox=dict(fc="white", alpha=0.8, ec="none"),
                        arrowprops=dict(arrowstyle="->", color="darkgreen"))
        std = np.std(wf, axis=0)
        stds[(rate, name)] = std
        ax.set_title(f"{name}, rotation {rate:g} rev/s", fontsize=11)
        ax.set_ylabel("distance [m]")
        if row == len(RATES_WF) - 1:
            ax.set_xlabel("time [ms]")
fig.colorbar(im, ax=fig.axes, shrink=0.5, label="phase rate [rad/frame]",
             location="right", pad=0.012)

# std vs distance (running median to suppress interference-fading spread)
def runmed(x, w=15):
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + w]) for i in range(x.size)])


ax1 = fig.add_subplot(gs[len(RATES_WF), 0])
cmap = plt.get_cmap("viridis")
for i, rate in enumerate(RATES_WF):
    col = cmap(0.15 + 0.7 * i / (len(RATES_WF) - 1))
    ax1.plot(z_mid, runmed(stds[(rate, "Single-pol RX")]), color=col, lw=1.4,
             label=f"single-pol, {rate:g} rev/s")
    ax1.plot(z_mid, runmed(stds[(rate, "Pol-diversity RX (RVS)")]), color=col,
             lw=1.4, ls="--", label=f"pol-div, {rate:g} rev/s")
ax1.axvline(ROT_POS_M, color="m", ls="--", lw=1)
for z0, *_ in EVENTS:
    ax1.axvline(z0, color="k", ls=":", lw=0.8)
ax1.set_yscale("log"), ax1.set_xlabel("distance [m]")
ax1.set_ylabel("phase-rate std [rad]\n(15-ch running median)")
ax1.set_title("std vs distance (magenta: rotator; dotted: events)", fontsize=10)
ax1.legend(fontsize=7, ncol=2), ax1.grid(alpha=0.3)

# rate sweep: downstream median std
ax2 = fig.add_subplot(gs[len(RATES_WF), 1])
sweep_sp, sweep_dv = [], []
for rate in RATES_SWEEP:
    e = received_field(rate)
    sweep_sp.append(np.median(np.std(wf_single_pol(e), axis=0)[down_mask]))
    sweep_dv.append(np.median(np.std(wf_pol_div(e), axis=0)[down_mask]))
ax2.loglog(RATES_SWEEP, sweep_sp, "o-", color="tab:blue", label="single-pol RX")
ax2.loglog(RATES_SWEEP, sweep_dv, "s-", color="tab:red",
           label="pol-diversity RX (RVS)")
ax2.axhline(sweep_dv[0], color="gray", ls=":", lw=1,
            label="receiver-noise floor")
ax2.set_xlabel("SOP rotation rate [Poincare rev/s]")
ax2.set_ylabel("downstream median phase-rate std [rad]")
ax2.set_title("Downstream (>250 m) phase noise vs rotation rate", fontsize=10)
ax2.legend(fontsize=8), ax2.grid(alpha=0.3, which="both")

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"saved {OUT_PNG}")
for rate, sp, dv in zip(RATES_SWEEP, sweep_sp, sweep_dv):
    print(f"rate {rate:6g} rev/s: single-pol {sp:.3f} rad | pol-div {dv:.3f} rad")
