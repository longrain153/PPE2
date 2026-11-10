"""DAS (phi-OTDR) simulation: effect of SOP on the phase waterfall.

Model
-----
* Discrete Rayleigh scatterer model: one complex circular-Gaussian
  backscatter coefficient per channel (natural interference fading).
* Distributed birefringence: per-channel Jones matrix built as a
  cumulative product of small random SU(2) rotations; round trip uses
  reciprocity, M_k = J_k^T J_k.
* Two vibration events (80 Hz @ 700 m, 25 Hz @ 1400 m) enter the
  cumulative optical phase; demodulation uses a spatial gauge and
  frame-to-frame phase rate (what commercial DAS waterfalls display).
* Receivers: single-polarization (x only) vs polarization-diversity
  with rotated-vector-sum (RVS) combining h = sum_pol E_{k+g} E_k*.
* SOP conditions: static input SOP vs dynamically rotating input SOP
  (wanders over the Poincare sphere during the record).

Output: results/fig_das_sop_waterfall.png -- 2x2 waterfalls plus
per-channel phase-rate std, and printed bad-channel statistics.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---------------- parameters ----------------
NCH, DX = 1000, 2.0          # channels, channel pitch [m]
NT, PRF = 600, 2000.0        # frames, pulse repetition rate [Hz]
GAUGE = 5                    # gauge in channels (10 m)
BIREF_STEP_RAD = 0.25        # birefringence random-walk per channel
SNR_DB = 20.0                # receiver SNR on a healthy channel
EVENTS = [                   # (position m, width m, freq Hz, total rad)
    (700.0, 6.0, 80.0, 2.0),
    (1400.0, 6.0, 160.0, 1.5),
]

t = np.arange(NT) / PRF
z = np.arange(NCH) * DX

# ---------------- fiber realization ----------------
r_scat = (rng.standard_normal(NCH) + 1j * rng.standard_normal(NCH)) / np.sqrt(2)

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)


def su2(delta: float, axis: np.ndarray) -> np.ndarray:
    h = axis[0] * SX + axis[1] * SY + axis[2] * SZ
    return np.cos(delta / 2) * np.eye(2) - 1j * np.sin(delta / 2) * h


jones = np.empty((NCH, 2, 2), complex)
acc = np.eye(2, dtype=complex)
for k in range(NCH):
    ax = rng.standard_normal(3)
    ax /= np.linalg.norm(ax)
    acc = su2(rng.normal(0.0, BIREF_STEP_RAD), ax) @ acc
    jones[k] = acc
m_rt = jones.transpose(0, 2, 1) @ jones  # reciprocal round trip J^T J

# ---------------- vibration phase ----------------
phi_local = np.zeros((NT, NCH))
for z0, width, freq, amp in EVENTS:
    prof = np.exp(-0.5 * ((z - z0) / width) ** 2)
    prof = prof / prof.sum() * amp
    phi_local += np.sin(2 * np.pi * freq * t)[:, None] * prof[None, :]
phi_cum = np.cumsum(phi_local, axis=1)  # one-way cumulative phase


# ---------------- input SOP trajectories ----------------
def sop_static(_t):
    p = np.zeros((_t.size, 2), complex)
    p[:, 0] = 1.0
    return p


def sop_dynamic(_t):
    """Wanders over the Poincare sphere during the record (~3 rad/s)."""
    a = 2 * np.pi * 0.9 * _t          # polar sweep
    b = 2 * np.pi * 0.35 * _t         # relative-phase sweep
    return np.stack([np.cos(a), np.sin(a) * np.exp(1j * b)], axis=1)


# ---------------- receive + demodulate ----------------
def received_field(p_in: np.ndarray) -> np.ndarray:
    """E[t, k, pol] = r_k e^{j2phi} (M_k p(t)) + noise."""
    mp = np.einsum("kij,tj->tki", m_rt, p_in)           # (NT, NCH, 2)
    e = mp * (r_scat * np.ones((NT, 1)) * np.exp(2j * phi_cum))[:, :, None]
    sig_rms = np.sqrt(np.mean(np.abs(e) ** 2))
    sigma = sig_rms * 10 ** (-SNR_DB / 20)
    e += sigma / np.sqrt(2) * (rng.standard_normal(e.shape)
                               + 1j * rng.standard_normal(e.shape))
    return e


def waterfall_single_pol(e: np.ndarray) -> np.ndarray:
    s = e[:, :, 0]                                       # x branch only
    h = s[:, GAUGE:] * np.conj(s[:, :-GAUGE])            # gauge interferometer
    return np.angle(h[1:] * np.conj(h[:-1]))             # frame-to-frame rate


def waterfall_pol_diversity(e: np.ndarray) -> np.ndarray:
    h = np.sum(e[:, GAUGE:, :] * np.conj(e[:, :-GAUGE, :]), axis=2)  # RVS
    return np.angle(h[1:] * np.conj(h[:-1]))


scenarios = [
    ("Single-pol RX, static SOP", sop_static, waterfall_single_pol),
    ("Single-pol RX, rotating SOP", sop_dynamic, waterfall_single_pol),
    ("Pol-diversity RX (RVS), static SOP", sop_static, waterfall_pol_diversity),
    ("Pol-diversity RX (RVS), rotating SOP", sop_dynamic, waterfall_pol_diversity),
]

fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.2], hspace=0.35, wspace=0.18)
axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
ax_std = fig.add_subplot(gs[2, :])

stats = {}
for ax, (title, sop_fn, demod_fn) in zip(axes, scenarios):
    wf = demod_fn(received_field(sop_fn(t)))
    im = ax.imshow(wf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-1.0, vmax=1.0,
                   extent=[t[1] * 1e3, t[-1] * 1e3, z[0], z[-GAUGE - 1]])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("time [ms]"), ax.set_ylabel("distance [m]")
    std = np.std(wf, axis=0)
    stats[title] = std
    bad = np.mean(std > 1.0) * 100
    ax.text(0.015, 0.97, f"channels with phase-rate std > 1 rad: {bad:.1f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", alpha=0.85, ec="none"))
fig.colorbar(im, ax=axes, shrink=0.6, label="phase rate [rad/frame]",
             location="right", pad=0.012)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for (title, std), col in zip(stats.items(), colors):
    ax_std.plot(z[: NCH - GAUGE], std, color=col, lw=1, alpha=0.85, label=title)
for z0, *_ in EVENTS:
    ax_std.axvline(z0, color="k", ls=":", lw=1)
ax_std.set_yscale("log")
ax_std.set_xlabel("distance [m]")
ax_std.set_ylabel("phase-rate std [rad]")
ax_std.set_title("Per-channel phase-rate std (events at dotted lines)")
ax_std.legend(fontsize=8, ncol=2), ax_std.grid(alpha=0.3)

fig.savefig("results/fig_das_sop_waterfall.png", dpi=150, bbox_inches="tight")
print("saved results/fig_das_sop_waterfall.png")
for title, std in stats.items():
    print(f"{title:42s}: median std {np.median(std):.3f} rad | "
          f">1 rad channels {np.mean(std > 1.0) * 100:5.1f}%")
