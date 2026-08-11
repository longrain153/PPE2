"""Diagnostic: does a *perfect* reference profile remove the ~0.55-km
matched-filter bias?  Compares, on identical fault captures:
  (1) estimated reference (12-capture LS mean, as in the main run)
  (2) true physical profile as reference
  (3) true profile with a global scale fitted to the LS mean
Same seed/sequence as run_experiment.py so trial data match the report.
"""
import sys, time
import numpy as np

sys.path.insert(0, "/home/user/PPE2")
from ppe.link_sim import (LinkConfig, SignalConfig, gen_qam_waveform,
                          propagate, true_power_profile)
from ppe.ppe_core import (build_g_matrix, differential_sparse_localize,
                          ls_ppe, matched_filter_localize, rx_to_perturbation)

FAULT = 72.35
rng = np.random.default_rng(2026)
sig = SignalConfig(n_symbols=81920)
link_ok = LinkConfig()
link_bad = LinkConfig(fault_pos_km=FAULT, fault_loss_db=1.0)
z = np.arange(0.5, 150, 1.0)
t0 = time.time()


def capture(link):
    u = gen_qam_waveform(sig, rng)
    a = propagate(u, link, sig, rng, rx_snr_db=18.0)
    y = rx_to_perturbation(a, u, link, sig)
    g = build_g_matrix(u, z, link, sig)
    return u, y, g


xs = []
for i in range(12):
    _, y, g = capture(link_ok)
    xs.append(ls_ppe(g, y))
    print(f"ref {i + 1}/12 ({time.time() - t0:.0f}s)", flush=True)
x_ref = np.mean(xs, 0)

p0 = 1e-3 * 10 ** (sig.launch_power_dbm / 10)
x_true = link_ok.gamma_w_km * p0 * true_power_profile(z, link_ok, sig)
alpha = float(x_ref @ x_true / (x_true @ x_true))
print(f"scale fit alpha = {alpha:.4f} (LS mean vs physical truth)", flush=True)

for t in range(3):
    u, y, g = capture(link_bad)
    for label, ref in (("est-ref  ", x_ref),
                       ("true-ref ", x_true),
                       ("scaled-tr", alpha * x_true)):
        _, _, k = differential_sparse_localize(g, y, ref, z)
        _, _, zmf, loss = matched_filter_localize(
            u, y, ref, z, g, link_bad, sig, z_center_km=float(z[k]))
        print(f"trial{t + 1} {label}: sparse {z[k] - FAULT:+.2f} km | "
              f"MF {1e3 * (zmf - FAULT):+.0f} m | loss {loss:.3f} dB "
              f"({time.time() - t0:.0f}s)", flush=True)
