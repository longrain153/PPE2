"""Split-step Fourier simulation of a multi-span coherent link.

Single-polarization scalar NLSE, periodic boundary (FFT-based), lumped
EDFAs with ASE, optional lumped anomaly loss inside a span. Units:
km, ps, W (field in sqrt(W)).
"""

from dataclasses import dataclass

import numpy as np

C_LIGHT_KM_S = 2.99792458e5  # km/s
H_PLANCK = 6.62607015e-34    # J*s


@dataclass
class LinkConfig:
    n_spans: int = 3
    span_length_km: float = 50.0
    alpha_db_km: float = 0.2
    beta2_ps2_km: float = -21.4          # ~17 ps/nm/km at 1550 nm
    gamma_w_km: float = 1.3              # 1/(W*km)
    edfa_nf_db: float = 5.0
    carrier_freq_hz: float = 193.4e12
    ssfm_step_km: float = 0.1
    # anomaly: lumped loss inserted at fault_pos_km (absolute distance)
    fault_pos_km: float | None = None
    fault_loss_db: float = 0.0

    @property
    def total_length_km(self) -> float:
        return self.n_spans * self.span_length_km

    @property
    def alpha_np_km(self) -> float:
        return self.alpha_db_km / (10.0 / np.log(10.0))


@dataclass
class SignalConfig:
    baud_ghz: float = 64.0
    n_symbols: int = 8192
    sps: int = 4
    rolloff: float = 0.1
    mod_order: int = 16                  # square QAM
    launch_power_dbm: float = 6.0

    @property
    def fs_ghz(self) -> float:
        return self.baud_ghz * self.sps

    @property
    def dt_ps(self) -> float:
        return 1e3 / self.fs_ghz

    @property
    def n_samples(self) -> int:
        return self.n_symbols * self.sps


def rrc_freq_response(n: int, sps: int, rolloff: float) -> np.ndarray:
    """Root-raised-cosine filter, frequency domain, on the FFT grid."""
    f = np.fft.fftfreq(n, d=1.0)  # cycles/sample
    fn = np.abs(f) * sps          # normalized to symbol rate
    h = np.zeros(n)
    h[fn <= (1 - rolloff) / 2] = 1.0
    trans = (fn > (1 - rolloff) / 2) & (fn <= (1 + rolloff) / 2)
    if rolloff > 0:
        h[trans] = np.sqrt(
            0.5 * (1 + np.cos(np.pi / rolloff * (fn[trans] - (1 - rolloff) / 2)))
        )
    return h


def gen_qam_waveform(sig: SignalConfig, rng: np.random.Generator) -> np.ndarray:
    """Unit-average-power RRC-shaped QAM waveform (complex baseband)."""
    m_side = int(np.sqrt(sig.mod_order))
    levels = 2 * np.arange(m_side) - (m_side - 1)
    syms = rng.choice(levels, sig.n_symbols) + 1j * rng.choice(levels, sig.n_symbols)
    syms = syms / np.sqrt(np.mean(np.abs(syms) ** 2))
    up = np.zeros(sig.n_samples, dtype=complex)
    up[:: sig.sps] = syms
    u = np.fft.ifft(np.fft.fft(up) * rrc_freq_response(sig.n_samples, sig.sps, sig.rolloff))
    return u / np.sqrt(np.mean(np.abs(u) ** 2))


def dispersion_op(n: int, dt_ps: float, beta2_ps2_km: float, z_km: float) -> np.ndarray:
    """Frequency-domain linear propagation operator over z_km."""
    w = 2 * np.pi * np.fft.fftfreq(n, d=dt_ps)  # rad/ps
    return np.exp(0.5j * beta2_ps2_km * w**2 * z_km)


def _ssfm_section(a: np.ndarray, length_km: float, link: LinkConfig,
                  dt_ps: float) -> np.ndarray:
    """Propagate field through length_km of fiber (symmetric SSFM)."""
    n_steps = max(1, int(round(length_km / link.ssfm_step_km)))
    dz = length_km / n_steps
    half = dispersion_op(a.size, dt_ps, link.beta2_ps2_km, dz / 2)
    att_field = np.exp(-link.alpha_np_km * dz / 2)  # power decays e^{-alpha dz}
    for _ in range(n_steps):
        a = np.fft.ifft(np.fft.fft(a) * half)
        a = a * np.exp(1j * link.gamma_w_km * np.abs(a) ** 2 * dz) * att_field
        a = np.fft.ifft(np.fft.fft(a) * half)
    return a


def propagate(u_tx: np.ndarray, link: LinkConfig, sig: SignalConfig,
              rng: np.random.Generator, rx_snr_db: float | None = None) -> np.ndarray:
    """Transmit unit-power waveform over the link, return received field.

    EDFAs (constant gain = span loss) restore launch power after every
    span and add ASE. The anomaly is a lumped loss which is NOT
    compensated (constant-gain amplifiers), so it shadows the rest of
    the link. Optionally add receiver-side AWGN at rx_snr_db (in the
    full simulation bandwidth) to model transceiver noise.
    """
    p0 = 1e-3 * 10 ** (sig.launch_power_dbm / 10)
    a = np.sqrt(p0) * u_tx
    fs_hz = sig.fs_ghz * 1e9
    gain_lin = 10 ** (link.alpha_db_km * link.span_length_km / 10)
    nf_lin = 10 ** (link.edfa_nf_db / 10)
    # single-pol ASE power in the simulated bandwidth per EDFA
    p_ase = 0.5 * nf_lin * (gain_lin - 1) * H_PLANCK * link.carrier_freq_hz * fs_hz

    for s in range(link.n_spans):
        z0 = s * link.span_length_km
        z1 = z0 + link.span_length_km
        fault_here = (
            link.fault_pos_km is not None
            and z0 < link.fault_pos_km <= z1
            and link.fault_loss_db > 0
        )
        if fault_here:
            a = _ssfm_section(a, link.fault_pos_km - z0, link, sig.dt_ps)
            a = a * 10 ** (-link.fault_loss_db / 20)
            a = _ssfm_section(a, z1 - link.fault_pos_km, link, sig.dt_ps)
        else:
            a = _ssfm_section(a, link.span_length_km, link, sig.dt_ps)
        a = a * np.sqrt(gain_lin)
        noise = np.sqrt(p_ase / 2) * (
            rng.standard_normal(a.size) + 1j * rng.standard_normal(a.size)
        )
        a = a + noise

    if rx_snr_db is not None:
        p_sig = np.mean(np.abs(a) ** 2)
        p_n = p_sig / 10 ** (rx_snr_db / 10)
        a = a + np.sqrt(p_n / 2) * (
            rng.standard_normal(a.size) + 1j * rng.standard_normal(a.size)
        )
    return a


def true_power_profile(z_km: np.ndarray, link: LinkConfig, sig: SignalConfig) -> np.ndarray:
    """Analytic launch-power-normalized profile P(z)/P(0) (linear units)."""
    p = np.zeros_like(z_km, dtype=float)
    for i, z in enumerate(z_km):
        span = min(int(z // link.span_length_km), link.n_spans - 1)
        z_in = z - span * link.span_length_km
        val_db = -link.alpha_db_km * z_in
        if link.fault_pos_km is not None and z > link.fault_pos_km and link.fault_loss_db > 0:
            val_db -= link.fault_loss_db
        p[i] = 10 ** (val_db / 10)
    return p
