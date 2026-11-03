"""Partially coherent brightfield model and 3D quantitative phase retrieval, as in Descloux et al. 2018, SI §1–2.

  H(g)  = ∫ S(ω)/k_z δ(g − K) dk_i dk dω,  K = k − k_i             (eq. S19, Köhler illumination, Streibl)
  I(g)  = I_DC δ(g) + Γ(g) + Γ*(−g)                                   (eq. S21)
  Γ₊(g) = I(g) K(g_z),  K = 1 if g_z > g_z,c else 0, inside the CTF    (eq. S22–S23, plus the CTF-shaped mask of §2.2 step 3)
  φ(x)  = atan( α Im Γ₊ / (I₀ + α Re Γ₊) )                            (eq. S28)
The stack is mirrored along z before the FFT (§2.2 step 2).
"""
import numpy as np
from scipy import fft as sfft, ndimage

from .params import REG


def source_weight(lam_um, spectrum_nm=None):
    """Relative source spectrum S(λ) (per unit wavelength) for eq. S19. spectrum_nm: [[λ_nm, I], ...] or None (flat)."""
    if spectrum_nm is None:
        return np.ones_like(lam_um)
    tab = np.asarray(spectrum_nm, float)
    return np.interp(1e3 * lam_um, tab[:, 0], tab[:, 1])


def transfer_function(na_det, na_ill, n_medium, band_um, spectrum_nm=None, n_samples=2_000_000, pupil_phase=None, seed=0):
    """Polychromatic CTF H(|g⊥|, g_z) of eq. S19 by Monte-Carlo over wavelength, illumination and detection directions
    (uniform discs of NA_ill and NA_det), binned radially. pupil_phase(sinθ, λ) -> detection-pupil phase (rad), optional.
    Returns (radial bin edges, axial bin edges, complex H, support mask)."""
    rng = np.random.default_rng(seed)
    lam = rng.uniform(*band_um, n_samples); k = 2 * np.pi * n_medium / lam
    def disc(na):
        s = np.sqrt(rng.uniform(0, (na / n_medium) ** 2, n_samples)); a = rng.uniform(0, 2 * np.pi, n_samples)
        return s * np.cos(a), s * np.sin(a)
    ix, iy = disc(na_ill); dx, dy = disc(na_det)
    kzd, kzi = np.sqrt(1 - dx ** 2 - dy ** 2), np.sqrt(1 - ix ** 2 - iy ** 2)
    Kr = k * np.hypot(dx - ix, dy - iy); Kz = k * (kzd - kzi)
    w = source_weight(lam, spectrum_nm) / (k * kzd) + 0j                     # S(ω) / k_z
    if pupil_phase is not None:
        w = w * np.exp(1j * (pupil_phase(np.hypot(dx, dy), lam) - pupil_phase(np.hypot(ix, iy), lam)))
    rb = np.linspace(0, 2 * np.pi * n_medium / band_um[0] * (na_det + na_ill) / n_medium * 1.02, 140)
    zb = np.linspace(Kz.min() * 1.02, 0.8, 200)
    Hr = np.histogram2d(Kr, Kz, [rb, zb], weights=w.real)[0]; Hi = np.histogram2d(Kr, Kz, [rb, zb], weights=w.imag)[0]
    cnt = np.histogram2d(Kr, Kz, [rb, zb])[0]
    return rb, zb, Hr + 1j * Hi, ndimage.binary_dilation(cnt > 0, iterations=2)


def gz_cutoff(na_ill, band_um, n_medium, convention):
    """Axial cut-off g_z,c of eq. S24. The SI is not self-consistent: the text quotes ≈ 0.4 rad/µm for NA_ill 0.26 (formula
    without n̄), eq. S24 as printed has the factor n̄ (0.53 rad/µm, the value eq. S29's '≈ 18 planes' needs).
    convention: "si_text" (no n̄) or "s24" (with n̄)."""
    k0max = 2 * np.pi / band_um[0]
    base = k0max * (1 - np.sqrt(1 - na_ill ** 2))
    if convention == "si_text":
        return base
    if convention == "s24":
        return n_medium * base
    raise ValueError(convention)


def grids(nz, ny, nx, dz, px):
    gz = 2 * np.pi * sfft.fftfreq(nz, dz)
    gr = np.hypot(*np.meshgrid(2 * np.pi * sfft.fftfreq(ny, px), 2 * np.pi * sfft.fftfreq(nx, px), indexing="ij"))
    return gz, gr


def lookup(tf, gz_phys, gr):
    """Map the (|g⊥|, K_z) table onto a 2D (ny, nx) slice at axial frequency gz_phys: (H, support)."""
    rb, zb, H, supp = tf
    ir = np.clip(np.digitize(gr, rb) - 1, 0, len(rb) - 2); jz = np.digitize(gz_phys, zb) - 1
    if not 0 <= jz < len(zb) - 1:
        return np.zeros(gr.shape, complex), np.zeros(gr.shape, bool)
    return H[ir, jz], supp[ir, jz]


def cross_spectral_density(c, dz, px, tf, sgn, gz_c):
    """Γ₊(x) of a normalised stack c (background 1, axis 0 = z, planes in increasing focus depth for sgn = +1):
    mirror along z, 3D FFT, keep sgn·g_z > g_z,c inside the CTF support, inverse FFT, first half (§2.2 steps 2–4)."""
    nz, ny, nx = c.shape
    s = np.concatenate([c - 1, (c - 1)[::-1]], 0).astype(np.complex64)
    F = sfft.fftn(s, workers=-1)
    gz, gr = grids(2 * nz, ny, nx, dz, px)
    for iz, g in enumerate(gz):
        F[iz] = 0 if sgn * g <= gz_c else F[iz] * lookup(tf, -abs(g), gr)[1]
    return sfft.ifftn(F, workers=-1)[:nz]


def phase(G, alpha, I0=1.0):
    """Quantitative phase, eq. S28."""
    return np.arctan2(alpha * G.imag, I0 + alpha * G.real).astype(np.float32)


def normalise(stack, flat):
    """Intensity normalised to the reference (unscattered) intensity, background = 1. flat: 2D I₀(x, y) of the same channel."""
    return (np.asarray(stack, np.float32) / np.asarray(flat, np.float32)[None]).astype(np.float32)


def choose_sign(c, dz, px, tf, gz_c, alpha):
    """Which half of the axial spectrum holds Γ depends on the scan direction. Pick the sign that makes the strongest features
    positive (objects denser than the medium, e.g. polystyrene beads in water). Heuristic: declare its use where called."""
    score = {}
    for sg in (+1, -1):
        ph = phase(cross_spectral_density(c, dz, px, tf, sg, gz_c), alpha).max(0)
        score[sg] = np.percentile(ph, 99.95) - abs(np.percentile(ph, 0.05))
    return max(score, key=score.get)


def k0_dn_h(h_um, n_object, n_medium, lambda0_um):
    """Phase of a thin step of height h: k₀ Δn h (SI §4)."""
    return 2 * np.pi / lambda0_um * (n_object - n_medium) * h_um


def bead_phase_theory(d_um, n_bead, n_medium, lambda0_um, factor):
    """Expected peak phase of a bead smaller than the PSF: factor · k₀ Δn d (SI §4: factor ≈ 0.6 for 200 nm beads, 0.34 rad)."""
    return factor * k0_dn_h(d_um, n_bead, n_medium, lambda0_um)
