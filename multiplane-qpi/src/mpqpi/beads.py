"""Polystyrene-bead validation of 3D quantitative phase (Descloux et al. 2018, SI §3.2–3.3, §4, §5, Fig. S1d,e, S3, S4e).

  forward model      point scatterer (or sphere) ⊗ H(g) -> Γ(x), I = 1 + 2 Re Γ          (SI §3.2, eq. S18)
  bead phase         per-bead peak of the retrieved phase, mean ± SD over beads            (SI §4: 0.33 ± 0.014 rad, 15 beads)
  bead average       re-centred (exact Fourier sub-voxel shifts) and averaged xz sections   (SI §3.3, Fig. S3a,b)
  single shot        8 registered planes ordered by Δ, retrieved as one volume              (SI §5, §6.4)
The SI does not say how beads were detected, selected or read out: every such choice is an explicit argument that the notebook
declares as a heuristic. Functions marked "extension" are not in the SI.
"""
import json
from pathlib import Path
import numpy as np
from scipy import fft as sfft, ndimage
from scipy.spatial import cKDTree

from . import qpi, geometry


# ------------------------------------------------------------------------------------------------------------- inputs
def read_alpha(path):
    """α of eq. S28 from notebook 03's alpha.json (SI §4). Accepts {"alpha": x} or {"alpha": {"value": x, "sd": s}}.
    Returns (alpha, sd or None) or None if the file does not exist."""
    if path is None or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    a = d["alpha"]
    if isinstance(a, dict):
        u = a.get("uncertainty", a.get("sd"))                       # notebook 03 writes "uncertainty"
        return float(a["value"]), (None if u is None else float(u))
    return float(a), (None if d.get("alpha_sd") is None else float(d["alpha_sd"]))


def flat_field(stack, sigma_px, statistic="median"):
    """Per-channel reference intensity I₀(x, y) from the channel's own brightfield stack (SI §6.4: 'for phase imaging, the
    channels transmission correction is based on bright field images'; the SI gives no procedure): z-median (or mean) of the
    offset-subtracted stack, Gaussian-smoothed with sigma_px so that beads and their defocus rings do not imprint on it."""
    s = np.asarray(stack, np.float32)
    m = np.median(s, 0) if statistic == "median" else s.mean(0)
    return ndimage.gaussian_filter(m, sigma_px, mode="nearest").astype(np.float32)


def frame_means(stack):
    """Per-frame mean / overall mean (lamp stability over the scan; reported, not corrected unless the notebook says so)."""
    m = np.asarray(stack, np.float32).mean((1, 2))
    return m / m.mean()


# --------------------------------------------------------------------------------------------- spectrum vs CTF (Fig. S1d,e)
def spectrum_rz(stacks, dz, px, r_edges):
    """Azimuthally averaged |3D FFT| of the z-mirrored (c − 1) stacks, averaged over channels (SI Fig. S1d: measured spectrum).
    Each frame's mean is removed first (g⊥ = 0 holds only lamp drift, which the CTF mask removes anyway, and would dominate).
    Returns g_z (fftshifted, rad/µm), radial bin centres (rad/µm), spec (n_gz, n_r)."""
    nz, ny, nx = stacks[0].shape
    gz = 2 * np.pi * sfft.fftshift(sfft.fftfreq(2 * nz, dz))
    gr = qpi.grids(2 * nz, ny, nx, dz, px)[1]
    ir = np.digitize(gr.ravel(), r_edges) - 1
    inside = (ir >= 0) & (ir < len(r_edges) - 1)
    cnt = np.bincount(ir[inside], minlength=len(r_edges) - 1)
    spec = np.zeros((2 * nz, len(r_edges) - 1))
    for c in stacks:
        d = c - c.mean((1, 2), keepdims=True)
        s = np.concatenate([d, d[::-1]], 0).astype(np.complex64)
        F = np.abs(sfft.fftshift(sfft.fftn(s, workers=-1), axes=0))
        for iz in range(2 * nz):
            spec[iz] += np.bincount(ir[inside], F[iz].ravel()[inside], len(r_edges) - 1) / np.maximum(cnt, 1)
    return gz, 0.5 * (r_edges[1:] + r_edges[:-1]), spec / len(stacks)


def support_rz(tf, gz, rcen):
    """CTF support of eq. S19 on the (g_z, |g⊥|) grid of spectrum_rz, both halves (the mirrored stack has an even spectrum)."""
    return np.array([qpi.lookup(tf, -abs(g), rcen[None, :])[1][0] for g in gz])


def support_check(spec, supp, gz, gz_c, noise_rows):
    """Fraction of the above-noise spectral power (|g_z| > g_z,c) that lies inside the CTF support. The noise power per radial
    bin is the mean of spec² over noise_rows (rows beyond the axial extent of the CTF) and is subtracted without clipping, so
    that pure-noise regions sum to ≈ 0. A value well below 1 means NA_det / NA_ill / pixel / Δz do not describe the data."""
    ex = spec ** 2 - np.mean(spec[noise_rows] ** 2, 0)[None]
    rows = np.abs(gz) > gz_c
    tot = ex[rows].sum()
    return float(ex[rows][supp[rows]].sum() / tot) if tot > 0 else np.nan


# --------------------------------------------------------------------------------------------------- forward model (§3.2)
def sphere_ft(g, radius_um):
    """Fourier transform of a uniform sphere, normalised to 1 at g = 0."""
    u = np.maximum(np.asarray(g) * radius_um, 1e-6)
    return 3 * (np.sin(u) - u * np.cos(u)) / u ** 3


def simulate_stack(tf, shape, dz, px, beads_zyx_um, sgn, amplitude, obj="point", d_um=None, pad_z=None, amplitudes=None):
    """SI §3.2: scatterers ⊗ H(g) -> Γ(x); intensity I = |1 + Γ|² ≈ 1 + 2 Re Γ (weak scattering, eq. S18).
    shape (nz, ny, nx); beads_zyx_um: (n, 3) continuous positions (µm) from voxel (0, 0, 0); obj "point" (Dirac, SI §3.2) or
    "sphere" (d_um). The object is placed by exact Fourier phase ramps (no interpolation). pad_z (frames, default nz) extends
    the periodic z grid so that beads do not wrap around. amplitude: max |Γ| of one isolated bead (amplitudes: per bead).
    H is read at K_z = −sgn·g_z so that the stack is retrieved with the same sgn. Returns I (nz, ny, nx) float32."""
    nz, ny, nx = shape
    pad = nz if pad_z is None else pad_z
    nzl = nz + 2 * pad
    gz, gr = qpi.grids(nzl, ny, nx, dz, px)
    gy, gx = 2 * np.pi * sfft.fftfreq(ny, px), 2 * np.pi * sfft.fftfreq(nx, px)
    H = np.stack([qpi.lookup(tf, -sgn * g, gr)[0] for g in gz])
    if obj == "sphere":
        H = H * sphere_ft(np.sqrt(gr[None] ** 2 + gz[:, None, None] ** 2), d_um / 2)
    elif obj != "point":
        raise ValueError(obj)
    norm = np.abs(sfft.ifftn(1j * H, workers=-1)).max()
    beads = np.atleast_2d(np.asarray(beads_zyx_um, float))
    amps = np.full(len(beads), amplitude) if amplitudes is None else np.asarray(amplitudes, float)
    ramp = np.zeros(H.shape, complex)
    for (z, y, x), a in zip(beads, amps):
        ramp += a * (np.exp(-1j * gz * (z + pad * dz))[:, None, None] * np.exp(-1j * gy * y)[None, :, None]
                     * np.exp(-1j * gx * x)[None, None, :])
    gam = sfft.ifftn(1j * H * ramp / norm, workers=-1)
    return (1 + 2 * gam.real[pad:pad + nz]).astype(np.float32)


# --------------------------------------------------------------------------------------- detection and per-bead readout
def detect(mp, k_sigma, footprint_px, edge_px):
    """Local maxima of a phase max projection above median + k_sigma · robust σ (1.4826 MAD), away from the border.
    SI: no procedure given. Returns (n, 2) integer (x, y)."""
    med = np.median(mp); sig = 1.4826 * np.median(np.abs(mp - med))
    pk = (mp == ndimage.maximum_filter(mp, footprint_px)) & (mp > med + k_sigma * sig)
    pk[:edge_px] = pk[mp.shape[0] - edge_px:] = False
    pk[:, :edge_px] = pk[:, mp.shape[1] - edge_px:] = False
    y, x = np.nonzero(pk)
    return np.c_[x, y]


def _parabolic_offsets(v, k, y, x):
    """Sub-voxel offset of a maximum along each axis from the 3-point parabola through (k, y, x), clipped to ±0.5."""
    o = []
    for lo, c0, hi in ((v[k - 1, y, x], v[k, y, x], v[k + 1, y, x]), (v[k, y - 1, x], v[k, y, x], v[k, y + 1, x]),
                       (v[k, y, x - 1], v[k, y, x], v[k, y, x + 1])):
        den = lo - 2 * c0 + hi
        o.append(float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5)) if den < 0 else 0.0)
    return np.array(o)


def peak_fourier(vol, k, y, x, half_kyx, n_iter=2, locate_sigma=0.0, field=None, alpha=None):
    """Peak value and sub-voxel position of a local maximum at voxel (k, y, x) by band-limited (Fourier) interpolation.
    A crop of ±half_kyx voxels is shifted in Fourier space by the sub-voxel offset (3-point parabolas, refined n_iter times,
    found on the crop Gaussian-smoothed by locate_sigma voxels so that the position does not follow the noise of single voxels).
    The value is read at the centre of the shifted crop: of `vol` itself, or — better — of the complex Γ₊ (`field`, same grid)
    turned into a phase with eq. S28 and `alpha`. Γ₊ is linear and band-limited (CTF mask), whereas the phase is a non-linear
    function of it: on a simulated 200 nm bead with α Γ₊ ≈ 0.3 the Γ₊ route is stable to 0.4 % for any sub-voxel offset,
    interpolating the phase overshoots by up to 2 %, the max voxel misses up to −28 %.
    Returns (value, kf, yf, xf, ok, g): g = interpolated Γ₊ at the vertex (NaN without field); ok = False (voxel values,
    offsets 0) when the crop leaves the volume."""
    h = np.asarray(half_kyx, int); c = np.array([k, y, x])
    if np.any(c - h < 0) or np.any(c + h >= vol.shape):
        if field is None:
            return float(vol[k, y, x]), float(k), float(y), float(x), False, complex(np.nan)
        g = complex(field[k, y, x])
        return float(qpi.phase(np.asarray([g]), alpha)[0]), float(k), float(y), float(x), False, g
    sl = (slice(k - h[0], k + h[0] + 1), slice(y - h[1], y + h[1] + 1), slice(x - h[2], x + h[2] + 1))
    F = sfft.fftn(vol[sl].astype(np.float64))
    Fl = ndimage.fourier_gaussian(F, locate_sigma) if locate_sigma else F
    d = np.zeros(3); sh = np.real(sfft.ifftn(Fl))
    for _ in range(n_iter):
        d = np.clip(d + _parabolic_offsets(sh, *h), -1, 1)
        sh = np.real(sfft.ifftn(ndimage.fourier_shift(Fl, -d)))
    if field is None:
        v = np.real(sfft.ifftn(ndimage.fourier_shift(F, -d)))[tuple(h)]; g = complex(np.nan)
    else:
        g = complex(sfft.ifftn(ndimage.fourier_shift(sfft.fftn(field[sl].astype(np.complex128)), -d))[tuple(h)])
        v = qpi.phase(np.asarray([g]), alpha)[0]
    return float(v), k + d[0], y + d[1], x + d[2], True, g


def measure_beads(phi, xy, search_px, estimator, isolation_px, z_edge_frames, interp_half_kyx, locate_sigma=0.0, G=None,
                  alpha=None):
    """Per-bead readout on a retrieved phase stack phi (nz, ny, nx) at detected (x, y). The peak voxel is the maximum within
    ±search_px laterally over all z (of phi smoothed by locate_sigma voxels, if > 0), then the value by `estimator`:
    "fourier" = peak_fourier (Γ₊ route if G and alpha are given), "max_voxel" = the largest raw voxel in the box. The sub-voxel
    centre always comes from peak_fourier. Selection flags (declared in the notebook): `isolated` (no other detection within
    isolation_px) and `z_ok` (focus ≥ z_edge_frames from both stack ends, where the z-mirroring of §2.2 distorts the retrieval).
    With G, Re/Im of Γ₊ interpolated at the peak are returned too (for the S28 inversion).
    Returns a dict of arrays: x, y, k (sub-voxel), peak, peak_max_voxel, isolated, z_ok, used, re_g, im_g."""
    nz, ny, nx = phi.shape
    xy = np.asarray(xy, int).reshape(-1, 2)
    iso = (cKDTree(xy).query(xy, k=2)[0][:, 1] > isolation_px) if len(xy) > 1 else np.ones(len(xy), bool)
    loc = ndimage.gaussian_filter(phi, locate_sigma) if locate_sigma else phi
    out = {k: [] for k in ("x", "y", "k", "peak", "peak_max_voxel", "z_ok", "re_g", "im_g")}
    for x, y in xy:
        y0, y1, x0, x1 = max(0, y - search_px), min(ny, y + search_px + 1), max(0, x - search_px), min(nx, x + search_px + 1)
        sub = loc[:, y0:y1, x0:x1]
        kk, yy, xx = np.unravel_index(np.argmax(sub), sub.shape)
        K, Y, X = kk, y0 + yy, x0 + xx
        v, kf, yf, xf, _, g = peak_fourier(phi, K, Y, X, interp_half_kyx, locate_sigma=locate_sigma, field=G, alpha=alpha)
        vmax = float(phi[:, y0:y1, x0:x1].max())
        if estimator == "max_voxel":
            v = vmax
        elif estimator != "fourier":
            raise ValueError(estimator)
        out["x"].append(xf); out["y"].append(yf); out["k"].append(kf); out["peak"].append(v)
        out["peak_max_voxel"].append(vmax)
        out["z_ok"].append(z_edge_frames <= kf <= nz - 1 - z_edge_frames)
        out["re_g"].append(g.real); out["im_g"].append(g.imag)
    r = {k: np.asarray(v, bool if k == "z_ok" else float) for k, v in out.items()}
    r["isolated"] = np.asarray(iso, bool)
    r["used"] = r["isolated"] & r["z_ok"]
    return r


def stats(values):
    """Bead statistics: the SI's mean ± SD over beads (§4), plus SEM, median and IQR."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    sd = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": sd, "sem": sd / np.sqrt(len(v)) if len(v) > 1 else np.nan,
            "median": float(q50), "q25": float(q25), "q75": float(q75)}


def trend(x, means, sems):
    """Weighted straight-line fit of per-channel means vs x (plane index or Δ): slope, its standard error and the number of
    decreasing consecutive steps (to report a monotonic P1 → P8 drop)."""
    x, m, s = (np.asarray(a, float) for a in (x, means, sems))
    ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
    if ok.sum() < 3:
        return {"slope": np.nan, "slope_se": np.nan, "n_decreasing_steps": 0, "n_steps": 0}
    w = 1 / s[ok] ** 2
    X = np.c_[np.ones(ok.sum()), x[ok]]
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = cov @ X.T @ (w * m[ok])
    chi2 = float(np.sum(w * (m[ok] - X @ beta) ** 2) / max(ok.sum() - 2, 1))
    d = np.diff(m[ok][np.argsort(x[ok])])
    return {"slope": float(beta[1]), "slope_se": float(np.sqrt(cov[1, 1] * max(chi2, 1.0))), "n_decreasing_steps": int((d < 0).sum()),
            "n_steps": int(len(d))}


def alpha_from_beads(phi_target, re_g, im_g, I0=1.0):
    """α that makes eq. S28 return phi_target, given the measured Re Γ₊, Im Γ₊ at the bead peak (the SI §4 inversion of S28,
    applied to beads). Diagnostic only: α is calibrated on the staircase (notebook 03)."""
    t = np.tan(phi_target)
    return t * I0 / (np.asarray(im_g) - t * np.asarray(re_g))


# --------------------------------------------------------------------------------------- bead average (§3.3, Fig. S3a,b)
def average_beads(vol, centres_kyx, half_kyx, margin):
    """Average of crops of vol centred on sub-voxel bead centres (SI §3.3: 'averaged over 15 individual measurements').
    Each crop (half-sizes half_kyx plus `margin` voxels) is moved by the exact sub-voxel shift in Fourier space (band-limited,
    no interpolation blur), then the margin is discarded. vol must have a zero background (phase, or intensity − 1).
    Returns (mean crop, number of beads used, list of used indices)."""
    hz, hy, hx = half_kyx; m = margin
    nz, ny, nx = vol.shape
    acc, used = [], []
    for i, (kf, yf, xf) in enumerate(np.asarray(centres_kyx, float).reshape(-1, 3)):
        k0, y0, x0 = int(round(kf)), int(round(yf)), int(round(xf))
        if not (hz + m <= k0 < nz - hz - m and hy + m <= y0 < ny - hy - m and hx + m <= x0 < nx - hx - m):
            continue
        crop = vol[k0 - hz - m:k0 + hz + m + 1, y0 - hy - m:y0 + hy + m + 1, x0 - hx - m:x0 + hx + m + 1].astype(np.float64)
        sh = np.real(sfft.ifftn(ndimage.fourier_shift(sfft.fftn(crop), (k0 - kf, y0 - yf, x0 - xf))))
        acc.append(sh[m:sh.shape[0] - m, m:sh.shape[1] - m, m:sh.shape[2] - m]); used.append(i)
    if not acc:
        return np.full((2 * hz + 1, 2 * hy + 1, 2 * hx + 1), np.nan), 0, used
    return np.mean(acc, 0), len(acc), used


def fwhm_1d(x, y):
    """FWHM of a sampled peak (linear interpolation at half maximum; no model fit). NaN if it does not fall below half."""
    y = np.asarray(y, float); i = int(np.nanargmax(y)); h = y[i] / 2
    l = i
    while l > 0 and y[l] > h:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] > h:
        r += 1
    if y[l] > h or y[r] > h:
        return np.nan
    xr = x[r - 1] + (h - y[r - 1]) * (x[r] - x[r - 1]) / (y[r] - y[r - 1])
    xl = x[l] + (h - y[l]) * (x[l + 1] - x[l]) / (y[l + 1] - y[l])
    return float(xr - xl)


def fwhm_xyz(vol, dz, px):
    """Lateral (x and y) and axial FWHM (µm) of a centred bead volume, through its central voxel (SI Fig. S3b)."""
    hz, hy, hx = (s // 2 for s in vol.shape)
    return {"x": fwhm_1d(px * np.arange(vol.shape[2]), vol[hz, hy, :]),
            "y": fwhm_1d(px * np.arange(vol.shape[1]), vol[hz, :, hx]),
            "z": fwhm_1d(dz * np.arange(vol.shape[0]), vol[:, hy, hx])}


def centred_crop(vol, half_kyx):
    """Crop around the maximum voxel (for a simulated, voxel-centred bead)."""
    k, y, x = np.unravel_index(np.nanargmax(vol), vol.shape); hz, hy, hx = half_kyx
    return vol[k - hz:k + hz + 1, y - hy:y + hy + 1, x - hx:x + hx + 1]


# --------------------------------------------------------------------------------------------- single-shot mode (SI §5)
def plane_spacing_fit(delta_um):
    """Robust inter-plane spacing from the plane positions Δ_p: sort Δ, then the Theil–Sen slope (median of pairwise slopes) of
    sorted Δ vs rank. Unlike mean(diff(Δ)) in label order, it uses every plane and tolerates one wrong Δ.
    Returns dict(order (indices by increasing Δ), spacing, intercept, residuals (µm, in sorted order))."""
    d = np.asarray(delta_um, float)
    order = np.argsort(d); ds = d[order]; r = np.arange(len(ds))
    i, j = np.triu_indices(len(ds), 1)
    slope = float(np.median((ds[j] - ds[i]) / (j - i)))
    icpt = float(np.median(ds - slope * r))
    return {"order": order, "spacing": slope, "intercept": icpt, "residuals": ds - (icpt + slope * r)}


def assemble_single_shot(signal_by_stream, planes, flats, delta_convention):
    """One camera frame -> 8-plane volume in reference coordinates (SI §5, §6.4: affine + bilinear warp of each channel).
    signal_by_stream: "A"/"B" -> offset-subtracted 2D frame; planes: calibration.json entries; flats: per-plane I₀ (panel
    coordinates). Planes are ordered so that the volume index runs like the stage z of a z-scan, i.e. the spectral half chosen on
    the z-scan applies: with delta_convention "stage_focus" (Δ_p = stage z at which plane p is in focus, minus that of the
    reference), plane p at stage z shows the sample like the z-scan frame at z − Δ_p, so the volume is in DEScending Δ.
    Pixels outside any warped panel are set to 1 (background) and flagged in `valid`.
    Returns (volume (8, H, W), plane labels in volume order, valid 2D mask)."""
    if delta_convention == "stage_focus":
        order = np.argsort([p["delta_z_um"] for p in planes])[::-1]
    elif delta_convention == "sample_depth":
        order = np.argsort([p["delta_z_um"] for p in planes])
    else:
        raise ValueError(delta_convention)
    ims = []
    for i in order:
        p = planes[i]
        c = geometry.panel(signal_by_stream, p) / flats[i]
        ims.append(geometry.warp(c, p["affine_from_reference"]))
    shapes = {a.shape for a in ims}
    if len(shapes) != 1:
        raise ValueError(f"panels of different sizes {shapes}: geometry.warp needs an output_shape argument")
    vol = np.stack(ims)
    valid = np.all(np.isfinite(vol), 0)
    return np.where(np.isfinite(vol), vol, 1.0).astype(np.float32), [planes[i]["plane"] for i in order], valid


def single_shot_frame(k_ref, delta_ref, delta_all, dz, delta_convention):
    """z-scan frame in which a bead focused at (sub-)frame k_ref in the reference channel sits mid-way through the 8 planes."""
    mid = 0.5 * (np.max(delta_all) + np.min(delta_all))
    s = 1 if delta_convention == "stage_focus" else -1
    return int(round(k_ref + s * (mid - delta_ref) / dz))


def ref_to_panel(A_from_ref, xy):
    xy = np.atleast_2d(xy); A = np.asarray(A_from_ref, float)
    return xy @ A[:, :2].T + A[:, 2]


def panel_to_ref(A_from_ref, xy):
    xy = np.atleast_2d(xy); A = np.asarray(A_from_ref, float)
    return (xy - A[:, 2]) @ np.linalg.inv(A[:, :2]).T


# ---------------------------------------------------------------------------------------------------------- extensions
def na_ill_scan(spec, gz, rcen, make_tf, grid_na_ill, grid_sigma_um, bead_radius_um, noise_rows, r_range, gz_range):
    """Extension (not in the SI): illumination NA from the measured spectrum (logic of check_04.py). Near the origin the
    brightfield CTF fills a double cone whose opening is set by NA_ill. Model |H| · |sphere FT| · exp(−(σ g⊥)²/2) (σ: blur
    nuisance) vs measured, log-amplitude misfit with a free scale, in r_range × gz_range. make_tf(na_ill) -> CTF table.
    Returns (best NA_ill, best σ, cost[n_na, n_sigma])."""
    floor = np.percentile(spec[noise_rows], 90, axis=0)
    meas = np.sqrt(np.clip(spec ** 2 - floor[None] ** 2, 0, None))
    reg = ((rcen[None, :] > r_range[0]) & (rcen[None, :] < r_range[1]) & (np.abs(gz)[:, None] > gz_range[0])
           & (np.abs(gz)[:, None] < gz_range[1]))
    m = reg & (meas > 0) & (spec > 1.5 * floor[None])
    gt = np.sqrt(rcen[None, :] ** 2 + gz[:, None] ** 2)
    C = np.full((len(grid_na_ill), len(grid_sigma_um)), np.inf)
    for i, na in enumerate(grid_na_ill):
        tf = make_tf(na)
        A = np.array([np.abs(qpi.lookup(tf, -abs(g), rcen[None, :])[0][0]) for g in gz]) * np.abs(sphere_ft(gt, bead_radius_um))
        for j, s in enumerate(grid_sigma_um):
            p = A * np.exp(-0.5 * (s * rcen[None, :]) ** 2)
            if (p[m] <= 0).mean() > 0.3:
                continue
            lr = np.log(meas[m]) - np.log(p[m] + 1e-9 * p.max())
            C[i, j] = float(np.mean((lr - lr.mean()) ** 2))
    i, j = np.unravel_index(np.argmin(C), C.shape)
    return float(grid_na_ill[i]), float(grid_sigma_um[j]), C


def plane_positions_brightfield(tables, planes, ref_index, dz, match_px, min_channels, clip_mad, n_iter=30):
    """Extension (not in the SI, which measures Δ with fluorescent beads, §6.2): plane positions from the brightfield bead
    foci. Beads are matched across channels via the calibration affines (≤ match_px in reference coordinates) and the focus
    stage z of bead b in plane p is solved as z_b + Δ_p (alternating medians, one clip at clip_mad robust σ half-way).
    Returns (Δ_bf relative to the reference plane, µm, number of beads used)."""
    ref = tables[ref_index]
    ref_xy = np.c_[ref["x"], ref["y"]]
    if len(ref_xy) == 0:
        return np.full(len(planes), np.nan), 0
    tree = cKDTree(ref_xy)
    zf = np.full((len(ref_xy), len(planes)), np.nan)
    for r, (t, p) in enumerate(zip(tables, planes)):
        if len(t["x"]) == 0:
            continue
        xy = panel_to_ref(p["affine_from_reference"], np.c_[t["x"], t["y"]])
        d, j = tree.query(xy, distance_upper_bound=match_px)
        for i, (dd, jj) in enumerate(zip(d, j)):
            if np.isfinite(dd):
                zf[jj, r] = dz * t["k"][i]
    zf = zf[np.isfinite(zf).sum(1) >= min_channels]
    if len(zf) == 0:
        return np.full(len(planes), np.nan), 0
    delta = np.zeros(len(planes))
    for it in range(n_iter):
        zb = np.nanmedian(zf - delta, 1); delta = np.nanmedian(zf - zb[:, None], 0)
        if it == n_iter // 2:
            res = zf - zb[:, None] - delta
            zf[np.abs(res) > clip_mad * 1.4826 * np.nanmedian(np.abs(res))] = np.nan
    return delta - delta[ref_index], len(zf)


def frame_shifts(stacks, highpass_sigma_px, lowpass_sigma_px, max_shift_px):
    """Extension (not in the SI): lateral shift between consecutive frames by cross-correlation of band-passed images
    (Gaussian low-pass lowpass_sigma_px minus high-pass highpass_sigma_px; not whitened, so that pixel noise does not dominate),
    summed over the given stacks (channels that see the same motion, same orientation and size), peak searched within
    ±max_shift_px, parabolic sub-pixel refinement. Returns the cumulative trajectory (nz, 2) = (x, y) px, frame 0 at the origin."""
    stacks = [stacks] if np.ndim(stacks) == 3 else stacks
    Fs = []
    for st in stacks:
        s = np.asarray(st, np.float32)
        bp = (ndimage.gaussian_filter(s, (0, lowpass_sigma_px, lowpass_sigma_px))
              - ndimage.gaussian_filter(s, (0, highpass_sigma_px, highpass_sigma_px)))
        Fs.append(sfft.fft2(bp, workers=-1))
    m = int(np.ceil(max_shift_px))
    traj = [np.zeros(2)]
    for k in range(1, len(Fs[0])):
        cc = sfft.fftshift(np.real(sfft.ifft2(sum(F[k] * np.conj(F[k - 1]) for F in Fs))))
        cy, cx = cc.shape[0] // 2, cc.shape[1] // 2
        win = cc[cy - m:cy + m + 1, cx - m:cx + m + 1]
        y, x = np.unravel_index(np.argmax(win), win.shape)
        off = []
        for i, line in ((y, win[:, x]), (x, win[y, :])):
            if 0 < i < len(line) - 1 and (line[i - 1] - 2 * line[i] + line[i + 1]) < 0:
                off.append(i - m + 0.5 * (line[i - 1] - line[i + 1]) / (line[i - 1] - 2 * line[i] + line[i + 1]))
            else:
                off.append(float(i - m))
        traj.append(traj[-1] + np.array([off[1], off[0]]))
    return np.array(traj)


def detrend(traj, z):
    """Remove the straight-line part of a trajectory (stage drift and defocus-dependent shifts are not separable from it)."""
    p = np.polyfit(z, traj, 1)
    return traj - (np.outer(z, p[0]) + p[1]), p


def apply_shifts(stack, traj_xy):
    """Undo per-frame shifts (exact Fourier shift of c − 1, periodic)."""
    out = np.empty_like(stack)
    for k, (sx, sy) in enumerate(traj_xy):
        out[k] = geometry.fourier_shift(stack[k] - 1, (-sy, -sx)) + 1
    return out
