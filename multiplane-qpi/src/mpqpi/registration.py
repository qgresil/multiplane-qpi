"""Fluorescent-bead calibration of the 8-plane prism: panel windows, plane offsets Δ, plane spacing, plane-to-reference affines and
channel transmission (Descloux et al. 2018, SI §6.1–6.4, Fig. S16, §8.3).

Paper method: beads scanned along the optical axis in 200 nm steps (SI §6.2, §6.4); the stage z at which a bead is in focus in
each plane gives the plane offsets Δ_p and the inter-plane distance (347 ± 11 nm, SI §6.2); an affine transformation per plane,
fitted on the bead positions and applied with bilinear interpolation, co-registers the planes (SI §6.4, Fig. S16); the same scan
gives the transmission of the 8 channels (SI §6.4). The SI does not say how panels, beads, focus or positions are found: those
procedures are implementation choices, and every threshold is a function argument that the notebook declares.
Extensions (not in the SI): focal-plane tilt fit, common lateral-slope (stage drift) correction, camera-B comb-offset diagnostic,
2nd-order polynomial comparison.

Conventions: positions (x, y) = (column, row) in the oriented panel, pixel centres at integers; a 2x3 affine M maps
src -> dst = M[:, :2] @ src + M[:, 2].
"""
import warnings
import numpy as np
from scipy import ndimage, optimize
from scipy.spatial import cKDTree

MAD_TO_SIGMA = 1.4826                       # σ = 1.4826 · median|x − median| for a Gaussian
RADIAL_MEDIAN = np.sqrt(2 * np.log(2))      # median of |r| for an isotropic 2D Gaussian of unit σ per axis


def px(um, pixel_um, odd=False, minimum=0):
    """Physical size (µm) -> pixels (int); odd=True gives an odd window size."""
    n = max(int(round(um / pixel_um)), minimum)
    return n + 1 - n % 2 if odd else n


def robust_sigma(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return MAD_TO_SIGMA * np.median(np.abs(x - np.median(x)))


# ------------------------------------------------------------------------------------------------------------------------- panels
def mean_image(stack, chunk=16):
    """Mean over frames, read in chunks (memory-mapped stacks)."""
    acc = np.zeros(stack.shape[1:], np.float64)
    for i in range(0, len(stack), chunk):
        acc += np.asarray(stack[i:i + chunk], np.float64).sum(0)
    return (acc / len(stack)).astype(np.float32)


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _longest(mask):
    r = _runs(mask)
    return max(r, key=lambda a: a[1] - a[0]) if r else (0, len(mask))


def detect_panels(img, n_panels, level, smooth_px, percentiles, min_lit_px, gap_search, min_gap_depth):
    """Panel windows [(top, left, height, width)] in raw camera coordinates, left to right (SI §6.1–6.2: the field stop gives
    n_panels images side by side per camera, at a constant lateral displacement d). Implementation (not in the SI): illuminated
    band at `level` of the (percentile) range of the z-mean image, split into n_panels equal pitches; the darkest column within
    ± gap_search·pitch of each expected boundary is a gap; panel edges where the column profile crosses half the gap depth.
    Raises RuntimeError if a gap is shallower than min_gap_depth (then give the windows by hand)."""
    lo, hi = np.percentile(img, percentiles)
    row_p = ndimage.gaussian_filter1d(img.mean(1), smooth_px)
    lo_r, hi_r = np.percentile(row_p, percentiles)
    r0, r1 = _longest(row_p > lo_r + level * (hi_r - lo_r))
    col_p = ndimage.gaussian_filter1d(np.median(img[r0:r1], 0), smooth_px)
    lo_c, hi_c = np.percentile(col_p, percentiles)
    lit = [r for r in _runs(col_p > lo_c + level * (hi_c - lo_c)) if r[1] - r[0] > min_lit_px]
    diag = {"col_profile": col_p, "gaps": [], "depths": [], "rows": (int(r0), int(r1))}
    if not lit:
        raise RuntimeError("panel detection: no illuminated columns")
    x0, x1 = lit[0][0], lit[-1][1]
    pitch = (x1 - x0) / n_panels
    plateau = np.median(col_p[x0:x1])
    edges = [(x0, x0)]
    for k in range(1, n_panels):
        c, half = int(round(x0 + k * pitch)), max(int(gap_search * pitch), 1)
        g = c - half + int(np.argmin(col_p[c - half:c + half]))
        diag["gaps"].append(g); diag["depths"].append(float((plateau - col_p[g]) / max(plateau - lo_c, 1e-9)))
        lev = col_p[g] + 0.5 * (plateau - col_p[g])
        a = b = g
        while a > x0 and col_p[a] < lev:
            a -= 1
        while b < x1 - 1 and col_p[b] < lev:
            b += 1
        edges.append((a, b))
    edges.append((x1, x1))
    if min(diag["depths"]) < min_gap_depth:
        raise RuntimeError(f"panel detection: gap depths {np.round(diag['depths'], 2).tolist()} < {min_gap_depth}; give the windows")
    wins = []
    for k in range(n_panels):
        c0, c1 = edges[k][1], edges[k + 1][0]
        rp = ndimage.gaussian_filter1d(np.median(img[:, c0:c1], 1), smooth_px)
        lo_k, hi_k = np.percentile(rp, percentiles)
        t0, t1 = _longest(rp > lo_k + level * (hi_k - lo_k))
        wins.append((int(t0), int(c0), int(t1 - t0), int(c1 - c0)))
    return wins, diag


def common_windows(windows_by_stream, margin_px):
    """Equal-size panels (needed to stack the planes): every window shrunk to the smallest one minus margin_px per side
    (even size), centred in its detected window. Returns panel dicts {id, stream, window_raw} with ids A0..A3, B0..B3
    numbered left to right on the raw camera image."""
    allw = [w for ws in windows_by_stream.values() for w in ws]
    h = min(w[2] for w in allw) - 2 * margin_px; w_ = min(w[3] for w in allw) - 2 * margin_px
    h -= h % 2; w_ -= w_ % 2
    panels = []
    for s, ws in windows_by_stream.items():
        for i, (t, l, hh, ww) in enumerate(sorted(ws, key=lambda a: a[1])):
            panels.append({"id": f"{s}{i}", "stream": s, "window_raw": {"top": int(t + (hh - h) // 2), "left": int(l + (ww - w_) // 2),
                                                                         "height": int(h), "width": int(w_)}})
    return panels


# -------------------------------------------------------------------------------------------------- projections and orientation
def dog(img, s1_px, s2_px):
    img = np.asarray(img, np.float32)
    return ndimage.gaussian_filter(img, s1_px) - ndimage.gaussian_filter(img, s2_px)


def bead_projection(stack, smooth_zyx, dog_px):
    """Bead map of a panel z-stack: smooth (z, y, x) before the max projection (a plain max turns noise into speckle), then a
    difference of Gaussians (σ in px). Used for orientation, bead detection and the overlay check (implementation choice)."""
    return dog(ndimage.gaussian_filter(np.asarray(stack, np.float32), smooth_zyx).max(0), *dog_px)


def _tukey(n, alpha):
    x = np.linspace(0, 1, n); w = np.ones(n); a = alpha / 2
    if a > 0:
        m = x < a; w[m] = 0.5 * (1 + np.cos(np.pi * (x[m] / a - 1)))
        m = x > 1 - a; w[m] = 0.5 * (1 + np.cos(np.pi * ((x[m] - 1) / a + 1)))
    return w


def phase_correlation(ref, mov, tukey_alpha):
    """Shift (dx, dy) such that features of `mov` sit at ref + (dx, dy), and the correlation-peak SNR (robust σ units)."""
    tk = np.outer(_tukey(ref.shape[0], tukey_alpha), _tukey(ref.shape[1], tukey_alpha)).astype(np.float32)
    R = np.fft.rfft2(mov * tk) * np.conj(np.fft.rfft2(ref * tk)); R /= np.abs(R) + 1e-12
    c = np.fft.irfft2(R, s=ref.shape)
    iy, ix = np.unravel_index(np.argmax(c), c.shape); ny, nx = c.shape

    def sub(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if d == 0 else 0.5 * (cm - cp) / d
    dy = iy + sub(c[(iy - 1) % ny, ix], c[iy, ix], c[(iy + 1) % ny, ix])
    dx = ix + sub(c[iy, (ix - 1) % nx], c[iy, ix], c[iy, (ix + 1) % nx])
    dy, dx = (dy - ny if dy > ny / 2 else dy), (dx - nx if dx > nx / 2 else dx)
    return np.array([dx, dy]), float((c[iy, ix] - np.median(c)) / robust_sigma(c))


def orientation_scores(mips, ref_id, candidates, tukey_alpha):
    """Phase-correlation SNR of every panel against the reference panel for each candidate orientation (flips only)."""
    from .geometry import ORIENT
    return {k: {o: phase_correlation(mips[ref_id], ORIENT[o](m), tukey_alpha)[1] for o in candidates}
            for k, m in mips.items() if k != ref_id}


def choose_orientation(scores, stream):
    """Orientation of `stream` relative to the reference stream: the candidate with the largest summed SNR over its panels.
    Returns (orientation, per-panel winners, smallest winner / runner-up ratio)."""
    ks = [k for k in scores if k.startswith(stream)]
    cands = list(scores[ks[0]])
    best = max(cands, key=lambda o: sum(scores[k][o] for k in ks))
    ratio = min(scores[k][best] / max(v for o, v in scores[k].items() if o != best) for k in ks)
    return best, {k: max(scores[k], key=scores[k].get) for k in ks}, float(ratio)


# ------------------------------------------------------------------------------------------------------ beads and matching
def detect_beads(img, thr_sigma, peak_px, border_px):
    """Local maxima (window peak_px) above median + thr_sigma robust σ, border_px from the edges; 3x3 centroid refinement."""
    med = np.median(img)
    pk = (img == ndimage.maximum_filter(img, size=peak_px)) & (img > med + thr_sigma * robust_sigma(img))
    pk[:border_px] = pk[-border_px:] = False; pk[:, :border_px] = pk[:, -border_px:] = False
    y, x = np.nonzero(pk)
    yy, xx = np.mgrid[-1:2, -1:2]
    nb = np.stack([img[y + dy, x + dx] for dy, dx in zip(yy.ravel(), xx.ravel())], 1)
    nb = np.clip(nb - nb.min(1, keepdims=True), 0, None) + 1e-9
    return np.c_[x + (nb * xx.ravel()).sum(1) / nb.sum(1), y + (nb * yy.ravel()).sum(1) / nb.sum(1)]


def fit_affine(src, dst):
    """Least-squares 2x3 affine dst ≈ M[:, :2] @ src + M[:, 2]."""
    return np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0].T


def apply_affine(M, pts):
    M = np.asarray(M, float)
    return np.asarray(pts, float) @ M[:, :2].T + M[:, 2]


def invert_affine(M):
    return np.linalg.inv(np.vstack([np.asarray(M, float), [0, 0, 1]]))[:2]


def mutual_nn(pred, pts, radius):
    """Index into pts of the mutual nearest neighbour of each predicted point within radius, else −1."""
    d, j = cKDTree(pts).query(pred, distance_upper_bound=radius)
    _, back = cKDTree(pred).query(pts)
    out = np.full(len(pred), -1); ok = np.isfinite(d)
    ok[ok] &= back[j[ok]] == np.flatnonzero(ok)
    out[ok] = j[ok]
    return out


def match_to_reference(ref_pts, pts, shift, radii_px):
    """Coarse shift, then affine prediction refined with shrinking mutual-nearest-neighbour radii."""
    idx = mutual_nn(ref_pts + shift, pts, radii_px[0])
    for r in radii_px[1:]:
        ok = idx >= 0
        if ok.sum() < 3:
            break
        idx = mutual_nn(apply_affine(fit_affine(ref_pts[ok], pts[idx[ok]]), ref_pts), pts, r)
    return idx


def match_beads(detections, ref_id, shifts, isolation_px, radii_px, min_panels):
    """Beads of the matching-reference panel with no neighbour closer than isolation_px, matched in every panel; beads found
    in fewer than min_panels panels are dropped. Returns (positions (NB, P, 2) NaN where unmatched, reference-panel detections,
    isolated mask)."""
    ids = list(detections)
    ref0 = detections[ref_id]
    iso = cKDTree(ref0).query(ref0, k=2)[0][:, 1] > isolation_px
    ref = ref0[iso]
    idx = np.stack([match_to_reference(ref, detections[k], shifts[k], radii_px) for k in ids], 1)
    keep = (idx >= 0).sum(1) >= min_panels
    idx = idx[keep]
    pos = np.full((len(idx), len(ids), 2), np.nan)
    for j, k in enumerate(ids):
        ok = idx[:, j] >= 0
        pos[ok, j] = detections[k][idx[ok, j]]
    return pos, ref0, iso


# ------------------------------------------------------------------------------------------------------------------------- axial
def roi_grid(half):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    return yy, xx, (np.abs(yy) == half) | (np.abs(xx) == half)


def extract_rois(stack, centres, half, frames=None):
    """ROIs around integer-rounded centres: (M, T, n, n) for all frames, or (M, F, n, n) for per-bead frame indices (M, F)."""
    yy, xx, _ = roi_grid(half)
    xi = np.rint(centres[:, 0]).astype(int); yi = np.rint(centres[:, 1]).astype(int)
    if frames is None:
        return np.moveaxis(np.asarray(stack[:, yi[:, None, None] + yy, xi[:, None, None] + xx], np.float32), 0, 1)
    return np.asarray(stack[frames[..., None, None], yi[:, None, None, None] + yy, xi[:, None, None, None] + xx], np.float32)


def focus_signal(stack, centres, roi_half_px, core_half_px):
    """On-axis bead signal per frame: mean of the (2·core+1)² core minus the median of the ROI border (local background).
    Implementation choice: the SI does not say how focus is found. A metric bias common to all planes cancels in Δ."""
    rois = extract_rois(stack, centres, roi_half_px)
    c, (_, _, ring) = roi_half_px, roi_grid(roi_half_px)
    return rois[..., c - core_half_px:c + core_half_px + 1, c - core_half_px:c + core_half_px + 1].mean((-1, -2)) - np.median(rois[..., ring], -1)


def _gauss_const(z, a, z0, s, c):
    return a * np.exp(-0.5 * ((z - z0) / s) ** 2) + c


def fit_focus(z, sig, half_window_um, width_bounds_um, sigma0_um, max_shift_steps, edge_frames):
    """Stage z of best focus: Gaussian + constant fitted within ± half_window_um (stage) of the maximum. Rejected (NaN) if the
    maximum is within edge_frames of the stack ends, the width is outside width_bounds_um, or the centre moved more than
    max_shift_steps from the maximum. Returns (z0, width, SNR = amplitude / residual SD)."""
    step = float(np.median(np.diff(z))); k = int(np.argmax(sig)); half = int(round(half_window_um / step))
    nan = (np.nan, np.nan, np.nan)
    if k < edge_frames or k > len(z) - 1 - edge_frames:
        return nan
    sl = slice(max(0, k - half), min(len(z), k + half + 1)); zz, ss = z[sl], sig[sl]
    if len(zz) < 5:
        return nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
            (a, z0, s, c), _ = optimize.curve_fit(_gauss_const, zz, ss, p0=[ss.max() - ss.min(), z[k], sigma0_um, ss.min()], maxfev=4000)
    except RuntimeError:
        return nan
    if a <= 0 or not (width_bounds_um[0] < abs(s) < width_bounds_um[1]) or abs(z0 - z[k]) > max_shift_steps * step:
        return nan
    return z0, abs(s), a / (np.std(ss - _gauss_const(zz, a, z0, s, c)) + 1e-12)


def plane_offsets(zf, n_iter, clip_sigma):
    """Model z_focus[b, p] = z_b + Δ_p (SI §6.2: Δ from the stage z at which each bead is in focus in each plane). Alternating
    medians (robust to outliers), one clipping pass at |r| > clip_sigma · robust σ halfway, then means on the kept values.
    Returns Δ relative to its minimum, bead depths z_b, and the clipped z_focus."""
    zf = np.array(zf, float); delta = np.zeros(zf.shape[1])
    for it in range(n_iter):
        zb = np.nanmedian(zf - delta, 1)
        delta = np.nanmedian(zf - zb[:, None], 0)
        if it == n_iter // 2:
            r = zf - zb[:, None] - delta
            zf[np.abs(r) > clip_sigma * MAD_TO_SIGMA * np.nanmedian(np.abs(r))] = np.nan
    for _ in range(n_iter):
        zb = np.nanmean(zf - delta, 1)
        delta = np.nanmean(zf - zb[:, None], 0)
    return delta - delta.min(), zb + delta.min(), zf


def bootstrap_offsets(zf, n_boot, seed, n_iter, clip_sigma):
    """Bootstrap over beads with the same estimator as plane_offsets: (n_boot, P) Δ samples."""
    rng = np.random.default_rng(seed)
    return np.array([plane_offsets(zf[rng.integers(0, len(zf), len(zf))], n_iter, clip_sigma)[0] for _ in range(n_boot)])


def per_bead_spacing(zf_clean, order):
    """Mean inter-plane distance of each bead in focus in the first and last plane: (z_last − z_first) / (P − 1). Mean ± SD over
    beads is the SI §6.2 statistic (347 ± 11 nm over 8 beads)."""
    a, b = zf_clean[:, order[0]], zf_clean[:, order[-1]]
    ok = np.isfinite(a) & np.isfinite(b)
    return (b[ok] - a[ok]) / (len(order) - 1)


def tilt_fit(bead, plane, z, x, y, n_planes, anchor, with_tilt=True, gauge_weight=1e3):
    """Extension (not in the SI). Joint least squares z = z_b + Δ_p + g_x,p x + g_y,p y (x, y from the panel centre).
    A gradient common to all planes is absorbed by the bead depths, so Σ g = 0 is imposed (soft constraint of weight gauge_weight;
    any large value fixes the gauge without changing the solution); Δ_anchor = 0. Returns Δ, g_x, g_y (µm/px), residuals."""
    ub, bi = np.unique(bead, return_inverse=True)
    nb, n, P = len(ub), len(z), n_planes
    A = np.zeros((n + 3, nb + P + (2 * P if with_tilt else 0))); rhs = np.r_[z, 0, 0, 0]; r = np.arange(n)
    A[r, bi] = 1; A[r, nb + plane] = 1; A[n, nb + anchor] = gauge_weight
    if with_tilt:
        A[r, nb + P + plane] = x; A[r, nb + 2 * P + plane] = y
        A[n + 1, nb + P:nb + 2 * P] = gauge_weight; A[n + 2, nb + 2 * P:] = gauge_weight
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    gx = sol[nb + P:nb + 2 * P] if with_tilt else np.zeros(P); gy = sol[nb + 2 * P:] if with_tilt else np.zeros(P)
    d = sol[nb:nb + P]
    return d - d.min(), gx, gy, z - A[:n] @ sol


def camera_comb(delta, streams, step_um, assumed_shift, shifts=(-1, 0, 1)):
    """Diagnostic for the camera-B frame pairing (extension). With the prism design (SI §6.2) the planes of one camera form a
    comb of pitch 2δ and the other camera's comb sits half-way (offset ±δ). A one-frame pairing error moves all B planes by one
    stage step. For each hypothesis `shift` (B frames recorded `shift` steps after A; the data were loaded with assumed_shift)
    returns the B-comb offset in units of δ (±1 = interleaved as designed, 0 = both cameras image the same planes)."""
    streams = np.asarray(streams)
    dA, dB = np.sort(delta[streams == "A"]), np.sort(delta[streams == "B"])
    i = np.arange(len(dA))
    d_unit = 0.25 * (np.polyfit(i, dA, 1)[0] + np.polyfit(np.arange(len(dB)), dB, 1)[0])
    out = {}
    for s in shifts:
        raw = float(np.mean(dB + (s - assumed_shift) * step_um - dA))
        out[int(s)] = {"offset_um": raw, "offset_delta": float(((raw + d_unit) % (2 * d_unit) - d_unit) / d_unit)}
    return out, float(d_unit)


def design_plane_spacing_um(prism_d_mm, prism_n, lateral_magnification):
    """SI §6.2: image-side inter-plane distance Δz_IP = d / (2n), object-side δz_IP = Δz_IP / M_a with M_a = M_l²."""
    return 1e3 * prism_d_mm / (2 * prism_n) / lateral_magnification ** 2


def axial_scale(n_sample, n_immersion):
    """Stage z -> focal shift in the sample, paraxial: Δz_sample = Δz_stage · n_sample / n_immersion (1 when index-matched).
    Not in the SI (the SI reports stage z with surface beads). High-NA focal-shift models give a different factor."""
    return n_sample / n_immersion


# ---------------------------------------------------------------------------------------------------------- lateral positions
def fit_gauss2d_batch(rois, sigma0_px, n_iter):
    """Batched Levenberg–Marquardt fit of A·exp(−r²/2s²) + bg on ROIs (M, n, n). Returns params (M, 5) [A, x0, y0, s, bg] with
    (x0, y0) relative to the ROI centre, SD of (x0, y0) from the covariance, and an acceptance mask."""
    M, n = rois.shape[0], rois.shape[-1]; half = n // 2
    yy, xx, ring = roi_grid(half); xx = xx.ravel().astype(float); yy = yy.ravel().astype(float)
    d = rois.reshape(M, -1).astype(np.float64)
    bg = np.median(d[:, ring.ravel()], 1)
    w = np.clip(d - bg[:, None], 0, None); tot = w.sum(1) + 1e-12
    p = np.c_[d.max(1) - bg, (w * xx).sum(1) / tot, (w * yy).sum(1) / tot, np.full(M, float(sigma0_px)), bg]

    def model(p):
        A, x0, y0, s, b = (p[:, i:i + 1] for i in range(5))
        dx, dy = xx - x0, yy - y0; r2 = dx ** 2 + dy ** 2; g = np.exp(-r2 / (2 * s ** 2))
        return A * g + b, np.stack([g, A * g * dx / s ** 2, A * g * dy / s ** 2, A * g * r2 / s ** 3, np.ones_like(g)], -1)
    lam = np.full(M, 1e-2)                                   # LM damping schedule: changes convergence speed, not the optimum
    f, J = model(p); cost = ((d - f) ** 2).sum(1)
    for _ in range(n_iter):
        JTJ = np.einsum("mni,mnj->mij", J, J); grad = np.einsum("mni,mn->mi", J, d - f)
        try:
            dp = np.linalg.solve(JTJ + lam[:, None, None] * (JTJ * np.eye(5)), grad[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        pn = p + dp; pn[:, 3] = np.clip(np.abs(pn[:, 3]), 0.3, n)
        fn, Jn = model(pn); cn = ((d - fn) ** 2).sum(1); ok = cn < cost
        p[ok], f[ok], J[ok], cost[ok] = pn[ok], fn[ok], Jn[ok], cn[ok]
        lam = np.where(ok, lam * 0.3, lam * 10)
    cov = np.linalg.pinv(np.einsum("mni,mnj->mij", J, J)) * (cost / (d.shape[1] - 5))[:, None, None]
    sd = np.sqrt(np.abs(np.c_[cov[:, 1, 1], cov[:, 2, 2]]))
    ok = (np.isfinite(p).all(1) & (p[:, 0] > 0) & (np.abs(p[:, 1]) < half - 1) & (np.abs(p[:, 2]) < half - 1) & (p[:, 3] < half)
          & np.isfinite(sd).all(1))
    return p, sd, ok


def positions_at_focus(stack, z, z_focus, centres, window_um, roi_half_px, sigma0_px, n_iter, min_frames):
    """Bead position in one plane at its focus: 2D Gaussian localisation in every frame within ± window_um (stage) of the bead's
    focus, then a weighted line x(z), y(z) evaluated at z_focus (implementation choice; the SI only says the affine is fitted on
    the bead calibration). Returns position (M, 2), its SD (M, 2), slope (M, 2) in px per µm stage z, and traces."""
    M = len(centres); pos = np.full((M, 2), np.nan); sd_out = np.full((M, 2), np.nan); slope = np.full((M, 2), np.nan)
    good = np.flatnonzero(np.isfinite(z_focus) & np.isfinite(centres[:, 0]))
    if not len(good):
        return pos, sd_out, slope, {}
    step = float(np.median(np.diff(z))); nw = max(int(np.floor(window_um / step)), 1); T = len(z)
    kc = np.clip(np.rint((z_focus[good] - z[0]) / step).astype(int), nw, T - 1 - nw)
    frames = kc[:, None] + np.arange(-nw, nw + 1)
    rois = extract_rois(stack, centres[good], roi_half_px, frames)
    nb, nf = frames.shape
    par, sd, ok = fit_gauss2d_batch(rois.reshape(nb * nf, *rois.shape[2:]), sigma0_px, n_iter)
    par, sd, ok = par.reshape(nb, nf, 5), sd.reshape(nb, nf, 2), ok.reshape(nb, nf)
    base = np.rint(centres[good])
    dz = z[frames] - z_focus[good][:, None]
    traces = {}
    for i, b in enumerate(good):
        m = ok[i]
        if m.sum() < min_frames:
            continue
        traces[int(b)] = (z[frames[i, m]], base[i] + par[i, m, 1:3])
        X = np.c_[np.ones(m.sum()), dz[i, m]]
        for a in (0, 1):
            wts = 1 / np.maximum(sd[i, m, a], 1e-3) ** 2           # floor: numerical guard against zero-variance fits
            cov = np.linalg.inv((X.T * wts) @ X); beta = cov @ ((X.T * wts) @ par[i, m, 1 + a])
            pos[b, a], sd_out[b, a], slope[b, a] = base[i, a] + beta[0], np.sqrt(cov[0, 0]), beta[1]
    return pos, sd_out, slope, traces


def common_slope_correction(pos, slope, z_focus):
    """Extension (not in the SI), off by default. In a stage scan each plane sees a bead at a different stage z (z_b + Δ_p); a
    lateral stage drift s (px/µm) then adds s·Δ_p to the positions and to the affine translations, which does not exist in a
    simultaneous multiplane acquisition. This refers every position to the bead's mean focus z with the median slope over all
    beads and planes. Wrong if the slope is a real (non-telecentric) property of the optics. Returns positions and the slope."""
    s = np.nanmedian(slope.reshape(-1, 2), 0)
    zmid = np.nanmean(z_focus, 1)
    return pos + s[None, None, :] * (zmid[:, None, None] - z_focus[..., None]), s


# ------------------------------------------------------------------------------------------------------------------ registration
def design_matrix(pts, degree, centre=None, scale=None):
    x, y = pts[:, 0], pts[:, 1]
    cols = [np.ones_like(x), x, y]
    if degree == 2:
        xs, ys = (x - centre[0]) / scale, (y - centre[1]) / scale          # conditioning only
        cols += [xs * xs, xs * ys, ys * ys]
    return np.stack(cols, 1)


def robust_fit(src, dst, degree, clip_sigma, n_iter):
    """Least squares dst ≈ f(src) (degree 1 = affine, SI §6.4; degree 2 = diagnostic) with iterative clipping of points whose
    residual norm exceeds clip_sigma · σ, σ = median|r| / √(2 ln 2) (per-axis σ of isotropic 2D Gaussian residuals).
    Returns coefficients (3 or 6, 2), kept mask, residuals and leave-one-out residuals r / (1 − h_ii) of the kept points."""
    centre, scale = src.mean(0), max(src.std(), 1e-9)
    X = design_matrix(src, degree, centre, scale)
    keep = np.ones(len(src), bool)
    for _ in range(n_iter):
        coef = np.linalg.lstsq(X[keep], dst[keep], rcond=None)[0]
        rn = np.hypot(*(dst - X @ coef).T)
        new = rn <= clip_sigma * max(np.median(rn[keep]) / RADIAL_MEDIAN, 1e-9)
        if (new == keep).all():
            break
        keep = new
    coef = np.linalg.lstsq(X[keep], dst[keep], rcond=None)[0]
    r = dst[keep] - X[keep] @ coef
    hat = np.einsum("ij,jk,ik->i", X[keep], np.linalg.pinv(X[keep].T @ X[keep]), X[keep])
    return coef, keep, r, r / (1 - hat)[:, None]


def affine_from_coef(coef):
    return np.asarray(coef).T[:, [1, 2, 0]]


def decompose(M):
    """Scale x, scale y, rotation (deg), shear and determinant of the linear part."""
    A = np.asarray(M)[:, :2]; sx = np.hypot(A[0, 0], A[1, 0]); det = np.linalg.det(A)
    return dict(scale_x=sx, scale_y=det / sx, rotation_deg=np.degrees(np.arctan2(A[1, 0], A[0, 0])),
                shear=(A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]) / det, det=det)


def register_planes(pos, ref_j, clip_sigma, n_iter, pos_sd=None):
    """Affine panel -> reference for every plane, fitted on bead positions at focus (SI §6.4, Fig. S16; one common reference
    plane rather than the SI's pairs of channels — equivalent by composition). Returns {j: M_to_ref}, {j: stats (px)}."""
    P = pos.shape[1]; aff, stats = {ref_j: np.array([[1.0, 0, 0], [0, 1.0, 0]])}, {}
    for j in range(P):
        if j == ref_j:
            continue
        ok = np.isfinite(pos[:, j, 0]) & np.isfinite(pos[:, ref_j, 0])
        src, dst = pos[ok, j], pos[ok, ref_j]
        coef, kept, r, r_loo = robust_fit(src, dst, 1, clip_sigma, n_iter)
        aff[j] = affine_from_coef(coef)
        r2_loo = robust_fit(src, dst, 2, clip_sigma, n_iter)[3]
        st = dict(n=int(kept.sum()), n_rejected=int((~kept).sum()), rms_px=float(np.sqrt((r ** 2).sum(1).mean())),
                  loo_rms_px=float(np.sqrt((r_loo ** 2).sum(1).mean())), loo_rms_poly2_px=float(np.sqrt((r2_loo ** 2).sum(1).mean())),
                  src=src[kept], resid=r, **{k: float(v) for k, v in decompose(aff[j]).items()})
        if pos_sd is not None:
            st["floor_px"] = float(np.sqrt(np.nanmean(pos_sd[ok, j][kept] ** 2 + pos_sd[ok, ref_j][kept] ** 2, 0).sum()))
        stats[j] = st
    return aff, stats


# ------------------------------------------------------------------------------------------------------------------ transmission
def bead_photometry(stack, centres, frames, radius_px, bg_width_px):
    """Integrated bead signal in a disc of radius_px minus the annulus median (width bg_width_px) times the disc area, at the
    given frames: (M, F). The stack must be offset-subtracted (io.to_signal)."""
    half = radius_px + bg_width_px
    yy, xx, _ = roi_grid(half); r = np.hypot(yy, xx)
    disc, ann = r <= radius_px, (r > radius_px) & (r <= half)
    rois = extract_rois(stack, centres, half, frames)
    return rois[..., disc].sum(-1) - disc.sum() * np.median(rois[..., ann], -1)


def channel_transmission(stacks, pos, frame_centre, half_frames, radius_px, bg_width_px, n_boot, seed):
    """Relative transmission of the 8 channels (SI §6.4: 'the beads calibration measurements are also used to correct the
    transmission variation among the 8 image channels'). All planes record the same emission at the same frame index, so the
    ratio of a bead's integrated signal between planes at a common frame is the channel-transmission ratio, independent of
    bleaching and bead brightness. Implementation: frames frame_centre ± half_frames, disc of radius_px (must contain the
    defocused images of the outer planes), ratio to the bead's mean over planes, median over beads and frames, normalised to
    mean 1. SD: bootstrap over beads. Uses beads found in all planes and far enough from the panel edges."""
    P = len(stacks); T, h, w = stacks[0].shape; half = radius_px + bg_width_px
    inside = np.all(np.isfinite(pos[..., 0]), 1)
    inside &= np.all((pos[..., 0] >= half) & (pos[..., 0] <= w - 1 - half) & (pos[..., 1] >= half) & (pos[..., 1] <= h - 1 - half), 1)
    inside &= (frame_centre >= half_frames) & (frame_centre <= T - 1 - half_frames)
    b = np.flatnonzero(inside)
    if len(b) < 2:
        return np.full(P, np.nan), np.full(P, np.nan), np.zeros((0, P)), b
    frames = frame_centre[b][:, None] + np.arange(-half_frames, half_frames + 1)
    S = np.stack([bead_photometry(stacks[j], pos[b, j], frames, radius_px, bg_width_px) for j in range(P)], -1)   # (M, F, P)
    ratio = (S / S.mean(-1, keepdims=True)).reshape(len(b), -1, P)
    per_bead = np.median(ratio, 1)

    def est(pb):
        t = np.median(pb, 0)
        return t / t.mean()
    rng = np.random.default_rng(seed)
    boot = np.array([est(per_bead[rng.integers(0, len(b), len(b))]) for _ in range(n_boot)])
    return est(per_bead), boot.std(0), per_bead, b


# --------------------------------------------------------------------------------------------------------------------------- QC
def ncc(a, b, thr_sigma, border_px):
    """Normalised cross-correlation of two bead maps restricted to bead pixels (> thr_sigma robust σ); NaN pixels ignored."""
    def bm(x):
        x = np.asarray(x, float)[border_px:-border_px, border_px:-border_px]
        return np.clip((x - np.nanmedian(x)) / robust_sigma(x) - thr_sigma, 0, None)
    a, b = bm(a), bm(b); m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
