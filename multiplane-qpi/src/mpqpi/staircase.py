"""Phase calibration factor α on an etched glass staircase, as in Descloux et al. 2018, SI §4 (Fig. S4, eq. S28).

SI §4: a brightfield stage z-stack (50 planes × 200 nm) of the staircase is retrieved (§2.2); the theoretical phase of a step of
AFM height h is φ_th = k₀ Δn h (k₀ = 2π/λ₀ = 10.8 µm⁻¹, Δn = 1.51 − 1 in air), and α is "estimated by solving equation (S28) for
α and using the expected theoretical phase and the measured imaginary and real parts of the cross-spectral density, and the
average intensity" (α = 3.15 in the paper). Only the edges carry the step phase (limited spatial coherence = lateral high-pass).

What the SI leaves open and this module makes explicit (every choice is a function argument the notebook declares):
  * the phase jump at an edge (operational definition): complex Γ₊ sampled on lines perpendicular to the edge and averaged along
    it (complex average, then eq. S28), then the peak-to-trough of φ(s) = atan2(α Im Γ̄₊(s), I₀ + α Re Γ̄₊(s)) within ± window
    of the edge;
  * α per edge = root of |jump(α)| = k₀ Δn h (eq. S28 is applied to the whole profile, so this is a 1D root find; for a single
    point it reduces to α = I₀ tan φ / (Im Γ₊ − Re Γ₊ tan φ));
  * in-focus plane per edge = plane of largest Im Γ₊ peak-to-trough (α-independent), optionally on profiles interpolated along z
    (band-limited), because the edge signal changes over ~0.2 µm in z and the 200 nm sampling alone biases α by up to ~15 %;
  * uncertainty per edge from the spread of α over segments along the edge; combination over edges (inverse variance or mean).
Edges are found automatically (Canny + probabilistic Hough, straight edges), labelled H1.. (horizontal, by y), V1.. (vertical, by
x), D1.. (other, by angle), and the AFM heights are assigned to labels by an explicit user mapping (`assign_heights`).
"""
import numpy as np
from scipy import ndimage, optimize

from . import qpi


# ---------------------------------------------------------------- acquisition checks and intensity normalisation

def check_zstack(z_um, n_planes_paper, step_um_paper, gz_c):
    """SI §4 / Table S2 (footnote d): the calibration uses a *stage* z-stack of 50 planes × 200 nm. Returns a list of warnings
    (empty if the stack matches) and the §5 axial sampling Δg_z = π / ((N − 1) Δz) compared with g_z,c (eq. S29 criterion)."""
    warns = []
    if z_um is None:
        return ["the input is not a stage z-stack (no commanded z per frame, e.g. a single-shot multiplane movie): SI §4 "
                "calibrates α on a 50 × 200 nm stage z-stack; α from a single-shot movie is a different quantity (extension a)"], None
    z = np.asarray(z_um, float); d = np.diff(z)
    step = float(np.median(np.abs(d)))
    if not (np.all(d > 0) or np.all(d < 0)):
        warns.append("commanded z is not monotonic: frames are sorted by z before the retrieval")
    if np.ptp(np.abs(d)) > 0.05 * step:
        warns.append(f"non-uniform z steps ({1e3 * np.abs(d).min():.0f}–{1e3 * np.abs(d).max():.0f} nm): the FFT assumes a uniform step")
    if len(z) != n_planes_paper or abs(step - step_um_paper) > 1e-3:
        warns.append(f"z-stack of {len(z)} planes × {1e3 * step:.0f} nm differs from the SI's {n_planes_paper} × "
                     f"{1e3 * step_um_paper:.0f} nm (§4): the axial sampling changes the §5 high-pass and therefore α")
    dgz = np.pi / ((len(z) - 1) * step)
    if dgz > gz_c:
        warns.append(f"Δg_z = π/((N−1)Δz) = {dgz:.2f} rad/µm > g_z,c = {gz_c:.2f} rad/µm (SI §5, eq. S29): the axial sampling is "
                     f"coarser than the overlap cut-off, extra lateral high-pass filtering (Fig. S5) can lower edge phases and so raise α")
    return warns, dgz


def flat_field(signal_stack, sigma_px):
    """I₀(x, y) of one channel from its own brightfield z-stack: z-mean, Gaussian-smoothed (σ = sigma_px). The SI (§6.4) says the
    channel transmission correction is "based on bright field images" without details: this choice is a heuristic (the
    smoothing must be wider than the coherence area so that the edge contrast is not removed)."""
    m = np.asarray(signal_stack, np.float32).mean(0)
    return ndimage.gaussian_filter(m, sigma_px, mode="nearest").astype(np.float32)


def retrieve_channel(signal_stack, flat, dz_um, px_um, tf, sgn, gz_c):
    """One channel of a stage z-stack (offset already subtracted, frames in increasing z) -> Γ₊ (nz, ny, nx) with I₀ = 1:
    normalise to the flat field (background 1) and apply qpi.cross_spectral_density (SI §2.2, eq. S20–S24)."""
    c = qpi.normalise(signal_stack, flat)
    return qpi.cross_spectral_density(c, dz_um, px_um, tf, sgn, gz_c).astype(np.complex64)


def lit_box(mean_img, fraction, min_cover, margin_px):
    """Crop of a panel to its illuminated part (heuristic, extension): pixels brighter than fraction × 90th percentile; rows and
    columns lit over at least min_cover of their length; minus margin_px. Returns (slice_y, slice_x)."""
    lit = mean_img > fraction * np.percentile(mean_img, 90)
    rows = np.flatnonzero(lit.mean(1) >= min_cover)
    cols = np.flatnonzero(lit[rows].mean(0) >= min_cover)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("no illuminated region found")
    return slice(rows.min() + margin_px, rows.max() + 1 - margin_px), slice(cols.min() + margin_px, cols.max() + 1 - margin_px)


def focus_plane(G, highpass_sigma_px):
    """Global in-focus plane of a retrieved stack: largest spatial standard deviation of the high-passed Im Γ₊ (α-independent).
    Used only to choose the image on which the edges are detected (each edge then gets its own focus plane)."""
    score = [np.std(g.imag - ndimage.gaussian_filter(g.imag, highpass_sigma_px)) for g in G]
    return int(np.argmax(score))


# ---------------------------------------------------------------- edges

def _seg(p0, p1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.hypot(*d); u = d / L
    return dict(p0=p0, p1=p1, u=u, n=np.array([-u[1], u[0]]), ang=float(np.degrees(np.arctan2(u[1], u[0])) % 180))


def edge_length_px(e):
    return float(np.hypot(*(e["p1"] - e["p0"])))


def detect_edges(img, px_um, smooth_sigma_px, canny_sigma_px, canny_quantiles, min_length_um, merge_angle_deg, merge_dist_um,
                 merge_gap_um, hough_threshold, hough_gap_um, border_px=0, n_hough_seeds=8):
    """Straight edges of the staircase in a 2D image (extension: the SI selects lines by hand, Fig. S4b–d). Canny edges
    (hysteresis thresholds = canny_quantiles (low, high) of the gradient magnitude), probabilistic Hough lines (union of n_hough_seeds fixed
    seeds = reproducible; accumulator threshold hough_threshold, gaps up to hough_gap_um bridged), merged when collinear
    (angle < merge_angle_deg, distance < merge_dist_um, gap < merge_gap_um).
    Returns a list of edges {p0, p1 (x, y px), u (unit direction), n (unit normal), ang (deg, [0, 180))}."""
    from skimage import feature, transform
    g = ndimage.gaussian_filter(np.asarray(img, float), smooth_sigma_px)
    e = feature.canny(g, sigma=canny_sigma_px, low_threshold=canny_quantiles[0], high_threshold=canny_quantiles[1], use_quantiles=True)
    if border_px:
        e[:border_px] = e[-border_px:] = False; e[:, :border_px] = e[:, -border_px:] = False
    lines = []
    kw = dict(threshold=hough_threshold, line_length=max(int(min_length_um / px_um / 2), 5), line_gap=max(int(hough_gap_um / px_um), 2))
    for seed in range(n_hough_seeds):
        try:
            lines += transform.probabilistic_hough_line(e, rng=seed, **kw)      # scikit-image >= 0.21
        except TypeError:
            lines += transform.probabilistic_hough_line(e, seed=seed, **kw)     # older scikit-image
    segs = [_seg(a, b) for a, b in lines if np.hypot(b[0] - a[0], b[1] - a[1]) * px_um >= min_length_um / 2]
    return [m for m in merge_edges(segs, px_um, merge_angle_deg, merge_dist_um, merge_gap_um) if edge_length_px(m) * px_um >= min_length_um]


def merge_edges(edges, px_um, angle_deg, dist_um, gap_um):
    """Merge collinear edges (angle difference < angle_deg, centre-to-line distance < dist_um, gap along the line < gap_um) into
    one, longest first; also removes the duplicates left when both lobes of one edge were detected and re-centred (extension)."""
    merged = []
    for s in sorted(edges, key=lambda s: -edge_length_px(s)):
        for m in merged:
            dang = min(abs(s["ang"] - m["ang"]), 180 - abs(s["ang"] - m["ang"]))
            dist = abs(np.dot((s["p0"] + s["p1"]) / 2 - m["p0"], m["n"]))
            t = [np.dot(q - m["p0"], m["u"]) for q in (s["p0"], s["p1"])]
            L = edge_length_px(m)
            if dang < angle_deg and dist * px_um < dist_um and min(t) < L + gap_um / px_um and max(t) > -gap_um / px_um:
                ts = [0, L] + t
                m.update(_seg(m["p0"] + min(ts) * m["u"], m["p0"] + max(ts) * m["u"]))
                break
        else:
            merged.append(dict(s))
    return merged


def split_at_crossings(edges, px_um, gap_um, reach_um, min_length_um, min_angle_deg=20.0):
    """Split an edge where another edge meets it in its interior (a crossing or a T-junction): the step height can change there
    (extension). The piece within ± gap_um of the junction is dropped (the profile there mixes both edges). reach_um: how far
    the other edge may stop short of the junction and still count; edges closer than min_angle_deg to parallel never cut."""
    out = []
    for i, a in enumerate(edges):
        La = edge_length_px(a); cuts = []
        for j, b in enumerate(edges):
            if i == j or abs(np.cross(a["u"], b["u"])) < np.sin(np.radians(min_angle_deg)):
                continue
            # a.p0 + t a.u = b.p0 + r b.u
            M = np.array([a["u"], -b["u"]]).T
            t, r = np.linalg.solve(M, b["p0"] - a["p0"])
            Lb = edge_length_px(b); reach = reach_um / px_um
            if -reach <= r <= Lb + reach and gap_um / px_um < t < La - gap_um / px_um:
                cuts.append(t)
        bounds = [0.0]
        for t in sorted(cuts):
            bounds += [t - gap_um / px_um, t + gap_um / px_um]
        bounds.append(La)
        for t0, t1 in zip(bounds[::2], bounds[1::2]):
            if (t1 - t0) * px_um >= min_length_um:
                out.append(_seg(a["p0"] + t0 * a["u"], a["p0"] + t1 * a["u"]))
    return out


def label_edges(edges, horizontal_tol_deg):
    """Stable, reviewable labels: H1, H2.. horizontal edges (|angle| < tol) by centre y; V1.. vertical by centre x; D1.. others by
    angle. Orientation is kept in e["orientation"] ("horizontal" / "vertical" / "oblique"). Returns the edges sorted by label."""
    groups = {"H": [], "V": [], "D": []}
    for e in edges:
        a = e["ang"]; c = (e["p0"] + e["p1"]) / 2
        if min(a, 180 - a) < horizontal_tol_deg:
            groups["H"].append((c[1], c[0], e)); e["orientation"] = "horizontal"
        elif abs(a - 90) < horizontal_tol_deg:
            groups["V"].append((c[0], c[1], e)); e["orientation"] = "vertical"
        else:
            groups["D"].append((a, c[0], e)); e["orientation"] = "oblique"
    out = []
    for k in "HVD":
        for i, (_, _, e) in enumerate(sorted(groups[k], key=lambda x: (x[0], x[1]))):
            e["label"] = f"{k}{i + 1}"; out.append(e)
    return out


def assign_heights(edges, mapping, steps_nm):
    """Explicit, reviewable AFM height assignment (SI §4, Fig. S4c). steps_nm = {step name: AFM height (nm)} (e.g. {"inner L":
    68.5}); mapping = {edge label: step name} (e.g. {"V2": "inner L"}), written by the user after looking at the labelled edge
    figure. Every label must have been detected and every step name must exist (KeyError otherwise, so that a changed detection
    is noticed). Returns (list of (edge, step name, h_um) in mapping order, list of detected labels without a height)."""
    by = {e["label"]: e for e in edges}
    missing = [k for k in mapping if k not in by]
    if missing:
        raise KeyError(f"labels {missing} of the height mapping were not detected (detected: {sorted(by)}); review the edge figure")
    unknown = sorted({v for v in mapping.values() if v not in steps_nm})
    if unknown:
        raise KeyError(f"step names {unknown} are not in the AFM step list {sorted(steps_nm)}")
    for k, h in steps_nm.items():
        if not (isinstance(h, (int, float)) and h > 0):
            raise ValueError(f"{k}: step height must be a positive number of nm, got {h!r}")
    used = [(by[k], name, 1e-3 * float(steps_nm[name])) for k, name in mapping.items()]
    return used, [e["label"] for e in edges if e["label"] not in mapping]


# ---------------------------------------------------------------- profiles, jump, α

def edge_profile(field, edge, px_um, half_um, end_margin_um, t_range=None, sample_px=0.25):
    """Complex profile of a 2D complex field perpendicular to the edge, averaged (complex mean) along it, ends excluded.
    s (µm) > 0 on the side of the edge normal. t_range = (t0, t1) in px along the edge (default: the whole edge minus the margins)."""
    L = edge_length_px(edge); m = end_margin_um / px_um
    t = np.arange(m, L - m, 1.0) if t_range is None else np.arange(t_range[0], t_range[1], 1.0)
    if len(t) < 3:
        return None, None
    s = np.arange(-half_um / px_um, half_um / px_um + 1e-9, sample_px)
    P = edge["p0"][None, None, :] + t[:, None, None] * edge["u"] + s[None, :, None] * edge["n"]
    coords = [P[..., 1].ravel(), P[..., 0].ravel()]
    re = ndimage.map_coordinates(np.real(field), coords, order=1, mode="nearest").reshape(len(t), len(s))
    im = ndimage.map_coordinates(np.imag(field), coords, order=1, mode="nearest").reshape(len(t), len(s))
    return s * px_um, (re + 1j * im).mean(0)


def peak_to_trough(s_um, y, window_um):
    """Signed peak-to-trough of y within |s| ≤ window: positive if the maximum lies on the + side (normal side) of the minimum."""
    w = np.abs(s_um) <= window_um
    i_max, i_min = np.argmax(np.where(w, y, -np.inf)), np.argmin(np.where(w, y, np.inf))
    return float((y[i_max] - y[i_min]) * np.sign(s_um[i_max] - s_um[i_min]))


def jump(s_um, prof, alpha, window_um, I0=1.0):
    """Phase jump at an edge (operational definition, see module doc): signed peak-to-trough of eq. S28 applied to the
    along-edge-averaged Γ₊ profile, within ± window_um."""
    return peak_to_trough(s_um, qpi.phase(prof, alpha, I0).astype(float), window_um)


def solve_alpha(s_um, prof, target_rad, window_um, I0=1.0, alpha_range=(1e-2, 1e2), n_grid=200):
    """Solve eq. S28 for α (SI §4): the α ≥ 0 at which |jump(α)| = target (the theoretical k₀ Δn h). The jump grows with α until
    atan saturates; the first crossing on a log grid is refined with Brent's method. NaN if the target is not reached."""
    f = lambda a: abs(jump(s_um, prof, a, window_um, I0)) - target_rad
    grid = np.geomspace(*alpha_range, n_grid)
    vals = np.array([f(a) for a in grid])
    k = np.flatnonzero(vals >= 0)
    if len(k) == 0 or k[0] == 0:
        return np.nan
    return float(optimize.brentq(f, grid[k[0] - 1], grid[k[0]], xtol=1e-6))


def alpha_single_point(G, phi_rad, I0=1.0):
    """Closed-form eq. S28 for one complex value: α = I₀ tan φ / (Im Γ₊ − Re Γ₊ tan φ) (reference / check)."""
    t = np.tan(phi_rad)
    return I0 * t / (np.imag(G) - np.real(G) * t)


def _steepest(s_um, y):
    i1, i2 = sorted((int(np.argmax(y)), int(np.argmin(y))))
    if i2 - i1 < 2:
        return np.nan
    k = i1 + int(np.argmax(np.abs(np.diff(y[i1:i2 + 1]))))
    return 0.5 * (s_um[k] + s_um[k + 1])


def recentre(field, edge, px_um, half_um, end_margin_um, block_um, n_iter=2):
    """Move the edge line onto the step: in blocks of block_um along the edge, the steepest point of the Im Γ₊ profile between its
    main maximum and minimum; a straight line fitted through these points (offset and angle) replaces the detected line (the
    detector traces a lobe boundary, not necessarily the step). α-independent. Extension (SI: lines chosen by hand)."""
    edge = dict(edge)
    for _ in range(n_iter):
        L = edge_length_px(edge); m = end_margin_um / px_um; b = block_um / px_um
        starts = np.arange(m, L - m - b + 1e-9, b)
        tc, s0 = [], []
        for t0 in starts:
            s_um, prof = edge_profile(field, edge, px_um, half_um, end_margin_um, (t0, t0 + b))
            if prof is not None:
                tc.append(t0 + b / 2); s0.append(_steepest(s_um, prof.imag) / px_um)
        tc, s0 = np.array(tc), np.array(s0); ok = np.isfinite(s0)
        if ok.sum() == 0:
            return edge
        c = np.polyfit(tc[ok], s0[ok], 1) if ok.sum() >= 3 else np.array([0.0, np.median(s0[ok])])
        edge.update(_seg(edge["p0"] + np.polyval(c, 0) * edge["n"], edge["p0"] + L * edge["u"] + np.polyval(c, L) * edge["n"]))
    return edge


def profile_stack(G, edge, px_um, half_um, end_margin_um, t_range=None, z_upsample=1):
    """Along-edge-averaged complex profiles of every plane of G (nz, ny, nx) -> (s_um, P (nz·z_upsample, ns)). z_upsample > 1:
    band-limited (Fourier) interpolation along z, so that the in-focus jump is not limited by where the surface falls between two
    planes (the edge signal of Γ₊ changes on a ~0.2 µm axial scale, comparable to the 200 nm step). Periodic interpolation:
    valid when the surface is not at the ends of the stack. Extension (the SI takes "the in-focus" plane)."""
    from scipy import signal
    rows = [edge_profile(g, edge, px_um, half_um, end_margin_um, t_range) for g in G]
    if rows[0][1] is None:
        return None, None
    P = np.array([r[1] for r in rows])
    if z_upsample > 1:
        P = signal.resample(P, len(P) * int(z_upsample), axis=0)
    return rows[0][0], P


def edge_focus(P, s_um, window_um):
    """In-focus plane of one edge from its profile stack: the (interpolated) z index of the largest |peak-to-trough| of Im Γ₊
    (α-independent; each edge has its own, the sample can be tilted)."""
    v = np.array([abs(peak_to_trough(s_um, p.imag, window_um)) for p in P])
    return int(np.argmax(v)), v


def measure_edge(G, edge, h_um, k0_dn, px_um, half_um, end_margin_um, window_um, segment_um, z_upsample=1, I0=1.0,
                 alpha_range=(1e-2, 1e2)):
    """SI §4 for one edge: in-focus plane (on z_upsample-interpolated profiles), along-edge-averaged Γ₊ profile, α solving
    |jump(α)| = k₀ Δn h (eq. S28), and α of consecutive segments of segment_um along the edge in the same plane (spread ->
    standard error). Returns a dict (z_index in units of the interpolated grid; z_planes = z_index / z_upsample)."""
    target = k0_dn * h_um
    s_um, P = profile_stack(G, edge, px_um, half_um, end_margin_um, None, z_upsample)
    kz, focus_curve = edge_focus(P, s_um, window_um)
    prof = P[kz]
    a = solve_alpha(s_um, prof, target, window_um, I0, alpha_range)
    L = edge_length_px(edge); m = end_margin_um / px_um; b = segment_um / px_um
    a_seg = []
    for t0 in np.arange(m, L - m - b + 1e-9, b):
        _, Ps = profile_stack(G, edge, px_um, half_um, end_margin_um, (t0, t0 + b), z_upsample)
        a_seg.append(solve_alpha(s_um, Ps[kz], target, window_um, I0, alpha_range) if Ps is not None else np.nan)
    a_seg = np.array(a_seg); ok = np.isfinite(a_seg)
    se = float(np.std(a_seg[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() >= 2 else np.nan
    return dict(label=edge["label"], orientation=edge.get("orientation"), h_nm=1e3 * h_um, phi_theory=float(target),
                z_index=kz, z_planes=kz / z_upsample, focus_curve=focus_curve, s_um=s_um, profile=prof, alpha=a,
                alpha_segments=a_seg, alpha_se=se, n_segments=int(ok.sum()), length_um=float(L * px_um),
                im_jump=peak_to_trough(s_um, prof.imag, window_um))


def combine(alphas, ses, method):
    """Combination of the per-edge α (the SI gives one α from 6 steps without saying how). method "inverse_variance": weights
    1/SE², uncertainty = max(internal 1/√Σw, external Birge scatter); an edge without SE (fewer than 2 segments) gets the largest
    SE of the others (conservative); without any SE it falls back to "mean": unweighted mean, SD/√n. Returns (α, uncertainty)."""
    a = np.asarray(alphas, float); s = np.asarray(ses, float); ok = np.isfinite(a)
    a, s = a[ok], s[ok]
    if len(a) == 0:
        return np.nan, np.nan
    good = np.isfinite(s) & (s > 0)
    if method == "mean" or not good.any():
        return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan)
    if method != "inverse_variance":
        raise ValueError(method)
    s = np.where(good, s, s[good].max())
    w = 1 / s ** 2; m = float(np.sum(w * a) / w.sum())
    internal = 1 / np.sqrt(w.sum())
    external = np.sqrt(np.sum(w * (a - m) ** 2) / ((len(a) - 1) * w.sum())) if len(a) > 1 else 0.0
    return m, float(max(internal, external))


def affine_points(A, pts):
    """Apply a 2x3 affine (x, y, 1) -> (x, y) to points (n, 2)."""
    A = np.asarray(A, float); pts = np.atleast_2d(pts)
    return pts @ A[:, :2].T + A[:, 2]


def invert_affine(A):
    A = np.asarray(A, float); M = np.linalg.inv(A[:, :2])
    return np.hstack([M, -M @ A[:, 2:3]])


def transform_edge(edge, fn):
    """Edge moved to another image's coordinates by fn((n, 2) points) -> (n, 2) points (e.g. reference -> panel affine and crop
    offset), label and orientation kept. Used to measure the same edges in several prism planes without resampling the data."""
    p0, p1 = fn(np.array([edge["p0"], edge["p1"]]))
    out = _seg(p0, p1)
    for k in ("label", "orientation"):
        if k in edge:
            out[k] = edge[k]
    return out


# ---------------------------------------------------------------- forward model (tests and synthetic data)

def simulate_thin_object(phi_xy, tf, z_um, px_um, scale, sgn=+1):
    """Extension (simulation, used by the tests): brightfield intensity of a weak thin phase object φ(x, y) located at z = 0 with
    the SI transfer function (eq. S19): Γ(g⊥, z) = i φ̂(g⊥) ∫ H(|g⊥|, K_z) e^{-i sgn K_z z} dK_z (continuous in z, so any z
    sampling works), normalised so that the strongest in-focus lateral frequency is transferred with gain 1, and
    I(z) = 1 + 2 · scale · Re Γ (eq. S21 for a weak object). `scale` stands for the unequal transfer that α must undo.
    sgn = +1: increasing z = increasing focus depth for qpi.cross_spectral_density(sgn=+1). Returns (I (nz, ny, nx), Γ)."""
    rb, zb, H, _ = tf
    kz = 0.5 * (zb[1:] + zb[:-1]); dkz = zb[1] - zb[0]
    ny, nx = phi_xy.shape
    gr = np.hypot(*np.meshgrid(2 * np.pi * np.fft.fftfreq(ny, px_um), 2 * np.pi * np.fft.fftfreq(nx, px_um), indexing="ij"))
    ir = np.clip(np.digitize(gr, rb) - 1, 0, len(rb) - 2)
    norm = np.abs(H.sum(1) * dkz).max()
    F = np.fft.fft2(phi_xy)
    G = []
    for z in np.atleast_1d(z_um):
        Kr = (H * np.exp(-1j * sgn * kz * z)[None]).sum(1) * dkz / norm        # radial kernel at this z
        G.append(np.fft.ifft2(1j * F * Kr[ir]))
    G = np.array(G)
    return (1 + 2 * scale * G.real).astype(np.float32), G


def ideal_gamma_plus(phi_xy, tf, px_um, gz_c, sgn=+1):
    """Extension (simulation): the Γ₊ the algorithm aims for at the object plane (z = 0), from the model of
    simulate_thin_object with scale 1: only the axial frequencies the mask K keeps (sgn·g_z > g_z,c, i.e. K_z < −g_z,c), no
    finite z range, no mirroring. Used as the oracle in the tests."""
    rb, zb, H, _ = tf
    kz = 0.5 * (zb[1:] + zb[:-1]); dkz = zb[1] - zb[0]
    ny, nx = phi_xy.shape
    gr = np.hypot(*np.meshgrid(2 * np.pi * np.fft.fftfreq(ny, px_um), 2 * np.pi * np.fft.fftfreq(nx, px_um), indexing="ij"))
    ir = np.clip(np.digitize(gr, rb) - 1, 0, len(rb) - 2)
    norm = np.abs(H.sum(1) * dkz).max()
    Kr = (H * (kz < -gz_c)[None]).sum(1) * dkz / norm
    return np.fft.ifft2(1j * np.fft.fft2(phi_xy) * Kr[ir])
