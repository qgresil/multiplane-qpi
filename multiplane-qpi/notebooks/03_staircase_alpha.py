# %% [markdown]
# # 03 — Phase calibration factor α on the etched glass staircase
#
# **Purpose.** Measure the real positive factor α of eq. S28, φ = atan(α Im Γ₊ / (I₀ + α Re Γ₊)), that accounts for the unequal
# transfer of the reference and scattered fields (SI §2.1), exactly as in SI §4: a brightfield **stage z-stack** of a staircase
# etched in borosilicate (SI §8.5) is retrieved (§2.2), the phase jump at each step edge is compared with the theoretical
# φ_th = k₀ Δn h (k₀ = 2π/λ₀ = 10.8 µm⁻¹, Δn = 1.51 − 1 in air, h from AFM), and eq. S28 is solved for α (paper: α = 3.15 from
# 6 steps, Fig. S4e).
#
# **SI references.** §4 and Fig. S4 (method, 50 planes × 200 nm, α = 3.15), eq. S25–S28 (§2.1), §2.2 (algorithm), eq. S24 (g_z,c),
# §5 / eq. S29 / Fig. S5 (axial sampling and high-pass), §6.4 (transmission correction from brightfield), §8.5 (staircase, AFM),
# Table S2 (footnote d: stage scanning).
#
# **Inputs.** Brightfield staircase z-stack (TIFF folder with `acquisition.json`, or `.dat` z-stack), `calibration.json` of
# notebook 01 (panel windows, orientation, pixel size; affines and plane spacing only for extension a), AFM step heights (user list)
# and the explicit mapping of detected edges to AFM steps.
#
# **Outputs** (`<stack>_03_alpha/`): `alpha.json` (α, uncertainty, per-edge table, parameters with tags, warnings), figures `N_*.png`.
#
# **What the SI does not specify, and is declared here as heuristic:** the operational definition of the phase jump (peak-to-trough
# of eq. S28 on the along-edge-averaged complex Γ₊ profile within ± a window), the flat field I₀ (smoothed z-mean of the same
# channel), the in-focus plane per edge, the combination of the per-edge α, the uncertainty (spread over segments along each edge).
#
# **Extensions used** (not in the SI): automatic straight-edge detection with labels (the SI selects lines by hand), splitting of
# edges at junctions, lit-region crop. Optional, **off by default**: (a) α applied to the single-shot 8-plane movie;
# (b) horizontal vs vertical edges.

# %%
import json, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mpqpi import io, qpi, geometry, staircase
from mpqpi.config import Instrument, data_path, output_dir
from mpqpi.params import REG
from mpqpi.provenance import write_json

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
REG.clear()
inst = Instrument()

# ---------------------------------------------------------------- inputs (user)
STACK = REG.declare("stack", str(data_path("261026_staircase/staircase_zstack")), "user",
                    "brightfield stage z-stack of the staircase (TIFF folder + acquisition.json, or .dat z-stack)")
STACK = Path(STACK)
CALIBRATION = Path(REG.declare("calibration", str(data_path("261014_beads/beads_zstack_01_registration/calibration.json")), "user",
                               "calibration.json of notebook 01 (panel windows, orientation, pixel; affines + spacing for ext. a)",
                               note="adapt to the calibration of this session"))
COLOR = REG.declare("color", "BF", "user", "colour key of the TIFF acquisition (cam0_BF.tif / cam1_BF.tif)")
OBJECTIVE = REG.declare("objective", "60x_water", "user", "objective used for the staircase stack")
PLANES = REG.declare("planes", None, "user", "prism plane(s) whose panel is retrieved: None = reference plane of calibration.json, "
                     "a list (e.g. ['P5']) or 'all'. One plane suffices for a stage z-stack (SI §4); no plane registration needed")

# ---------------------------------------------------------------- sample (staircase) and AFM
N_GLASS = REG.declare("n_glass", 1.51, "paper", "SI §4: Δn = n_glass − n_air = 1.51 − 1 (borosilicate)", note="dispersion ignored")
N_ABOVE = REG.declare("n_above", 1.0, "user", "medium in contact with the etched surface (SI §4: air). Water would give Δn = 0.18")
LAMBDA0 = inst.value("brightfield.lambda0_um", "lambda0_um")
K0 = REG.declare("k0", 2 * np.pi / LAMBDA0, "derived", "k₀ = 2π/λ₀ (SI §4: 10.8 µm⁻¹)", "rad/µm")
K0_DN = REG.declare("k0_dn", qpi.k0_dn_h(1.0, N_GLASS, N_ABOVE, LAMBDA0), "derived", "k₀ (n_glass − n_above): rad per µm of step", "rad/µm")
STEPS_NM = REG.declare("afm_steps_nm", {"inner L": 68.5, "outer L": 37.5, "long horizontal line": 51.5}, "measured",
                       "user's AFM of the staircase: 68–69 nm and 37–38 nm (two nested L-steps, both arms), 51–52 nm (long horizontal "
                       "line); midpoints", "nm", note="the SI used 6 steps (Fig. S4c)")
# Edge label -> AFM step. Fill AFTER looking at figure 4 (labels H1.. by y, V1.. by x); unlisted edges are not used.
EDGE_STEPS = REG.declare("edge_steps", {}, "user", "explicit mapping detected edge label -> AFM step name (review figure 4)")

# ---------------------------------------------------------------- acquisition, optics (paper / instrument)
N_PLANES_PAPER = REG.declare("n_planes_paper", 50, "paper", "SI §4 and Table S2: 50 planes")
STEP_PAPER = REG.declare("step_um_paper", 0.2, "paper", "SI §4 and Table S2: 200 nm inter-plane distance (stage scan, footnote d)", "µm")
ALPHA_PAPER = REG.declare("alpha_paper", 3.15, "paper", "SI §4: α = 3.15 for the paper's setup")
OFFSET = inst.value("cameras.offset_counts", "offset_counts")
BAND = inst.value("brightfield.band_um", "band_um")
SPECTRUM = inst.value("brightfield.lamp_spectrum_nm", "lamp_spectrum_nm")
NA_ILL = inst.value("brightfield.na_ill", "na_ill")
NA_OBJ = inst.value(f"objectives.{OBJECTIVE}.na", "na_objective")
N_MEDIUM_TF = REG.declare("n_medium_tf", N_ABOVE, "user", "medium of the transfer function (wave number, maximum NA) for the "
                          "staircase: n_above (the paper describes the surface 'in air')",
                          note="imaging through the glass would argue for 1.51; α absorbs the difference for this geometry only")
NA_CAP = REG.declare("na_cap_fraction", 0.99, "heuristic", "detection NA limited to this fraction of n_medium_tf (light cannot leave "
                     "a medium of index n at NA > n)")
NA_DET = REG.declare("na_det", min(NA_OBJ, NA_CAP * N_MEDIUM_TF), "derived", "min(NA_objective, na_cap_fraction · n_medium_tf)")
GZ_CONV = REG.declare("gz_convention", "si_text", "heuristic", "g_z,c convention: 'si_text' = SI §1 text value ≈ 0.4 rad/µm for "
                      "NA_ill 0.26 (no n̄); 's24' = eq. S24 as printed (×n̄, the value eq. S29's ≈18 planes needs)",
                      note="the SI is inconsistent between the text/S24 and S29")
GZ_C = REG.declare("gz_c", qpi.gz_cutoff(NA_ILL, BAND, N_MEDIUM_TF, GZ_CONV), "derived", f"qpi.gz_cutoff (eq. S24, {GZ_CONV})", "rad/µm")
SCAN_SIGN = REG.declare("scan_sign", +1, "user", "+1 if increasing commanded z = increasing focus depth (selects Γ₊ vs Γ₋)",
                        note="|jump| and α do not depend on it (Γ₋ = conj Γ₊); it only flips which side looks thicker")
TF_SAMPLES = REG.declare("tf_samples", 2_000_000, "heuristic", "Monte-Carlo samples of eq. S19 (statistical noise of H < 1 %)")

# ---------------------------------------------------------------- heuristics of the SI §4 measurement
FLAT_SIGMA_UM = REG.declare("flat_sigma_um", 10.0, "heuristic", "I₀ = z-mean of the channel smoothed with this Gaussian σ (SI §6.4 "
                            "gives no detail); ≫ coherence radius 0.61λ₀/NA_ill = 1.36 µm so that edge contrast is kept", "µm")
LIT_FRACTION = REG.declare("lit_fraction", 0.5, "heuristic", "panel crop: pixels > this × P90 of the z-mean are lit")
LIT_COVER = REG.declare("lit_cover", 0.8, "heuristic", "panel crop: keep rows/columns lit over this fraction")
CROP_MARGIN_UM = REG.declare("crop_margin_um", 1.0, "heuristic", "extra margin removed inside the lit region", "µm")
FOCUS_HP_UM = REG.declare("focus_highpass_um", 1.0, "heuristic", "high-pass σ of the global focus score (detection image only)", "µm")
EDGE_SMOOTH_PX = REG.declare("edge_smooth_px", 1.0, "heuristic", "Gaussian σ on Im Γ₊ before edge detection", "px")
CANNY_SIGMA_PX = REG.declare("canny_sigma_px", 1.5, "heuristic", "Canny σ", "px")
CANNY_Q = REG.declare("canny_quantiles", (0.85, 0.93), "heuristic", "Canny hysteresis thresholds as quantiles of the gradient")
HOUGH_THR = REG.declare("hough_threshold", 10, "heuristic", "probabilistic Hough accumulator threshold")
HOUGH_GAP_UM = REG.declare("hough_gap_um", 1.0, "heuristic", "gaps bridged within a Hough line", "µm")
MIN_EDGE_UM = REG.declare("min_edge_length_um", 4.0, "heuristic", "shortest edge kept (after margins it still gives ≥ 1 segment)", "µm")
MERGE_ANGLE = REG.declare("merge_angle_deg", 4.0, "heuristic", "collinear merge: angle tolerance", "deg")
MERGE_DIST_UM = REG.declare("merge_dist_um", 0.7, "heuristic", "collinear merge: distance (≈ separation of the two lobes)", "µm")
MERGE_GAP_UM = REG.declare("merge_gap_um", 1.5, "heuristic", "collinear merge: largest gap along the line", "µm")
CROSS_GAP_UM = REG.declare("crossing_gap_um", 1.5, "heuristic", "edge split at a junction: ± this removed (profiles mix both edges)", "µm")
CROSS_REACH_UM = REG.declare("crossing_reach_um", 1.5, "heuristic", "another edge ending this close to an edge counts as a junction", "µm")
HOR_TOL = REG.declare("orientation_tol_deg", 20.0, "heuristic", "edges within this of 0°/90° are labelled H/V", "deg")
PROFILE_HALF_UM = REG.declare("profile_half_um", 1.5, "heuristic", "profiles perpendicular to the edge over ± this", "µm")
PROFILE_SAMPLE_PX = REG.declare("profile_sample_px", 0.25, "heuristic", "profile sampling step (bilinear)", "px")
END_MARGIN_UM = REG.declare("end_margin_um", 1.0, "heuristic", "excluded at both ends of an edge (corners, junctions)", "µm")
RECENTRE_BLOCK_UM = REG.declare("recentre_block_um", 1.0, "heuristic", "block length of the line fit that puts the edge on the step", "µm")
JUMP_WINDOW_UM = REG.declare("jump_window_um", 0.8, "heuristic", "jump = peak-to-trough of eq. S28 within ± this of the edge (SI gives "
                             "no definition); ≈ 0.6 × coherence radius 1.36 µm, wide enough for both lobes, narrow enough to "
                             "exclude neighbouring edges ≥ 1.5 µm away", "µm")
Z_UPSAMPLE = REG.declare("z_upsample", 8, "heuristic", "edge profiles interpolated ×8 along z (band-limited) before taking the "
                         "in-focus plane: the Γ₊ edge signal changes over ~0.2 µm in z, so with 200 nm planes alone α depends by up to "
                         "~15 % on where the surface falls between planes (simulation, tests/test_staircase.py); 1 = planes only")
SEGMENT_UM = REG.declare("segment_um", 3.0, "heuristic", "α per segment of this length along each edge: spread -> SE of the edge", "µm")
ALPHA_RANGE = REG.declare("alpha_search_range", (0.01, 100.0), "heuristic", "α root search interval (log grid + Brent)")
COMBINE = REG.declare("combine", "inverse_variance", "heuristic", "per-edge α combined with weights 1/SE² (uncertainty = max(internal, "
                      "Birge external); edges with < 2 segments get the largest SE); 'mean' = unweighted. The SI does not say how its 6 steps were combined")

# ---------------------------------------------------------------- extensions (off by default)
RUN_EXT_SINGLESHOT = REG.declare("run_ext_singleshot", False, "user", "extension (a): apply α to the single-shot 8-plane movie")
RUN_EXT_ORIENTATION = REG.declare("run_ext_orientation", False, "user", "extension (b): horizontal vs vertical edges")
if RUN_EXT_SINGLESHOT:
    SS_MOVIE = Path(REG.declare("ss_movie", str(data_path("261026_staircase/20261026_152033_staircase.dat")), "user",
                                "single-shot 8-plane brightfield movie of the staircase (extension a)"))
    SS_MAX_FRAMES = REG.declare("ss_max_frames", 200, "heuristic", "frames averaged from the movie (ext. a)")
    SS_DZ_SCALE = REG.declare("ss_dz_scale", 1.0, "user", "plane spacing used = calibration plane_spacing_um.mean × this (ext. a)",
                              note="calibration spacing is in the bead sample's space (n_sample of calibration.json); rescaling to "
                                   "the staircase medium is a model choice: state it if ≠ 1")
    SS_SIGN = REG.declare("ss_sign", SCAN_SIGN, "user", "+1 if increasing delta_z_um = increasing focus depth (ext. a)")
    SS_EDGE_STEPS = REG.declare("ss_edge_steps", None, "user", "edge -> AFM step mapping for the movie (None = reuse edge_steps; "
                                "labels may differ, review figure a2)")

WARNINGS = []
def warn(msg):
    WARNINGS.append(msg); print("WARNING:", msg)

warn(f"NA_ill = {NA_ILL} is the paper's value (tag '{REG.params['na_ill'].tag}'): it must equal the condenser setting used for this "
     "stack, and α is only valid for that setting (SI §4: same illumination aperture for all imaging)")
REG.banner()

# %% [markdown]
# ## 1. Stage z-stack: commanded z, checks against SI §4
# The paper calibrates on a stage z-stack of 50 planes × 200 nm (Table S2, footnote d). The axial sampling matters: SI §5 gives
# Δg_z = π / ((N − 1) Δz); when Δg_z > g_z,c (eq. S29) the retrieval adds a lateral high-pass (Fig. S5), which lowers edge phases
# and therefore raises α. A stack that differs from 50 × 200 nm gives an α valid for that sampling only.

# %%
acq = io.load(STACK, inst, color=COLOR)
z = acq.z_um["A"]
msgs, DGZ = staircase.check_zstack(z, N_PLANES_PAPER, STEP_PAPER, GZ_C)
for m in msgs:
    warn(m)
if z is None:
    raise RuntimeError("the paper method needs a stage z-stack (commanded z per frame); for a single-shot movie use extension (a)")
order = np.argsort(z)
DZ = REG.declare("dz_um", float(np.median(np.abs(np.diff(z)))), "derived", "median step of the commanded stage z (io.load)", "µm")
print(f"{STACK.name}: {acq.fmt}, {acq.n_frames} frames, commanded z {z.min():.2f}…{z.max():.2f} µm, step {1e3 * DZ:.0f} nm")
print(f"Δg_z = π/((N−1)Δz) = {DGZ:.2f} rad/µm vs g_z,c = {GZ_C:.2f} rad/µm ({GZ_CONV}); paper stack: "
      f"{np.pi / ((N_PLANES_PAPER - 1) * STEP_PAPER):.2f} rad/µm")

calib = json.loads(CALIBRATION.read_text())
PX = REG.declare("pixel_um", float(calib["pixel_um"]), "derived", f"calibration.json ({CALIBRATION.parent.name})", "µm")
PL = {p["plane"]: p for p in calib["planes"]}
REF = calib["reference_plane"]
plane_list = [REF] if PLANES is None else ([REF] + [p for p in calib["plane_order"] if p != REF] if PLANES == "all" else list(PLANES))
PRIMARY = plane_list[0]
print(f"pixel {1e3 * PX:.1f} nm; planes retrieved: {plane_list} (edges detected on {PRIMARY}, reference plane {REF})")

out = output_dir(STACK, "03_alpha")
print("output:", out)

# %% [markdown]
# ## 2. Retrieval per channel (SI §2.2)
# Per prism plane: panel cut and oriented (`geometry.panel`), camera offset subtracted (DESIGN rule 5), cropped to the lit region,
# divided by its own flat field I₀ (smoothed z-mean, heuristic), then mirrored, 3D FFT, mask g_z > g_z,c inside the CTF, inverse FFT
# (`qpi.cross_spectral_density`). After normalisation I₀ = 1 in eq. S28. No registration between planes: each panel is its own
# z-stack (edges are carried between panels by the calibration affines, not by resampling the data).

# %%
TF = qpi.transfer_function(NA_DET, NA_ILL, N_MEDIUM_TF, BAND, spectrum_nm=SPECTRUM, n_samples=TF_SAMPLES)
crop_px = int(round(CROP_MARGIN_UM / PX))

def retrieve_plane(name):
    sig = io.to_signal(geometry.panel(acq.streams, PL[name])[order], OFFSET)
    box = staircase.lit_box(sig.mean(0), LIT_FRACTION, LIT_COVER, crop_px)
    sig = sig[:, box[0], box[1]]
    flat = staircase.flat_field(sig, FLAT_SIGMA_UM / PX)
    G = staircase.retrieve_channel(sig, flat, DZ, PX, TF, SCAN_SIGN, GZ_C)
    return G, qpi.normalise(sig, flat), box

G0, C0, BOX0 = retrieve_plane(PRIMARY)
KF = staircase.focus_plane(G0, FOCUS_HP_UM / PX)
print(f"{PRIMARY}: panel crop rows {BOX0[0].start}:{BOX0[0].stop}, cols {BOX0[1].start}:{BOX0[1].stop}; "
      f"global focus plane {KF} (z = {z[order][KF]:.2f} µm)")
if KF in (0, len(z) - 1):
    warn("the focus is at the end of the stack: the surface must be inside the scanned range")

fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
ph0 = qpi.phase(G0[KF], ALPHA_PAPER)
for a, img, t, cm in ((ax[0], C0[KF], "intensity / I₀ (focus plane)", "gray"), (ax[1], G0[KF].imag, "Im Γ₊", "RdBu_r"),
                      (ax[2], ph0, f"phase with α = {ALPHA_PAPER} (paper, for display)", "RdBu_r")):
    v = np.percentile(np.abs(img - np.median(img)), 99.5)
    im = a.imshow(img, cmap=cm, vmin=np.median(img) - v, vmax=np.median(img) + v); a.set_title(t); a.axis("off")
    fig.colorbar(im, ax=a, shrink=0.7)
fig.suptitle(f"{PRIMARY}, z-stack retrieval (Fig. S4a,b analogue)")
fig.savefig(out / "2_retrieval.png", dpi=200); plt.show()

# %% [markdown]
# ## 3. Edges (extension: automatic detection; the SI selects lines by hand)
# Straight edges are detected on Im Γ₊ of the focus plane (α-independent), re-centred on the step (line fit through the steepest
# point of the Im Γ₊ profile in 1 µm blocks), de-duplicated (both lobes of an edge), split where another edge meets them (the
# step height can change there), and labelled H1.. (horizontal, by y) / V1.. (vertical, by x) / D1.. (oblique).

# %%
edge_kw = dict(half_um=PROFILE_HALF_UM, end_margin_um=END_MARGIN_UM)
edges = staircase.detect_edges(G0[KF].imag, PX, EDGE_SMOOTH_PX, CANNY_SIGMA_PX, CANNY_Q, MIN_EDGE_UM, MERGE_ANGLE, MERGE_DIST_UM,
                               MERGE_GAP_UM, HOUGH_THR, HOUGH_GAP_UM, border_px=crop_px)
edges = [staircase.recentre(G0[KF], e, PX, block_um=RECENTRE_BLOCK_UM, **edge_kw) for e in edges]
edges = staircase.merge_edges(edges, PX, MERGE_ANGLE, MERGE_DIST_UM, MERGE_GAP_UM)
edges = staircase.split_at_crossings(edges, PX, CROSS_GAP_UM, CROSS_REACH_UM, MIN_EDGE_UM)
edges = staircase.label_edges(edges, HOR_TOL)
if not edges:
    raise RuntimeError("no straight edges found: check focus, crop and the detection heuristics")
print(f"{len(edges)} edges on {PRIMARY}:")
for e in edges:
    c = PX * (e["p0"] + e["p1"]) / 2
    print(f"  {e['label']:4s} {e['orientation']:10s} length {PX * staircase.edge_length_px(e):5.1f} µm, centre (x, y) = "
          f"({c[0]:5.1f}, {c[1]:5.1f}) µm, angle {e['ang']:5.1f}°")

# %% [markdown]
# ## 4. AFM heights → edges (explicit mapping, to review)
# Look at the figure: every edge carries its label. Write `EDGE_STEPS = {"V1": "outer L", "H3": "long horizontal line", ...}` in
# the parameter cell (step names from `STEPS_NM`) and re-run. Mapped edges are drawn in colour with their height; unmapped edges
# in grey are not used. The mapping is stored in alpha.json.

# %%
used, unassigned = staircase.assign_heights(edges, EDGE_STEPS, STEPS_NM)
step_col = {n: f"C{i}" for i, n in enumerate(STEPS_NM)}
fig, ax = plt.subplots(figsize=(10, 10 * C0.shape[1] / C0.shape[2] + 0.5), layout="constrained")
v = np.percentile(np.abs(G0[KF].imag), 99.5)
ax.imshow(G0[KF].imag, cmap="RdBu_r", vmin=-v, vmax=v)
names = {e["label"]: n for e, n, _ in used}
for e in edges:
    col = step_col[names[e["label"]]] if e["label"] in names else "0.5"
    ax.plot(*np.array([e["p0"], e["p1"]]).T, color=col, lw=2.5 if e["label"] in names else 1)
    c = (e["p0"] + e["p1"]) / 2
    txt = e["label"] + (f"\n{names[e['label']]} ({STEPS_NM[names[e['label']]]} nm)" if e["label"] in names else "")
    ax.text(*c, txt, color="k", fontsize=8, ha="center", va="center", bbox=dict(fc="w", ec=col, alpha=0.8, pad=1))
ax.set_title(f"{PRIMARY}: detected edges on Im Γ₊ (focus plane); colour = AFM step assigned, grey = not used")
ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
fig.savefig(out / "4_edges_labels.png", dpi=200); plt.show()
print("unassigned edges (not used):", unassigned)
if not used:
    raise RuntimeError("EDGE_STEPS is empty: assign AFM steps to the edge labels of figure 4 in the parameter cell, then re-run")

# %% [markdown]
# ## 5. Phase jump per edge and α (SI §4, eq. S28)
# For each mapped edge: in-focus plane (largest Im Γ₊ peak-to-trough, on profiles interpolated ×`Z_UPSAMPLE` along z), complex
# Γ₊ profile perpendicular to the edge averaged along
# it, φ(s; α) = atan2(α Im Γ̄₊, I₀ + α Re Γ̄₊) with I₀ = 1, jump(α) = peak-to-trough of φ within ± `JUMP_WINDOW_UM`.
# α solves |jump(α)| = k₀ Δn h (for one point this is α = I₀ tan φ / (Im Γ₊ − Re Γ₊ tan φ)). The edge's SE comes from the
# spread of α over `SEGMENT_UM` segments along it.

# %%
mkw = dict(window_um=JUMP_WINDOW_UM, segment_um=SEGMENT_UM, z_upsample=Z_UPSAMPLE, alpha_range=ALPHA_RANGE, **edge_kw)

def measure_all(G, used_edges):
    return [staircase.measure_edge(G, e, h, K0_DN, PX, **mkw) | {"step": n} for e, n, h in used_edges]

res = measure_all(G0, used)
alpha, alpha_u = staircase.combine([r["alpha"] for r in res], [r["alpha_se"] for r in res], COMBINE)
for r in res:
    r["jump_at_alpha"] = staircase.jump(r["s_um"], r["profile"], alpha, JUMP_WINDOW_UM)
print(f"{'edge':5s} {'step':22s} {'h (nm)':>7s} {'φ_th':>6s} {'z':>5s} {'L (µm)':>6s} {'α':>7s} {'SE':>6s} {'n_seg':>5s} {'jump(α̂)':>8s}")
for r in res:
    print(f"{r['label']:5s} {r['step']:22s} {r['h_nm']:7.1f} {r['phi_theory']:6.3f} {r['z_planes']:5.1f} {r['length_um']:6.1f} "
          f"{r['alpha']:7.3f} {r['alpha_se']:6.3f} {r['n_segments']:5d} {r['jump_at_alpha']:+8.3f}")
print("z = in-focus position in plane units (interpolated)")
print(f"\nα = {alpha:.3f} ± {alpha_u:.3f} ({COMBINE}, {np.isfinite([r['alpha'] for r in res]).sum()} edges)   paper: {ALPHA_PAPER}")
if any(not np.isfinite(r["alpha"]) for r in res):
    warn("α not found for some edges (target phase not reached in the α search range): check the mapping and the focus")
spread = np.nanstd([r["alpha"] for r in res]) / alpha
if spread > 0.15:
    warn(f"per-edge α scatter {100 * spread:.0f} % (> 15 %): check the height mapping, the focus and edges near junctions")
signs = {r["label"]: np.sign(r["jump_at_alpha"]) for r in res}
print("jump signs (+ = thicker on the normal side, check against the AFM):", signs)

# %% [markdown]
# ### Fig. S4d/e analogue: profiles, measured vs theoretical phase, residuals

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
for r in res:
    ph = qpi.phase(r["profile"], alpha)
    ax[0].plot(r["s_um"], ph * np.sign(r["jump_at_alpha"]), color=step_col[r["step"]], lw=1, label=f"{r['label']} ({r['h_nm']:.1f} nm)")
ax[0].axvspan(-JUMP_WINDOW_UM, JUMP_WINDOW_UM, color="0.9", zorder=-1)
ax[0].set(xlabel="distance across the edge (µm)", ylabel=f"phase (rad), α = {alpha:.2f}", title="edge profiles (window shaded)")
ax[0].legend(fontsize=7, ncol=2)
th = np.array([r["phi_theory"] for r in res]); me = np.array([abs(r["jump_at_alpha"]) for r in res])
err = np.array([abs(staircase.jump(r["s_um"], r["profile"], alpha + alpha_u, JUMP_WINDOW_UM)) - abs(r["jump_at_alpha"]) for r in res])
for r, t, m in zip(res, th, me):
    ax[1].plot(t, m, "o", color=step_col[r["step"]]); ax[1].annotate(r["label"], (t, m), fontsize=7, xytext=(3, 3), textcoords="offset points")
lim = [0, 1.15 * max(th.max(), me.max())]
ax[1].plot(lim, lim, "k--", lw=0.8, label="1:1")
ax[1].set(xlim=lim, ylim=lim, xlabel="theoretical phase k₀Δn h (rad)", ylabel="measured jump (rad)",
          title=f"measured vs theory, α = {alpha:.2f} ± {alpha_u:.2f} (Fig. S4e)"); ax[1].legend()
ax[2].errorbar(th, me - th, yerr=np.abs(err), fmt="o", color="k", ms=4)
ax[2].axhline(0, color="k", lw=0.6)
for r, t, m in zip(res, th, me):
    ax[2].annotate(r["label"], (t, m - t), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[2].set(xlabel="theoretical phase (rad)", ylabel="measured − theory (rad)", title="residuals (bars: effect of ± uncertainty of α)")
fig.savefig(out / "5_measured_vs_theory.png", dpi=200); plt.show()

fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
phf = qpi.phase(G0[KF], alpha); v = np.percentile(np.abs(phf), 99.7)
im = ax.imshow(phf, cmap="RdBu_r", vmin=-v, vmax=v); fig.colorbar(im, ax=ax, label="rad", shrink=0.8)
ax.set_title(f"{PRIMARY}: calibrated phase, α = {alpha:.2f} (focus plane {KF})"); ax.axis("off")
fig.savefig(out / "5_phase_calibrated.png", dpi=200); plt.show()

# %% [markdown]
# ## 6. Other prism planes (only if `PLANES` lists several)
# The same edges are mapped into each panel with the calibration affines (reference → panel; no resampling of the data) and
# re-centred there. α per plane shows how much the calibration depends on the detection path. The output α is the first plane's.

# %%
per_plane = {PRIMARY: dict(alpha=alpha, u=alpha_u, edges=[r["label"] for r in res])}
if len(plane_list) > 1:
    A_ref_from_primary = staircase.invert_affine(PL[PRIMARY]["affine_from_reference"])
    for name in plane_list[1:]:
        G, _, box = retrieve_plane(name)
        A = np.asarray(PL[name]["affine_from_reference"], float)
        def to_plane(pts, A=A, box=box):
            p = pts + [BOX0[1].start, BOX0[0].start]                            # primary crop -> primary panel
            p = staircase.affine_points(A, staircase.affine_points(A_ref_from_primary, p))
            return p - [box[1].start, box[0].start]
        kf = staircase.focus_plane(G, FOCUS_HP_UM / PX)
        ue = [(staircase.recentre(G[kf], staircase.transform_edge(e, to_plane), PX, block_um=RECENTRE_BLOCK_UM, **edge_kw), n, h)
              for e, n, h in used]
        rp = measure_all(G, ue)
        a, u = staircase.combine([r["alpha"] for r in rp], [r["alpha_se"] for r in rp], COMBINE)
        per_plane[name] = dict(alpha=a, u=u, per_edge={r["label"]: r["alpha"] for r in rp})
        del G
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    names_p = list(per_plane)
    ax.errorbar(range(len(names_p)), [per_plane[n]["alpha"] for n in names_p], yerr=[per_plane[n]["u"] for n in names_p], fmt="o")
    ax.axhline(ALPHA_PAPER, color="0.5", ls=":", label=f"paper {ALPHA_PAPER}")
    ax.set(xticks=range(len(names_p)), xticklabels=names_p, ylabel="α", title="α per prism plane (stage z-stack of each panel)")
    ax.legend(); fig.savefig(out / "6_alpha_per_plane.png", dpi=200); plt.show()
for n, d in per_plane.items():
    print(f"{n}: α = {d['alpha']:.3f} ± {d['u']:.3f}")

# %% [markdown]
# ## 7. Output `alpha.json`

# %%
def edge_row(r):
    return {k: (float(r[k]) if isinstance(r[k], (float, np.floating)) else r[k]) for k in
            ("label", "step", "orientation", "h_nm", "phi_theory", "z_planes", "length_um", "alpha", "alpha_se", "n_segments",
             "jump_at_alpha", "im_jump")} | {"alpha_segments": [float(x) for x in r["alpha_segments"]]}

payload = {
    "alpha": {"value": alpha, "uncertainty": alpha_u, "combination": COMBINE, "plane": PRIMARY,
              "definition": "eq. S28 solved so that the peak-to-trough phase jump at each mapped step edge equals k0*dn*h (SI §4)"},
    "acquisition": {"mode": "stage z-stack", "n_planes": int(len(z)), "dz_um": DZ, "delta_gz": DGZ, "gz_c": GZ_C,
                    "gz_convention": GZ_CONV, "na_ill": NA_ILL, "na_det": NA_DET, "n_medium_tf": N_MEDIUM_TF, "scan_sign": SCAN_SIGN},
    "sample": {"n_glass": N_GLASS, "n_above": N_ABOVE, "lambda0_um": LAMBDA0, "afm_steps_nm": STEPS_NM, "edge_steps": EDGE_STEPS},
    "edges": [edge_row(r) for r in res],
    "per_plane": per_plane,
    "warnings": WARNINGS,
}
write_json(out / "alpha.json", payload, "03_staircase_alpha", {"stack": STACK, "calibration": CALIBRATION})
print("wrote", out / "alpha.json")
print("\nheuristics used:", ", ".join(p.name for p in REG.heuristics()))

# %% [markdown]
# ## Extension (b) — not in the paper: horizontal vs vertical edges
# In single-shot 8-plane mode horizontal edges were seen to vanish, but not in single-panel z-scans. Here α (and Im Γ₊ jump per nm
# of step) is compared between horizontal and vertical edges of the z-stack: equal values mean the z-scan retrieval treats both
# orientations alike, so the single-shot loss comes from the multiplane path (registration, plane spacing), not from the sample.

# %%
if RUN_EXT_ORIENTATION:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for i, o in enumerate(("horizontal", "vertical")):
        rr = [r for r in res if r["orientation"] == o]
        if not rr:
            print(f"no mapped {o} edge"); continue
        a, u = staircase.combine([r["alpha"] for r in rr], [r["alpha_se"] for r in rr], COMBINE)
        g = np.array([abs(r["im_jump"]) / r["h_nm"] for r in rr])
        print(f"{o:10s}: {len(rr)} edges, α = {a:.3f} ± {u:.3f}; |Im Γ₊ jump| per nm = {1e3 * g.mean():.3f} ± {1e3 * g.std():.3f} mrad/nm")
        ax[0].errorbar([i] * len(rr), [r["alpha"] for r in rr], yerr=[r["alpha_se"] for r in rr], fmt="o", alpha=0.6)
        ax[0].errorbar(i + 0.15, a, yerr=u, fmt="s", color="k")
        ax[1].plot([i] * len(rr), 1e3 * g, "o")
    for a_ in ax:
        a_.set_xticks([0, 1]); a_.set_xticklabels(["horizontal", "vertical"]); a_.set_xlim(-0.5, 1.5)
    ax[0].set(ylabel="α", title="α per edge orientation"); ax[1].set(ylabel="|Im Γ₊ jump| / h (mrad/nm)", title="α-independent")
    fig.savefig(out / "b_orientation.png", dpi=200); plt.show()

# %% [markdown]
# ## Extension (a) — not in the paper's calibration: α applied to the single-shot 8-plane movie
# The movie is averaged over frames, each panel is offset-subtracted, co-registered onto the reference plane with the calibration
# affines (bilinear, SI §6.4), divided by its own smoothed brightfield image (per-channel transmission, SI §6.4), stacked by
# `delta_z_um` with the calibration plane spacing, and retrieved. With 8 planes the §5 high-pass is stronger (Fig. S5), so the
# z-stack α is not expected to make these edge phases quantitative; the α that would be needed is reported as α₈ (a different,
# geometry-specific quantity, not to be used as the paper's α).

# %%
if RUN_EXT_SINGLESHOT:
    mv = io.load(SS_MOVIE, inst)
    if mv.z_um["A"] is not None:
        warn(f"{SS_MOVIE.name} is a z-stack, not a single-shot movie")
    T = min(mv.n_frames, SS_MAX_FRAMES)
    mean = {s: io.to_signal(np.asarray(v[:T], np.float32).mean(0), OFFSET) for s, v in mv.streams.items()}
    names8 = sorted(PL, key=lambda n: SS_SIGN * PL[n]["delta_z_um"])
    regd = np.array([geometry.warp(geometry.panel(mean, PL[n]), PL[n]["affine_from_reference"]) for n in names8])
    ok = np.isfinite(regd).all(0) & (regd > 0).all(0)
    box = staircase.lit_box(np.where(ok, regd.min(0), 0), LIT_FRACTION, LIT_COVER, crop_px)
    regd = regd[:, box[0], box[1]]
    c8 = np.array([qpi.normalise(p[None], staircase.flat_field(p[None], FLAT_SIGMA_UM / PX))[0] for p in regd])
    dz8 = REG.declare("ss_dz_um", calib["plane_spacing_um"]["mean"] * SS_DZ_SCALE, "derived",
                      "calibration plane_spacing_um.mean × ss_dz_scale", "µm")
    _, dgz8 = staircase.check_zstack(np.arange(len(names8)) * dz8, N_PLANES_PAPER, STEP_PAPER, GZ_C)
    print(f"{SS_MOVIE.name}: {T} frames averaged; planes {names8}; Δz = {1e3 * dz8:.0f} nm, Δg_z = {dgz8:.2f} rad/µm "
          f"(g_z,c = {GZ_C:.2f})")
    G8 = qpi.cross_spectral_density(c8, dz8, PX, TF, +1, GZ_C)
    kf8 = staircase.focus_plane(G8, FOCUS_HP_UM / PX)
    e8 = staircase.detect_edges(G8[kf8].imag, PX, EDGE_SMOOTH_PX, CANNY_SIGMA_PX, CANNY_Q, MIN_EDGE_UM, MERGE_ANGLE, MERGE_DIST_UM,
                                MERGE_GAP_UM, HOUGH_THR, HOUGH_GAP_UM, border_px=crop_px)
    e8 = [staircase.recentre(G8[kf8], e, PX, block_um=RECENTRE_BLOCK_UM, **edge_kw) for e in e8]
    e8 = staircase.merge_edges(e8, PX, MERGE_ANGLE, MERGE_DIST_UM, MERGE_GAP_UM)
    e8 = staircase.label_edges(staircase.split_at_crossings(e8, PX, CROSS_GAP_UM, CROSS_REACH_UM, MIN_EDGE_UM), HOR_TOL)
    mapping8 = EDGE_STEPS if SS_EDGE_STEPS is None else SS_EDGE_STEPS
    if SS_EDGE_STEPS is None:
        warn("extension a reuses the z-stack edge mapping: the labels of the movie may differ, check figure a2")
    mapping8 = {k: v for k, v in mapping8.items() if k in {e["label"] for e in e8}}
    used8, _ = staircase.assign_heights(e8, mapping8, STEPS_NM)
    r8 = measure_all(G8, used8)
    for r in r8:
        r["jump_at_alpha"] = staircase.jump(r["s_um"], r["profile"], alpha, JUMP_WINDOW_UM)     # at the z-stack α
    a8, u8 = staircase.combine([r["alpha"] for r in r8], [r["alpha_se"] for r in r8], COMBINE)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
    v = np.percentile(np.abs(G8[kf8].imag), 99.5)
    ax[0].imshow(qpi.phase(G8[kf8], alpha), cmap="RdBu_r"); ax[0].set_title(f"8-plane phase, z-stack α = {alpha:.2f}"); ax[0].axis("off")
    ax[1].imshow(G8[kf8].imag, cmap="RdBu_r", vmin=-v, vmax=v); ax[1].set_title("a2: Im Γ₊ and edge labels"); ax[1].axis("off")
    for e in e8:
        ax[1].plot(*np.array([e["p0"], e["p1"]]).T, color="k", lw=1); ax[1].text(*(e["p0"] + e["p1"]) / 2, e["label"], fontsize=8)
    for r in r8:
        m = abs(r["jump_at_alpha"])
        ax[2].plot(r["phi_theory"], m, "o", color="C1" if r["orientation"] == "horizontal" else "C0")
        ax[2].annotate(r["label"], (r["phi_theory"], m), fontsize=7)
    lim = [0, 1.15 * max([r["phi_theory"] for r in r8] + [0.1])]
    ax[2].plot(lim, lim, "k--", lw=0.8)
    ax[2].set(xlabel="k₀Δn h (rad)", ylabel=f"8-plane jump at α = {alpha:.2f} (rad)",
              title=f"single shot (orange: horizontal); α₈ needed = {a8:.2f} ± {u8:.2f}")
    fig.savefig(out / "a_singleshot.png", dpi=200); plt.show()
    print(f"single-shot 8 planes: α₈ = {a8:.3f} ± {u8:.3f} (z-stack α = {alpha:.3f}); per edge:",
          {r["label"]: round(r["alpha"], 3) for r in r8})
    payload["extension_a_singleshot"] = {"movie": str(SS_MOVIE), "frames": int(T), "dz_um": dz8, "alpha_8": a8, "u": u8,
                                         "edges": [edge_row(r) for r in r8], "note": "jump_at_alpha uses the z-stack α"}
    write_json(out / "alpha.json", payload, "03_staircase_alpha", {"stack": STACK, "calibration": CALIBRATION, "movie": SS_MOVIE})
    print("alpha.json updated with extension (a) (the paper α above is unchanged)")
