# %% [markdown]
# # 04 — Brightfield 200 nm polystyrene beads: 3D quantitative phase and validation of α
#
# **Purpose.** Retrieve the 3D quantitative phase of 200 nm polystyrene beads in agarose from a brightfield stage z-scan
# (50 × 200 nm), with the α calibrated on the staircase (notebook 03), and check it against the SI. This is a validation step:
# nothing here re-calibrates α.
#
# **SI references** (Descloux et al., Nat. Photonics 12, 165 (2018), Supplementary Information)
# - §1–2, eq. S19–S28: CTF, cross-spectral density, z-mirroring, cut-off g_z,c, phase (core module `mpqpi.qpi`).
# - Fig. S1d,e: measured 3D spectrum vs CTF support.
# - §3.2: forward model (point scatterer ⊗ H). §3.3, Fig. S3a,b: bead-averaged xz intensity and phase, FWHM lateral
#   theory 0.30 / experiment 0.36 µm, axial 0.55 / 0.56 µm. Fig. S3c: colour-coded max projection.
# - §4: 200 nm beads, theory 0.6 · k₀Δn · d = 0.34 rad, measured 0.33 ± 0.014 rad (mean of 15 beads); Fig. S4e.
# - §5: Fourier sampling (eq. S29–S30); 8-plane single-shot mode gave 0.33 rad. §6.4: affine + bilinear registration,
#   brightfield-based channel transmission. §8.4: sample. Table S2: 50 frames, 200 nm, BrightLine 582/75, 20 ms.
#
# **Inputs.** Bead z-scan (`.dat` + `.info` or TIFF folder), `calibration.json` (notebook 01), `alpha.json` (notebook 03),
# optionally a single-shot 8-plane movie. Instrument facts from `instruments/lob_multiplane.yaml`.
#
# **Outputs** (`<stem>_04_beads/`): `beads.json` (results + parameters with tags + provenance), figures `NN_*.png`.
#
# **Extensions (not in the paper, all OFF by default; they never feed the paper results unless switched on):**
# E1 NA_ill estimated from the measured spectrum (§2b), E2 brightfield plane positions vs fluorescence Δ (§7),
# E3 sample-motion correction between z-steps (§1b; if switched on it changes every downstream number).
# Diagnostics that do not change any result: implied α from the beads (§4), peak phase vs focus position (§5b).

# %%
from pathlib import Path
import json, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from mpqpi import io, qpi, geometry, beads
from mpqpi.config import Instrument, data_path, output_dir
from mpqpi.params import REG
from mpqpi.provenance import write_json

REG.clear()
inst = Instrument()
WARNINGS = []
def warn(msg):
    WARNINGS.append(msg); print("WARNING:", msg)

# ---------------------------------------------------------------- inputs (user)
DATA_FILE = data_path(REG.declare("data_file", "261022/20261022_145141_PSBeads_200nm_Stock_1-100_Agarose_2_DIC5-1.dat", "user",
                                  "brightfield 200 nm PS bead z-scan, 50 × 200 nm (SI §8.4, Table S2)"))
CALIBRATION = data_path(REG.declare("calibration_json", "CHANGE_ME_fluorescent_bead_stack_01_registration/calibration.json", "user",
                                    "notebook 01 output for this session (affines, Δ, transmission)"))
ALPHA_JSON = data_path(REG.declare("alpha_json", "CHANGE_ME_staircase_stack_03_alpha/alpha.json", "user",
                                   "notebook 03 output (staircase α, SI §4)"))
SINGLE_SHOT_FILE = REG.declare("single_shot_file", None, "user",
                               "single-shot 8-plane movie of the same sample (SI §5); None = use single frames of the z-scan")
OBJECTIVE = REG.declare("objective", "60x_water", "user", "objective used (key in the instrument file)")

# ---------------------------------------------------------------- sample (SI §4, §8.4)
N_MEDIUM = REG.declare("n_medium", 1.33, "user", "agarose 0.5 % ≈ water (SI §4: n_agarose = 1.33)",
                       note="also the mean index n̄ of eq. S19 (Ewald radius, NA/n̄)")
N_BEAD = REG.declare("n_bead", 1.59, "paper", "SI §4 (n_beads = 1.59)")
BEAD_D_UM = REG.declare("bead_d_um", 0.2, "manufacturer", "Polybead 0.2 µm (SI §8.4)", "µm")
BEAD_FACTOR = REG.declare("bead_factor", 0.6, "paper", "SI §4: OPL averaged over the bead, 'estimated correction factor ≈ 0.6'",
                          note="no derivation in the SI")
PHI_PAPER = REG.declare("phi_paper_rad", [0.33, 0.014, 15], "paper", "SI §4: 0.33 ± 0.014 rad, average over 15 beads", "rad")
PHI_PAPER_8PLANE = REG.declare("phi_paper_8plane_rad", 0.33, "paper", "SI §5: 8-plane prism mode, average phase 0.33 rad", "rad")
FWHM_PAPER = REG.declare("fwhm_paper_um", {"lateral": {"theory": 0.30, "experiment": 0.36}, "axial": {"theory": 0.55, "experiment": 0.56}},
                         "paper", "SI Fig. S3b (titles of the line plots)", "µm")
STAGE_TO_SAMPLE_Z = REG.declare("stage_to_sample_z", 1.0, "user",
                                "sample z per stage z; 1 for a water objective in an aqueous sample (no focal shift)")

# ---------------------------------------------------------------- optics (instrument file)
NA_DET = inst.value(f"objectives.{OBJECTIVE}.na", "na_det")
NA_ILL_USER = REG.declare("na_ill_user", None, "user", "condenser NA actually set on this microscope; None = not recorded")
BAND_UM = tuple(inst.value("brightfield.band_um"))
LAMBDA0_UM = inst.value("brightfield.lambda0_um")
SPECTRUM = inst.value("brightfield.lamp_spectrum_nm")
OFFSET = inst.value("cameras.offset_counts")
GZ_C_CONVENTION = REG.declare("gz_c_convention", "si_text", "heuristic",
                              "eq. S24 without n̄ reproduces the SI text's ≈ 0.4 rad/µm; 's24' (with n̄) gives 0.53",
                              note="for 50 × 200 nm both remove the same g_z rows")
TF_SAMPLES = REG.declare("tf_samples", 2_000_000, "heuristic", "Monte-Carlo samples of eq. S19 (support stable above ~5·10⁵)")

# ---------------------------------------------------------------- processing choices (the SI gives none)
FLAT_SIGMA_PX = REG.declare("flat_sigma_px", 25, "heuristic",
                            "I₀(x, y) = Gaussian-smoothed z-median of the channel's own stack (SI §6.4 'based on bright field "
                            "images', no procedure given); 25 px ≈ 2.8 µm ≫ bead image, ≪ vignetting scale", "px")
FLAT_STAT = REG.declare("flat_statistic", "median", "heuristic", "z-median: beads occupy few frames per pixel, so it ignores them")
LAMP_CORRECTION = REG.declare("lamp_correction", False, "heuristic",
                              "divide each frame by its mean (lamp drift); off: the drift is reported, and a uniform offset per frame "
                              "only reaches g⊥ ≈ 0, which the CTF mask removes")
SIGN_RULE = REG.declare("sign_rule", "qpi.choose_sign", "heuristic",
                        "spectral half (eq. S22–S23; the SI does not say which) chosen so that the beads come out positive "
                        "(n_bead > n_medium) on the reference channel")
R_BINS = REG.declare("spectrum_r_bins", 90, "heuristic", "radial bins of the azimuthal spectrum average (§2)")
NOISE_ROWS_FACTOR = REG.declare("spectrum_noise_rows", 1.2, "heuristic",
                                "noise floor from rows |g_z| > factor × axial extent of the CTF support")
SUPPORT_MIN_FRACTION = REG.declare("support_min_fraction", 0.7, "heuristic",
                                   "warn if less above-noise power lies inside the CTF support (simulated data from the same CTF: "
                                   "≈ 1.0 noise-free, ≈ 0.8 at the noise level of real stacks)")
DETECT_K_SIGMA = REG.declare("detect_k_sigma", 8.0, "heuristic", "detection on the phase max projection: median + k · robust σ")
FOOTPRINT_PX = REG.declare("detect_footprint_px", 9, "heuristic", "local-maximum neighbourhood (≈ 1 µm)", "px")
EDGE_PX = REG.declare("edge_px", 20, "heuristic", "ignore detections within this distance of the panel border", "px")
SEARCH_PX = REG.declare("search_px", 2, "heuristic", "lateral search box (±) around a detection for the 3D peak", "px")
ISOLATION_UM = REG.declare("isolation_um", 1.7, "heuristic", "beads with a neighbour closer than this are not used", "µm")
Z_EDGE_FRAMES = REG.declare("z_edge_frames", 5, "heuristic",
                            "beads focused within N frames of either stack end are not used (z-mirroring, §2.2 step 2); "
                            "§5b shows the simulated peak vs focus position", "frames")
PEAK_ESTIMATOR = REG.declare("peak_estimator", "fourier", "heuristic",
                             "peak of the phase from Γ₊ interpolated in Fourier space (band-limited) at the sub-voxel vertex, then "
                             "eq. S28: unbiased (< 1 %) for sub-voxel offsets and noise in simulation; the max voxel is biased low by "
                             "sampling (≈ −10 % on average, up to −28 %). SI §4 does not say how the peak was read")
LOCATE_SIGMA = REG.declare("locate_sigma_vox", 0.0, "heuristic",
                           "smoothing (voxels) for locating the peak; 0: simulation shows smoothing biases the value low", "vox")
INTERP_HALF = REG.declare("interp_half_vox", [8, 8, 8], "heuristic", "crop (±z, ±y, ±x voxels) for the Fourier peak interpolation")
AVG_CHANNELS = REG.declare("average_channels", "reference", "heuristic",
                           "bead average on the reference channel only ('all' pools channels whose bead phase differs)")
AVG_HALF_UM = REG.declare("average_half_um", [2.0, 1.33], "heuristic", "bead-average crop half size (axial, lateral), Fig. S3b span", "µm")
AVG_MARGIN = REG.declare("average_margin_vox", 3, "heuristic", "extra voxels around each crop for the Fourier shift")
SIM_OBJECT = REG.declare("sim_object", "point", "paper", "SI §3.2 (Dirac point scatterer); 'sphere' = 200 nm sphere FT (option)")
SIM_SPHERE_TOO = REG.declare("sim_sphere_too", True, "heuristic", "also show the 200 nm sphere model for comparison")
SIM_LATERAL = REG.declare("sim_lateral_px", 64, "heuristic", "lateral size of the simulation grid", "px")
SIM_NORMALISATION = REG.declare("sim_normalisation", "peak_to_measured", "heuristic",
                                "the SI states no absolute normalisation of the simulated phase: simulated curves are scaled so that "
                                "their peak equals the measured average peak (display only); only the FWHMs are compared")
DELTA_CONVENTION = REG.declare("delta_convention", "stage_focus", "heuristic",
                               "calibration Δ_p = stage z at which plane p is in focus minus that of the reference (must match notebook 01)")
SPACING_RULE = REG.declare("single_shot_spacing_rule", "theil_sen_sorted_delta", "heuristic",
                           "inter-plane distance = robust slope of sorted Δ vs rank (not mean(diff) in label order)")
SPACING_TOL = REG.declare("spacing_tolerance", 0.1, "heuristic", "warn if the fitted spacing differs from calibration.json by more")
SS_MAX_FRAMES = REG.declare("single_shot_max_frames", 10, "heuristic", "frames of a single-shot movie processed")
SS_HALF = REG.declare("single_shot_interp_half_vox", [3, 8, 8], "heuristic", "Fourier peak crop for the 8-plane volumes")
SS_Z_EDGE = REG.declare("single_shot_z_edge_planes", 2, "heuristic",
                        "movie mode: beads peaking within N planes of either end of the 8-plane volume are not used "
                        "(they are cut by the volume; the z-scan emulation picks the frame with the bead mid-volume instead)")

# ---------------------------------------------------------------- extensions (not in the paper), OFF by default
EXT_NA_ILL_FIT = REG.declare("ext_na_ill_fit", False, "user", "Extension E1: estimate NA_ill from the measured spectrum (check_04.py)")
EXT_PLANES_BF = REG.declare("ext_planes_brightfield", False, "user", "Extension E2: plane positions from brightfield bead foci")
EXT_MOTION = REG.declare("ext_motion_correction", False, "user", "Extension E3: correct lateral sample motion between z-steps")

if NA_ILL_USER is None:
    NA_ILL = inst.value("brightfield.na_ill")
    warn(f"NA_ill = {NA_ILL} is the PAPER's condenser setting (SI §2), not yours: set NA_ILL_USER to your condenser reading. "
         "It sets g_z,c and the CTF support (§2 shows whether it fits the data).")
else:
    NA_ILL = REG.declare("na_ill", NA_ILL_USER, "user", "condenser setting on this microscope")
PHI_THEORY = REG.declare("phi_theory_rad", qpi.bead_phase_theory(BEAD_D_UM, N_BEAD, N_MEDIUM, LAMBDA0_UM, BEAD_FACTOR), "derived",
                         "bead_factor · 2π/λ₀ · (n_bead − n_medium) · d (SI §4: 0.34 rad)", "rad")
GZ_C = REG.declare("gz_c", qpi.gz_cutoff(NA_ILL, BAND_UM, N_MEDIUM, GZ_C_CONVENTION), "derived", f"eq. S24 ({GZ_C_CONVENTION})", "rad/µm")

ALPHA_READ = beads.read_alpha(ALPHA_JSON)
if ALPHA_READ is None:
    ALPHA = REG.declare("alpha", 3.15, "paper", "SI §4 (staircase + AFM on THEIR microscope)",
                        note="fallback: alpha.json not found")
    ALPHA_SD = None
    warn("!!! alpha.json not found: using α = 3.15 from the PAPER (another instrument). The bead phases below are NOT a validation "
         "of this microscope's calibration. Run notebook 03 and set ALPHA_JSON. !!!")
else:
    ALPHA, ALPHA_SD = ALPHA_READ
    REG.declare("alpha", ALPHA, "measured", f"notebook 03 staircase calibration: {ALPHA_JSON}", note=f"sd {ALPHA_SD}")

assert Path(DATA_FILE).exists(), DATA_FILE
assert Path(CALIBRATION).exists(), f"{CALIBRATION}: run notebook 01 first"
OUT = output_dir(DATA_FILE, "04_beads")
def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200); plt.show()
print(f"theory 0.6·k₀Δn·d = {PHI_THEORY:.3f} rad; α = {ALPHA}; NA_det {NA_DET}, NA_ill {NA_ILL}; g_z,c = {GZ_C:.3f} rad/µm")
print("OUT:", OUT)

# %%
REG.banner()

# %% [markdown]
# ## 1. Load, camera offset, per-channel flat field (SI §6.4, eq. S28)
# Each channel is cut from its camera stream and oriented like the reference (calibration.json), the camera offset is subtracted
# (`io.to_signal`), then the stack is divided by its own reference intensity I₀(x, y) so that the background is 1 (I₀ = 1 in
# eq. S28). This equalises the channel transmissions; the relative transmissions are reported and compared with the
# fluorescence ones of calibration.json. **Heuristic** (declared): I₀ = Gaussian-smoothed z-median of the channel's own stack.

# %%
acq = io.load(DATA_FILE, inst)
cal = json.loads(Path(CALIBRATION).read_text())
planes = cal["planes"]; PL = [p["plane"] for p in planes]; P = len(planes)
REF = PL.index(cal["reference_plane"])
PX = REG.declare("pixel_um", cal["pixel_um"], "derived", f"calibration.json (notebook 01) {CALIBRATION.name}", "µm")
assert acq.z_um["A"] is not None, "a z-scan is expected (SeqType ZSTACK / z_offsets_nm)"
DZ = REG.declare("dz_um", acq.step_um() * STAGE_TO_SAMPLE_Z, "derived", "commanded stage step × stage_to_sample_z", "µm")
z_um = acq.z_um["A"] * STAGE_TO_SAMPLE_Z
n_z = len(z_um)
print(f"{P} channels, {n_z} frames, step {1e3 * DZ:.0f} nm, pixel {1e3 * PX:.1f} nm (instrument nominal "
      f"{1e3 * inst.pixel_um(OBJECTIVE):.1f} nm)")
if abs(n_z - 50) > 0 or abs(DZ - 0.2) > 0.01:
    print(f"note: the SI used 50 × 200 nm (Table S2); here {n_z} × {1e3 * DZ:.0f} nm")

stacks, flats, lamp = [], [], []
for p in planes:
    sig = io.to_signal(geometry.panel(acq.streams, p)[:n_z], OFFSET)
    f0 = beads.flat_field(sig, FLAT_SIGMA_PX, FLAT_STAT)
    lamp.append(beads.frame_means(sig))
    c = qpi.normalise(sig, f0)
    if LAMP_CORRECTION:
        c = c / lamp[-1][:, None, None]
    stacks.append(c); flats.append(f0)
ny, nx = stacks[REF].shape[1:]
trans = np.array([f.mean() for f in flats]); trans_rel = trans / trans.mean()
trans_fl = np.array([p.get("transmission", np.nan) for p in planes], float)
lamp_ptp = 100 * np.ptp(np.mean(lamp, 0))
raw_max = max(float(np.asarray(acq.streams[s][n_z // 2]).max()) for s in acq.streams)
print("relative transmission, brightfield (this stack):", dict(zip(PL, np.round(trans_rel, 3).tolist())))
print("relative transmission, fluorescence (calibration.json):", dict(zip(PL, np.round(trans_fl / np.nanmean(trans_fl), 3).tolist())))
print(f"lamp drift over the scan: {lamp_ptp:.2f} % peak-to-peak (corrected: {LAMP_CORRECTION}); max raw count {raw_max:.0f}")
if raw_max > 0.95 * 65535:
    warn("raw counts close to saturation")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1, 1]})
mos = np.concatenate([np.pad(stacks[r][n_z // 2], ((0, 0), (0, 6)), constant_values=np.nan) for r in range(P)], 1)
axes[0].imshow(mos, cmap="gray", vmin=0.95, vmax=1.05, aspect="auto")
axes[0].set_xticks([(nx + 6) * r + nx / 2 for r in range(P)]); axes[0].set_xticklabels(PL); axes[0].set_yticks([])
axes[0].set_title(f"normalised brightfield, frame {n_z // 2} (0.95–1.05)")
axes[1].bar(np.arange(P) - 0.2, trans_rel, 0.4, label="brightfield I₀")
axes[1].bar(np.arange(P) + 0.2, trans_fl / np.nanmean(trans_fl), 0.4, color="0.6", label="fluorescence (01)")
axes[1].set(xticks=range(P), xticklabels=PL, ylabel="relative transmission", ylim=(0.5, 1.3)); axes[1].legend(fontsize=8)
for r in range(P):
    axes[2].plot(z_um, 100 * (lamp[r] - 1), lw=1, color=plt.cm.viridis(r / P))
axes[2].set(xlabel="stage z (µm)", ylabel="frame mean − mean (%)", title="lamp stability")
savefig(fig, "01_channels")

# %% [markdown]
# ### 1b. Extension E3 — sample-motion correction (not in the paper; off by default)
# Lateral shift between consecutive frames by cross-correlation of the band-passed intensity, averaged per camera; the straight-line part
# (stage drift + defocus-dependent shifts, not separable) is kept, only the frame-to-frame jitter is removed, by exact Fourier
# shifts. Applied only if `EXT_MOTION` is True **and** the two cameras agree (they see the same sample).

# %%
motion = None
if EXT_MOTION:
    HP_SIGMA = REG.declare("ext_motion_highpass_px", 5, "heuristic", "E3: band-pass, high-pass Gaussian σ", "px")
    LP_SIGMA = REG.declare("ext_motion_lowpass_px", 2, "heuristic",
                           "E3: band-pass, low-pass Gaussian σ (pixel noise; broadens the correlation peak for the parabolic "
                           "sub-pixel step: 0.03 px step error in simulation vs 0.07 px at σ = 1)", "px")
    MAX_SHIFT = REG.declare("ext_motion_max_shift_px", 3, "heuristic", "E3: largest frame-to-frame shift searched", "px")
    MIN_CORR = REG.declare("ext_motion_min_corr", 0.5, "heuristic", "E3: apply only if camera A/B step correlation exceeds this")
    tr = {s: beads.frame_shifts([stacks[r] for r in range(P) if planes[r]["stream"] == s], HP_SIGMA, LP_SIGMA, MAX_SHIFT) for s in "AB"}
    jit = {s: beads.detrend(tr[s], z_um)[0] for s in tr}
    corr = [float(np.corrcoef(np.diff(jit["A"][:, a]), np.diff(jit["B"][:, a]))[0, 1]) for a in (0, 1)]
    rms = 1e3 * PX * np.std(0.5 * (jit["A"] + jit["B"]), 0)
    print(f"E3: jitter RMS x {rms[0]:.1f}, y {rms[1]:.1f} nm; camera A/B step correlation x {corr[0]:+.2f}, y {corr[1]:+.2f}")
    applied = min(corr) > MIN_CORR
    if applied:
        j = 0.5 * (jit["A"] + jit["B"])
        stacks = [beads.apply_shifts(s, j) for s in stacks]
    print("E3 applied:", applied)
    motion = {"rms_nm": rms.tolist(), "corr_A_B": corr, "applied": bool(applied)}

# %% [markdown]
# ## 2. Measured 3D spectrum vs the CTF support of eq. S19 (SI Fig. S1d,e)
# Azimuthal average of |3D FFT| of the z-mirrored (c − 1) stacks (all channels), with the support of the polychromatic CTF
# (NA_det from the instrument file, NA_ill as declared above, lamp spectrum of Fig. S8, BrightLine 582/75) and ±g_z,c.
# The number reported is the fraction of the above-noise power (|g_z| > g_z,c) inside the support (each frame's mean removed
# first: g⊥ = 0 only carries lamp drift). The retrieval keeps only
# what is inside the support: if the data extend beyond it, NA_det / NA_ill / pixel / Δz are wrong.

# %%
TF = qpi.transfer_function(NA_DET, NA_ILL, N_MEDIUM, BAND_UM, SPECTRUM, n_samples=TF_SAMPLES)
rb, zb = TF[0], TF[1]
r_edges = np.linspace(0, min(np.pi / PX, 1.6 * rb[-1]), R_BINS + 1)
gz_s, rcen, spec = beads.spectrum_rz(stacks, DZ, PX, r_edges)
supp_rz = beads.support_rz(TF, gz_s, rcen)
noise_rows = np.abs(gz_s) > NOISE_ROWS_FACTOR * np.abs(zb).max()
if noise_rows.sum() < 3:
    noise_rows = np.abs(gz_s) >= np.sort(np.abs(gz_s))[-4]
    warn("few g_z rows beyond the CTF support (Δz too large): noise floor from the highest |g_z| rows")
frac_in = beads.support_check(spec, supp_rz, gz_s, GZ_C, noise_rows)
print(f"above-noise spectral energy inside the CTF support: {100 * frac_in:.0f} %")
if not frac_in >= SUPPORT_MIN_FRACTION:
    warn(f"only {100 * frac_in:.0f} % of the measured spectrum is inside the CTF support: check NA_det, NA_ill, pixel and Δz")
fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
pos = spec[spec > 0]
ax.imshow(spec, origin="lower", aspect="auto", extent=[r_edges[0], r_edges[-1], gz_s[0], gz_s[-1]], cmap="viridis",
          norm=LogNorm(np.percentile(pos, 20), np.percentile(pos, 99.9)))
ax.contour(rcen, gz_s, supp_rz.astype(float), [0.5], colors="w", linewidths=1.2)
for s_ in (1, -1):
    ax.axhline(s_ * GZ_C, color="r", ls=":", lw=1)
ax.set(xlabel="|g⊥| (rad/µm)", ylabel="g_z (rad/µm)",
       title=f"|3D FFT| (log), white = CTF support (NA_det {NA_DET}, NA_ill {NA_ILL}); {100 * frac_in:.0f} % inside")
savefig(fig, "02_spectrum_vs_ctf")

# %% [markdown]
# ### 2b. Extension E1 — NA_ill from the measured spectrum (not in the paper; off by default)
# Near the origin the brightfield CTF fills a double cone whose opening is set by NA_ill. The measured spectrum is compared with
# |H| · |sphere FT| · exp(−(σ g⊥)²/2) for a grid of NA_ill (σ = blur, nuisance). Diagnostic only: the result is NOT used by
# the retrieval; if it disagrees with your condenser reading, check the condenser and set `NA_ILL_USER`.

# %%
na_ill_fit = None
if EXT_NA_ILL_FIT:
    GRID_ILL = REG.declare("ext_na_ill_grid", np.round(np.arange(0.10, 0.91, 0.05), 2).tolist(), "heuristic", "E1 NA_ill grid")
    GRID_SIG = REG.declare("ext_na_ill_blur_um", [0.0, 0.05, 0.1], "heuristic", "E1 blur nuisance grid", "µm")
    R_RANGE = REG.declare("ext_na_ill_r_range", [0.8, 7.0], "heuristic", "E1 fitted |g⊥| range (low-g cone)", "rad/µm")
    GZ_RANGE = REG.declare("ext_na_ill_gz_range", [0.15, 4.0], "heuristic", "E1 fitted |g_z| range", "rad/µm")
    best, sig_b, C = beads.na_ill_scan(spec, gz_s, rcen, lambda a: qpi.transfer_function(NA_DET, a, N_MEDIUM, BAND_UM, SPECTRUM,
                                                                                         n_samples=600_000),
                                       GRID_ILL, GRID_SIG, BEAD_D_UM / 2, noise_rows, R_RANGE, GZ_RANGE)
    na_ill_fit = {"best": best, "blur_um": sig_b, "cost": dict(zip(map(str, GRID_ILL), C.min(1).tolist()))}
    print(f"E1: best NA_ill = {best:.2f} (used: {NA_ILL}), blur σ {1e3 * sig_b:.0f} nm")
    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    ax.plot(GRID_ILL, C.min(1), "o-"); ax.axvline(NA_ILL, color="0.5", ls=":")
    ax.set(xlabel="NA_ill", ylabel="misfit (log amplitude)", title="E1 (extension): NA_ill from the spectrum")
    savefig(fig, "02b_ext_na_ill_fit")

# %% [markdown]
# ## 3. Phase retrieval per channel (SI §2.2, eq. S20–S28) and max projection (Fig. S3c)
# Spectral half chosen on the reference channel by `qpi.choose_sign` (**heuristic**: beads denser than the medium must come out
# positive), the same for all channels. Each channel is retrieved independently with α from notebook 03.

# %%
t0 = time.time()
SGN = REG.declare("sgn", int(qpi.choose_sign(stacks[REF], DZ, PX, TF, GZ_C, ALPHA)), "derived", "qpi.choose_sign on the reference channel")
ISO_PX = ISOLATION_UM / PX
phis, tables, noise = [], [], []
for r in range(P):
    G = qpi.cross_spectral_density(stacks[r], DZ, PX, TF, SGN, GZ_C)
    ph = qpi.phase(G, ALPHA)
    xy = beads.detect(ph.max(0), DETECT_K_SIGMA, FOOTPRINT_PX, EDGE_PX)
    tables.append(beads.measure_beads(ph, xy, SEARCH_PX, PEAK_ESTIMATOR, ISO_PX, Z_EDGE_FRAMES, INTERP_HALF, LOCATE_SIGMA, G, ALPHA))
    sl = ph[n_z // 2]; noise.append(1.4826 * np.median(np.abs(sl - np.median(sl))))
    phis.append(ph); del G
print(f"spectral half {SGN:+d}; retrieval of {P} channels in {time.time() - t0:.0f} s; background phase noise per voxel (mrad): "
      + ", ".join(f"{PL[r]} {1e3 * noise[r]:.1f}" for r in range(P)))

mp = phis[REF].max(0); kz = phis[REF].argmax(0)
v_ = np.clip(mp / max(np.percentile(mp, 99.7), 1e-6), 0, 1) ** 0.7
rgb = plt.cm.turbo(kz / (n_z - 1))[..., :3] * v_[..., None]
tr = tables[REF]
yA = int(np.median(tr["y"])) if len(tr["y"]) else ny // 2
fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.4]})
axes[0].imshow(rgb, extent=[0, nx * PX, ny * PX, 0]); axes[0].axhline(yA * PX, color="w", lw=0.8, ls="--")
axes[0].set(xlabel="x (µm)", ylabel="y (µm)", title=f"{PL[REF]}: phase max projection, colour = depth")
fig.colorbar(plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(z_um[0], z_um[-1])), ax=axes[0], shrink=0.6, label="stage z (µm)")
axes[1].imshow(phis[REF][:, max(0, yA - 2):yA + 3, :].max(1), cmap="magma", vmin=0, vmax=max(np.percentile(mp, 99.9), 1e-3),
               extent=[0, nx * PX, z_um[-1], z_um[0]], aspect="equal")
axes[1].set(xlabel="x (µm)", ylabel="stage z (µm)", title="orthogonal slice A–B (max over ±2 px)")
savefig(fig, "03_max_projection")

# %% [markdown]
# ## 4. Bead peak phase vs theory and the SI (SI §4, Fig. S4e)
# Per bead: the peak of the retrieved phase (declared estimator). Used beads: isolated (> `isolation_um`) and focused at least
# `z_edge_frames` from the stack ends. Reported as in the SI (**mean ± SD over beads**) and as median [IQR]; per channel and
# pooled. A straight-line fit of the channel means vs plane index tests the P1 → P8 drop seen in earlier data.
#
# Diagnostic (not a calibration): the α that would make eq. S28 return the theory value at each bead, from the measured
# Re Γ₊ and Im Γ₊ at its peak voxel (the SI §4 inversion of S28, applied to beads).

# %%
used = [t["used"] for t in tables]
peak = [t["peak"][u] for t, u in zip(tables, used)]
st = [beads.stats(p) for p in peak]
pooled = beads.stats(np.concatenate(peak))
plane_idx = np.array([int("".join(ch for ch in lab if ch.isdigit()) or r) for r, lab in enumerate(PL)])
delta_fl = np.array([p["delta_z_um"] for p in planes], float)
tr_idx = beads.trend(plane_idx, [s["mean"] for s in st], [s["sem"] for s in st])
tr_delta = beads.trend(delta_fl, [s["mean"] for s in st], [s["sem"] for s in st])
a_imp = [beads.alpha_from_beads(PHI_THEORY, t["re_g"][u], t["im_g"][u]) for t, u in zip(tables, used)]
print(f"theory {PHI_THEORY:.3f} rad; SI {PHI_PAPER[0]} ± {PHI_PAPER[1]} rad (mean ± SD of {PHI_PAPER[2]} beads); α = {ALPHA}")
print("channel  detected  used   mean ± SD (rad)     median [IQR]            implied α (median)")
for r in range(P):
    s = st[r]
    print(f"{PL[r]:6s}   {len(tables[r]['x']):5d}  {s['n']:5d}   {s['mean']:.3f} ± {s['sd']:.3f}     "
          f"{s['median']:.3f} [{s['q25']:.3f}–{s['q75']:.3f}]     {np.nanmedian(a_imp[r]) if len(a_imp[r]) else np.nan:.2f}")
print(f"pooled   {sum(len(t['x']) for t in tables):5d}  {pooled['n']:5d}   {pooled['mean']:.3f} ± {pooled['sd']:.3f}     "
      f"{pooled['median']:.3f} [{pooled['q25']:.3f}–{pooled['q75']:.3f}]")
print(f"reference channel {PL[REF]}: {st[REF]['mean']:.3f} ± {st[REF]['sd']:.3f} rad (n = {st[REF]['n']}) vs theory {PHI_THEORY:.3f}, "
      f"SI {PHI_PAPER[0]} ± {PHI_PAPER[1]}")
print(f"trend of channel means vs plane index: {1e3 * tr_idx['slope']:+.1f} ± {1e3 * tr_idx['slope_se']:.1f} mrad/plane; "
      f"{tr_idx['n_decreasing_steps']}/{tr_idx['n_steps']} consecutive steps decrease; vs Δ: "
      f"{1e3 * tr_delta['slope']:+.1f} ± {1e3 * tr_delta['slope_se']:.1f} mrad/µm")
if tr_idx["n_steps"] and (tr_idx["n_decreasing_steps"] == tr_idx["n_steps"] or tr_idx["slope"] < -3 * tr_idx["slope_se"]):
    warn(f"bead phase drops from {PL[0]} to {PL[-1]} ({1e3 * tr_idx['slope']:+.1f} ± {1e3 * tr_idx['slope_se']:.1f} mrad/plane): "
         "the phase depends on the channel (aberrations / transmission / focus position), not only on α")
if abs(pooled["mean"] - PHI_THEORY) > 0.15 * PHI_THEORY:
    warn(f"pooled bead phase {pooled['mean']:.3f} rad differs from theory {PHI_THEORY:.3f} by more than 15 %")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.3), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
rng_ = np.random.default_rng(0)
for r in range(P):
    axes[0].plot(r + (rng_.random(len(peak[r])) - 0.5) * 0.35, peak[r], "o", ms=3, alpha=0.4, color=f"C{0 if planes[r]['stream'] == 'A' else 1}")
    axes[0].errorbar(r + 0.3, st[r]["mean"], st[r]["sd"], fmt="ks", ms=4, capsize=3)
    axes[0].plot([r - 0.2, r + 0.2], [st[r]["median"]] * 2, "k-", lw=2)
axes[0].axhline(PHI_THEORY, color="g", lw=1.5, label=f"theory {PHI_THEORY:.2f} rad")
axes[0].axhspan(PHI_PAPER[0] - PHI_PAPER[1], PHI_PAPER[0] + PHI_PAPER[1], color="0.6", alpha=0.3, label="SI 0.33 ± 0.014")
axes[0].set(xticks=range(P), xticklabels=PL, ylabel="bead peak phase (rad)",
            title=f"per channel (bar = median, square = mean ± SD; α = {ALPHA})"); axes[0].legend(fontsize=8)
for r in range(P):
    t, u = tables[r], used[r]
    axes[1].plot(z_um[0] + DZ * t["k"][u], t["peak"][u], "o", ms=3, alpha=0.5, color=plt.cm.viridis(r / P))
axes[1].axhline(PHI_THEORY, color="g")
axes[1].set(xlabel="stage z of the bead focus (µm)", ylabel="peak phase (rad)", title="vs depth in the scan")
savefig(fig, "04_bead_phase")

# %% [markdown]
# ## 5. Bead-averaged xz intensity and phase vs the SI forward model (SI §3.2–3.3, Fig. S3a,b)
# Used beads of the declared channel(s) are re-centred on their sub-voxel peak by **exact Fourier shifts** (no interpolation
# blur) and averaged. The forward model of SI §3.2 (Dirac point scatterer ⊗ H → Γ, I = 1 + 2 Re Γ, retrieved with the same
# code, CTF, g_z,c, sign and α) gives the theory curves. **Normalisation**: the SI states none; the simulated phase is scaled so
# that its peak equals the measured average peak (display only), so only the FWHMs (x, y, z) are model-independent numbers.

# %%
HZ, HXY = int(round(AVG_HALF_UM[0] / DZ)), int(round(AVG_HALF_UM[1] / PX))
avg_ch = [REF] if AVG_CHANNELS == "reference" else list(range(P))
acc_p, acc_i, n_avg = [], [], 0
for r in avg_ch:
    t, u = tables[r], used[r]
    cen = np.c_[t["k"], t["y"], t["x"]][u]
    a_p, n_p, _ = beads.average_beads(phis[r], cen, (HZ, HXY, HXY), AVG_MARGIN)
    a_i, _, _ = beads.average_beads(stacks[r] - 1, cen, (HZ, HXY, HXY), AVG_MARGIN)
    if n_p:
        acc_p.append(a_p * n_p); acc_i.append(a_i * n_p); n_avg += n_p
if n_avg == 0:
    raise RuntimeError("no bead available for the average: check the selection parameters")
avg_phi, avg_int = np.sum(acc_p, 0) / n_avg, np.sum(acc_i, 0) / n_avg

sims = {}
for obj in ["point", "sphere"] if SIM_SPHERE_TOO else [SIM_OBJECT]:
    n_s = SIM_LATERAL
    I_s = beads.simulate_stack(TF, (n_z, n_s, n_s), DZ, PX, [[(n_z // 2) * DZ, (n_s // 2) * PX, (n_s // 2) * PX]], SGN,
                               1e-3, obj=obj, d_um=BEAD_D_UM)
    ph_s = qpi.phase(qpi.cross_spectral_density(I_s, DZ, PX, TF, SGN, GZ_C), ALPHA)
    k_, y_, x_ = np.unravel_index(np.argmax(ph_s), ph_s.shape)
    sl = (slice(k_ - HZ, k_ + HZ + 1), slice(y_ - HXY, y_ + HXY + 1), slice(x_ - HXY, x_ + HXY + 1))
    scale = avg_phi.max() / ph_s[sl].max()
    sims[obj] = ((I_s[sl] - 1) / np.abs(I_s[sl] - 1).max() * np.abs(avg_int).max(), ph_s[sl] * scale, scale)
fw = {"experiment": beads.fwhm_xyz(avg_phi, DZ, PX)}
fw.update({f"model ({k})": beads.fwhm_xyz(v[1], DZ, PX) for k, v in sims.items()})
print(f"averaged beads: {n_avg} (channels {[PL[r] for r in avg_ch]}); average peak {avg_phi.max():.3f} rad")
print("phase FWHM (nm)            x      y      z")
for k, f in fw.items():
    print(f"  {k:22s} {1e3 * f['x']:5.0f}  {1e3 * f['y']:5.0f}  {1e3 * f['z']:5.0f}")
print(f"  SI Fig. S3b theory       {1e3 * FWHM_PAPER['lateral']['theory']:5.0f}         {1e3 * FWHM_PAPER['axial']['theory']:5.0f}")
print(f"  SI Fig. S3b experiment   {1e3 * FWHM_PAPER['lateral']['experiment']:5.0f}         {1e3 * FWHM_PAPER['axial']['experiment']:5.0f}")

panels_ = [("experiment", avg_int, avg_phi)] + [(f"model ({k}, scaled)", v[0], v[1]) for k, v in sims.items()]
fig, axes = plt.subplots(2, len(panels_), figsize=(4.2 * len(panels_), 7.5), layout="constrained", squeeze=False)
ext = [-HXY * PX, HXY * PX, HZ * DZ, -HZ * DZ]
for i, (name, Iv, Pv) in enumerate(panels_):
    m_ = max(np.abs(Iv[:, HXY, :]).max(), 1e-9)
    axes[0, i].imshow(Iv[:, HXY, :] / m_, cmap="RdBu_r", vmin=-1, vmax=1, extent=ext, aspect="equal")
    axes[0, i].set_title(f"{name}\nintensity − 1 (normalised)", fontsize=9)
    axes[1, i].imshow(Pv[:, HXY, :], cmap="magma", vmin=0, vmax=avg_phi.max(), extent=ext, aspect="equal")
    axes[1, i].set_title(f"phase (0 – {avg_phi.max():.2f} rad)", fontsize=9); axes[1, i].set_xlabel("x (µm)")
axes[0, 0].set_ylabel("defocus (µm)"); axes[1, 0].set_ylabel("defocus (µm)")
fig.suptitle(f"bead-averaged xz sections ({n_avg} beads) vs SI §3.2 model (model scaled to the measured peak: shapes only)")
savefig(fig, "05a_bead_xz")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
xl, zl = PX * np.arange(-HXY, HXY + 1), DZ * np.arange(-HZ, HZ + 1)
for name, _, Pv in panels_:
    st_ = dict(color="k", marker="o", ms=4) if name == "experiment" else dict(lw=2, ls="--")
    axes[0].plot(xl, Pv[HZ, HXY, :], label=name, **st_)
    axes[1].plot(xl, Pv[HZ, :, HXY], label=name, **st_)
    axes[2].plot(zl, Pv[:, HXY, HXY], label=name, **st_)
for ax, lab in zip(axes, ("x (µm)", "y (µm)", "defocus (µm)")):
    ax.set(xlabel=lab, ylabel="phase (rad)")
axes[0].set_title(f"x — SI: theory {FWHM_PAPER['lateral']['theory']}, exp. {FWHM_PAPER['lateral']['experiment']} µm FWHM")
axes[1].set_title("y")
axes[2].set_title(f"z — SI: theory {FWHM_PAPER['axial']['theory']}, exp. {FWHM_PAPER['axial']['experiment']} µm FWHM")
axes[0].legend(fontsize=8)
savefig(fig, "05b_bead_profiles")

# %% [markdown]
# ### 5b. Model check: simulated peak phase vs where the bead's focus sits in the stack (diagnostic)
# The z-mirroring of §2.2 assumes the stack continues symmetrically; beads focused near the ends are distorted. Same forward
# model and retrieval as above, bead focus moved through the stack. This justifies (or not) `z_edge_frames`.

# %%
edge_check = {}
for kf in sorted({1, 2, 3, Z_EDGE_FRAMES, Z_EDGE_FRAMES + 2, n_z // 2}):
    I_s = beads.simulate_stack(TF, (n_z, 48, 48), DZ, PX, [[kf * DZ, 24 * PX, 24 * PX]], SGN, 1e-3, obj=SIM_OBJECT, d_um=BEAD_D_UM)
    edge_check[kf] = float(qpi.phase(qpi.cross_spectral_density(I_s, DZ, PX, TF, SGN, GZ_C), ALPHA).max())
mid_ = edge_check[n_z // 2]
print("focus frame: simulated peak / peak at mid-stack  " + "  ".join(f"{k}: {v / mid_:.2f}" for k, v in edge_check.items()))
if abs(edge_check[Z_EDGE_FRAMES] / mid_ - 1) > 0.05:
    warn(f"beads at the z-edge limit ({Z_EDGE_FRAMES} frames) lose {100 * (1 - edge_check[Z_EDGE_FRAMES] / mid_):.0f} % in the model: "
         "increase z_edge_frames")

# %% [markdown]
# ## 6. Single-shot 8-plane mode (SI §5, §6.4)
# Each plane of one camera frame pair is normalised by its flat field, warped onto the reference with the calibration.json
# affine (`geometry.warp`, bilinear, SI §6.4), and the 8 planes are ordered by Δ (declared convention: the volume runs like the
# z-scan stage, so the spectral half of §3 applies). The inter-plane distance is the **robust slope of sorted Δ vs rank**
# (Theil–Sen), not mean(diff(Δ)) in label order. The 8-plane volume is retrieved with the same CTF, g_z,c, sign and α.
#
# Without a single-shot movie, single shots are taken from the z-scan itself: every z-scan frame pair *is* an 8-plane
# single shot. For each used bead of the reference channel, the frame where it sits mid-way through the 8 planes is
# retrieved and the bead phase is compared with its z-scan value (paired). With a movie, beads are detected in each volume and
# the distribution is compared with the z-scan.

# %%
fit = beads.plane_spacing_fit(delta_fl)
DZ_MP = REG.declare("dz_single_shot_um", fit["spacing"], "derived", "Theil–Sen slope of sorted calibration Δ vs rank", "µm")
sp_cal = cal.get("plane_spacing_um", {}).get("mean")
order_lab = [PL[i] for i in fit["order"]]
print(f"planes by increasing Δ: {order_lab}; spacing {1e3 * DZ_MP:.0f} nm (calibration.json mean "
      f"{'n/a' if sp_cal is None else f'{1e3 * sp_cal:.0f} nm'}; SI §6.2: 347 ± 11 nm); residuals "
      + ", ".join(f"{1e3 * v:+.0f}" for v in fit["residuals"]) + " nm")
if order_lab != PL and order_lab != PL[::-1]:
    print("note: Δ order differs from the label order")
if sp_cal is not None and abs(DZ_MP / sp_cal - 1) > SPACING_TOL:
    warn(f"robust spacing {1e3 * DZ_MP:.0f} nm differs from calibration.json mean {1e3 * sp_cal:.0f} nm by > {100 * SPACING_TOL:.0f} %")
if np.max(np.abs(fit["residuals"])) > 0.25 * DZ_MP:
    warn("the planes are not evenly spaced (residual > 25 % of the spacing): the 8-plane FFT assumes even spacing")
REF_AFF = np.asarray(planes[REF]["affine_from_reference"], float)
if np.abs(REF_AFF - np.eye(2, 3)).max() > 1e-6:
    warn("the reference plane's affine is not the identity: z-scan bead positions are mapped into reference coordinates")
print("Nyquist note (SI eq. S29): planes needed for full overlap removal ≈ "
      f"{1 + np.pi / (DZ_MP * GZ_C):.0f} at this spacing; the single shot has {P}")

def retrieve_single(vol):
    G8 = qpi.cross_spectral_density(vol, DZ_MP, PX, TF, SGN, GZ_C)
    return qpi.phase(G8, ALPHA), G8

single = {"mode": None}
if SINGLE_SHOT_FILE is None:
    single["mode"] = "z-scan frames"
    t = tables[REF]; u = np.flatnonzero(used[REF])
    xy_ref = beads.panel_to_ref(REF_AFF, np.c_[t["x"], t["y"]]) if len(t["x"]) else np.zeros((0, 2))
    cache, rows = {}, []
    for i in u:
        k = beads.single_shot_frame(t["k"][i], delta_fl[REF], delta_fl, DZ, DELTA_CONVENTION)
        if not 0 <= k < n_z:
            continue
        if k not in cache:
            sig = {s: io.to_signal(acq.streams[s][k], OFFSET) for s in acq.streams}
            vol, labels, valid = beads.assemble_single_shot(sig, planes, flats, DELTA_CONVENTION)
            cache[k] = (*retrieve_single(vol), valid, labels, vol)
        ph8, G8, valid, labels, _ = cache[k]
        x, y = (int(round(v)) for v in xy_ref[i])
        y0, x0 = max(0, y - SEARCH_PX), max(0, x - SEARCH_PX)
        sub = ph8[:, y0:y + SEARCH_PX + 1, x0:x + SEARCH_PX + 1]
        if sub.size == 0 or not valid[min(max(y, 0), ny - 1), min(max(x, 0), nx - 1)]:
            continue
        kk, yy, xx = np.unravel_index(np.argmax(sub), sub.shape)
        v8 = beads.peak_fourier(ph8, kk, y0 + yy, x0 + xx, SS_HALF, locate_sigma=LOCATE_SIGMA, field=G8, alpha=ALPHA)
        rows.append([t["peak"][i], v8[0] if PEAK_ESTIMATOR == "fourier" else float(sub.max()), k, kk])
    rows = np.array(rows).reshape(-1, 4)
    sgn8 = int(qpi.choose_sign(next(iter(cache.values()))[4], DZ_MP, PX, TF, GZ_C, ALPHA)) if cache else None
    single.update(n_beads=len(rows), n_volumes=len(cache), zscan=beads.stats(rows[:, 0]), single_shot=beads.stats(rows[:, 1]),
                  ratio=beads.stats(rows[:, 1] / rows[:, 0]) if len(rows) else None, sign_check=sgn8)
    ss_vals, zs_vals = rows[:, 1], rows[:, 0]
else:
    single["mode"] = "movie"
    mov = io.load(data_path(SINGLE_SHOT_FILE), inst)
    n_t = min(mov.n_frames, SS_MAX_FRAMES)
    sig0 = {s: io.to_signal(mov.streams[s][:n_t], OFFSET) for s in mov.streams}
    flats_mv = [beads.flat_field(geometry.panel(sig0, p), FLAT_SIGMA_PX, FLAT_STAT) for p in planes]   # the movie's own I₀
    vals = []
    for k in range(n_t):
        vol, labels, valid = beads.assemble_single_shot({s: v[k] for s, v in sig0.items()}, planes, flats_mv, DELTA_CONVENTION)
        ph8, G8 = retrieve_single(vol)
        mp8 = np.where(valid, ph8.max(0), np.median(ph8.max(0)))
        xy = beads.detect(mp8, DETECT_K_SIGMA, FOOTPRINT_PX, EDGE_PX)
        t8 = beads.measure_beads(ph8, xy, SEARCH_PX, PEAK_ESTIMATOR, ISO_PX, SS_Z_EDGE, SS_HALF, LOCATE_SIGMA, G8, ALPHA)
        vals.append(t8["peak"][t8["used"]])
    ss_vals = np.concatenate(vals) if vals else np.zeros(0)
    zs_vals = peak[REF]
    sgn8 = int(qpi.choose_sign(vol, DZ_MP, PX, TF, GZ_C, ALPHA))
    single.update(n_frames=n_t, n_beads=len(ss_vals), zscan=beads.stats(zs_vals), single_shot=beads.stats(ss_vals), sign_check=sgn8,
                  note="different beads in the two measurements: distributions compared, not pairs")
print(f"single shot ({single['mode']}): {single['n_beads']} beads; z-scan {single['zscan']['mean']:.3f} ± {single['zscan']['sd']:.3f} rad, "
      f"8 planes {single['single_shot']['mean']:.3f} ± {single['single_shot']['sd']:.3f} rad (medians {single['zscan']['median']:.3f} / "
      f"{single['single_shot']['median']:.3f}); SI §5: {PHI_PAPER_8PLANE} rad")
if single.get("ratio"):
    print(f"paired ratio single-shot / z-scan: {single['ratio']['mean']:.2f} ± {single['ratio']['sd']:.2f} (median {single['ratio']['median']:.2f})")
if single["sign_check"] is not None and single["sign_check"] != SGN:
    warn(f"choose_sign on an 8-plane volume gives {single['sign_check']:+d}, the z-scan {SGN:+d}: check delta_convention")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
if single["mode"] == "z-scan frames" and len(ss_vals):
    lim_ = max(0.45, 1.05 * np.nanmax(np.r_[ss_vals, zs_vals]))
    axes[0].plot(zs_vals, ss_vals, "o", ms=4, alpha=0.6, color="k"); axes[0].plot([0, lim_], [0, lim_], "k:")
    axes[0].set(xlabel="z-scan phase (rad)", ylabel="single-shot 8-plane phase (rad)", xlim=(0, lim_), ylim=(0, lim_), aspect="equal",
                title="same beads")
bins = np.linspace(0, max(0.5, np.nanmax(np.r_[ss_vals, zs_vals, 0.0]) * 1.05), 30)
axes[1].hist(zs_vals, bins, alpha=0.5, label=f"z-scan {PL[REF]}"); axes[1].hist(ss_vals, bins, alpha=0.5, label="single shot")
axes[1].axvline(PHI_THEORY, color="g"); axes[1].axvline(PHI_PAPER_8PLANE, color="0.4", ls="--", label="SI §5: 0.33")
axes[1].set(xlabel="bead peak phase (rad)", ylabel="beads"); axes[1].legend(fontsize=8)
savefig(fig, "06_single_shot")

# %% [markdown]
# ## 7. Extension E2 — brightfield plane positions vs fluorescence Δ (not in the paper; off by default)
# The SI measures Δ with fluorescent beads only (§6.2). Here the brightfield bead foci of all channels are matched through the
# calibration affines and solved as z_b + Δ_p. Diagnostic: it never replaces calibration.json. Different wavelength and
# sample from the fluorescence calibration, so differences are expected (chromatic focal shift, index).

# %%
planes_bf = None
if EXT_PLANES_BF:
    MATCH_PX = REG.declare("ext_planes_match_px", 3.0, "heuristic", "E2: match radius in reference coordinates", "px")
    MIN_CH = REG.declare("ext_planes_min_channels", 6, "heuristic", "E2: beads seen in at least this many channels")
    CLIP = REG.declare("ext_planes_clip_mad", 4.0, "heuristic", "E2: one outlier clip (robust σ)")
    tabs_used = [{k: v[u] for k, v in t.items()} for t, u in zip(tables, used)]
    d_bf, n_bf = beads.plane_positions_brightfield(tabs_used, planes, REF, DZ, MATCH_PX, MIN_CH, CLIP)
    d_fl = delta_fl - delta_fl[REF]
    for r in range(P):
        print(f"{PL[r]}: Δ brightfield {1e3 * d_bf[r]:+6.0f} nm, fluorescence {1e3 * d_fl[r]:+6.0f} nm, difference {1e3 * (d_bf[r] - d_fl[r]):+5.0f}")
    print(f"E2: {n_bf} beads; robust spacing brightfield {1e3 * beads.plane_spacing_fit(d_bf)['spacing']:.0f} nm vs fluorescence {1e3 * DZ_MP:.0f} nm")
    planes_bf = {"n_beads": n_bf, "delta_bf_um": dict(zip(PL, d_bf.tolist())), "delta_fl_um": dict(zip(PL, d_fl.tolist()))}

# %% [markdown]
# ## 8. Output

# %%
result = {
    "data_file": str(DATA_FILE), "calibration": str(CALIBRATION), "alpha_source": "notebook 03" if ALPHA_READ else "paper fallback (3.15)",
    "alpha": ALPHA, "alpha_sd": ALPHA_SD, "na_det": NA_DET, "na_ill": NA_ILL, "na_ill_is_users_setting": NA_ILL_USER is not None,
    "gz_c_rad_per_um": GZ_C, "sgn": SGN, "dz_um": DZ, "pixel_um": PX, "n_frames": n_z,
    "channel_transmission_brightfield": dict(zip(PL, trans_rel.tolist())), "lamp_drift_percent_ptp": lamp_ptp,
    "spectrum_fraction_inside_ctf": frac_in, "phase_noise_rad": dict(zip(PL, np.asarray(noise, float).tolist())),
    "theory_rad": PHI_THEORY, "si_rad": {"mean": PHI_PAPER[0], "sd": PHI_PAPER[1], "n": PHI_PAPER[2]},
    "bead_phase_per_channel": dict(zip(PL, st)), "bead_phase_pooled": pooled,
    "bead_phase_reference_channel": st[REF], "trend_vs_plane_index": tr_idx, "trend_vs_delta": tr_delta,
    "implied_alpha_median_per_channel (diagnostic)": dict(zip(PL, [float(np.nanmedian(a)) if len(a) else None for a in a_imp])),
    "bead_average": {"n_beads": n_avg, "channels": [PL[r] for r in avg_ch], "peak_rad": float(avg_phi.max()),
                     "fwhm_um": fw, "fwhm_si_um": FWHM_PAPER, "model_normalisation": SIM_NORMALISATION},
    "edge_check_sim_peak_rad": {str(k): v for k, v in edge_check.items()},
    "single_shot": {**single, "spacing_um": DZ_MP, "plane_order_increasing_delta": order_lab,
                    "spacing_residuals_um": fit["residuals"].tolist(), "spacing_calibration_um": sp_cal},
    "beads": {PL[r]: {k: v.tolist() for k, v in tables[r].items()} for r in range(P)},
    "extensions": {"E1_na_ill_fit": na_ill_fit, "E2_planes_brightfield": planes_bf, "E3_motion": motion},
    "warnings": WARNINGS,
}
write_json(OUT / "beads.json", result, "04_ps_beads_phase",
           {"data_file": DATA_FILE, "calibration": CALIBRATION, "alpha_json": ALPHA_JSON if ALPHA_READ else None,
            "single_shot_file": SINGLE_SHOT_FILE})
print("wrote", OUT / "beads.json")
print(f"{len(WARNINGS)} warning(s):")
for w_ in WARNINGS:
    print(" -", w_)
